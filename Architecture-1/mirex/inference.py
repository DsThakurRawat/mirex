"""
MusicScope-CL — Inference Pipeline (MIREX 2026 Submission)

End-to-end scoring: loads a test audio → extracts Branch A (surface) +
Branch B (structural narrative) → fuses → classifies → outputs P(AI).

Supports Multiple Instance Learning (MIL): slices each track into
multiple 30s chunks, scores independently, and averages for a more robust
final prediction.

Usage:
    # Score a single file
    python inference.py --input track.wav --simclr_ckpt checkpoints/simclr_best.pt \\
                        --fusion_ckpt checkpoints/fusion_best.pt

    # Score a directory (MIREX submission mode)
    python inference.py --input_dir /data/input --output_csv /data/output/scores.csv \\
                        --simclr_ckpt checkpoints/simclr_best.pt \\
                        --fusion_ckpt checkpoints/fusion_best.pt

    # With SupCon head
    python inference.py --input_dir /data/input --output_csv scores.csv \\
                        --simclr_ckpt checkpoints/simclr_best.pt \\
                        --supcon_ckpt checkpoints/supcon_best.pt \\
                        --fusion_ckpt checkpoints/fusion_best.pt
"""
import argparse
import csv
import time
from pathlib import Path

import librosa
import numpy as np
import torch

import config
from track_b_style import StyleEncoder
from track_a_narrative import (extract_narrative_vector, get_feature_columns,
                                feats_to_vector)
from fusion_model import FusionMLP, TemperatureScaler
from train_supcon import SupConProjectionHead
from dataset import load_audio_chunk, audio_to_logmel, discover_audio_files


class MusicScopeCLScorer:
    """End-to-end scorer combining both branches + fusion head.

    Loads all checkpoints and provides a single `score(path) -> float`
    method returning P(AI-generated) in [0, 1].
    """

    def __init__(self, simclr_ckpt: str, fusion_ckpt: str,
                 supcon_ckpt: str | None = None,
                 temp_scaler_ckpt: str | None = None,
                 device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[Scorer] Device: {self.device}")

        # --- Branch A: Surface encoder (frozen SimCLR backbone) ---
        self.encoder = StyleEncoder(pretrained=False).to(self.device)
        ckpt = torch.load(simclr_ckpt, map_location=self.device,
                          weights_only=True)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.encoder.eval()

        # --- Branch B: Narrative feature extractor (no model to load) ---
        # We'll extract features on-the-fly and cache the column order
        self._feature_cols = None

        # --- SupCon projection head (optional) ---
        self.supcon_head = None
        if supcon_ckpt and Path(supcon_ckpt).exists():
            self.supcon_head = SupConProjectionHead().to(self.device)
            sc = torch.load(supcon_ckpt, map_location=self.device,
                            weights_only=True)
            self.supcon_head.load_state_dict(sc["proj_head_state"])
            self.supcon_head.eval()
            print(f"[Scorer] SupCon head loaded from {supcon_ckpt}")

        # --- Fusion classifier head ---
        fc = torch.load(fusion_ckpt, map_location=self.device,
                        weights_only=True)
        in_dim = fc.get("in_dim", config.FUSED_DIM)
        self.fusion = FusionMLP(in_dim=in_dim).to(self.device)
        self.fusion.load_state_dict(fc["model_state"])
        self.fusion.eval()

        # --- Temperature scaler (optional) ---
        self.scaler = None
        if temp_scaler_ckpt and Path(temp_scaler_ckpt).exists():
            self.scaler = TemperatureScaler().to(self.device)
            ts = torch.load(temp_scaler_ckpt, map_location=self.device,
                            weights_only=True)
            self.scaler.load_state_dict(ts["scaler_state"])
            self.scaler.eval()
        elif fc.get("scaler_state"):
            self.scaler = TemperatureScaler().to(self.device)
            self.scaler.load_state_dict(fc["scaler_state"])
            self.scaler.eval()

    def _get_narrative_vec(self, path: str) -> np.ndarray:
        """Extract the 128-dim structural narrative vector."""
        feats = extract_narrative_vector(path)
        if self._feature_cols is None:
            self._feature_cols = get_feature_columns(feats)
        return feats_to_vector(feats, self._feature_cols)

    def _get_surface_emb(self, y: np.ndarray) -> np.ndarray:
        """Extract the 576-dim surface embedding from audio waveform."""
        log_mel = audio_to_logmel(y)
        spec = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)  # (1,1,M,T)
        spec = spec.to(self.device)
        with torch.no_grad():
            emb = self.encoder(spec)
        return emb.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def score_chunk(self, path: str, offset: float = 0.0) -> float:
        """Score a single 30s chunk. Returns P(AI) in [0, 1]."""
        # Branch B: Structural
        narr_vec = self._get_narrative_vec(path)

        # Branch A: Surface (load specific chunk)
        y = load_audio_chunk(path, offset=offset)
        surface_emb = self._get_surface_emb(y)

        # Fuse
        fused = np.concatenate([surface_emb, narr_vec])
        fused_t = torch.from_numpy(fused).unsqueeze(0).to(self.device)

        # Optional SupCon projection (not used for final classification,
        # but the fusion head may expect SupCon-shaped input if trained that way)
        if self.supcon_head is not None:
            fused_t = self.supcon_head(fused_t)

        # Classify
        logit = self.fusion(fused_t)

        # Optional temperature scaling
        if self.scaler is not None:
            logit = self.scaler(logit)

        prob = torch.sigmoid(logit).item()
        return prob

    def score(self, path: str, n_chunks: int = config.MIL_CHUNKS) -> float:
        """Score a track using Multiple Instance Learning (MIL).

        Slices the track into n_chunks 30s segments, scores each
        independently, and returns the averaged probability.
        """
        try:
            total_duration = librosa.get_duration(path=str(path))
        except Exception:
            total_duration = config.CHUNK_SECONDS

        if total_duration <= config.CHUNK_SECONDS or n_chunks <= 1:
            return self.score_chunk(path, offset=0.0)

        # Evenly space chunks across the track
        max_start = total_duration - config.CHUNK_SECONDS
        offsets = np.linspace(0, max(0, max_start), n_chunks)
        scores = [self.score_chunk(path, offset=off) for off in offsets]
        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# CLI — MIREX-compatible entrypoint
# ---------------------------------------------------------------------------

def score_directory(scorer: MusicScopeCLScorer, input_dir: str,
                    output_csv: str, n_chunks: int = config.MIL_CHUNKS):
    """Score all audio files in a directory and write results to CSV."""
    root = Path(input_dir)
    files = discover_audio_files(root)
    print(f"[Inference] Scoring {len(files)} files from {root}")

    results = []
    for i, fpath in enumerate(files):
        t0 = time.time()
        try:
            prob = scorer.score(str(fpath), n_chunks=n_chunks)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] ERROR {fpath.name}: {e}")
            prob = config.FALLBACK_SCORE
        elapsed = time.time() - t0
        results.append({"filename": fpath.name, "score": prob})
        print(f"  [{i+1}/{len(files)}] {fpath.name}: "
              f"P(AI)={prob:.4f} ({elapsed:.1f}s)")

    # Write CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "score"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[Inference] Wrote {len(results)} scores to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="MusicScope-CL inference (MIREX 2026 submission)")
    parser.add_argument("--input", type=str, help="Single audio file to score")
    parser.add_argument("--input_dir", type=str,
                        help="Directory of audio files to score")
    parser.add_argument("--output_csv", type=str, default="scores.csv",
                        help="Output CSV path")
    parser.add_argument("--simclr_ckpt", required=True,
                        help="SimCLR encoder checkpoint")
    parser.add_argument("--supcon_ckpt", default=None,
                        help="SupCon projection head checkpoint (optional)")
    parser.add_argument("--fusion_ckpt", required=True,
                        help="Fusion MLP classifier checkpoint")
    parser.add_argument("--n_chunks", type=int, default=config.MIL_CHUNKS,
                        help="Number of MIL chunks per track")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    scorer = MusicScopeCLScorer(
        simclr_ckpt=args.simclr_ckpt,
        fusion_ckpt=args.fusion_ckpt,
        supcon_ckpt=args.supcon_ckpt,
        device=args.device,
    )

    if args.input:
        prob = scorer.score(args.input, n_chunks=args.n_chunks)
        print(f"\n  {args.input}: P(AI-generated) = {prob:.4f}")
    elif args.input_dir:
        score_directory(scorer, args.input_dir, args.output_csv, args.n_chunks)
    else:
        parser.error("Provide --input (single file) or --input_dir (directory)")


if __name__ == "__main__":
    main()
