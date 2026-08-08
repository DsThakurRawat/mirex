"""MIREX submission entrypoint (plan §10).

  python src/inference.py --input_dir /data/test --output_csv /data/out.csv \
      [--mode ensemble|rank_average|single:a|tta] [--device cuda]

Contract: directory of WAVs (44.1/48 kHz, mono/stereo) in, CSV out with
`track_id,ai_generated_score` (score in [0,1], track_id = filename stem).
Fully offline (set HF_HUB_OFFLINE=1 in the container). Per-track hard timeout
with config.FALLBACK_SCORE fallback so the >5%-failure exclusion rule can
never trigger. Two-pass design: per-branch scoring first, fusion second (the
rank-average mode needs the whole score matrix).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                             # noqa: E402
from aggregate import aggregate_chunks                    # noqa: E402
from datasets import load_audio                           # noqa: E402
from fusion import BRANCH_ORDER, StackedFusion            # noqa: E402
from simulator import _resample                           # noqa: E402

logger = logging.getLogger("inference")

MAX_CHUNKS_PER_TRACK = 24            # cap compute on very long tracks


def find_checkpoint(branch: str) -> Path | None:
    d = config.CHECKPOINT_DIR / branch / "full"
    if not d.exists():
        return None
    cands = sorted(d.glob("*.ckpt"))
    return cands[-1] if cands else None


class BranchScorer:
    def __init__(self, branch: str, ckpt: Path, device: str):
        from train import BranchModule
        self.branch = branch
        self.cfg = config.BRANCHES[branch]
        self.device = device
        self.module = BranchModule.load_from_checkpoint(
            str(ckpt), map_location=device)
        self.module.eval().to(device)

    @torch.no_grad()
    def score_track(self, wave: torch.Tensor, sr: int) -> float:
        x = _resample(wave, sr, self.cfg["input_sr"]).mean(dim=0)
        n = int(self.cfg["chunk_s"] * self.cfg["input_sr"])
        if x.shape[0] < n:
            x = torch.nn.functional.pad(x, (0, n - x.shape[0]))
        starts = list(range(0, x.shape[0] - n + 1, n)) or [0]
        if len(starts) > MAX_CHUNKS_PER_TRACK:
            idx = np.linspace(0, len(starts) - 1,
                              MAX_CHUNKS_PER_TRACK).astype(int)
            starts = [starts[i] for i in idx]
        chunks = torch.stack([x[s:s + n] for s in starts]).to(self.device)
        outs = []
        bs = max(1, self.cfg["batch_size"])
        for i in range(0, len(chunks), bs):
            batch = chunks[i:i + bs]
            if self.branch == "e":
                outs.append(self.module.model(batch).float().cpu())
            else:
                outs.append(torch.sigmoid(
                    self.module.model(batch)).float().cpu())
        z = torch.cat(outs).numpy()
        return aggregate_chunks(z)


def run(input_dir: Path, output_csv: Path, mode: str, device: str,
        timeout_s: int = config.PER_TRACK_TIMEOUT_S):
    wavs = sorted([p for p in input_dir.iterdir()
                   if p.suffix.lower() in (".wav", ".flac", ".mp3")])
    if not wavs:
        raise SystemExit(f"No audio files found in {input_dir}")
    logger.info("%d tracks to score, mode=%s", len(wavs), mode)

    wanted = [mode.split(":", 1)[1]] if mode.startswith("single") \
        else BRANCH_ORDER
    scorers: dict[str, BranchScorer] = {}
    for b in wanted:
        ckpt = find_checkpoint(b)
        if ckpt is None:
            logger.warning("No checkpoint for branch %s — skipping", b)
            continue
        scorers[b] = BranchScorer(b, ckpt, device)
    if not scorers:
        raise SystemExit("No branch checkpoints available under "
                         f"{config.CHECKPOINT_DIR} — cannot score.")

    use_tta = mode == "tta"
    if use_tta:
        from tta import apply_tta_transform, TTA_TRANSFORMS

    # Pass 1: per-branch scores per track (with per-track hard timeout).
    matrix: dict[str, dict[str, float]] = {}
    pool = ThreadPoolExecutor(max_workers=1)

    def _score_one(path: Path) -> dict[str, float]:
        wave, sr = load_audio(path, max_s=600)
        out = {}
        for b, sc in scorers.items():
            if use_tta:
                vals = [sc.score_track(
                    apply_tta_transform(wave, sr, t), sr)
                    for t in TTA_TRANSFORMS]
                out[b] = float(np.mean(vals))
            else:
                out[b] = sc.score_track(wave, sr)
        return out

    n_fallback = 0
    for i, path in enumerate(wavs):
        fut = pool.submit(_score_one, path)
        try:
            matrix[path.stem] = fut.result(timeout=timeout_s)
        except (FutTimeout, Exception) as e:          # noqa: B902
            fut.cancel()
            n_fallback += 1
            logger.error("FALLBACK for %s: %s", path.name, e)
            matrix[path.stem] = {}
        if (i + 1) % 50 == 0:
            logger.info("scored %d/%d", i + 1, len(wavs))

    # Pass 2: fusion.
    rows = [{"scores": matrix[t]} for t in matrix]
    ids = list(matrix)
    if mode == "rank_average":
        final = StackedFusion.rank_average(rows)
    elif mode.startswith("single"):
        b = mode.split(":", 1)[1]
        final = np.array([r["scores"].get(b, config.FALLBACK_SCORE)
                          for r in rows])
    else:                                   # ensemble / tta
        fdir = config.FUSION_DIR
        if (fdir / "fusion.joblib").exists():
            final = StackedFusion.load(fdir).predict(rows)
        else:
            logger.warning("No fitted fusion found — rank-averaging instead")
            final = StackedFusion.rank_average(rows)
    # Empty score dicts (timeouts) -> fallback.
    final = np.array([config.FALLBACK_SCORE if not rows[i]["scores"]
                      else float(np.clip(final[i], 0, 1))
                      for i in range(len(rows))])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "ai_generated_score"])
        for t, s in zip(ids, final):
            w.writerow([t, f"{s:.6f}"])
    logger.info("Wrote %s (%d rows, %d fallbacks)",
                output_csv, len(ids), n_fallback)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--mode", default="ensemble",
                    choices=["ensemble", "rank_average", "tta",
                             *[f"single:{b}" for b in BRANCH_ORDER]])
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run(a.input_dir, a.output_csv, a.mode, a.device)
