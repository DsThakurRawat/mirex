"""
Track A — Narrative Branch (Branch B in the architecture spec)
Extracts a 128-dim "Musical Narrative Vector" from an audio chunk using
pure MIR/DSP methods. No training required.

The 128-dim vector is built from 5 feature groups, each expanded into
multi-statistics (mean, std, skew, kurtosis, percentiles, temporal
derivatives) to capture both the magnitude and the *temporal evolution*
of each musical property.

Feature Groups:
    1. Harmonic Entropy & Tension      (chroma_cqt)           → 26 dims
    2. Rhythmic Micro-timing (IOI)     (onset_strength)       → 20 dims
    3. Motivic Recurrence Rate         (self-similarity matrix)→ 22 dims
    4. Spectral Density Variance       (flatness/rolloff/cent) → 36 dims
    5. Dynamic Envelope Variance       (RMS energy)            → 24 dims
                                                        Total: 128 dims

Usage:
    python track_a_narrative.py --input_dir ./data/raw --output narrative_vectors.parquet
"""
import argparse
import glob
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import config

SR = config.SAMPLE_RATE
CHUNK_SECONDS = config.CHUNK_SECONDS


# ---------------------------------------------------------------------------
# Statistical summary helpers
# ---------------------------------------------------------------------------

def _stats(x: np.ndarray, prefix: str, n_percentiles: int = 3) -> dict:
    """Compute a rich statistical summary of a 1-D signal.

    Returns: mean, std, skew, kurtosis, min, max, and n_percentiles
    evenly-spaced percentiles. Plus delta-mean and delta-std (temporal
    derivatives) if the signal has enough samples.
    """
    if len(x) < 2:
        # Degenerate case: return zeros
        keys = ([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_skew",
                 f"{prefix}_kurtosis", f"{prefix}_min", f"{prefix}_max",
                 f"{prefix}_delta_mean", f"{prefix}_delta_std"] +
                [f"{prefix}_p{int(p)}" for p in
                 np.linspace(10, 90, n_percentiles)])
        return {k: 0.0 for k in keys}

    d = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_skew": float(stats.skew(x)),
        f"{prefix}_kurtosis": float(stats.kurtosis(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
    }
    # Percentiles
    for p in np.linspace(10, 90, n_percentiles):
        d[f"{prefix}_p{int(p)}"] = float(np.percentile(x, p))
    # Temporal derivatives (how the feature changes over time)
    delta = np.diff(x)
    d[f"{prefix}_delta_mean"] = float(np.mean(delta)) if len(delta) > 0 else 0.0
    d[f"{prefix}_delta_std"] = float(np.std(delta)) if len(delta) > 0 else 0.0
    return d


# ---------------------------------------------------------------------------
# Feature Group 1: Harmonic Entropy & Tension (26 dims)
# ---------------------------------------------------------------------------

def harmonic_features(y: np.ndarray, sr: int) -> dict:
    """Chroma-based harmonic entropy and tension over time."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Normalize each frame to a probability distribution over 12 pitch classes
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    # Per-frame entropy (higher = more ambiguous key)
    entropy = -np.sum(chroma_norm * np.log(chroma_norm + 1e-9), axis=0)
    # Per-frame tension (std across pitch classes — proxy for dissonance)
    tension = np.std(chroma, axis=0)
    # Key clarity: max chroma activation per frame (higher = clearer key)
    key_clarity = np.max(chroma_norm, axis=0)

    feats = {}
    feats.update(_stats(entropy, "harm_entropy"))      # 11 dims
    feats.update(_stats(tension, "harm_tension"))       # 11 dims
    # Key clarity summary (just mean + std to stay in budget)
    feats["harm_key_clarity_mean"] = float(np.mean(key_clarity))
    feats["harm_key_clarity_std"] = float(np.std(key_clarity))
    # Chroma flux (frame-to-frame harmonic change)
    chroma_flux = np.sqrt(np.sum(np.diff(chroma, axis=1) ** 2, axis=0))
    feats["harm_chroma_flux_mean"] = float(np.mean(chroma_flux))
    feats["harm_chroma_flux_std"] = float(np.std(chroma_flux))
    return feats  # 26 dims


# ---------------------------------------------------------------------------
# Feature Group 2: Rhythmic Micro-timing (20 dims)
# ---------------------------------------------------------------------------

def rhythmic_features(y: np.ndarray, sr: int) -> dict:
    """Onset-based rhythmic micro-timing and groove analysis."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units="time")

    feats = {}
    # Onset strength envelope statistics
    feats.update(_stats(onset_env, "rhythm_onset_env"))  # 11 dims

    if len(onsets) >= 3:
        ioi = np.diff(onsets)
        feats.update(_stats(ioi, "rhythm_ioi"))          # 11 dims → but we'll
    else:
        # Not enough onsets — zero-fill the IOI stats
        feats.update(_stats(np.array([0.0]), "rhythm_ioi"))

    # Trim to 20 dims: keep onset_env stats (11) + ioi stats (9 selected)
    # We keep all 11 onset_env + 9 from ioi (drop delta_mean and delta_std
    # from ioi since they're less meaningful with few onsets)
    keep = {}
    for k, v in feats.items():
        if k.startswith("rhythm_onset_env"):
            keep[k] = v
        elif k.startswith("rhythm_ioi") and "delta" not in k:
            keep[k] = v
    return keep  # 20 dims


# ---------------------------------------------------------------------------
# Feature Group 3: Motivic Recurrence (22 dims)
# ---------------------------------------------------------------------------

def motivic_features(y: np.ndarray, sr: int) -> dict:
    """Self-similarity matrix analysis for motif detection."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    feats = {}

    # Chroma-based self-similarity
    ssm_chroma = librosa.segment.recurrence_matrix(
        chroma, mode="affinity", sym=True)
    # Diagonal strength (how much the track repeats at fixed lag)
    diag_strengths = np.array([
        np.mean(np.diag(ssm_chroma, k=k))
        for k in range(1, min(ssm_chroma.shape[0], 50))
    ])
    feats.update(_stats(diag_strengths, "motif_chroma_diag"))  # 11 dims
    feats["motif_chroma_recurrence"] = float(np.mean(ssm_chroma))

    # Mel-based self-similarity (captures timbral repetition)
    ssm_mel = librosa.segment.recurrence_matrix(
        log_mel, mode="affinity", sym=True)
    diag_mel = np.array([
        np.mean(np.diag(ssm_mel, k=k))
        for k in range(1, min(ssm_mel.shape[0], 50))
    ])
    feats.update(_stats(diag_mel, "motif_mel_diag"))           # 11 dims
    # Cut to 22: keep chroma_diag stats (11) + chroma_recurrence (1) +
    # mel_diag selected (10)
    keep = {}
    for k, v in feats.items():
        if k.startswith("motif_chroma"):
            keep[k] = v
    # Add mel diag stats (keep first 10 to hit 22 total)
    mel_keys = [k for k in feats if k.startswith("motif_mel_diag")]
    for k in mel_keys[:10]:
        keep[k] = feats[k]
    return keep  # 22 dims


# ---------------------------------------------------------------------------
# Feature Group 4: Spectral Density (36 dims)
# ---------------------------------------------------------------------------

def spectral_features(y: np.ndarray, sr: int) -> dict:
    """Spectral flatness, rolloff, centroid, and bandwidth over time."""
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    feats = {}
    feats.update(_stats(flatness, "spec_flatness"))    # 11 dims
    feats.update(_stats(rolloff, "spec_rolloff"))      # 11 dims
    feats.update(_stats(centroid, "spec_centroid"))     # 11 dims

    # Bandwidth — keep just 3 to reach 36 total
    feats["spec_bandwidth_mean"] = float(np.mean(bandwidth))
    feats["spec_bandwidth_std"] = float(np.std(bandwidth))
    feats["spec_bandwidth_skew"] = float(stats.skew(bandwidth))
    return feats  # 36 dims


# ---------------------------------------------------------------------------
# Feature Group 5: Dynamic Envelope (24 dims)
# ---------------------------------------------------------------------------

def dynamic_features(y: np.ndarray, sr: int = SR) -> dict:
    """RMS energy and zero-crossing rate dynamics."""
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    feats = {}
    feats.update(_stats(rms, "dyn_rms"))       # 11 dims
    feats.update(_stats(zcr, "dyn_zcr"))       # 11 dims

    # Loudness range (difference between loud and quiet sections)
    feats["dyn_loudness_range"] = float(np.percentile(rms, 95) -
                                        np.percentile(rms, 5))
    feats["dyn_crest_factor"] = float(np.max(np.abs(y)) /
                                      (np.sqrt(np.mean(y ** 2)) + 1e-9))
    return feats  # 24 dims


# ---------------------------------------------------------------------------
# Full extraction pipeline → 128-dim vector
# ---------------------------------------------------------------------------

FEATURE_EXTRACTORS = [
    harmonic_features,    # 26 dims
    rhythmic_features,    # 20 dims
    motivic_features,     # 22 dims
    spectral_features,    # 36 dims
    dynamic_features,     # 24 dims
]                         # Total: 128 dims


def extract_narrative_vector(path: str, sr: int = SR,
                             duration: float = CHUNK_SECONDS) -> dict:
    """Extract the full 128-dim Musical Narrative Vector from one audio file."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)

    feats = {"path": path}
    for extractor in FEATURE_EXTRACTORS:
        feats.update(extractor(y, sr))
    return feats


def get_feature_columns(sample_feats: dict) -> list[str]:
    """Return the ordered list of feature column names (excluding 'path')."""
    return [k for k in sample_feats if k != "path"]


def feats_to_vector(feats: dict, columns: list[str]) -> np.ndarray:
    """Convert a feature dict to a fixed-order numpy vector."""
    return np.array([feats.get(c, 0.0) for c in columns], dtype=np.float32)


def _extract_single(item: tuple[str, int | None]) -> dict | None:
    """Helper for parallel feature extraction. item is (path_str, label)."""
    path_str, label = item
    try:
        feats = extract_narrative_vector(path_str)
        if label is not None:
            feats["label"] = int(label)
        return feats
    except Exception as e:
        print(f"[warn] failed on {path_str}: {e}")
        return None


def main():
    import concurrent.futures
    import sqlite3

    parser = argparse.ArgumentParser(
        description="Extract 128-dim Musical Narrative Vectors (Branch B)")
    parser.add_argument("--input_dir", default="",
                        help="Directory of audio files (used if --db not given)")
    parser.add_argument("--db", default="",
                        help="Path to metadata.db for balanced labeled sampling")
    parser.add_argument("--output", required=True,
                        help="Output parquet path")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Max tracks to extract (0 = all; default: 2000)")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8),
                        help="Number of parallel worker processes")
    parser.add_argument("--ext", default="mp3",
                        help="Audio file extension to glob for")
    args = parser.parse_args()

    items: list[tuple[str, int | None]] = []

    # Strategy 1: Load balanced human & AI from SQLite DB
    if args.db and Path(args.db).exists():
        print(f"[Narrative] Loading tracks from database: {args.db}")
        conn = sqlite3.connect(args.db)
        # Fetch human (0) and AI (1) tracks
        half = args.limit // 2 if args.limit > 0 else 50000
        q_human = "SELECT file_path, is_ai FROM tracks WHERE is_ai = 0 AND file_path IS NOT NULL"
        q_ai = "SELECT file_path, is_ai FROM tracks WHERE is_ai = 1 AND file_path IS NOT NULL"
        human_rows = conn.execute(q_human).fetchall()
        ai_rows = conn.execute(q_ai).fetchall()
        conn.close()

        # Check file existence
        valid_human = [(fp, label) for fp, label in human_rows if Path(fp).exists()]
        valid_ai = [(fp, label) for fp, label in ai_rows if Path(fp).exists()]
        print(f"[Narrative] Found on disk: {len(valid_human)} human, {len(valid_ai)} AI")

        selected = []
        if args.limit > 0:
            selected.extend(valid_human[:half])
            selected.extend(valid_ai[:half])
        else:
            selected.extend(valid_human)
            selected.extend(valid_ai)
        items = selected
    else:
        # Strategy 2: Scan audio directory
        root = args.input_dir or str(config.RAW_DATA_DIR)
        print(f"[Narrative] Scanning audio files under: {root}")
        files = glob.glob(os.path.join(root, f"**/*.{args.ext}"), recursive=True)
        if not files:
            for ext in config.AUDIO_EXTS:
                files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
        files = sorted(set(files))
        if args.limit > 0:
            files = files[:args.limit]
        items = [(f, None) for f in files]

    print(f"[Narrative] Extracting narrative vectors for {len(items)} tracks using {args.workers} workers...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # Use ProcessPoolExecutor for true multi-core parallel CPU DSP processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(executor.map(_extract_single, items), total=len(items), desc="Extracting features"):
            if res is not None:
                rows.append(res)
                # Checkpoint every 100 tracks
                if len(rows) % 100 == 0:
                    pd.DataFrame(rows).to_parquet(out_path, index=False)

    if not rows:
        print("[error] No features extracted.")
        return

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ("path", "label")]
    print(f"Extracted {len(df)} tracks | Feature dimensionality: {len(feature_cols)} (target: {config.NARRATIVE_DIM})")
    df.to_parquet(out_path, index=False)
    print(f"Saved narrative vectors to {out_path}")


if __name__ == "__main__":
    main()
