#!/usr/bin/env python3
"""
Patch and finalize the 28,134-track held-out validation evaluation.
Re-scores the 8,045 NaN rows caused by FP16 Mel overflow using the fixed FP32 engine,
updates the checkpoint and results CSV, and generates the publication scorecard.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ARCH1_DIR = _SCRIPTS_DIR.parent
_MIREX_DIR = _ARCH1_DIR / "mirex"

for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer
from evaluate_val_28k import generate_publication_scorecard


def patch_eval_28k(
    results_csv: Path = _SCRIPTS_DIR / "eval_28k_results.csv",
    ckpt_parquet: Path = config.PROCESSED_DATA_DIR / "eval_28k_checkpoint.parquet",
    output_png: Path = _SCRIPTS_DIR / "eval_28k_analysis.png",
    batch_size: int = 64,
    workers: int = 12,
    chunk_size: int = 512,
):
    print("=" * 80)
    print("   MIREX 2026: 28,134-Track Validation Set NaN Patching & Scorecard")
    print("=" * 80)

    # 1. Load DataFrame
    if ckpt_parquet.exists():
        df = pd.read_parquet(ckpt_parquet)
        print(f"[Loaded] Checkpoint parquet: {ckpt_parquet} ({len(df):,d} tracks)")
    elif results_csv.exists():
        df = pd.read_csv(results_csv)
        print(f"[Loaded] Results CSV: {results_csv} ({len(df):,d} tracks)")
    else:
        raise FileNotFoundError(f"Neither {ckpt_parquet} nor {results_csv} found.")

    nan_mask = df["score"].isna()
    nan_indices = df[nan_mask].index.tolist()
    total_nans = len(nan_indices)
    print(f"Total tracks: {len(df):,d} | Already Valid: {len(df) - total_nans:,d} | NaNs to patch: {total_nans:,d}")

    if total_nans == 0:
        print("[Complete] No NaNs found! All 28,134 tracks have valid scores.")
    else:
        scorer = FastMusicScopeScorer()
        t_start = time.perf_counter()
        patched_count = 0

        for start_idx in range(0, total_nans, chunk_size):
            chunk_indices = nan_indices[start_idx : start_idx + chunk_size]
            sub_paths = df.loc[chunk_indices, "file_path"].tolist()

            print(f"\n--- Processing Chunk [{start_idx + 1:,d} - {start_idx + len(chunk_indices):,d}] of {total_nans:,d} ---")
            chunk_results = scorer.score_directory(sub_paths, batch_size=batch_size, workers=workers)

            new_scores = [r["score"] for r in chunk_results]
            df.loc[chunk_indices, "score"] = new_scores
            patched_count += len(chunk_indices)

            # Checkpoint save
            df.to_parquet(ckpt_parquet, index=False)
            df.to_csv(results_csv, index=False)

            elapsed = time.perf_counter() - t_start
            speed = patched_count / max(elapsed, 0.001)
            remaining_nans = total_nans - patched_count
            eta_mins = (remaining_nans / max(speed, 0.001)) / 60
            print(f"  >>> Checkpoint Saved: Patched {patched_count:,d}/{total_nans:,d} ({patched_count/total_nans*100:.1f}%) | Speed: {speed:.1f} trk/s | ETA: {eta_mins:.1f} min")

    # Final Verification
    final_nans = df["score"].isna().sum()
    print("\n" + "=" * 80)
    print(f"   VALIDATION SET STATUS: Total: {len(df):,d} | NaNs Remaining: {final_nans:,d}")
    print("=" * 80)

    # Save finalized files
    df.to_parquet(ckpt_parquet, index=False)
    df.to_csv(results_csv, index=False)
    print(f"Saved finalized dataset to:\n  - {ckpt_parquet}\n  - {results_csv}")

    # Generate Publication Scorecard Dashboard
    print(f"\nRendering Publication Scorecard Dashboard to: {output_png}...")
    generate_publication_scorecard(df, output_png)
    print(f"[Done] Publication dashboard successfully rendered and saved to {output_png}!")


if __name__ == "__main__":
    patch_eval_28k()
