"""
Fast Verification on 28k Held-Out Audio Benchmark with New Retrained Models
Evaluates 100 tracks from each generator family (700 tracks total) using
FastMusicScopeScorer (audio I/O + TorchAudio + SupCon + Calibrated FusionMLP).
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer

def main():
    print("=" * 70)
    print("  EVALUATING NEW MODEL ON 28k HELD-OUT RAW AUDIO BENCHMARK")
    print("  Sample Size: 100 tracks per generator family (700 tracks total)")
    print("=" * 70)

    df_28k = pd.read_parquet("d:/mirex/data/processed/eval_28k_checkpoint.parquet")

    # Sample 100 tracks per family
    sample_dfs = []
    for gen, grp in df_28k.groupby("generator_family"):
        valid = grp[grp["file_path"].apply(os.path.exists)]
        sample_dfs.append(valid.sample(n=min(len(valid), 100), random_state=42))
    df_sample = pd.concat(sample_dfs).reset_index(drop=True)

    print(f"\n[1/3] Sampled {len(df_sample):,d} tracks across {df_sample['generator_family'].nunique()} families:")
    for gen, grp in df_sample.groupby("generator_family"):
        print(f"        - {gen:12s}: {len(grp):3d} tracks")

    # Initialize Fast Scorer with newly retrained checkpoints
    print("\n[2/3] Initializing FastMusicScopeScorer with new 200k checkpoints...")
    scorer = FastMusicScopeScorer(
        simclr_ckpt=str(config.CHECKPOINT_DIR / "simclr_200k_best.pt"),
        supcon_ckpt=str(config.CHECKPOINT_DIR / "supcon_200k_best.pt"),
        fusion_ckpt=str(config.CHECKPOINT_DIR / "fusion_200k_best.pt"),
        threshold=config.DEFAULT_DECISION_THRESHOLD,
    )

    # Score raw audio tracks with multi-processing pool
    print("\n[3/3] Running audio decoding & GPU forward pass...")
    t0 = time.time()
    results = scorer.score_directory(
        df_sample["file_path"].tolist(),
        batch_size=64,
        workers=12,
    )
    elapsed = time.time() - t0
    print(f"      Scored {len(results):,d} raw audio tracks in {elapsed:.1f}s ({len(results)/elapsed:.1f} trk/s)")

    scores = [r["score"] for r in results]
    df_sample["new_score"] = scores
    y_true = df_sample["is_ai"].values.astype(int)
    y_pred = (np.array(scores) >= scorer.threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, scores)
    f1 = f1_score(y_true, y_pred)

    print("\n" + "=" * 70)
    print(f"  HELD-OUT 28k SAMPLE BENCHMARK RESULTS (Threshold T = {scorer.threshold:.2f})")
    print("=" * 70)
    print(f"  Macro-AUROC      : {auc:.4f}")
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Overall F1 Score : {f1:.4f}")
    print("-" * 70)
    print(f"{'Family':12s} | {'Tracks':6s} | {'Mean Score':10s} | {'Detected / Accuracy':18s}")
    print("-" * 70)
    for gen, grp in df_sample.groupby("generator_family"):
        g_scores = grp["new_score"].values
        m = g_scores.mean()
        if gen == "human":
            acc_g = (g_scores < scorer.threshold).mean() * 100
            print(f"{gen:12s} | {len(grp):6d} | {m:10.4f} | {acc_g:6.2f}% Human Acc")
        else:
            det_g = (g_scores >= scorer.threshold).mean() * 100
            print(f"{gen:12s} | {len(grp):6d} | {m:10.4f} | {det_g:6.2f}% Caught")
    print("=" * 70)

if __name__ == "__main__":
    main()
