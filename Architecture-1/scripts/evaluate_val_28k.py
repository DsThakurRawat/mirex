"""
Comprehensive 28,134 Track Held-Out Validation Benchmark
Architecture-1 (MusicScope-CL)

Evaluates the final 200k trained models across all 28,134 unseen held-out validation tracks
using the maximum-throughput batched inference engine.

Metrics:
  - Macro-AUROC, Average Precision (AP), F1, Accuracy
  - Equal Error Rate (EER) and Youden's J Optimal Threshold
  - 8-Class Strata Breakdown (Human FMA, Suno, Udio, Mustango, MusicGen, AudioLDM, Echoes)
  - 6-Panel Publication Scorecard: eval_28k_analysis.png
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Link mirex and scripts
_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer


def generate_publication_scorecard(df_results: pd.DataFrame, output_png: Path):
    """Generates a 6-panel publication-quality analysis dashboard."""
    y_true = df_results["is_ai"].values
    y_scores = df_results["score"].values
    y_pred_default = (y_scores >= 0.50).astype(int)

    # 1. Metrics
    acc = accuracy_score(y_true, y_pred_default)
    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred_default)

    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    print("=" * 70)
    print(f"  28,134 HELD-OUT TRACK VALIDATION SCORECARD")
    print("=" * 70)
    print(f"  Total Unseen Tracks : {len(df_results):,d}")
    print(f"  Macro-AUROC         : {auroc:.4f}")
    print(f"  Average Precision   : {ap:.4f}")
    print(f"  Accuracy (P >= 0.50): {acc*100:.2f}%")
    print(f"  F1 Score            : {f1:.4f}")
    print(f"  Equal Error Rate    : {eer*100:.2f}%")
    print(f"  Optimal Cutoff (J)  : {best_thresh:.4f}")
    print("-" * 70)

    # Per-Generator Breakdown
    print("  ACCURACY BY GENERATOR FAMILY:")
    strat_results = {}
    for strat, grp in df_results.groupby("strat_class"):
        grp_acc = accuracy_score(grp["is_ai"], (grp["score"] >= best_thresh).astype(int))
        strat_results[strat] = grp_acc
        print(f"    {strat:20s}: {grp_acc*100:6.2f}% ({len(grp):,d} tracks)")
    print("=" * 70)

    # Plot 6-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)
    fig.patch.set_facecolor('#0d1117')
    for ax in axes.flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Panel 1: ROC Curve
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color='#58a6ff', lw=2.5, label=f'ROC (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='#8b949e', ls='--', lw=1.5)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color='#f78166', s=80, zorder=5, label=f'Optimal J ({best_thresh:.2f})')
    ax.set_title('Receiver Operating Characteristic', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate (Human False Alarms)')
    ax.set_ylabel('True Positive Rate (AI Detection Rate)')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax.grid(color='#30363d', alpha=0.5)

    # Panel 2: Precision-Recall Curve
    ax = axes[0, 1]
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    ax.plot(rec_curve, prec_curve, color='#7ee787', lw=2.5, label=f'PR (AP = {ap:.4f})')
    ax.set_title('Precision-Recall Curve', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax.grid(color='#30363d', alpha=0.5)

    # Panel 3: Confusion Matrix
    ax = axes[0, 2]
    cm = confusion_matrix(y_true, (y_scores >= best_thresh).astype(int))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'], color='white')
    ax.set_yticklabels(['Human', 'AI'], color='white')
    ax.set_title(f'Confusion Matrix (Threshold: {best_thresh:.2f})', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    for i in range(2):
        for j in range(2):
            val_pct = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]:,d})"
            color = "white" if cm_norm[i, j] > 0.5 else "#c9d1d9"
            ax.text(j, i, val_pct, ha="center", va="center", color=color, fontweight='bold')

    # Panel 4: Per-Generator Accuracy
    ax = axes[1, 0]
    names = [s.replace("ai_", "").replace("human_", "") for s in strat_results.keys()]
    accs = [v * 100 for v in strat_results.values()]
    colors = ['#58a6ff' if 'fma' in k else '#bc8cff' for k in strat_results.keys()]
    bars = ax.barh(names, accs, color=colors, edgecolor='#30363d')
    ax.set_xlim(0, 105)
    ax.set_title('Accuracy across Generator Families', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Accuracy (%)')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', color='white', fontsize=9)
    ax.grid(color='#30363d', alpha=0.5, axis='x')

    # Panel 5: Score Distribution Histogram
    ax = axes[1, 1]
    human_scores = y_scores[y_true == 0]
    ai_scores = y_scores[y_true == 1]
    ax.hist(human_scores, bins=40, alpha=0.7, color='#58a6ff', label='Human Tracks', density=True)
    ax.hist(ai_scores, bins=40, alpha=0.7, color='#bc8cff', label='AI Tracks', density=True)
    ax.axvline(best_thresh, color='#f78166', ls='--', lw=2, label=f'Optimal Cutoff ({best_thresh:.2f})')
    ax.set_title('P(AI) Output Probability Distribution', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Probability P(AI)')
    ax.set_ylabel('Density')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax.grid(color='#30363d', alpha=0.5)

    # Panel 6: Calibration Curve
    ax = axes[1, 2]
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
    ax.plot([0, 1], [0, 1], color='#8b949e', ls='--', label='Perfect Calibration')
    ax.plot(prob_pred, prob_true, marker='o', color='#ffa657', lw=2, label='MusicScope-CL (Platt Scaled)')
    ax.set_title('Platt Temperature Calibration', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax.grid(color='#30363d', alpha=0.5)

    plt.suptitle("MusicScope-CL 200k Model: 28,134 Unseen Track Held-Out Benchmark",
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    plt.close()
    print(f"\n[Saved Scorecard] Publication dashboard saved to: {output_png}")


def main():
    split_path = config.PROCESSED_DATA_DIR / "train_val_split_200k.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found at {split_path}")

    print(f"[1/4] Loading held-out validation partition from {split_path.name}...")
    df_all = pd.read_parquet(split_path)
    df_val = df_all[df_all["split"] == "val"].reset_index(drop=True)
    total_val = len(df_val)
    print(f"      Loaded {total_val:,d} unseen tracks ({df_val['is_ai'].sum():,d} AI vs {(df_val['is_ai']==0).sum():,d} Human).\n")

    # Check for existing checkpoint results to allow auto-resume
    results_csv = _SCRIPTS_DIR / "eval_28k_results.csv"
    ckpt_parquet = config.PROCESSED_DATA_DIR / "eval_28k_checkpoint.parquet"

    existing_df = None
    if ckpt_parquet.exists():
        existing_df = pd.read_parquet(ckpt_parquet)
        nan_count = existing_df["score"].isna().sum()
        print(f"[Auto-Resume] Found existing evaluation checkpoint with {len(existing_df):,d} tracks ({nan_count:,d} NaNs).")
        if len(existing_df) >= total_val and nan_count == 0:
            print("      Evaluation already completed with 100% valid scores! Generating scorecard...")
            generate_publication_scorecard(existing_df, _SCRIPTS_DIR / "eval_28k_analysis.png")
            return

    # Initialize Fast Scorer
    scorer = FastMusicScopeScorer()

    # Determine remaining tracks
    if existing_df is not None:
        if existing_df["score"].isna().sum() > 0:
            # We already have all 28k rows, but some have NaNs that need patching
            valid_df = existing_df[~existing_df["score"].isna()].copy()
            df_to_score = existing_df[existing_df["score"].isna()].copy().reset_index(drop=True)
            print(f"      Preserving {len(valid_df):,d} valid tracks. Re-scoring {len(df_to_score):,d} tracks with patched engine...")
            all_scored = valid_df.to_dict('records')
        else:
            done_paths = set(existing_df["file_path"].tolist())
            df_to_score = df_val[~df_val["file_path"].isin(done_paths)].reset_index(drop=True)
            all_scored = existing_df.to_dict('records')
    else:
        df_to_score = df_val
        all_scored = []

    print(f"\n[2/4] Scoring {len(df_to_score):,d} remaining tracks with 20 CPU Workers + GPU TorchAudio...")
    paths_to_score = df_to_score["file_path"].tolist()

    # Score in high-throughput chunks of 2,048 tracks with milestone saving
    chunk_sz = 2048
    t_start = time.perf_counter()

    for c_idx in range(0, len(paths_to_score), chunk_sz):
        sub_paths = paths_to_score[c_idx:c_idx + chunk_sz]
        chunk_results = scorer.score_directory(sub_paths, batch_size=128, workers=20)

        # Merge metadata with direct 1-to-1 order mapping
        sub_df = df_to_score.iloc[c_idx:c_idx + len(sub_paths)].copy()
        sub_df["score"] = [r["score"] for r in chunk_results]

        all_scored.extend(sub_df.to_dict('records'))

        # Save milestone checkpoint
        df_progress = pd.DataFrame(all_scored)
        df_progress.to_parquet(ckpt_parquet, index=False)
        elapsed = time.perf_counter() - t_start
        speed = (c_idx + len(sub_paths)) / max(elapsed, 0.001)
        print(f"\n  >>> Milestone Saved: [{len(all_scored):,d}/{total_val:,d}] ({len(all_scored)/total_val*100:.1f}%) | Speed: {speed:.1f} trk/s\n")

    # Final Save
    df_final = pd.DataFrame(all_scored)
    df_final.to_csv(results_csv, index=False)
    print(f"\n[3/4] Wrote final predictions to {results_csv}")

    # Generate Publication Scorecard
    print("\n[4/4] Generating Publication Scorecard Dashboard...")
    generate_publication_scorecard(df_final, _SCRIPTS_DIR / "eval_28k_analysis.png")


if __name__ == "__main__":
    main()
