"""
Fast Evaluation on All Generators (10k Tracks Per Generator + 10k Human)
Evaluates the newly retrained SupCon + Calibrated FusionMLP model on the
RTX 2000 Ada GPU across all MIREX generator families.
"""
import time
from pathlib import Path
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
import sys
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from train_supcon import SupConProjectionHead
from fusion_model import FusionMLP, TemperatureScaler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  FAST COMPREHENSIVE BENCHMARK: ALL GENERATORS + HUMAN (Up to 10k/Family)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print("=" * 75)

    t0 = time.time()
    # 1. Load Metadata
    meta_path = config.PROCESSED_DATA_DIR / "train_val_split_200k.parquet"
    narr_path = config.PROCESSED_DATA_DIR / "narrative_vectors_200k_train.parquet"
    surf_path = config.PROCESSED_DATA_DIR / "surface_embeddings_200k_train.npy"

    print("[1/4] Loading cached representations & metadata...")
    df_meta = pd.read_parquet(meta_path)[["file_path", "generator_family", "is_ai"]]
    df_narr = pd.read_parquet(narr_path)
    surface_X = np.load(surf_path)
    
    # Feature columns
    feature_cols = [c for c in df_narr.columns if c not in ("path", "is_ai", "strat_class", "file_path", "generator_family")]
    feature_cols = [c for c in feature_cols if np.issubdtype(df_narr[c].dtype, np.number)]
    narr_X = df_narr[feature_cols].values.astype(np.float32)
    narr_X = np.nan_to_num(narr_X, nan=0.0, posinf=50.0, neginf=-50.0)
    narr_X = np.clip(narr_X, -50.0, 50.0)
    surface_X = np.nan_to_num(surface_X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X_fused = np.concatenate([surface_X, narr_X], axis=1).astype(np.float32)
    
    # Map back to generator families via inner merge on path
    df_narr_meta = df_narr[["path", "is_ai"]].copy()
    df_narr_meta["row_idx"] = np.arange(len(df_narr))
    merged = df_narr_meta.merge(df_meta, left_on="path", right_on="file_path", how="inner")
    
    print(f"      Mapped {len(merged):,d} tracks with 704-D fused representations in {time.time()-t0:.1f}s")

    # 2. Sample 10k tracks per generator (or all available if < 10k)
    print("\n[2/4] Sampling test set: Up to 10k tracks per generator family + 10k human...")
    sample_indices = []
    counts_dict = {}
    for gen, group in merged.groupby("generator_family"):
        n_take = min(len(group), 10000)
        chosen = group.sample(n=n_take, random_state=42)
        sample_indices.extend(chosen["row_idx"].values)
        counts_dict[gen] = n_take

    sample_indices = np.array(sample_indices)
    df_test = merged[merged["row_idx"].isin(sample_indices)].copy()
    X_test_np = X_fused[df_test["row_idx"].values]
    y_test_np = df_test["is_ai_x"].values.astype(int)

    total_test = len(df_test)
    print(f"      Total Evaluation Volume: {total_test:,d} Tracks across {len(counts_dict)} families:")
    for gen in sorted(counts_dict.keys()):
        print(f"        - {gen:15s}: {counts_dict[gen]:,d} tracks")

    # 3. Load Trained Model Weights
    print("\n[3/4] Loading newly retrained SupCon & Calibrated Fusion checkpoints...")
    supcon_head = SupConProjectionHead(in_dim=704, out_dim=128).to(device)
    sup_ckpt = torch.load(config.CHECKPOINT_DIR / "supcon_200k_best.pt", map_location=device, weights_only=False)
    supcon_head.load_state_dict(sup_ckpt["proj_head_state"])
    supcon_head.eval()

    fus_ckpt = torch.load(config.CHECKPOINT_DIR / "fusion_200k_best.pt", map_location=device, weights_only=False)
    fusion_mlp = FusionMLP(in_dim=fus_ckpt.get("in_dim", 128)).to(device)
    fusion_mlp.load_state_dict(fus_ckpt["model_state"])
    fusion_mlp.eval()

    scaler = TemperatureScaler().to(device)
    if "scaler_state" in fus_ckpt:
        scaler.load_state_dict(fus_ckpt["scaler_state"])
    scaler.eval()

    # 4. High-Speed Batched Evaluation in VRAM
    print("\n[4/4] Executing batch forward pass on GPU...")
    t_eval = time.time()
    batch_size = 4096
    all_probs = []

    X_test_gpu = torch.from_numpy(X_test_np).to(device)

    with torch.no_grad():
        for b in range(0, total_test, batch_size):
            bx = X_test_gpu[b:b+batch_size]
            z = supcon_head(bx)
            logits = fusion_mlp(z)
            scaled_logits = scaler(logits)
            probs = torch.sigmoid(scaled_logits).cpu().numpy().flatten()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    df_test["score"] = all_probs
    eval_elapsed = time.time() - t_eval
    print(f"      GPU Inference Complete in {eval_elapsed:.2f}s ({total_test / eval_elapsed:,.1f} tracks/sec!)")

    # Metrics Summary
    threshold = config.DEFAULT_DECISION_THRESHOLD  # 0.18
    preds = (all_probs >= threshold).astype(int)
    overall_acc = accuracy_score(y_test_np, preds)
    overall_auc = roc_auc_score(y_test_np, all_probs)
    overall_f1 = f1_score(y_test_np, preds)
    overall_rec = recall_score(y_test_np, preds)

    print("\n" + "=" * 75)
    print("  OVERALL BENCHMARK RESULTS (Calibrated Threshold T = 0.18)")
    print("=" * 75)
    print(f"  Total Tracks Evaluated : {total_test:,d}")
    print(f"  Macro-AUROC            : {overall_auc:.4f}")
    print(f"  Overall Accuracy       : {overall_acc*100:.2f}%")
    print(f"  AI Recall (Catch Rate) : {overall_rec*100:.2f}%")
    print(f"  Overall F1 Score       : {overall_f1:.4f}")
    print("=" * 75)

    print("\n=== PER-GENERATOR PERFORMANCE BREAKDOWN ===")
    print(f"{'Family':15s} | {'Tracks':7s} | {'Mean Score':10s} | {'T=0.18 Caught':14s} | {'T=0.05 (Strict)':14s}")
    print("-" * 75)
    
    summary_rows = []
    for gen in ["human", "suno", "udio", "audioldm", "musicgen", "mustango", "echoes"]:
        sub = df_test[df_test["generator_family"] == gen]
        if len(sub) == 0:
            continue
        scores = sub["score"].values
        mean_s = scores.mean()
        caught_18 = (scores >= 0.18).mean() * 100
        caught_05 = (scores >= 0.05).mean() * 100
        
        # For human, caught means false alarm, so accuracy is 100 - caught
        if gen == "human":
            acc_18_str = f"{100 - caught_18:5.2f}% Acc"
            acc_05_str = f"{100 - caught_05:5.2f}% Acc"
        else:
            acc_18_str = f"{caught_18:5.2f}% Detected"
            acc_05_str = f"{caught_05:5.2f}% Detected"
            
        print(f"{gen:15s} | {len(sub):7,d} | {mean_s:10.4f} | {acc_18_str:14s} | {acc_05_str:14s}")
        summary_rows.append({
            "Generator": gen,
            "Count": len(sub),
            "Mean_Score": mean_s,
            "Detected_T018": caught_18 if gen != "human" else (100 - caught_18),
            "Detected_T005": caught_05 if gen != "human" else (100 - caught_05),
        })

    # Render High-Contrast Performance Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161922")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#ffffff")

    gens = [r["Generator"].capitalize() for r in summary_rows]
    det_18 = [r["Detected_T018"] for r in summary_rows]
    
    colors = ["#2ecc71" if g == "Human" else "#e74c3c" for g in gens]
    bars = ax.bar(gens, det_18, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    ax.axhline(99.0, color="#f1c40f", linestyle="--", linewidth=1.5, alpha=0.8, label="99% Target Baseline")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")

    ax.set_ylim(0, 115)
    ax.set_title(f"Comprehensive MIREX Benchmark Across All Generators ({total_test:,d} Tracks)", fontsize=13, pad=14, color="white", fontweight="bold")
    ax.set_ylabel("Accuracy / Detection Rate (%)", fontsize=11, color="#cccccc")
    ax.tick_params(colors="#cccccc", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333842")

    ax.legend(loc="upper right", facecolor="#1c202a", edgecolor="#333842", labelcolor="white", fontsize=9.5)
    plt.tight_layout()
    
    out_chart = Path("d:/mirex/Architecture-1/scripts/eval_all_generators_chart.png")
    plt.savefig(out_chart, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Saved] Performance chart written to: {out_chart}")

if __name__ == "__main__":
    main()
