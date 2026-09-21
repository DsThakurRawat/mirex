"""
Detailed Analysis and Visualization for 5,000 Unseen Track Evaluation
Architecture-1 (MusicScope-CL)
"""
import os
import sys
import shutil

# Ensure utf-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

# 1. Load data
from pathlib import Path
csv_path = Path(__file__).resolve().parent / "eval_5000_results.csv"
if not csv_path.exists():
    # Fallback to mirex if in original location
    csv_path = Path(__file__).resolve().parent.parent / "mirex" / "eval_5000_results.csv"
df = pd.read_csv(str(csv_path))

print("=" * 60)
print(f"Loaded {len(df)} predictions from {csv_path.name}")
print("=" * 60)

# Basic metrics
y_true = df["true_label"].values
y_scores = df["score"].values
y_pred = df["predicted_label"].values

acc = accuracy_score(y_true, y_pred)
auroc = roc_auc_score(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
f1 = f1_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Optimal threshold analysis (Youden's J: TPR - FPR)
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

# Optimal F1 threshold
thresh_range = np.linspace(0.01, 0.99, 200)
f1_curves = [f1_score(y_true, (y_scores >= t).astype(int)) for t in thresh_range]
acc_curves = [accuracy_score(y_true, (y_scores >= t).astype(int)) for t in thresh_range]
best_f1_idx = np.argmax(f1_curves)
best_f1_thresh = thresh_range[best_f1_idx]
best_f1_val = f1_curves[best_f1_idx]

# Equal Error Rate (EER)
eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
eer_thresh = thresholds[eer_idx]
eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2.0

print(f"Overall Accuracy (thr=0.50): {acc*100:.2f}%")
print(f"Overall AUROC:              {auroc:.4f}")
print(f"Average Precision (AP):     {ap:.4f}")
print(f"F1 Score (thr=0.50):        {f1:.4f}")
print(f"Precision:                  {prec:.4f}")
print(f"Recall:                     {rec:.4f}")
print(f"Equal Error Rate (EER):     {eer*100:.2f}% (at threshold {eer_thresh:.3f})")
print(f"Youden's J Optimal Thresh:  {best_thresh:.4f} (TPR={best_tpr:.3f}, FPR={best_fpr:.3f})")
print(f"Max F1 Threshold:           {best_f1_thresh:.4f} (F1={best_f1_val:.4f}, Acc={acc_curves[best_f1_idx]*100:.2f}%)")
print()

# Breakdown by Family
print("-" * 60)
print("Performance Breakdown by Class / Generator Family:")
print("-" * 60)
summary_rows = []
for fam, g in df.groupby("family"):
    fam_n = len(g)
    fam_acc = accuracy_score(g["true_label"], g["predicted_label"]) * 100
    mean_s = g["score"].mean()
    median_s = g["score"].median()
    std_s = g["score"].std()
    p25 = np.percentile(g["score"], 25)
    p75 = np.percentile(g["score"], 75)
    fp_or_fn = (g["predicted_label"] != g["true_label"]).sum()
    summary_rows.append({
        "Family": fam,
        "Tracks": fam_n,
        "Accuracy (%)": f"{fam_acc:.2f}%",
        "Errors": fp_or_fn,
        "Mean Score": f"{mean_s:.4f}",
        "Median Score": f"{median_s:.4f}",
        "Score IQR": f"[{p25:.3f}, {p75:.3f}]"
    })
print(pd.DataFrame(summary_rows).to_string(index=False))
print()

# Top 5 False Positives (Human tracks with highest AI probability)
fp_tracks = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)].sort_values("score", ascending=False)
print("-" * 60)
print(f"Top 5 False Positives (Total: {len(fp_tracks)} Human tracks misclassified as AI):")
for _, r in fp_tracks.head(5).iterrows():
    safe_fn = str(r['filename']).encode('ascii', 'replace').decode()
    print(f"  P(AI) = {r['score']:.4f} | {safe_fn}")

# Top 5 False Negatives (AI tracks with lowest AI probability)
fn_tracks = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)].sort_values("score", ascending=True)
print("-" * 60)
print(f"Top 5 False Negatives (Total: {len(fn_tracks)} AI tracks misclassified as Human):")
for _, r in fn_tracks.head(5).iterrows():
    safe_fn = str(r['filename']).encode('ascii', 'replace').decode()
    print(f"  P(AI) = {r['score']:.4f} | {r['family']:<12} | {safe_fn}")
print("=" * 60)

# Set style for publication-quality visuals
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(18, 12), dpi=150)

# Color scheme
c_human = "#2b5c8f"    # Deep Blue
c_suno = "#d95f02"     # Rich Orange
c_echoes = "#7570b3"   # Purple
c_accent = "#1b9e77"   # Teal

# -------------------------------------------------------------
# 1. Score Distribution per Family (Histogram / Stepped Density)
# -------------------------------------------------------------
ax1 = plt.subplot(2, 3, 1)
bins = np.linspace(0, 1, 40)

human_scores = df[df["family"] == "Human (FMA)"]["score"]
suno_scores = df[df["family"] == "Suno/Sonics"]["score"]
echoes_scores = df[df["family"] == "Echoes"]["score"]

ax1.hist(human_scores, bins=bins, alpha=0.6, color=c_human, label="Human (FMA)", density=True)
ax1.hist(suno_scores, bins=bins, alpha=0.6, color=c_suno, label="Suno / Sonics", density=True)
if len(echoes_scores) > 0:
    ax1.hist(echoes_scores, bins=bins, alpha=0.5, color=c_echoes, label="Echoes", density=True)

ax1.axvline(0.5, color="red", linestyle="--", linewidth=1.8, label="Decision Boundary (0.50)")
ax1.set_title("Prediction Score Density by Family", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Predicted P(AI)", fontsize=11)
ax1.set_ylabel("Density", fontsize=11)
ax1.legend(loc="upper center", frameon=True, fontsize=9)
ax1.set_xlim(0, 1)

# -------------------------------------------------------------
# 2. ROC Curve
# -------------------------------------------------------------
ax2 = plt.subplot(2, 3, 2)
ax2.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"MusicScope-CL (AUROC = {auroc:.4f})")
ax2.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.2, label="Random Guess (0.5000)")

# Mark default threshold (0.50)
idx_05 = np.argmin(np.abs(thresholds - 0.5))
ax2.scatter(fpr[idx_05], tpr[idx_05], color="red", s=60, zorder=5, label=f"Default 0.50 (TPR={tpr[idx_05]:.2f}, FPR={fpr[idx_05]:.2f})")
# Mark Youden's optimal
ax2.scatter(best_fpr, best_tpr, color="green", s=60, marker="^", zorder=5, label=f"Optimal J={best_thresh:.2f} (TPR={best_tpr:.2f})")

ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax2.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax2.legend(loc="lower right", frameon=True, fontsize=9)
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.02])

# -------------------------------------------------------------
# 3. Precision-Recall Curve
# -------------------------------------------------------------
ax3 = plt.subplot(2, 3, 3)
p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
ax3.plot(r_curve, p_curve, color=c_accent, lw=2.5, label=f"PR Curve (AP = {ap:.4f})")
ax3.scatter(rec, prec, color="red", s=60, zorder=5, label=f"Operating Point 0.50\n(P={prec:.2f}, R={rec:.2f})")

ax3.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Recall (True Positive Rate)", fontsize=11)
ax3.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
ax3.legend(loc="lower left", frameon=True, fontsize=9)
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.4, 1.02])

# -------------------------------------------------------------
# 4. Confusion Matrix Heatmap
# -------------------------------------------------------------
ax4 = plt.subplot(2, 3, 4)
cax = ax4.matshow(cm, cmap="Blues", alpha=0.85)

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = val / len(df) * 100
        color = "white" if val > 1500 else "black"
        ax4.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Pred Human", "Pred AI"], fontsize=10)
ax4.set_yticklabels(["True Human", "True AI"], fontsize=10)
ax4.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
ax4.set_title("Confusion Matrix (Threshold = 0.50)", fontsize=13, fontweight="bold", pad=15)

# -------------------------------------------------------------
# 5. Threshold vs Accuracy & F1 Curve
# -------------------------------------------------------------
ax5 = plt.subplot(2, 3, 5)
ax5.plot(thresh_range, acc_curves, color="#2b5c8f", lw=2, label="Accuracy")
ax5.plot(thresh_range, f1_curves, color="#e7298a", lw=2, linestyle="--", label="F1-Score")
ax5.axvline(0.50, color="red", linestyle=":", lw=1.5, label="Default (0.50)")
ax5.axvline(best_f1_thresh, color="green", linestyle=":", lw=1.5, label=f"Peak F1 ({best_f1_thresh:.2f})")

ax5.set_title("Performance vs Decision Threshold", fontsize=13, fontweight="bold", pad=10)
ax5.set_xlabel("Classification Threshold", fontsize=11)
ax5.set_ylabel("Score", fontsize=11)
ax5.legend(loc="lower center", frameon=True, fontsize=9)
ax5.set_xlim(0, 1)
ax5.set_ylim(0.5, 1.0)

# -------------------------------------------------------------
# 6. Generator Family Accuracy & Score Spread (Violin / Box)
# -------------------------------------------------------------
ax6 = plt.subplot(2, 3, 6)
families = ["Human (FMA)", "Suno/Sonics", "Echoes"]
acc_vals = [
    accuracy_score(df[df["family"] == f]["true_label"], df[df["family"] == f]["predicted_label"]) * 100
    for f in families
]
colors = [c_human, c_suno, c_echoes]

bars = ax6.bar(families, acc_vals, color=colors, alpha=0.85, width=0.55, edgecolor="black", linewidth=1.2)
ax6.axhline(50, color="gray", linestyle=":", label="Chance (50%)")
ax6.axhline(acc * 100, color="red", linestyle="--", linewidth=1.5, label=f"Mean Acc ({acc*100:.1f}%)")

for bar in bars:
    yval = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax6.set_title("Accuracy by Generator Family", fontsize=13, fontweight="bold", pad=10)
ax6.set_ylabel("Accuracy (%)", fontsize=11)
ax6.set_ylim(0, 110)
ax6.legend(loc="upper right", frameon=True, fontsize=9)

plt.tight_layout()

# Save output plots
output_img_local = str(Path(__file__).resolve().parent / "eval_5000_analysis.png")
plt.savefig(output_img_local, bbox_inches="tight")
plt.close()
print(f"\n[Saved] Visual analysis dashboard saved to: {output_img_local}")

# Also copy to artifact directory if available
artifact_dir = r"C:\Users\RF AND SIMULATION\.gemini\antigravity-ide\brain\2489741e-4526-42f7-a5a7-38a49df61ff0"
if os.path.exists(artifact_dir):
    dest_path = os.path.join(artifact_dir, "eval_5000_analysis.png")
    shutil.copyfile(output_img_local, dest_path)
    print(f"[Copied] Dashboard copied to artifact directory: {dest_path}")
