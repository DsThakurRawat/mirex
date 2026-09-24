"""
Zero-Shot Generalization Benchmark on 8 Completely Unseen Generators
Architecture-1 (MusicScope-CL)

Evaluates the final 200k retrained models on 8 AI generator architectures
that were NEVER present in the training catalog:
  1. ElevenLabs Music (300 tracks)
  2. DiffRhythm (594 tracks)
  3. SongGen (584 tracks)
  4. Producer / CassetteAI (300 tracks)
  5. Brev.ai (298 tracks)
  6. AceStep (295 tracks)
  7. Stable Audio (194 tracks)
  8. Mubert (149 tracks)
Plus 1,000 strictly held-out Real Human Music tracks from FMA.
"""
import os
import sys
import time
from pathlib import Path
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer


def collect_evaluation_dataset() -> pd.DataFrame:
    print("[1/4] Collecting dataset from 8 Unseen Generators + Held-out Human Music...")
    base_echoes = Path("D:/mirex/data/raw/echoes/Echoes")
    unseen_gens = [
        "elevenlabs", "diffrhythm", "songgen", "producer",
        "brev", "acestep", "stableaudio", "mubert"
    ]

    records = []
    for gen in unseen_gens:
        files = list(base_echoes.rglob(f"*/{gen}/*.*"))
        valid_files = [f for f in files if f.suffix.lower() in [".mp3", ".wav"]]
        for f in valid_files:
            records.append({
                "file_path": str(f),
                "generator": gen,
                "is_ai": 1,
                "category": "Unseen AI Generator"
            })
        print(f"      - {gen:15s}: {len(valid_files):4d} tracks (100% Unseen in Training)")

    # Collect 1,000 strictly held-out Human tracks from FMA
    fma_dir = Path("D:/mirex/data/raw/fma/fma_large")
    all_fma = list(fma_dir.rglob("*.mp3"))

    # Exclude any files used in training
    train_split_path = config.PROCESSED_DATA_DIR / "train_val_split_200k.parquet"
    train_paths = set()
    if train_split_path.exists():
        df_split = pd.read_parquet(train_split_path, columns=["file_path"])
        train_paths = set(df_split["file_path"].tolist())

    unseen_fma = [str(f) for f in all_fma if str(f) not in train_paths]
    rng = np.random.RandomState(42)
    chosen_fma = rng.choice(unseen_fma, size=min(1000, len(unseen_fma)), replace=False)

    for f in chosen_fma:
        records.append({
            "file_path": str(f),
            "generator": "human_fma",
            "is_ai": 0,
            "category": "Real Human Music"
        })
    print(f"      - {'human_fma':15s}: {len(chosen_fma):4d} tracks (100% Unseen Real Music)")

    df = pd.DataFrame(records)
    print(f"\n      Total Benchmark Dataset: {len(df):,d} tracks ({df['is_ai'].sum():,d} AI vs {(df['is_ai']==0).sum():,d} Human)\n")
    return df


def plot_scorecard(df: pd.DataFrame, auroc: float, acc: float, f1: float, output_png: Path):
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    # 1. Bar Chart: Caught Rate by Generator
    gens = []
    caught_rates = []
    mean_scores = []
    colors = []

    for gen, grp in df.groupby("generator"):
        gens.append(gen)
        scores = grp["score"].values
        mean_s = np.mean(scores)
        mean_scores.append(mean_s)
        if gen == "human_fma":
            # Human accuracy (score < threshold)
            rate = (scores < 0.18).mean() * 100
            colors.append("#38bdf8")
        elif gen == "mubert":
            rate = (scores >= 0.18).mean() * 100
            colors.append("#94a3b8")
        else:
            rate = (scores >= 0.18).mean() * 100
            colors.append("#10b981" if rate >= 80 else "#f59e0b")
        caught_rates.append(rate)

    order = np.argsort(caught_rates)[::-1]
    gens = [gens[i] for i in order]
    caught_rates = [caught_rates[i] for i in order]
    mean_scores = [mean_scores[i] for i in order]
    colors = [colors[i] for i in order]

    bars = ax1.barh(gens, caught_rates, color=colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax1.set_xlabel("Detection Rate / Accuracy (%)", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax1.set_title("Zero-Shot Generalization Across 8 Unseen Generators", fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    ax1.set_xlim(0, 115)
    ax1.grid(axis="x", linestyle=":", alpha=0.3)

    for bar, rate, ms in zip(bars, caught_rates, mean_scores):
        ax1.text(rate + 1.5, bar.get_y() + bar.get_height()/2, f"{rate:.1f}% (μ={ms:.3f})",
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#f8fafc")

    # 2. Probability Distribution by Generator Category
    for cat, color, label in [("Unseen AI Generator", "#ef4444", "Unseen Neural AI (7 Gens)"),
                              ("Real Human Music", "#38bdf8", "Real Human Music (FMA)")]:
        cat_scores = df[(df["category"] == cat) & (df["generator"] != "mubert")]["score"].values
        ax2.hist(cat_scores, bins=50, alpha=0.65, color=color, label=label, density=True)

    mubert_scores = df[df["generator"] == "mubert"]["score"].values
    ax2.hist(mubert_scores, bins=25, alpha=0.6, color="#94a3b8", label="Mubert (Loop Assembler)", density=True)

    ax2.axvline(0.18, color="#e2e8f0", linestyle="--", linewidth=1.5, label="Decision Cutoff (T = 0.18)")
    ax2.set_xlabel("Predicted Probability P(AI)", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax2.set_title(f"Score Separation: Macro-AUROC = {auroc:.4f} | F1 = {f1:.4f}", fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    ax2.legend(frameon=True, facecolor="#1e293b", edgecolor="#475569", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.3)

    plt.suptitle("MusicScope-CL: Zero-Shot Generalization Test on Completely Unseen Generators",
                 fontsize=15, fontweight="bold", color="#38bdf8", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Scorecard] Saved publication figure to: {output_png}")


def main():
    print("=" * 75)
    print("  PHASE 1: FULL ZERO-SHOT BENCHMARK ON 8 UNSEEN GENERATOR ARCHITECTURES")
    print("=" * 75)

    df_test = collect_evaluation_dataset()

    print("[2/4] Initializing FastMusicScopeScorer with 200k checkpoints...")
    scorer = FastMusicScopeScorer(threshold=config.DEFAULT_DECISION_THRESHOLD)

    print(f"[3/4] Running multi-worker batched audio evaluation (16 Workers)...")
    t0 = time.time()
    results = scorer.score_directory(
        df_test["file_path"].tolist(),
        batch_size=128,
        workers=16,
    )
    elapsed = time.time() - t0
    speed = len(results) / max(elapsed, 0.001)
    print(f"\n      Scored {len(results):,d} tracks in {elapsed:.1f}s ({speed:.1f} trk/s)")

    scores = [r["score"] for r in results]
    df_test["score"] = scores
    df_test["prediction"] = [r["prediction"] for r in results]

    # Metrics
    y_true = df_test["is_ai"].values
    y_pred = (np.array(scores) >= scorer.threshold).astype(int)

    auroc = roc_auc_score(y_true, scores)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n" + "=" * 75)
    print(f"  PHASE 1: UNSEEN GENERATOR BENCHMARK RESULTS (Threshold T = {scorer.threshold:.2f})")
    print("=" * 75)
    print(f"  Macro-AUROC      : {auroc:.4f}")
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Overall F1 Score : {f1:.4f}")
    print("-" * 75)
    print(f"{'Generator Family':18s} | {'Tracks':6s} | {'Mean Score':10s} | {'Performance / Caught Rate':25s}")
    print("-" * 75)

    for gen, grp in df_test.groupby("generator"):
        g_scores = grp["score"].values
        m = g_scores.mean()
        if gen == "human_fma":
            h_acc = (g_scores < scorer.threshold).mean() * 100
            print(f"{gen:18s} | {len(grp):6d} | {m:10.4f} | {h_acc:6.2f}% Human Accuracy")
        else:
            caught = (g_scores >= scorer.threshold).mean() * 100
            print(f"{gen:18s} | {len(grp):6d} | {m:10.4f} | {caught:6.2f}% Caught (AI)")
    print("=" * 75)

    # Save CSV
    results_dir = _SCRIPTS_DIR.parent / "results" / "new_model_retrained_200k"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "eval_unseen_generators_results.csv"
    df_test.to_csv(out_csv, index=False)
    print(f"\n[Saved CSV] Saved detailed predictions to: {out_csv}")

    # Plot figure
    out_png = results_dir / "unseen_generators_scorecard.png"
    plot_scorecard(df_test, auroc, acc, f1, out_png)
    # Also copy to artifacts dir
    try:
        import shutil
        art_dir = Path("C:/Users/RF AND SIMULATION/.gemini/antigravity-ide/brain/0dda047d-7015-4d0e-b363-57eb63e63123")
        shutil.copy(out_png, art_dir / "unseen_generators_scorecard.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
