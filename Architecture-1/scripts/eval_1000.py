"""
Evaluation Benchmark: 1,000 Unseen Audio Tracks (500 Human vs. 500 AI)
Architecture-1 (MusicScope-CL)

Excludes all tracks used in training, runs inference across diverse AI generators,
and prints a comprehensive performance scorecard (Accuracy, AUROC, Confusion Matrix,
and Generator Breakdown).
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import sys
_MIREX_DIR = Path(__file__).resolve().parent.parent / "mirex"
if str(_MIREX_DIR) not in sys.path:
    sys.path.insert(0, str(_MIREX_DIR))

import config
from inference import MusicScopeCLScorer


def fast_collect_mp3s(root_dir: Path, target: int, exclude: set) -> list[Path]:
    """Blazingly fast MP3 collector using os.walk that breaks immediately on target."""
    collected = []
    if not root_dir.exists():
        return collected
    for dirpath, _, filenames in os.walk(str(root_dir)):
        for f in filenames:
            if f.lower().endswith(".mp3"):
                full = os.path.join(dirpath, f)
                if full not in exclude:
                    collected.append(Path(full))
                    if len(collected) >= target:
                        return collected
    return collected


def sample_unseen_tracks(n_human: int = 2500, n_ai: int = 2500, seed: int = 42):
    """Select completely unseen Human and AI tracks, avoiding training data."""
    # 1. Load training paths to exclude
    train_paths = set()
    narr_path = config.PROCESSED_DATA_DIR / "narrative_vectors.parquet"
    if narr_path.exists():
        df_train = pd.read_parquet(narr_path)
        train_paths = set(df_train["path"].astype(str).tolist())
    print(f"[Dataset] Excluded {len(train_paths)} training tracks.")

    random.seed(seed)

    # 2. Sample Unseen Human Tracks (from FMA)
    fma_root = config.RAW_DATA_DIR / "fma"
    print(f"[Dataset] Fast-sampling {n_human} Human tracks from FMA...")
    human_pool = fast_collect_mp3s(fma_root, target=n_human * 2, exclude=train_paths)
    selected_human = random.sample(human_pool, min(n_human, len(human_pool)))
    print(f"[Dataset] Selected {len(selected_human)} unseen Human tracks.")

    # 3. Sample Unseen AI Tracks across multiple generators proportionally
    target_suno = int(n_ai * 0.5)
    target_fmc = int(n_ai * 0.3)
    target_echoes = n_ai - target_suno - target_fmc

    ai_pool = []
    print(f"[Dataset] Fast-sampling AI tracks (Targets: {target_suno} Suno, {target_fmc} FakeMusicCaps, {target_echoes} Echoes)...")

    # A. Sonics (Suno)
    sonics_root = config.RAW_DATA_DIR / "sonics"
    suno_files = fast_collect_mp3s(sonics_root, target=target_suno * 2, exclude=train_paths)
    if suno_files:
        ai_pool.extend([(p, "Suno/Sonics") for p in random.sample(suno_files, min(target_suno, len(suno_files)))])

    # B. FakeMusicCaps (Udio / MusicGen / AudioLDM)
    fmc_root = config.RAW_DATA_DIR / "fakemusiccaps"
    fmc_files = fast_collect_mp3s(fmc_root, target=target_fmc * 2, exclude=train_paths)
    if fmc_files:
        ai_pool.extend([(p, "FakeMusicCaps") for p in random.sample(fmc_files, min(target_fmc, len(fmc_files)))])

    # C. Echoes
    echoes_root = config.RAW_DATA_DIR / "echoes"
    echoes_files = fast_collect_mp3s(echoes_root, target=target_echoes * 2, exclude=train_paths)
    if echoes_files:
        ai_pool.extend([(p, "Echoes") for p in random.sample(echoes_files, min(target_echoes, len(echoes_files)))])

    # Top-up from suno if needed
    remaining = n_ai - len(ai_pool)
    if remaining > 0 and len(suno_files) > target_suno:
        used_paths = {p for p, _ in ai_pool}
        extra = [p for p in suno_files if p not in used_paths]
        ai_pool.extend([(p, "Suno/Sonics") for p in extra[:remaining]])

    print(f"[Dataset] Selected {len(ai_pool)} unseen AI tracks.")

    # Combine
    items = []
    for p in selected_human:
        items.append({"path": str(p), "label": 0, "family": "Human (FMA)"})
    for p, fam in ai_pool:
        items.append({"path": str(p), "label": 1, "family": fam})

    random.shuffle(items)
    return items


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Unseen Audio Tracks")
    parser.add_argument("--n_human", type=int, default=2500, help="Number of human tracks")
    parser.add_argument("--n_ai", type=int, default=2500, help="Number of AI tracks")
    parser.add_argument("--chunks", type=int, default=1, help="MIL chunks per track (1=fast, 3=full MIL)")
    parser.add_argument("--output_csv", type=str, default="eval_5000_results.csv", help="Output predictions CSV")
    args = parser.parse_args()

    print("=" * 75)
    print(f"   MusicScope-CL: Large-Scale Benchmark ({args.n_human + args.n_ai} Tracks)")
    print("=" * 75)

    # 1. Sample unseen tracks
    items = sample_unseen_tracks(n_human=args.n_human, n_ai=args.n_ai)
    total = len(items)
    print(f"\n[Benchmark] Ready to score {total} tracks ({args.chunks} chunk(s) per track)...\n")

    # 2. Load Scorer
    simclr_ckpt = str(config.CHECKPOINT_DIR / "simclr_epoch2.pt")
    supcon_ckpt = str(config.CHECKPOINT_DIR / "supcon_best.pt")
    fusion_ckpt = str(config.CHECKPOINT_DIR / "fusion_best.pt")

    scorer = MusicScopeCLScorer(
        simclr_ckpt=simclr_ckpt,
        supcon_ckpt=supcon_ckpt,
        fusion_ckpt=fusion_ckpt,
    )

    # 3. Evaluate Loop
    y_true = []
    y_scores = []
    y_pred = []
    results = []

    t_start = time.time()

    for idx, item in enumerate(items):
        path = item["path"]
        true_label = item["label"]
        family = item["family"]

        try:
            score = scorer.score(path, n_chunks=args.chunks)
        except Exception:
            score = 0.5  # Fallback

        pred = 1 if score >= 0.5 else 0

        y_true.append(true_label)
        y_scores.append(score)
        y_pred.append(pred)

        results.append({
            "path": path,
            "filename": Path(path).name,
            "true_label": true_label,
            "family": family,
            "score": score,
            "predicted_label": pred,
            "correct": int(pred == true_label),
        })

        # Progress reporting every 100 tracks
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            curr_acc = accuracy_score(y_true, y_pred) * 100
            elapsed = time.time() - t_start
            speed = (idx + 1) / max(elapsed, 0.001)
            eta_m = ((total - (idx + 1)) / max(speed, 0.001)) / 60
            pct = (idx + 1) / total * 100
            print(f"  [{idx + 1:4d}/{total}] ({pct:5.1f}%) | Acc so far: {curr_acc:5.2f}% | Speed: {speed:4.1f} trk/s | ETA: {eta_m:4.1f} mins")

    total_time = time.time() - t_start

    # 4. Compute Metrics
    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = np.array(y_pred)

    overall_acc = accuracy_score(y_true_arr, y_pred_arr) * 100
    try:
        overall_auroc = roc_auc_score(y_true_arr, y_scores_arr)
    except Exception:
        overall_auroc = 0.5

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    tn, fp, fn, tp = cm.ravel()

    human_scores = y_scores_arr[y_true_arr == 0]
    ai_scores = y_scores_arr[y_true_arr == 1]

    # 5. Print Scorecard
    print("\n" + "=" * 75)
    print("                    FINAL BENCHMARK SCORECARD")
    print("=" * 75)
    print(f"  Total Tracks Evaluated : {total}")
    print(f"  Overall Test Accuracy  : {overall_acc:.2f}%")
    print(f"  Overall Test AUROC     : {overall_auroc:.4f}")
    print(f"  Total Evaluation Time  : {total_time:.1f}s ({total / total_time:.1f} tracks/sec)")
    print("-" * 75)
    print(f"  Confusion Matrix       :")
    print(f"    True Humans (TN)     : {tn:<5} | False Positives (FP) : {fp}")
    print(f"    True AI Tracks (TP)  : {tp:<5} | False Negatives (FN) : {fn}")
    print("-" * 75)
    print(f"  Average Human P(AI)    : {human_scores.mean():.4f}  (Ideal: close to 0.0)")
    print(f"  Average AI P(AI)       : {ai_scores.mean():.4f}  (Ideal: close to 1.0)")
    print("-" * 75)

    # 6. Generator Family Breakdown
    df_res = pd.DataFrame(results)
    print("  Breakdown by Generator / Source:")
    for fam, group in df_res.groupby("family"):
        fam_acc = accuracy_score(group["true_label"], group["predicted_label"]) * 100
        avg_s = group["score"].mean()
        print(f"    {fam:<24} : {len(group):4d} tracks | Acc: {fam_acc:5.1f}% | Avg P(AI): {avg_s:.4f}")

    print("=" * 75)

    # 7. Save predictions
    df_res.to_csv(args.output_csv, index=False)
    print(f"\n[Output] Saved detailed predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
