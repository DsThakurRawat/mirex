"""
Quick Evaluation Script — Test trained Architecture-1 on unseen Human & AI tracks.
"""
import random
from pathlib import Path
import pandas as pd
import numpy as np

import config
from inference import MusicScopeCLScorer

def main():
    print("=" * 70)
    print("   MusicScope-CL — Unseen Test Set Evaluation")
    print("=" * 70)

    # 1. Load the tracks that were in training to exclude them
    narrative_path = config.PROCESSED_DATA_DIR / "narrative_vectors.parquet"
    train_paths = set()
    if narrative_path.exists():
        df = pd.read_parquet(narrative_path)
        train_paths = set(df["path"].astype(str).tolist())
    print(f"Excluded {len(train_paths)} training tracks.")

    # 2. Find candidate Human files (FMA / MTG Jamendo)
    human_candidates = []
    fma_dir = config.RAW_DATA_DIR / "fma"
    if fma_dir.exists():
        for p in fma_dir.rglob("*.mp3"):
            if str(p) not in train_paths:
                human_candidates.append(p)
                if len(human_candidates) >= 50:
                    break

    # 3. Find candidate AI files (Sonics / FakeMusicCaps / Echoes)
    ai_candidates = []
    for ai_sub in ["sonics", "fakemusiccaps", "echoes"]:
        ai_dir = config.RAW_DATA_DIR / ai_sub
        if ai_dir.exists():
            for p in ai_dir.rglob("*.mp3"):
                if str(p) not in train_paths:
                    ai_candidates.append(p)
                    if len(ai_candidates) >= 50:
                        break
        if len(ai_candidates) >= 50:
            break

    # Pick 5 unseen Human and 5 unseen AI tracks
    random.seed(42)
    selected_human = random.sample(human_candidates, min(5, len(human_candidates)))
    selected_ai = random.sample(ai_candidates, min(5, len(ai_candidates)))

    test_items = [(p, 0, "Human") for p in selected_human] + \
                 [(p, 1, "AI-Generated") for p in selected_ai]

    print(f"\nEvaluating on {len(test_items)} unseen tracks (5 Human + 5 AI)...\n")

    # 4. Initialize Scorer
    simclr_ckpt = str(config.CHECKPOINT_DIR / "simclr_epoch2.pt")
    supcon_ckpt = str(config.CHECKPOINT_DIR / "supcon_best.pt")
    fusion_ckpt = str(config.CHECKPOINT_DIR / "fusion_best.pt")

    scorer = MusicScopeCLScorer(
        simclr_ckpt=simclr_ckpt,
        supcon_ckpt=supcon_ckpt,
        fusion_ckpt=fusion_ckpt,
    )

    results = []
    print("\n" + "-" * 75)
    print(f"{'True Label':<14} | {'P(AI)':<8} | {'Predicted':<10} | {'Status':<6} | {'Filename'}")
    print("-" * 75)

    correct_count = 0
    for path, true_label, label_name in test_items:
        try:
            score = scorer.score(str(path), n_chunks=config.MIL_CHUNKS)
            pred_label = 1 if score >= 0.5 else 0
            pred_name = "AI" if pred_label == 1 else "Human"
            is_correct = (pred_label == true_label)
            if is_correct:
                correct_count += 1
            status_str = "CORRECT" if is_correct else "WRONG"

            print(f"{label_name:<14} | {score:<8.4f} | {pred_name:<10} | {status_str:<7} | {path.name[:30]}")
            results.append({
                "path": str(path),
                "true_label": true_label,
                "score": score,
                "correct": is_correct
            })
        except Exception as e:
            print(f"{label_name:<14} | ERROR    | ERROR      | FAIL    | {path.name[:30]} ({e})")

    accuracy = (correct_count / len(test_items)) * 100
    print("-" * 75)
    print(f"\n>> Final Test Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_items)} correct)")
    print("=" * 70)

if __name__ == "__main__":
    main()
