"""
Step 2: Multi-Process 128-D Narrative Feature Extraction for 200k Tracks
Architecture-1 (MusicScope-CL)

Features:
  - 18–20 parallel worker processes utilizing Intel Core i9-14900K.
  - Chunked parquet output (every 5,000 tracks) for 100% crash recovery / auto-resume.
  - Smoke test mode for quick verification.
  - Final concatenation into narrative_vectors_200k.parquet.
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reconfigure stdout for utf-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add Architecture-1/mirex to sys.path
_MIREX_DIR = Path(__file__).resolve().parent.parent / "mirex"
if str(_MIREX_DIR) not in sys.path:
    sys.path.insert(0, str(_MIREX_DIR))

import config
from track_a_narrative import (
    extract_narrative_vector,
    get_feature_columns,
    feats_to_vector,
)


def process_single_track(item: tuple[str, int, str]) -> dict | None:
    """Worker function: extracts 128-D narrative features from one audio track."""
    path, is_ai, strat_class = item
    try:
        feats = extract_narrative_vector(path, sr=config.SAMPLE_RATE, duration=config.CHUNK_SECONDS)
        feats["is_ai"] = int(is_ai)
        feats["strat_class"] = str(strat_class)
        return feats
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Process Narrative Feature Extraction")
    parser.add_argument("--workers", type=int, default=28, help="Number of CPU worker processes")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Tracks per chunk parquet file")
    parser.add_argument("--smoke", action="store_true", help="Run quick verification on a small batch")
    parser.add_argument("--smoke_count", type=int, default=200, help="Track count for smoke test")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"], help="Which split to extract")
    args = parser.parse_args()

    print("=" * 70)
    print("  High-Throughput Parallel 128-D Narrative Feature Extraction")
    print(f"  Workers: {args.workers} | Chunk Size: {args.chunk_size} | Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}")
    print("=" * 70)

    # 1. Load Split Data
    split_parquet = config.PROCESSED_DATA_DIR / "train_val_split_200k.parquet"
    if not split_parquet.exists():
        raise FileNotFoundError(f"{split_parquet} not found. Run split_dataset_200k.py first.")

    df_all = pd.read_parquet(split_parquet)
    if args.split in ("train", "val"):
        df_target = df_all[df_all["split"] == args.split].copy().reset_index(drop=True)
    else:
        df_target = df_all.copy().reset_index(drop=True)

    if args.smoke:
        df_target = df_target.head(args.smoke_count).copy()

    total_tracks = len(df_target)
    print(f"[Dataset] Target split: '{args.split}' ({total_tracks:,d} tracks)")

    # 2. Setup Chunks Directory & Auto-Resume
    chunks_dir = config.PROCESSED_DATA_DIR / f"narrative_chunks_{args.split}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Check already processed paths from existing chunks
    processed_paths = set()
    existing_chunk_files = sorted(glob.glob(str(chunks_dir / "chunk_*.parquet")))
    if existing_chunk_files:
        print(f"[Resume] Found {len(existing_chunk_files)} existing chunk files. Scanning processed tracks...")
        for cf in existing_chunk_files:
            try:
                cdf = pd.read_parquet(cf, columns=["path"])
                processed_paths.update(cdf["path"].tolist())
            except Exception as e:
                print(f"  [Warning] Could not read {cf}: {e}")
        print(f"[Resume] Already processed: {len(processed_paths):,d} tracks. Resuming remaining tracks.")

    # Filter out already processed tracks
    df_remaining = df_target[~df_target["file_path"].isin(processed_paths)].copy().reset_index(drop=True)
    remaining_count = len(df_remaining)

    if remaining_count == 0:
        print("[Done] All target tracks are already processed in chunks!")
    else:
        print(f"[Queue] Ready to process {remaining_count:,d} tracks using {args.workers} CPU processes...\n")

        # Prepare work items: (file_path, is_ai, strat_class)
        items = [
            (row["file_path"], row["is_ai"], row["strat_class"])
            for _, row in df_remaining.iterrows()
        ]

        # 3. Parallel Extraction with Chunked Output
        t_start = time.time()
        chunk_buffer = []
        chunk_idx = len(existing_chunk_files)
        total_extracted_this_session = 0

        with mp.Pool(processes=args.workers) as pool:
            # Using imap_unordered for optimal load balancing
            for res in pool.imap_unordered(process_single_track, items, chunksize=16):
                if res is not None:
                    chunk_buffer.append(res)
                    total_extracted_this_session += 1

                # Progress reporting every 100 tracks
                if total_extracted_this_session % 100 == 0 or total_extracted_this_session == remaining_count:
                    elapsed = time.time() - t_start
                    speed = total_extracted_this_session / max(elapsed, 0.001)
                    eta_mins = (remaining_count - total_extracted_this_session) / max(speed, 0.001) / 60
                    pct = (len(processed_paths) + total_extracted_this_session) / total_tracks * 100
                    print(
                        f"  [{len(processed_paths) + total_extracted_this_session:6,d}/{total_tracks:,d}] "
                        f"({pct:5.1f}%) | Speed: {speed:4.1f} trk/s | ETA: {eta_mins:5.1f} min"
                    )

                # Save chunk when buffer reaches chunk_size
                if len(chunk_buffer) >= args.chunk_size or (total_extracted_this_session == remaining_count and chunk_buffer):
                    chunk_df = pd.DataFrame(chunk_buffer)
                    chunk_path = chunks_dir / f"chunk_{chunk_idx:05d}.parquet"
                    chunk_df.to_parquet(chunk_path, index=False)
                    print(f"  >>> [Saved Milestone] Chunk {chunk_idx:05d} ({len(chunk_df)} tracks) saved to {chunk_path.name}")
                    chunk_buffer = []
                    chunk_idx += 1

    # 4. Concatenate All Chunks into Final Dataset
    all_chunks = sorted(glob.glob(str(chunks_dir / "chunk_*.parquet")))
    if all_chunks:
        print(f"\n[Consolidate] Merging {len(all_chunks)} chunk files into consolidated dataset...")
        dfs = [pd.read_parquet(f) for f in all_chunks]
        final_df = pd.concat(dfs, ignore_index=True)

        final_parquet = config.PROCESSED_DATA_DIR / f"narrative_vectors_200k_{args.split}.parquet"
        final_df.to_parquet(final_parquet, index=False)
        print(f"[Success] Consolidated {len(final_df):,d} narrative vectors saved to: {final_parquet}")
        print(f"          Feature columns: {len(final_df.columns) - 3} (128 DSP features + metadata)")

    print("=" * 70)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
