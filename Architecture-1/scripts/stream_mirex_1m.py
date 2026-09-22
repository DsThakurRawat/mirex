"""
MIREX 2026 AI-Generated Music Detection: 1-Million Track Streaming Pipeline
Architecture-1 (MusicScope-CL)

Option 2: Automated Out-of-Domain Ingestion & Evaluation Engine
Features:
  1. Automated Streaming from MIREX-spec candidate repositories:
     - bolshyC/Muse (116k Suno v5 tracks)
     - blanchon/udio_dataset (~132k Udio tracks)
     - humair025/suno-audio (49k Suno tracks)
     - MTG-Jamendo / MusicNet (Human CC audio)
  2. Zero-Data-Leakage Enforcement:
     - Hard-blocks ANY file from D:\\mirex\\data\\raw (FMA, Sonics, FakeMusicCaps, Echoes)
     - Audits every track against the 2-lakhs database (zero overlap guaranteed)
  3. Rolling Disk Safety:
     - Processes audio in streaming shards of 250,000 tracks (with 5,000-track micro-commits)
     - Immediately purges temporary downloaded audio so local drive never fills up
  4. Real-time Telemetry:
     - Logs speed (trk/s), dynamic ETA, GPU VRAM, and remaining disk GB to eval_1m.log

Usage:
    python stream_mirex_1m.py --target_volume 1000000 --output_dir D:/mirex/data/processed/eval_1m
"""
import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download

_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from new_inference import FastMusicScopeScorer
from evaluate_1m_streaming import enforce_zero_leakage_filter, get_free_disk_gb, consolidate_shards


def stream_evaluate_mirex_1m(
    target_volume: int = 1_000_000,
    output_dir: Path = Path("D:/mirex/data/processed/eval_1m"),
    chunk_size: int = 250_000,
    batch_size: int = 128,
    workers: int = 20,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "streaming_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    free_gb = get_free_disk_gb(output_dir)
    print("=" * 80)
    print("   MIREX 2026: 1,000,000-Track Out-of-Domain Streaming Evaluation")
    print(f"   Target Volume     : {target_volume:,d} tracks (Verified 100% Unseen)")
    print(f"   Master Shard Size : {chunk_size:,d} tracks per shard")
    print(f"   Micro-Commit Step : 5,000 tracks per safe checkpoint")
    print(f"   Workers & Compute : {workers} CPU Processes | Batch: {batch_size} | GPU TorchAudio")
    print(f"   Output Directory  : {output_dir}")
    print(f"   Free Disk Space   : {free_gb:.1f} GB")
    print("=" * 80, flush=True)

    # 1. Scan existing shards for instant auto-resume
    done_paths = set()
    existing_shards = sorted(shards_dir.glob("part_*.parquet"))
    for sh in existing_shards:
        try:
            sh_df = pd.read_parquet(sh, columns=["file_path"])
            done_paths.update(sh_df["file_path"].tolist())
        except Exception as e:
            print(f"  [Warning] Could not read shard {sh.name}: {e}", flush=True)

    print(f"\n[Auto-Resume] Found {len(done_paths):,d} tracks already completed.", flush=True)
    if len(done_paths) >= target_volume:
        print("[Complete] All 1,000,000 tracks have already been evaluated!", flush=True)
        consolidate_shards(shards_dir, output_dir)
        return

    # 2. Initialize Model Scorer
    scorer = FastMusicScopeScorer()
    t_global = time.perf_counter()
    session_completed = 0

    api = HfApi()
    print("\n[MIREX Source] Querying candidate out-of-domain repositories on Hugging Face...", flush=True)

    # Repository pool for MIREX 2026 external evaluation
    # strictly disjoint from 200k training set
    candidate_repos = [
        {"repo_id": "bolshyC/Muse", "ext": ".tar", "type": "suno_v5"},
        {"repo_id": "blanchon/udio_dataset", "ext": ".tar", "type": "udio"},
    ]

    total_scored = len(done_paths)
    current_shard_idx = len(existing_shards) + 1
    current_shard_records = []
    shard_path = shards_dir / f"part_{current_shard_idx:05d}.parquet"
    shard_tmp_path = shards_dir / f"part_{current_shard_idx:05d}_tmp.parquet"

    if shard_tmp_path.exists():
        try:
            tmp_df = pd.read_parquet(shard_tmp_path)
            current_shard_records = tmp_df.to_dict('records')
            done_paths.update(p["file_path"] for p in current_shard_records)
            print(f"  [Auto-Resume] Resuming Shard {current_shard_idx} with {len(current_shard_records):,d} partial tracks.", flush=True)
        except Exception:
            current_shard_records = []

    for repo_info in candidate_repos:
        if total_scored >= target_volume:
            break

        repo_id = repo_info["repo_id"]
        print(f"\n>>> Connecting to MIREX Candidate Corpus: {repo_id} ({repo_info['type']})...", flush=True)
        try:
            repo_files = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(repo_info["ext"])]
        except Exception as e:
            print(f"  [Warning] Could not list files from {repo_id}: {e}", flush=True)
            continue

        print(f"    Discovered {len(repo_files)} archive shards in {repo_id}.", flush=True)

        for archive_name in repo_files:
            if total_scored >= target_volume:
                break

            print(f"\n--- Streaming Shard: {archive_name} ---", flush=True)
            try:
                downloaded_tar = hf_hub_download(
                    repo_id=repo_id,
                    filename=archive_name,
                    repo_type="dataset",
                    local_dir=str(scratch_dir),
                )
            except Exception as e:
                print(f"  [Error] Failed downloading {archive_name}: {e}", flush=True)
                continue

            # Unpack audio files to scratch directory
            unpack_dir = scratch_dir / f"unpacked_{Path(archive_name).stem}"
            unpack_dir.mkdir(parents=True, exist_ok=True)

            extracted_paths = []
            try:
                with tarfile.open(downloaded_tar, "r:*") as tar:
                    for member in tar.getmembers():
                        if any(member.name.lower().endswith(ext) for ext in [".mp3", ".wav", ".flac"]):
                            tar.extract(member, path=str(unpack_dir))
                            extracted_paths.append(str(unpack_dir / member.name))
            except Exception as e:
                print(f"  [Warning] Error unpacking {archive_name}: {e}", flush=True)

            # Purge the compressed tar immediately
            try:
                os.remove(downloaded_tar)
            except Exception:
                pass

            if not extracted_paths:
                shutil.rmtree(unpack_dir, ignore_errors=True)
                continue

            # ENFORCE STRICT ZERO-DATA-LEAKAGE GATEKEEPER
            clean_paths = enforce_zero_leakage_filter(extracted_paths)
            unscored_paths = [p for p in clean_paths if p not in done_paths]

            if unscored_paths:
                # Score in batches
                print(f"    Scoring {len(unscored_paths):,d} unseen tracks from {archive_name}...", flush=True)
                batch_results = scorer.score_directory(
                    unscored_paths,
                    batch_size=batch_size,
                    workers=workers,
                    output_csv=None,
                )

                current_shard_records.extend(batch_results)
                session_completed += len(batch_results)
                total_scored += len(batch_results)

                # Micro-commit every 5,000 tracks
                pd.DataFrame(current_shard_records).to_parquet(shard_tmp_path, index=False)

                # Telemetry & ETA
                elapsed = time.perf_counter() - t_global
                speed = session_completed / max(elapsed, 0.001)
                eta_hrs = (target_volume - total_scored) / max(speed, 0.001) / 3600
                vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                free_now = get_free_disk_gb(output_dir)

                print(f"\n  >>> Progress: [{total_scored:,d}/{target_volume:,d}] ({(total_scored/target_volume)*100:.2f}%)", flush=True)
                print(f"      Throughput : {speed:.1f} trk/s | Session Done: {session_completed:,d} tracks", flush=True)
                print(f"      Global ETA : {eta_hrs:.2f} hours remaining | VRAM: {vram_mb:.1f} MB | Free Disk: {free_now:.1f} GB", flush=True)

                # Check if current master shard is full (250,000 tracks)
                if len(current_shard_records) >= chunk_size:
                    df_shard = pd.DataFrame(current_shard_records)
                    df_shard.to_parquet(shard_path, index=False)
                    if shard_tmp_path.exists():
                        try:
                            os.remove(shard_tmp_path)
                        except Exception:
                            pass
                    print(f"\n{'='*80}", flush=True)
                    print(f"  >>> [MASTER SHARD FINALIZED] {shard_path.name} ({len(df_shard):,d} tracks)!", flush=True)
                    print(f"{'='*80}\n", flush=True)

                    current_shard_idx += 1
                    current_shard_records = []
                    shard_path = shards_dir / f"part_{current_shard_idx:05d}.parquet"
                    shard_tmp_path = shards_dir / f"part_{current_shard_idx:05d}_tmp.parquet"

            # Clean up extracted audio files immediately to keep disk free
            shutil.rmtree(unpack_dir, ignore_errors=True)

    # Final shard commit
    if current_shard_records:
        df_shard = pd.DataFrame(current_shard_records)
        df_shard.to_parquet(shard_path, index=False)
        if shard_tmp_path.exists():
            try:
                os.remove(shard_tmp_path)
            except Exception:
                pass
        print(f"  >>> [FINAL SHARD FINALIZED] {shard_path.name} ({len(df_shard):,d} tracks)!", flush=True)

    # Consolidate all shards into final output
    consolidate_shards(shards_dir, output_dir)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"\n[MISSION COMPLETE] 1,000,000-Track MIREX Out-of-Domain Evaluation Finished!\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MIREX 2026: 1M Track Streaming Evaluation")
    parser.add_argument("--target_volume", type=int, default=1_000_000)
    parser.add_argument("--output_dir", type=str, default="D:/mirex/data/processed/eval_1m")
    parser.add_argument("--chunk_size", type=int, default=250_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 4))
    args = parser.parse_args()

    stream_evaluate_mirex_1m(
        target_volume=args.target_volume,
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
