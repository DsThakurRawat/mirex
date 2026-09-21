"""
Step 1: Stratified Dataset Splitter for 200,000+ Tracks
Architecture-1 (MusicScope-CL)

Splits available audio on disk into:
  - 85% Training (~183,000 tracks)
  - 15% Validation (~32,000 tracks)
Stratified across all generator families (FMA Human, Suno, Udio, AudioLDM, MusicGen, Mustango, Echoes).
"""
import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import sys
_MIREX_DIR = Path(__file__).resolve().parent.parent / "mirex"
if str(_MIREX_DIR) not in sys.path:
    sys.path.insert(0, str(_MIREX_DIR))

import config

def main():
    print("=" * 70)
    print("  Dataset Partitioning: 200,000+ Track Stratified Split")
    print("=" * 70)

    db_path = config.PROCESSED_DATA_DIR / "metadata.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    print(f"[1/4] Querying tracks from {db_path}...")
    conn = sqlite3.connect(str(db_path))
    query = """
    SELECT track_id, source_dataset, generator_family, is_ai, file_path, duration_s
    FROM tracks
    WHERE file_path IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"      Retrieved {len(df):,} records from database.")

    # Filter for files that exist on disk and are valid audio (> 5KB, no macOS sidecars)
    print("[2/4] Verifying file existence on disk and filtering corrupted/macOS sidecar files...")
    def is_valid_audio_file(fp: str) -> bool:
        if not fp or not os.path.exists(fp):
            return False
        if "__MACOSX" in fp or os.path.basename(fp).startswith("._"):
            return False
        try:
            return os.path.getsize(fp) > 5000
        except OSError:
            return False

    df["valid_audio"] = df["file_path"].apply(is_valid_audio_file)
    df_valid = df[df["valid_audio"]].copy().reset_index(drop=True)
    total_valid = len(df_valid)
    print(f"      Found {total_valid:,} genuine, valid audio files on disk.")

    # Clean family labels for clear stratification
    def get_strat_label(row):
        is_ai = row["is_ai"]
        src = str(row["source_dataset"]).lower()
        fam = str(row["generator_family"]).lower()
        if is_ai == 0:
            return "human_fma"
        if "echoes" in src or "echoes" in fam:
            return "ai_echoes"
        if "sonics" in src:
            if "suno" in fam:
                return "ai_sonics_suno"
            return "ai_sonics_udio"
        if "fakemusiccaps" in src:
            return f"ai_fmc_{fam}"
        return f"ai_{fam}"

    df_valid["strat_class"] = df_valid.apply(get_strat_label, axis=1)

    print("\n[3/4] Stratified Breakdown of Available Files:")
    strat_counts = df_valid["strat_class"].value_counts()
    for sc, cnt in strat_counts.items():
        pct = cnt / total_valid * 100
        print(f"      {sc:<22} : {cnt:6,d} tracks ({pct:5.2f}%)")

    # Filter out any ultra-rare classes with < 10 samples if any
    valid_classes = strat_counts[strat_counts >= 10].index
    df_valid = df_valid[df_valid["strat_class"].isin(valid_classes)].copy().reset_index(drop=True)

    # Perform stratified 85 / 15 split
    print("\n[4/4] Performing 85% Train / 15% Validation stratified split (seed=42)...")
    train_df, val_df = train_test_split(
        df_valid,
        test_size=0.15,
        random_state=config.SEED,
        stratify=df_valid["strat_class"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"

    split_df = pd.concat([train_df, val_df], ignore_index=True)

    # Verify no path leakage
    train_paths = set(train_df["file_path"])
    val_paths = set(val_df["file_path"])
    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage detected between train and val sets!"
    print("      Verification passed: 0% data leakage between train and val.")

    # Save to parquet
    out_parquet = config.PROCESSED_DATA_DIR / "train_val_split_200k.parquet"
    split_df.to_parquet(out_parquet, index=False)
    print(f"\n[Success] Split saved to: {out_parquet}")
    print(f"  Training set   : {len(train_df):,d} tracks ({(len(train_df)/len(split_df)*100):.1f}%)")
    print(f"  Validation set : {len(val_df):,d} tracks ({(len(val_df)/len(split_df)*100):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
