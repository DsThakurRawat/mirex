"""Dedicated downloader for Option 1: SONICS (30.4 GB) + AIME (58.0 GB).

Total download: ~88.4 GB of AI-generated music across 12+ architectures,
perfectly balancing the ~93.4 GB FMA-Large real music dataset.
Pins HF cache to D:\\mirex\\hf_cache to protect C: drive space.
"""
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Pin HuggingFace cache to D: drive
os.environ["HF_HOME"] = str(ROOT / "hf_cache")

import config
from data_fetch import register_aime, register_sonics
from metadata_db import MetadataDatabase
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("fetch_option1")


def download_sonics():
    dest_dir = config.RAW_DATA_DIR / "sonics"
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("==================================================")
    logger.info("=== STEP 1: Downloading SONICS (~30.4 GB) ===")
    logger.info("==================================================")

    max_retries = 50
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id="awsaf49/sonics",
                repo_type="dataset",
                local_dir=dest_dir,
                max_workers=4,
            )
            logger.info("SONICS snapshot download complete.")
            break
        except Exception as e:
            logger.warning("SONICS download attempt %d/%d interrupted: %s. Retrying in 10s...", attempt, max_retries, e)
            time.sleep(10)
    else:
        raise RuntimeError("Failed to download SONICS after max retries.")

    # Unpack any fake_songs part zip archives if present to make audio accessible
    zip_files = list(dest_dir.glob("**/*.zip"))
    if zip_files:
        logger.info("Found %d zip archive(s) in SONICS. Unpacking to make audio files ready...", len(zip_files))
        for zf in sorted(zip_files):
            extract_target = zf.parent
            marker = extract_target / ".extracted" / f"{zf.name}.done"
            if marker.exists():
                logger.info("Archive %s already unpacked, skipping.", zf.name)
                continue
            logger.info("Extracting %s -> %s ...", zf.name, extract_target)
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(extract_target)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("done")
            try:
                zf.unlink()
                logger.info("Extracted and removed %s to free disk space.", zf.name)
            except Exception as ex:
                logger.warning("Could not delete %s: %s", zf.name, ex)

    logger.info("=== Registering SONICS into Metadata DB ===")
    db = MetadataDatabase()
    try:
        n = register_sonics(db)
        logger.info("Successfully registered %d tracks for SONICS.", n)
        db.census(write=True)
    except Exception as e:
        logger.warning("SONICS registration completed with note: %s", e)


def download_aime():
    dest_dir = config.RAW_DATA_DIR / "aime"
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("==================================================")
    logger.info("=== STEP 2: Downloading AIME (~58.0 GB) ===")
    logger.info("==================================================")

    max_retries = 50
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id="disco-eth/AIME",
                repo_type="dataset",
                local_dir=dest_dir,
                max_workers=4,
            )
            logger.info("AIME snapshot download complete.")
            break
        except Exception as e:
            logger.warning("AIME download attempt %d/%d interrupted: %s. Retrying in 10s...", attempt, max_retries, e)
            time.sleep(10)
    else:
        raise RuntimeError("Failed to download AIME after max retries.")

    logger.info("=== Registering AIME into Metadata DB ===")
    db = MetadataDatabase()
    try:
        n = register_aime(db)
        logger.info("Successfully registered %d tracks for AIME.", n)
        db.census(write=True)
    except Exception as e:
        logger.warning("AIME registration completed with note: %s", e)


def main():
    logger.info("Starting Option 1 balanced acquisition: SONICS + AIME (~88.4 GB total)")
    download_sonics()
    download_aime()
    logger.info("==================================================")
    logger.info("ALL DONE: Option 1 datasets acquired & registered!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
