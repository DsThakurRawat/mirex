"""Dedicated downloader for Muse (bolshyC/Muse on HuggingFace).

Downloads 116k Suno v5 songs (tar shards of mp3 + jsonl) directly to D:\\mirex\\data\\raw\\muse,
with HF cache pinned to D:\\mirex\\hf_cache to protect C: drive space, automatic resume,
and metadata registration.
"""
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Pin HuggingFace cache to D: drive before importing huggingface_hub
os.environ["HF_HOME"] = str(ROOT / "hf_cache")

import config
from data_fetch import register_muse
from metadata_db import MetadataDatabase
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("fetch_muse")


def main():
    dest_dir = config.RAW_DATA_DIR / "muse"
    dest_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = ROOT / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    logger.info("Target directory: %s", dest_dir)
    logger.info("HuggingFace cache directory: %s", hf_cache)
    logger.info("Starting snapshot download for bolshyC/Muse (~575 GB)...")

    max_retries = 50
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id="bolshyC/Muse",
                repo_type="dataset",
                local_dir=dest_dir,
                max_workers=4,
                resume_download=True
            )
            logger.info("=== Download of bolshyC/Muse completed successfully ===")
            break
        except Exception as e:
            logger.warning("Download attempt %d/%d interrupted: %s. Retrying in 15s...", attempt, max_retries, e)
            time.sleep(15)
    else:
        raise RuntimeError("Failed to complete Muse download after max retries.")

    logger.info("=== Registering Muse into Metadata DB ===")
    db = MetadataDatabase()
    try:
        n = register_muse(db)
        logger.info("Successfully registered %d tracks for Muse.", n)
        db.census(write=True)
    except Exception as e:
        logger.warning("Registration completed with note: %s", e)


if __name__ == "__main__":
    main()
