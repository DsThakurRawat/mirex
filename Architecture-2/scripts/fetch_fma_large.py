"""Dedicated robust downloader for FMA-Large (Kaggle brunosette/fma-large mirror).

Directly downloads fma_large.zip (93.4 GB) from the official UNIL switch cloud
repository with HTTP Range resume support, periodic progress logging, automatic
retries with backoff, checksum verification, automated archive extraction, and
metadata DB registration.
"""
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
from data_fetch import extract_archive, register_fma, verify_checksum
from metadata_db import MetadataDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("fetch_fma_large")

FMA_FILES = [
    {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "filename": "fma_metadata.zip",
        "checksum": "sha1:f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
        "desc": "FMA Metadata (tracks.csv, artists, genres)"
    },
    {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
        "filename": "fma_large.zip",
        "checksum": "sha1:497109f4dd721066b5ce5e5f250ec604dc78939e",
        "desc": "FMA Large Audio (93.4 GB, 106,574 30-second clips)"
    }
]


def download_file_with_progress(url: str, dest: Path, checksum: str | None = None) -> Path:
    """Resumable download with periodic progress logging."""
    import requests

    if dest.exists():
        logger.info("Already present: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    logger.info("Connecting to %s (resume offset: %.2f GB)...", url, offset / (1024**3))

    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if r.status_code == 416:
            logger.info("Server reported range beyond EOF — file already fully downloaded: %s", part)
        else:
            r.raise_for_status()
            content_length = r.headers.get("Content-Length")
            total_expected = (int(content_length) + offset) if content_length else None

            mode = "ab" if (offset and r.status_code == 206) else "wb"
            last_log_time = time.time()
            last_log_bytes = offset
            current_bytes = offset

            with open(part, mode) as f:
                for block in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4MB chunks
                    if not block:
                        continue
                    f.write(block)
                    current_bytes += len(block)

                    now = time.time()
                    if now - last_log_time >= 15.0:  # Log every 15 seconds
                        speed_mb = ((current_bytes - last_log_bytes) / (now - last_log_time)) / (1024 * 1024)
                        if total_expected:
                            pct = (current_bytes / total_expected) * 100.0
                            rem_gb = (total_expected - current_bytes) / (1024**3)
                            eta_sec = (total_expected - current_bytes) / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                            logger.info(
                                "Progress: %.2f / %.2f GB (%.1f%%) | Speed: %.2f MB/s | ETA: %.1f min",
                                current_bytes / (1024**3),
                                total_expected / (1024**3),
                                pct,
                                speed_mb,
                                eta_sec / 60.0
                            )
                        else:
                            logger.info(
                                "Downloaded: %.2f GB | Speed: %.2f MB/s",
                                current_bytes / (1024**3),
                                speed_mb
                            )
                        last_log_time = now
                        last_log_bytes = current_bytes

    if checksum:
        logger.info("Verifying checksum for %s...", part.name)
        verify_checksum(part, checksum)

    part.rename(dest)
    logger.info("Download completed successfully: %s", dest)
    return dest


def robust_download(url: str, dest: Path, checksum: str | None = None, max_retries: int = 50):
    for attempt in range(1, max_retries + 1):
        try:
            return download_file_with_progress(url, dest, checksum)
        except Exception as e:
            logger.warning("Attempt %d/%d failed: %s. Retrying in 10s with resume...", attempt, max_retries, e)
            time.sleep(10)
    raise RuntimeError(f"Failed to download {url} after {max_retries} retries.")


def main():
    dest_dir = config.RAW_DATA_DIR / "fma"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Target directory: %s", dest_dir)
    for item in FMA_FILES:
        target_file = dest_dir / item["filename"]
        marker = dest_dir / ".extracted" / f"{item['filename']}.done"

        if marker.exists():
            logger.info("=== %s already extracted, skipping ===", item["desc"])
            continue

        logger.info("=== Downloading %s ===", item["desc"])
        robust_download(item["url"], target_file, item["checksum"])

        logger.info("=== Extracting %s ===", item["filename"])
        extract_archive(target_file, dest_dir, keep_archive=False)

    logger.info("=== Registering FMA-Large into Metadata DB ===")
    db = MetadataDatabase()
    n = register_fma(db)
    logger.info("Successfully registered %d tracks for fma_large.", n)
    db.census(write=True)


if __name__ == "__main__":
    main()
