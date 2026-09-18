"""Quarantine protocol — plan §4.1, executed BEFORE anything is trained.

CLI:
    python src/quarantine.py build    # construct + apply the blocklist
    python src/quarantine.py verify   # prove zero overlap; exit 1 otherwise

build (plan §4.1 steps 1-5):
  1. Block all SDD track ids (Zenodo 10072001, full 706-track set) — these
     ARE Jamendo track ids, so both the ``sdd`` namespace and the underlying
     MTG-Jamendo ``track_00xxxxx`` form are blocked.
  2. Block the entire MTG-Jamendo split-0 TEST split (SDD's parent pool).
  3. Resolve every SDD track's artist (SDD's own ``artist_id`` column, plus
     raw.meta.tsv lookup) and block ALL tracks by those artists.
  4. Block AIME's 500 real MTG-Jamendo tracks and Echoes' bona-fide FMA
     tracks (rows already registered by data_fetch with is_ai=0).
  5. Cross-dataset audio dedup of the real class by fingerprint: ``fpcalc``
     (chromaprint) when on PATH, else sha1 of ffmpeg-decoded mono/8 kHz/60 s
     PCM. Identical fingerprints across sources -> keep one, quarantine rest.

verify: recomputes overlap of every trainable (non-quarantined, non-frozen)
row against the blocklist ids/artists/fingerprints and checks fingerprint
uniqueness, writes ``config.QUARANTINE_REPORT`` with ``gate_passed``.

Missing inputs always raise loudly with instructions — this gate must never
pass silently (failure F5 of the plan; the previous implementation's sin).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import config
from data_fetch import MTG_RAW_META_TSV, MTG_SPLIT0_TEST_TSV, SDD_METADATA_CSV
from metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)

FingerprintFn = Callable[[Path], str]


# --- ID canonicalization --------------------------------------------------
# MTG-Jamendo uses "track_0000214" / "artist_000014"; SDD uses bare numeric
# Jamendo ids ("1004034" with artist "368880"). We block every alias form.

def track_aliases(raw: str) -> set[str]:
    """All equivalent spellings of one (possibly Jamendo) track id."""
    raw = str(raw).strip()
    digits = raw[len("track_"):] if raw.startswith("track_") else raw
    if digits.isdigit():
        n = int(digits)
        return {raw, str(n), f"track_{n:07d}"}
    return {raw}


def artist_aliases(raw: str) -> set[str]:
    """All equivalent spellings of one (possibly Jamendo) artist id."""
    raw = str(raw).strip()
    digits = raw[len("artist_"):] if raw.startswith("artist_") else raw
    if digits.isdigit():
        n = int(digits)
        return {raw, str(n), f"artist_{n:06d}"}
    return {raw}


def native_id(db_track_id: str) -> str:
    """'mtg_jamendo:track_0000214' -> 'track_0000214' (plain ids pass through)."""
    return db_track_id.split(":", 1)[1] if ":" in db_track_id else db_track_id


def _require(path: Path, what: str, fix: str) -> Path:
    """Refuse to run on missing inputs — never silently pass the gate."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"QUARANTINE INPUT MISSING: {what} not found at {path}. "
            f"Fix: {fix}. Refusing to build a partial blocklist (plan §4.1).")
    return path


# --- Input readers --------------------------------------------------------

def load_sdd(sdd_csv: Path) -> tuple[set[str], set[str]]:
    """(track ids, artist ids) from song_describer.csv.

    Verified header (Zenodo 10072001): caption_id,track_id,caption,
    is_valid_subset,familiarity,artist_id,album_id,path,duration — track_id
    and artist_id are the underlying Jamendo ids.
    """
    tids: set[str] = set()
    aids: set[str] = set()
    with open(sdd_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"track_id", "artist_id"} - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"{sdd_csv} lacks expected columns {sorted(missing)} "
                f"(got {reader.fieldnames}). Re-download with data_fetch "
                "--dataset sdd.")
        for rec in reader:
            if rec.get("track_id"):
                tids.add(str(rec["track_id"]).strip())
            if rec.get("artist_id"):
                aids.add(str(rec["artist_id"]).strip())
    if not tids:
        raise RuntimeError(f"{sdd_csv} parsed to zero track ids — corrupt "
                           "download? Refusing to continue.")
    return tids, aids


def load_mtg_test_ids(test_tsv: Path) -> set[str]:
    """TRACK_IDs of MTG-Jamendo split-0 autotagging-test.tsv (col 0)."""
    with open(test_tsv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if not header or header[0] != "TRACK_ID":
            raise RuntimeError(f"{test_tsv}: unexpected header {header[:3]}")
        ids = {rec[0].strip() for rec in reader if rec and rec[0].strip()}
    if not ids:
        raise RuntimeError(f"{test_tsv} parsed to zero ids.")
    return ids


def load_mtg_track_to_artist(meta_tsv: Path) -> dict[str, str]:
    """TRACK_ID -> ARTIST_ID from raw.meta.tsv (verified header)."""
    mapping: dict[str, str] = {}
    with open(meta_tsv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # TRACK_ID ARTIST_ID ALBUM_ID TRACK_NAME ...
        for rec in reader:
            if len(rec) >= 2:
                mapping[rec[0].strip()] = rec[1].strip()
    return mapping


# --- Fingerprinting (§4.1 step 5) ----------------------------------------

def fingerprint_file(path: Path) -> str:
    """Chromaprint via ``fpcalc`` when available; else sha1 of ffmpeg-decoded
    (mono, 8 kHz, first 60 s) PCM. Raises when neither tool exists."""
    path = Path(path)
    if shutil.which("fpcalc"):
        out = subprocess.run(["fpcalc", "-length", "60", str(path)],
                             capture_output=True, text=True, check=True)
        for line in out.stdout.splitlines():
            if line.startswith("FINGERPRINT="):
                return "cp:" + line.split("=", 1)[1].strip()
        raise RuntimeError(f"fpcalc produced no FINGERPRINT for {path}")
    if shutil.which("ffmpeg"):
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
             "-ar", "8000", "-t", "60", "-f", "s16le", "-"],
            capture_output=True, check=True)
        return "pcm8k:" + hashlib.sha1(out.stdout).hexdigest()
    raise RuntimeError(
        "Neither fpcalc (chromaprint) nor ffmpeg is on PATH — cannot "
        "fingerprint for dedup. Install one (apt install libchromaprint-tools "
        "or ffmpeg) and re-run quarantine build.")


def _store_fingerprints(db: MetadataDatabase, fp_by_id: dict[str, str]) -> None:
    """Persist computed fingerprints (metadata_db has no setter; direct SQL)."""
    if not fp_by_id:
        return
    conn = sqlite3.connect(db.db_path)
    try:
        conn.executemany("UPDATE tracks SET fingerprint=? WHERE track_id=?",
                         [(fp, tid) for tid, fp in fp_by_id.items()])
        conn.commit()
    finally:
        conn.close()


# --- build ---------------------------------------------------------------

def build(db: MetadataDatabase,
          sdd_csv: Path = SDD_METADATA_CSV,
          mtg_test_tsv: Path = MTG_SPLIT0_TEST_TSV,
          mtg_meta_tsv: Path = MTG_RAW_META_TSV,
          blocklist_path: Path = config.QUARANTINE_FILE,
          fingerprint_fn: FingerprintFn = fingerprint_file,
          run_dedup: bool = True) -> dict:
    """Build the blocklist, apply it to the DB, persist it as JSON (§4.1)."""
    sdd_csv = _require(sdd_csv, "SDD metadata CSV",
                       "python src/data_fetch.py --dataset sdd")
    mtg_test_tsv = _require(mtg_test_tsv, "MTG-Jamendo split-0 test TSV",
                            "python src/data_fetch.py --dataset mtg_jamendo")
    mtg_meta_tsv = _require(mtg_meta_tsv, "MTG-Jamendo raw.meta.tsv",
                            "python src/data_fetch.py --dataset mtg_jamendo")

    blocked_ids: set[str] = set()
    blocked_artists: set[str] = set()

    # Steps 1-3: SDD ids + split-0 test ids + SDD artists (artist-level).
    sdd_tids, sdd_aids = load_sdd(sdd_csv)
    for tid in sdd_tids:
        blocked_ids |= track_aliases(tid)
    logger.info("Step 1: blocked %d SDD tracks (all alias forms)", len(sdd_tids))

    test_ids = load_mtg_test_ids(mtg_test_tsv)
    for tid in test_ids:
        blocked_ids |= track_aliases(tid)
    logger.info("Step 2: blocked %d MTG-Jamendo split-0 TEST tracks",
                len(test_ids))

    track2artist = load_mtg_track_to_artist(mtg_meta_tsv)
    resolved = {track2artist[a] for tid in sdd_tids
                for a in track_aliases(tid) if a in track2artist}
    for aid in sdd_aids | resolved:
        blocked_artists |= artist_aliases(aid)
    logger.info("Step 3: blocked %d SDD artists (%d via raw.meta.tsv lookup)",
                len(sdd_aids | resolved), len(resolved))

    # Step 4: AIME real (MTG-Jamendo) + Echoes bona-fide (FMA) rows.
    eval_real = db.fetch(
        "source_dataset IN ('aime','echoes') AND is_ai=0")
    for row in eval_real:
        blocked_ids |= track_aliases(native_id(row["track_id"]))
        extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
        for key in ("native_id", "jamendo_id", "fma_id", "original_audio"):
            if extra.get(key):
                blocked_ids |= track_aliases(Path(str(extra[key])).stem)
    logger.info("Step 4: blocked %d AIME/Echoes eval-real rows", len(eval_real))

    # Apply id/artist blocking to the DB.
    hit_ids = [row["track_id"] for row in db.fetch()
               if track_aliases(native_id(row["track_id"])) & blocked_ids]
    db.mark_quarantined(hit_ids)
    db.quarantine_by_artists(blocked_artists)
    logger.info("Applied: %d rows quarantined by id; artist filter over %d "
                "artist aliases", len(hit_ids), len(blocked_artists))

    # Fingerprints of id/artist-blocked rows guard against re-entry of the
    # same audio under a different id/source. Rows quarantined purely as
    # dedup duplicates are excluded: their kept twin is legitimately
    # trainable and must not trip the fingerprint gate on rebuilds.
    blocked_fps = {
        row["fingerprint"] for row in db.fetch("quarantined=1")
        if row["fingerprint"] and (
            track_aliases(native_id(row["track_id"])) & blocked_ids
            or (row["artist_id"]
                and artist_aliases(row["artist_id"]) & blocked_artists))}

    # Step 5: cross-dataset dedup of the surviving real class.
    duplicate_fps: set[str] = set()
    if run_dedup:
        duplicate_fps = _dedup_real_class(db, fingerprint_fn)

    blocklist = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan_section": "4.1",
        "counts": {
            "sdd_tracks": len(sdd_tids),
            "mtg_split0_test_tracks": len(test_ids),
            "blocked_artists": len(blocked_artists),
            "aime_echoes_real_rows": len(eval_real),
            "db_rows_quarantined_by_id": len(hit_ids),
            "duplicate_fingerprint_groups": len(duplicate_fps),
        },
        "blocked_track_ids": sorted(blocked_ids),
        "blocked_artist_ids": sorted(blocked_artists),
        "blocked_fingerprints": sorted(blocked_fps),
        "duplicate_fingerprints": sorted(duplicate_fps),
    }
    blocklist_path = Path(blocklist_path)
    blocklist_path.parent.mkdir(parents=True, exist_ok=True)
    blocklist_path.write_text(json.dumps(blocklist, indent=2))
    logger.info("Blocklist written to %s (%d ids, %d artists, %d fps)",
                blocklist_path, len(blocked_ids), len(blocked_artists),
                len(blocked_fps))
    return blocklist


def _dedup_real_class(db: MetadataDatabase,
                      fingerprint_fn: FingerprintFn) -> set[str]:
    """§4.1 step 5: fingerprint non-quarantined real tracks; for identical
    fingerprints across rows keep the lexicographically-first track_id,
    quarantine the rest. Returns the duplicated fingerprint values."""
    rows = db.fetch("is_ai=0 AND quarantined=0")
    fp_by_id: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for row in rows:
        fp = row["fingerprint"]
        if not fp:
            path = Path(row["file_path"])
            if not path.is_file():
                skipped += 1
                continue
            fp = fingerprint_fn(path)
            fp_by_id[row["track_id"]] = fp
        groups[fp].append(row["track_id"])
    _store_fingerprints(db, fp_by_id)
    if skipped:
        logger.warning("Dedup: %d real tracks had no local audio file and "
                        "were NOT fingerprinted — re-run quarantine build "
                        "after their audio lands.", skipped)
    dup_fps: set[str] = set()
    to_quarantine: list[str] = []
    for fp, tids in groups.items():
        if len(tids) > 1:
            keep, *rest = sorted(tids)
            dup_fps.add(fp)
            to_quarantine.extend(rest)
            logger.info("Dedup: fingerprint shared by %s — keeping %s, "
                        "quarantining %s", tids, keep, rest)
    db.mark_quarantined(to_quarantine)
    logger.info("Dedup: %d duplicate rows quarantined (%d groups)",
                len(to_quarantine), len(dup_fps))
    return dup_fps


# --- verify ---------------------------------------------------------------

def verify(db: MetadataDatabase,
           blocklist_path: Path = config.QUARANTINE_FILE,
           report_path: Path = config.QUARANTINE_REPORT) -> dict:
    """Acceptance check (§4.1): zero id/artist/fingerprint overlap between
    trainable rows and the blocklist, plus fingerprint uniqueness. Writes
    ``report_path`` with an explicit ``gate_passed`` bool."""
    blocklist_path = Path(blocklist_path)
    if not blocklist_path.is_file():
        raise RuntimeError(
            f"Blocklist {blocklist_path} does not exist — run "
            "`python src/quarantine.py build` first. verify() will NOT "
            "pass a gate that was never built.")
    blocklist = json.loads(blocklist_path.read_text())
    blocked_ids = set(blocklist["blocked_track_ids"])
    blocked_artists = set(blocklist["blocked_artist_ids"])
    blocked_fps = set(blocklist["blocked_fingerprints"])

    rows = db.trainable()
    violations: list[dict] = []
    counts = Counter()
    fp_counter: Counter = Counter(
        row["fingerprint"] for row in rows if row["fingerprint"])
    for row in rows:
        tid = row["track_id"]
        if track_aliases(native_id(tid)) & blocked_ids:
            counts["id_overlap"] += 1
            violations.append({"track_id": tid, "reason": "blocked_track_id"})
        if row["artist_id"] and \
                artist_aliases(row["artist_id"]) & blocked_artists:
            counts["artist_overlap"] += 1
            violations.append({"track_id": tid, "reason": "blocked_artist",
                               "artist_id": row["artist_id"]})
        fp = row["fingerprint"]
        if fp and fp in blocked_fps:
            counts["fingerprint_overlap"] += 1
            violations.append({"track_id": tid,
                               "reason": "blocked_fingerprint"})
        if fp and fp_counter[fp] > 1:
            counts["duplicate_fingerprint"] += 1
            violations.append({"track_id": tid,
                               "reason": "duplicate_fingerprint_in_pool"})

    gate_passed = not violations
    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan_section": "4.1 acceptance check",
        "trainable_rows_checked": len(rows),
        "counts": dict(counts),
        "violations_sample": violations[:50],
        "total_violations": len(violations),
        "gate_passed": gate_passed,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    if gate_passed:
        logger.info("QUARANTINE GATE PASSED: %d trainable rows, zero overlap "
                    "(report: %s)", len(rows), report_path)
    else:
        logger.error("QUARANTINE GATE FAILED: %d violations across %d rows "
                     "(report: %s)", len(violations), len(rows), report_path)
    return report


# --- CLI ------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine protocol (plan §4.1): build | verify")
    parser.add_argument("command", choices=["build", "verify"])
    args = parser.parse_args(argv)

    config.ensure_dirs()
    db = MetadataDatabase()
    if args.command == "build":
        build(db)
        report = verify(db)          # immediate acceptance check after build
    else:
        report = verify(db)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
