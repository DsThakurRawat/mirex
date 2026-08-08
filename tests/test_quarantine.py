"""Tests for the quarantine protocol (plan §4.1) with tiny synthetic fixtures.

Covers: artist-level blocking, cross-dataset dedup (duplicate quarantined,
original kept), verify() failing when a blocked id sneaks into trainable(),
verify() passing when clean, and loud failure on missing inputs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import quarantine  # noqa: E402
from metadata_db import MetadataDatabase  # noqa: E402

# --- Fixture data ---------------------------------------------------------
# SDD track 1004034 belongs to artist 368880 (real header layout verified
# against Zenodo record 10072001). MTG-Jamendo uses padded forms.

SDD_HEADER = ("caption_id,track_id,caption,is_valid_subset,familiarity,"
              "artist_id,album_id,path,duration")
SDD_ROWS = [
    '859,1004034,"electronic tune",True,0,368880,118196,34/1004034.mp3,202.5',
    '990,1007274,"acoustic guitar",True,1,429137,118548,74/1007274.mp3,140.2',
]

MTG_TEST_HEADER = "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS"
MTG_TEST_ROWS = [
    "track_0000214\tartist_000014\talbum_000031\t14/214.mp3\t124.6\tgenre---punkrock",
]

MTG_META_HEADER = ("TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME"
                   "\tALBUM_NAME\tRELEASEDATE\tURL")
MTG_META_ROWS = [
    "track_1004034\tartist_368880\talbum_118196\tSong A\tArtist X\tAlbum\t2020-01-01\tu",
    "track_0000999\tartist_368880\talbum_000001\tSong B\tArtist X\tAlbum\t2020-01-01\tu",
    "track_0000214\tartist_000014\talbum_000031\tIntro\tDavid\tPouce\t2004-12-28\tu",
    "track_0000777\tartist_000777\talbum_000002\tSong C\tArtist Y\tAlbum\t2021-01-01\tu",
]


def sha1_of_file(path: Path) -> str:
    """Deterministic stand-in fingerprint (no ffmpeg/fpcalc needed in CI)."""
    return "test:" + hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _track(track_id: str, source: str, *, is_ai: int = 0,
           artist_id: str | None = None, file_path: str = "/nonexistent.mp3",
           extra: dict | None = None) -> dict:
    return {"track_id": track_id, "source_dataset": source,
            "generator_family": "suno" if is_ai else "human",
            "is_ai": is_ai, "artist_id": artist_id, "file_path": file_path,
            "extra_json": json.dumps(extra) if extra else None}


@pytest.fixture()
def env(tmp_path: Path):
    """Temp DB + tiny fake SDD CSV / MTG TSVs + audio files for dedup."""
    db = MetadataDatabase(tmp_path / "meta.db")

    sdd_csv = tmp_path / "song_describer.csv"
    sdd_csv.write_text("\n".join([SDD_HEADER, *SDD_ROWS]) + "\n")
    test_tsv = tmp_path / "autotagging-test.tsv"
    test_tsv.write_text("\n".join([MTG_TEST_HEADER, *MTG_TEST_ROWS]) + "\n")
    meta_tsv = tmp_path / "raw.meta.tsv"
    meta_tsv.write_text("\n".join([MTG_META_HEADER, *MTG_META_ROWS]) + "\n")

    # Audio content: dup1 == dup2 (cross-dataset duplicate), solo distinct.
    dup1 = tmp_path / "dup1.mp3"
    dup2 = tmp_path / "dup2.mp3"
    solo = tmp_path / "solo.mp3"
    dup1.write_bytes(b"IDENTICAL-AUDIO-PAYLOAD")
    dup2.write_bytes(b"IDENTICAL-AUDIO-PAYLOAD")
    solo.write_bytes(b"UNIQUE-AUDIO-PAYLOAD")

    db.insert_tracks([
        # SDD leak (id-level) + split-0 test leak (id-level).
        _track("mtg_jamendo:track_1004034", "mtg_jamendo",
               artist_id="artist_368880"),
        _track("mtg_jamendo:track_0000214", "mtg_jamendo",
               artist_id="artist_000014"),
        # Same artist as an SDD track -> must be blocked at ARTIST level.
        _track("mtg_jamendo:track_0000999", "mtg_jamendo",
               artist_id="artist_368880"),
        # Clean real track (different artist), with real audio bytes.
        _track("mtg_jamendo:track_0000777", "mtg_jamendo",
               artist_id="artist_000777", file_path=str(solo)),
        # Cross-dataset duplicate pair ("fma:200" sorts before "musicnet:300").
        _track("fma:200", "fma", artist_id="fma_artist_1",
               file_path=str(dup1)),
        _track("musicnet:300", "musicnet", artist_id="composer_Bach",
               file_path=str(dup2)),
        # AIME eval-real row carrying its native Jamendo id (step 4).
        _track("aime:real_55", "aime", extra={"native_id": "1007274"}),
        # An AI-side row: untouched by real-class dedup / artist blocking.
        _track("muse:cn/000001", "muse", is_ai=1),
    ])

    def run_build():
        return quarantine.build(
            db, sdd_csv=sdd_csv, mtg_test_tsv=test_tsv, mtg_meta_tsv=meta_tsv,
            blocklist_path=tmp_path / "blocklist.json",
            fingerprint_fn=sha1_of_file)

    def run_verify():
        return quarantine.verify(
            db, blocklist_path=tmp_path / "blocklist.json",
            report_path=tmp_path / "report.json")

    return {"db": db, "tmp": tmp_path, "build": run_build,
            "verify": run_verify}


def _quarantined(db: MetadataDatabase) -> dict[str, int]:
    return {r["track_id"]: r["quarantined"] for r in db.fetch()}


# --- build ----------------------------------------------------------------

def test_missing_inputs_raise_loudly(env, tmp_path):
    """§4.1: a missing SDD CSV must abort, never silently pass the gate."""
    with pytest.raises(RuntimeError, match="QUARANTINE INPUT MISSING"):
        quarantine.build(
            env["db"], sdd_csv=tmp_path / "does_not_exist.csv",
            mtg_test_tsv=tmp_path / "autotagging-test.tsv",
            mtg_meta_tsv=tmp_path / "raw.meta.tsv",
            blocklist_path=tmp_path / "bl.json",
            fingerprint_fn=sha1_of_file)


def test_id_level_blocking(env):
    """SDD ids (Jamendo alias forms) + split-0 test ids are quarantined."""
    env["build"]()
    q = _quarantined(env["db"])
    assert q["mtg_jamendo:track_1004034"] == 1   # SDD track, padded alias
    assert q["mtg_jamendo:track_0000214"] == 1   # split-0 test track
    assert q["aime:real_55"] == 1                # AIME eval-real row (step 4)


def test_artist_level_blocking(env):
    """§4.1 step 3: ALL tracks by SDD artists are blocked; others are not."""
    env["build"]()
    q = _quarantined(env["db"])
    assert q["mtg_jamendo:track_0000999"] == 1   # same artist as SDD track
    assert q["mtg_jamendo:track_0000777"] == 0   # unrelated artist stays
    assert q["muse:cn/000001"] == 0              # AI side untouched


def test_dedup_quarantines_duplicate_not_original(env):
    """§4.1 step 5: identical audio across sources -> keep one copy."""
    env["build"]()
    q = _quarantined(env["db"])
    assert q["fma:200"] == 0          # kept (lexicographically first)
    assert q["musicnet:300"] == 1     # duplicate quarantined
    fps = {r["track_id"]: r["fingerprint"] for r in env["db"].fetch()}
    assert fps["fma:200"] == fps["musicnet:300"] is not None


def test_blocklist_file_contents(env, tmp_path):
    env["build"]()
    bl = json.loads((tmp_path / "blocklist.json").read_text())
    assert "track_1004034" in bl["blocked_track_ids"]     # padded alias
    assert "1004034" in bl["blocked_track_ids"]           # bare Jamendo id
    assert "artist_368880" in bl["blocked_artist_ids"]
    assert bl["counts"]["duplicate_fingerprint_groups"] == 1


# --- verify ---------------------------------------------------------------

def test_verify_passes_when_clean(env, tmp_path):
    env["build"]()
    report = env["verify"]()
    assert report["gate_passed"] is True
    assert report["total_violations"] == 0
    assert json.loads((tmp_path / "report.json").read_text())["gate_passed"]


def test_verify_fails_when_blocked_id_sneaks_into_trainable(env):
    """Un-quarantining a blocked track must flip the gate to FAILED."""
    env["build"]()
    conn = sqlite3.connect(env["db"].db_path)
    conn.execute("UPDATE tracks SET quarantined=0 "
                 "WHERE track_id='mtg_jamendo:track_1004034'")
    conn.commit()
    conn.close()
    assert any(r["track_id"] == "mtg_jamendo:track_1004034"
               for r in env["db"].trainable())
    report = env["verify"]()
    assert report["gate_passed"] is False
    reasons = {v["reason"] for v in report["violations_sample"]}
    assert "blocked_track_id" in reasons
    # artist-level leak is caught too (same row has a blocked artist)
    assert "blocked_artist" in reasons


def test_verify_fails_on_duplicate_reentry(env):
    """Re-admitting the deduped copy trips the fingerprint-uniqueness check."""
    env["build"]()
    conn = sqlite3.connect(env["db"].db_path)
    conn.execute(
        "UPDATE tracks SET quarantined=0 WHERE track_id='musicnet:300'")
    conn.commit()
    conn.close()
    report = env["verify"]()
    assert report["gate_passed"] is False
    reasons = {v["reason"] for v in report["violations_sample"]}
    assert "duplicate_fingerprint_in_pool" in reasons


def test_verify_without_blocklist_raises(env, tmp_path):
    """verify() must never pass a gate that was never built."""
    with pytest.raises(RuntimeError, match="run"):
        quarantine.verify(env["db"],
                          blocklist_path=tmp_path / "missing.json",
                          report_path=tmp_path / "report.json")


def test_build_is_idempotent(env):
    env["build"]()
    first = _quarantined(env["db"])
    env["build"]()
    assert _quarantined(env["db"]) == first
