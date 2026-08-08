"""SQLite metadata database: provenance for every audio file in the project.

Every track that enters any training/eval pool MUST be registered here with
source, generator family/version, license, and (once computed) fingerprint.
The quarantine gate and the census report both read from this DB.
"""
import json
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

from config import METADATA_DB, CENSUS_REPORT

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,          -- globally unique: "<source>:<native id>"
    source_dataset TEXT NOT NULL,       -- e.g. muse, sonics, fma, mtg_jamendo, gen_ace_step
    generator_family TEXT NOT NULL,     -- suno/udio/.../human
    generator_version TEXT,             -- e.g. v5, v3.5, unknown
    is_ai INTEGER NOT NULL,             -- 0 real, 1 AI
    artist_id TEXT,                     -- real music only; for artist-level splits
    license TEXT,
    file_path TEXT NOT NULL,
    duration_s REAL,
    sample_rate INTEGER,
    channels INTEGER,
    fingerprint TEXT,                   -- chromaprint (or fallback content hash)
    quarantined INTEGER DEFAULT 0,      -- 1 = blocked, never load for training
    split TEXT DEFAULT 'unassigned',    -- train / dev_frozen / unassigned
    extra_json TEXT                     -- generator settings, prompts, etc.
);
CREATE INDEX IF NOT EXISTS idx_tracks_family ON tracks(generator_family);
CREATE INDEX IF NOT EXISTS idx_tracks_source ON tracks(source_dataset);
CREATE INDEX IF NOT EXISTS idx_tracks_split ON tracks(split);
CREATE INDEX IF NOT EXISTS idx_tracks_fp ON tracks(fingerprint);
"""


class MetadataDatabase:
    def __init__(self, db_path: Path = METADATA_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_tracks(self, rows: list[dict]):
        """Bulk insert. Each row needs: track_id, source_dataset,
        generator_family, is_ai, file_path. The rest is optional."""
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO tracks
                   (track_id, source_dataset, generator_family, generator_version,
                    is_ai, artist_id, license, file_path, duration_s, sample_rate,
                    channels, fingerprint, quarantined, split, extra_json)
                   VALUES (:track_id, :source_dataset, :generator_family,
                           :generator_version, :is_ai, :artist_id, :license,
                           :file_path, :duration_s, :sample_rate, :channels,
                           :fingerprint, :quarantined, :split, :extra_json)""",
                [{"generator_version": None, "artist_id": None, "license": None,
                  "duration_s": None, "sample_rate": None, "channels": None,
                  "fingerprint": None, "quarantined": 0, "split": "unassigned",
                  "extra_json": None, **r} for r in rows])

    def mark_quarantined(self, track_ids: list[str]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(
                "UPDATE tracks SET quarantined=1 WHERE track_id=?",
                [(t,) for t in track_ids])
            return cur.rowcount

    def quarantine_by_artists(self, artist_ids: set[str]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(
                "UPDATE tracks SET quarantined=1 WHERE artist_id=?",
                [(a,) for a in artist_ids])
            return cur.rowcount

    def fetch(self, where: str = "1=1", params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM tracks WHERE {where}", params).fetchall()
            return [dict(r) for r in rows]

    def trainable(self) -> list[dict]:
        """Everything eligible for training: not quarantined, not frozen dev."""
        return self.fetch("quarantined=0 AND split != 'dev_frozen'")

    def assign_split(self, track_ids: list[str], split: str):
        with self._conn() as conn:
            conn.executemany("UPDATE tracks SET split=? WHERE track_id=?",
                             [(split, t) for t in track_ids])

    def census(self, write: bool = True) -> dict:
        """Per source x family counts/hours + quarantine totals (P1 gate)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT source_dataset, generator_family, is_ai, quarantined,
                          COUNT(*) AS n, SUM(COALESCE(duration_s,0))/3600.0 AS hours
                   FROM tracks
                   GROUP BY source_dataset, generator_family, is_ai, quarantined
                """).fetchall()
        report = {"by_source": [dict(r) for r in rows]}
        report["total_tracks"] = sum(r["n"] for r in rows)
        report["total_quarantined"] = sum(r["n"] for r in rows if r["quarantined"])
        report["ai_tracks"] = sum(r["n"] for r in rows
                                  if r["is_ai"] and not r["quarantined"])
        report["real_tracks"] = sum(r["n"] for r in rows
                                    if not r["is_ai"] and not r["quarantined"])
        if write:
            CENSUS_REPORT.parent.mkdir(parents=True, exist_ok=True)
            CENSUS_REPORT.write_text(json.dumps(report, indent=2))
            logger.info("Census written to %s", CENSUS_REPORT)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = MetadataDatabase()
    print(json.dumps(db.census(write=False), indent=2))
