"""Dataset acquisition + metadata registration (plan §4.2 AI pool, §4.3 real pool).

CLI:
    python src/data_fetch.py --dataset <name|all> [--metadata-only] [--subset-gb N]
                             [--register-only]

Every dataset lands under ``config.RAW_DATA_DIR/<name>`` (SDD and MTG-Jamendo
use their pinned config dirs) and is then registered into the
``MetadataDatabase`` by its ``register_<dataset>()`` function. Registrars are
independently callable and idempotent (INSERT OR IGNORE keyed on track_id).

All URLs below were verified live on 2026-08-08 (HTTP 200 / repo pages).
Third-party imports (requests, huggingface_hub, pyarrow, pandas, soundfile)
are deliberately lazy so this module imports with stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import config
from metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}
_SMALL_FILE_BYTES = 32 * 2**20  # always fetched by --subset-gb (metadata sidecars)
HF_METADATA_PATTERNS = ["*.json", "*.jsonl", "*.csv", "*.tsv", "*.md", "*.txt"]

# --- Pinned on-disk locations shared with quarantine.py -------------------
MTG_AUTOTAGGING_TSV = config.MTG_JAMENDO_DIR / "autotagging.tsv"
MTG_RAW_META_TSV = config.MTG_JAMENDO_DIR / "raw.meta.tsv"
MTG_SPLIT0_TEST_TSV = config.MTG_JAMENDO_DIR / "autotagging-test.tsv"
SDD_METADATA_CSV = config.SDD_DIR / "song_describer.csv"

_MTG_RAW = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master"


@dataclass(frozen=True)
class RemoteFile:
    """One HTTP-downloadable file. checksum format: '<algo>:<hex>' or None."""
    url: str
    filename: str
    checksum: Optional[str] = None
    metadata: bool = True            # False = bulk audio payload
    approx_gb: float = 0.0           # used by --subset-gb ordering


@dataclass(frozen=True)
class DatasetSpec:
    """Registry entry for one source (plan §4.2/§4.3 tables)."""
    name: str
    kind: str                        # 'hf_dataset' | 'http' | 'metadata_only'
    license: str
    is_ai: Optional[bool]            # None = mixed real/AI content
    generator_family: Optional[str]  # default family; registrars may refine
    generator_version: Optional[str] = None
    repo_id: Optional[str] = None    # HF dataset repo id
    files: tuple[RemoteFile, ...] = ()
    notes: str = ""

    @property
    def dest(self) -> Path:
        if self.name == "sdd":
            return config.SDD_DIR
        if self.name == "mtg_jamendo":
            return config.MTG_JAMENDO_DIR
        return config.RAW_DATA_DIR / self.name


REGISTRY: dict[str, DatasetSpec] = {s.name: s for s in [
    # ------------------------- AI side (§4.2) -----------------------------
    DatasetSpec("muse", "hf_dataset", "MIT", True, "suno", "v5",
                repo_id="bolshyC/Muse",
                notes="116k Suno v5 songs (CN+EN), tar shards of mp3 + jsonl."),
    DatasetSpec("suno_audio", "hf_dataset", "MIT", True, "suno", None,
                repo_id="humair025/suno-audio",
                notes="49.7k tracks, parquet with per-track model_name."),
    DatasetSpec("udio", "hf_dataset",
                "CC0-1.0 (compilation/metadata; audio rights vary)",
                True, "udio", "v1/v1.5",
                repo_id="blanchon/udio_dataset",
                notes="~132k tracks, 289 WebDataset tar shards (mp3+json)."),
    DatasetSpec("echoes", "hf_dataset", "CC-BY-SA-4.0", None, None,
                repo_id="Octavian97/Echoes",
                notes="4.5k AI tracks from ~10 systems + FMA bona-fide refs; "
                      "folder names encode the generator."),
    DatasetSpec("aime", "hf_dataset",
                "CC-BY-4.0 (generated) / per-track (Jamendo real)",
                None, None, repo_id="disco-eth/AIME",
                notes="6,000 AI (12 models) + 500 real MTG-Jamendo tracks; "
                      "'model' column, model=='MTG-Jamendo' marks real."),
    DatasetSpec("sonics", "hf_dataset", "CC-BY-NC-4.0", None, None,
                repo_id="awsaf49/sonics",
                notes="97k songs (49k Suno/Udio fakes); real/fake CSV "
                      "manifests with 'algorithm'/'source' columns."),
    DatasetSpec("fakemusiccaps", "http", "CC-BY-NC-4.0", True, None,
                files=(RemoteFile(
                    "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1",
                    "FakeMusicCaps.zip", None, metadata=False, approx_gb=12.9),),
                notes="27,605 10 s clips from 5 TTM models; folder per model."),
    # ------------------------- Real side (§4.3) ---------------------------
    DatasetSpec("mtg_jamendo", "http", "CC (per-track, see Jamendo)",
                False, "human",
                files=(
                    # NB: data/autotagging.tsv in the repo is a git symlink to
                    # this file; raw.githubusercontent serves the link text,
                    # so we fetch the real target and keep the local name.
                    RemoteFile(f"{_MTG_RAW}/data/raw_30s_cleantags_50artists.tsv",
                               "autotagging.tsv"),
                    RemoteFile(f"{_MTG_RAW}/data/raw.meta.tsv",
                               "raw.meta.tsv"),
                    RemoteFile(f"{_MTG_RAW}/data/splits/split-0/autotagging-test.tsv",
                               "autotagging-test.tsv"),
                ),
                notes="Metadata TSVs only. Audio: use the official "
                      "scripts/download/download.py from "
                      "github.com/MTG/mtg-jamendo-dataset "
                      "(--dataset raw_30s --type audio) into "
                      f"{config.MTG_JAMENDO_DIR / 'audio'}."),
    DatasetSpec("fma", "http", "CC (per-track, see fma_metadata)",
                False, "human",
                files=(
                    RemoteFile("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
                               "fma_metadata.zip",
                               "sha1:f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
                               metadata=True, approx_gb=0.35),
                    RemoteFile("https://os.unil.cloud.switch.ch/fma/fma_small.zip",
                               "fma_small.zip",
                               "sha1:ade154f733639d52e35e32f5593efe5be76c6d70",
                               metadata=False, approx_gb=7.2),
                    RemoteFile("https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
                               "fma_medium.zip",
                               "sha1:c67b69ea232021025fca9231fc1c7c1a063ab50b",
                               metadata=False, approx_gb=22.0),
                    RemoteFile("https://os.unil.cloud.switch.ch/fma/fma_large.zip",
                               "fma_large.zip",
                               "sha1:497109f4dd721066b5ce5e5f250ec604dc78939e",
                               metadata=False, approx_gb=93.0),
                    RemoteFile("https://os.unil.cloud.switch.ch/fma/fma_full.zip",
                               "fma_full.zip",
                               "sha1:0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab",
                               metadata=False, approx_gb=879.0),
                ),
                notes="artist id lives in tracks.csv multi-index column "
                      "('artist','id')."),
    DatasetSpec("musicnet", "http", "CC-BY-4.0", False, "human",
                files=(
                    RemoteFile("https://zenodo.org/records/5120004/files/musicnet_metadata.csv?download=1",
                               "musicnet_metadata.csv",
                               "md5:1caef62cee9c875235e62aac368b49d8"),
                    RemoteFile("https://zenodo.org/records/5120004/files/musicnet.tar.gz?download=1",
                               "musicnet.tar.gz",
                               "md5:844764911fa0d5b97c97da944a057590",
                               metadata=False, approx_gb=11.1),
                ),
                notes="330 classical recordings, Zenodo record 5120004."),
    # --------------- QUARANTINED source — metadata only (§4.1) -----------
    DatasetSpec("sdd", "metadata_only", "CC-BY-SA-4.0", False, "human",
                files=(RemoteFile(
                    "https://zenodo.org/records/10072001/files/song_describer.csv?download=1",
                    "song_describer.csv"),),
                notes="Song Describer Dataset, Zenodo record 10072001. "
                      "METADATA ONLY — the audio is the hidden-eval real "
                      "pool's parent and is strictly quarantined."),
]}

# --- Generator-family normalization (plan §4.2, config.TEST_FAMILIES) -----
_FAMILY_SUBSTRINGS: list[tuple[str, str]] = [
    ("ace", "ace-step"), ("suno", "suno"), ("chirp", "suno"), ("udio", "udio"),
    ("mureka", "mureka"), ("minimax", "minimax"), ("yue", "yue"),
    ("diffrhythm", "diffrhythm"), ("riffusion", "riffusion"),
    ("stable", "stable-audio"), ("songgen", "songgen"), ("mubert", "mubert"),
    ("musicgen", "musicgen"), ("audiogen", "musicgen"),
    ("audioldm", "audioldm"), ("musicldm", "audioldm"),
    ("mustango", "mustango"), ("jamendo", "human"), ("bonafide", "human"),
    ("bona_fide", "human"), ("bona-fide", "human"), ("original", "human"),
    ("real", "human"), ("fma", "human"),
]


def normalize_family(raw: str) -> str:
    """Map a free-form model/folder name onto a canonical family key."""
    low = raw.strip().lower().replace(" ", "")
    for needle, family in _FAMILY_SUBSTRINGS:
        if needle in low:
            return family
    return low or "unknown"


# --- Download plumbing ----------------------------------------------------

def _file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksum(path: Path, checksum: str) -> None:
    """checksum is '<algo>:<hexdigest>'. Raises on mismatch."""
    algo, _, expected = checksum.partition(":")
    got = _file_digest(path, algo)
    if got != expected.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path}: {algo} {got} != {expected}. "
            "Delete the file and re-run data_fetch.")
    logger.info("Checksum OK (%s) for %s", algo, path.name)


def download_file(url: str, dest: Path, checksum: Optional[str] = None) -> Path:
    """Streaming HTTP download with resume (Range) + optional checksum."""
    import requests
    if dest.exists():
        logger.info("Already present, skipping: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    logger.info("Downloading %s -> %s (resume at %d)", url, dest, offset)
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if r.status_code == 416:          # range beyond EOF: already complete
            logger.info("Server reports file already complete: %s", part)
        else:
            r.raise_for_status()
            mode = "ab" if (offset and r.status_code == 206) else "wb"
            with open(part, mode) as f:
                for block in r.iter_content(chunk_size=1 << 20):
                    f.write(block)
    if checksum:
        verify_checksum(part, checksum)
    part.rename(dest)
    return dest


def extract_archive(path: Path, dest_dir: Path) -> None:
    """Unpack .zip / .tar.gz next to the archive (idempotent-ish: skips if a
    same-stem directory already exists)."""
    marker = dest_dir / (path.name.split(".")[0])
    if marker.exists():
        logger.info("Extraction target %s exists, skipping unpack", marker)
        return
    logger.info("Extracting %s -> %s", path, dest_dir)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest_dir)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(dest_dir)
    else:
        logger.warning("Unknown archive format, leaving as-is: %s", path)


def _hf_subset_patterns(repo_id: str, subset_gb: float) -> list[str]:
    """allow_patterns covering all small (metadata) files plus the first
    shards up to ``subset_gb`` GB, for dev boxes (task spec: --subset-gb)."""
    from huggingface_hub import HfApi
    files = [(e.path, e.size) for e in
             HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
             if getattr(e, "size", None) is not None]
    allow = [p for p, s in files if s <= _SMALL_FILE_BYTES]
    budget, used = subset_gb * 2**30, 0
    for p, s in sorted(f for f in files if f[1] > _SMALL_FILE_BYTES):
        if used + s > budget:
            break
        allow.append(p)
        used += s
    logger.info("--subset-gb %.1f: fetching %d files (%.2f GB of shards) "
                "from %s", subset_gb, len(allow), used / 2**30, repo_id)
    return allow


def fetch_hf(spec: DatasetSpec, metadata_only: bool,
             subset_gb: Optional[float]) -> None:
    """snapshot_download with resume; honors --metadata-only / --subset-gb."""
    from huggingface_hub import snapshot_download
    allow: Optional[list[str]] = None
    if metadata_only:
        allow = HF_METADATA_PATTERNS
    elif subset_gb is not None:
        allow = _hf_subset_patterns(spec.repo_id, subset_gb)
    spec.dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec.repo_id, repo_type="dataset",
                      local_dir=spec.dest, allow_patterns=allow,
                      max_workers=4)
    logger.info("HF snapshot complete: %s -> %s", spec.repo_id, spec.dest)


def fetch_http(spec: DatasetSpec, metadata_only: bool,
               subset_gb: Optional[float]) -> None:
    """Download the spec's RemoteFiles (streaming, resumable, checksummed)."""
    budget = None if subset_gb is None else subset_gb
    used = 0.0
    for rf in spec.files:
        if metadata_only and not rf.metadata:
            logger.info("--metadata-only: skipping %s", rf.filename)
            continue
        if (budget is not None and not rf.metadata
                and used + rf.approx_gb > budget):
            logger.info("--subset-gb %.1f: skipping %s (~%.1f GB)",
                        budget, rf.filename, rf.approx_gb)
            continue
        path = download_file(rf.url, spec.dest / rf.filename, rf.checksum)
        if not rf.metadata:
            used += rf.approx_gb
        if path.suffix in {".zip", ".gz", ".tar"}:
            extract_archive(path, spec.dest)
    if spec.name == "mtg_jamendo":
        logger.warning("MTG-Jamendo AUDIO is not auto-fetched. %s", spec.notes)


def fetch_sdd(spec: DatasetSpec) -> None:
    """SDD: metadata CSV only — the audio is quarantined (plan §4.1)."""
    for rf in spec.files:
        download_file(rf.url, spec.dest / rf.filename, rf.checksum)
    logger.warning(
        "=" * 70 + "\nSDD AUDIO IS QUARANTINED (plan §4.1 / MIREX rules). "
        "Only song_describer.csv was fetched, into %s. Never download "
        "audio.zip from Zenodo record 10072001 into this project.\n" + "=" * 70,
        spec.dest)


# --- Registration helpers -------------------------------------------------

def _audio_info(path: Path) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """(duration_s, sample_rate, channels) via soundfile.info, else Nones."""
    try:
        import soundfile
        info = soundfile.info(str(path))
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception:
        return None, None, None


def _walk_audio(root: Path) -> Iterator[Path]:
    yield from (p for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _insert(db: MetadataDatabase, rows: list[dict], source: str) -> int:
    if not rows:
        logger.warning("register_%s: nothing to register — did data_fetch "
                       "run for it?", source)
        return 0
    for i in range(0, len(rows), 5000):
        db.insert_tracks(rows[i:i + 5000])
    logger.info("register_%s: %d rows (idempotent upsert)", source, len(rows))
    return len(rows)


def _require(path: Path, dataset: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python src/data_fetch.py --dataset "
            f"{dataset}` first.")
    return path


def _row(spec: DatasetSpec, native_id: str, file_path: str, *,
         family: Optional[str] = None, version: Optional[str] = None,
         is_ai: Optional[bool] = None, artist_id: Optional[str] = None,
         duration: Optional[float] = None, sample_rate: Optional[int] = None,
         channels: Optional[int] = None, extra: Optional[dict] = None) -> dict:
    ai = spec.is_ai if is_ai is None else is_ai
    return {"track_id": f"{spec.name}:{native_id}",
            "source_dataset": spec.name,
            "generator_family": family or spec.generator_family or "unknown",
            "generator_version": version or spec.generator_version,
            "is_ai": int(bool(ai)),
            "artist_id": artist_id,
            "license": spec.license,
            "file_path": file_path,
            "duration_s": duration, "sample_rate": sample_rate,
            "channels": channels,
            "extra_json": json.dumps(extra) if extra else None}


def _parquet_rows(pq_path: Path, columns: list[str]) -> Iterator[dict]:
    """Stream selected columns of a parquet file (never decodes audio blobs)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(pq_path)
    present = [c for c in columns if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(columns=present, batch_size=1024):
        yield from batch.to_pylist()


# --- Registrars (one per source; independently callable, idempotent) ------

def register_mtg_jamendo(db: MetadataDatabase) -> int:
    """Real-side core (§4.3): every autotagging.tsv track, with ARTIST_ID."""
    spec = REGISTRY["mtg_jamendo"]
    tsv = _require(MTG_AUTOTAGGING_TSV, spec.name)
    audio_root = config.MTG_JAMENDO_DIR / "audio"
    rows = []
    with open(tsv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS...
        for rec in reader:
            if len(rec) < 5:
                continue
            tid, artist, _album, rel, dur = rec[:5]
            rows.append(_row(spec, tid, str(audio_root / rel),
                             artist_id=artist,
                             duration=float(dur) if dur else None))
    if not audio_root.exists():
        logger.warning("MTG-Jamendo audio dir %s missing — rows registered "
                       "from metadata only (paths are expected locations).",
                       audio_root)
    return _insert(db, rows, spec.name)


def _fma_artist_map(meta_dir: Path) -> dict[int, str]:
    """track id -> artist id from tracks.csv multi-index column ('artist','id')."""
    import pandas as pd
    tracks_csv = meta_dir / "tracks.csv"
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    series = df[("artist", "id")]
    return {int(t): f"fma_artist_{int(a)}"
            for t, a in series.items() if pd.notna(a)}


def register_fma(db: MetadataDatabase) -> int:
    """FMA audio + artist_id from fma_metadata/tracks.csv (§4.3)."""
    spec = REGISTRY["fma"]
    meta_dir = _require(spec.dest / "fma_metadata", spec.name)
    artists = _fma_artist_map(meta_dir)
    rows = []
    for sub in ("fma_full", "fma_large", "fma_medium", "fma_small"):
        root = spec.dest / sub
        if not root.exists():
            continue
        for p in _walk_audio(root):
            try:
                tid = int(p.stem)
            except ValueError:
                continue
            dur, sr, ch = _audio_info(p)
            rows.append(_row(spec, str(tid), str(p),
                             artist_id=artists.get(tid),
                             duration=dur, sample_rate=sr, channels=ch,
                             extra={"subset": sub}))
        break  # largest available subset wins; ids overlap across subsets
    return _insert(db, rows, spec.name)


def register_musicnet(db: MetadataDatabase) -> int:
    """MusicNet classical recordings (§4.3); artist_id = composer."""
    spec = REGISTRY["musicnet"]
    root = _require(spec.dest / "musicnet", spec.name)
    composers: dict[str, str] = {}
    meta = spec.dest / "musicnet_metadata.csv"
    if meta.exists():
        with open(meta, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                composers[str(rec.get("id", ""))] = \
                    f"composer_{rec.get('composer', 'unknown')}"
    rows = []
    for p in _walk_audio(root):
        dur, sr, ch = _audio_info(p)
        rows.append(_row(spec, p.stem, str(p),
                         artist_id=composers.get(p.stem),
                         duration=dur, sample_rate=sr, channels=ch))
    return _insert(db, rows, spec.name)


def register_muse(db: MetadataDatabase) -> int:
    """Muse: Suno v5 (§4.2). Walks extracted mp3s (tar shards need unpack)."""
    spec = REGISTRY["muse"]
    root = _require(spec.dest, spec.name)
    rows = [_row(spec, p.relative_to(root).with_suffix("").as_posix(), str(p))
            for p in _walk_audio(root)]
    if not rows:
        logger.warning("register_muse: no audio found under %s — extract the "
                       ".tar shards (suno_cn_songs/, suno_en_songs/) first.",
                       root)
    return _insert(db, rows, spec.name)


def register_suno_audio(db: MetadataDatabase) -> int:
    """suno-audio parquet shards; per-track model_name -> generator_version."""
    spec = REGISTRY["suno_audio"]
    root = _require(spec.dest, spec.name)
    rows = []
    for pq_path in sorted(root.rglob("*.parquet")):
        for i, rec in enumerate(_parquet_rows(
                pq_path, ["id", "model_name", "duration", "title"])):
            native = str(rec.get("id") or f"{pq_path.stem}_{i}")
            dur = rec.get("duration")
            rows.append(_row(spec, native, f"{pq_path}#row={i}",
                             version=str(rec.get("model_name") or "unknown"),
                             duration=float(dur) if dur else None,
                             extra={"title": rec.get("title")}))
    return _insert(db, rows, spec.name)


def register_udio(db: MetadataDatabase) -> int:
    """Udio WebDataset shards: walk extracted mp3s, else index tar members."""
    spec = REGISTRY["udio"]
    root = _require(spec.dest, spec.name)
    rows = [_row(spec, p.relative_to(root).with_suffix("").as_posix(), str(p))
            for p in _walk_audio(root)]
    if not rows:
        for shard in sorted(root.rglob("*.tar")):
            with tarfile.open(shard) as t:
                for name in t.getnames():
                    if Path(name).suffix.lower() in AUDIO_EXTS:
                        rows.append(_row(spec, Path(name).stem,
                                         f"{shard}::{name}"))
    return _insert(db, rows, spec.name)


def register_echoes(db: MetadataDatabase) -> int:
    """Echoes: folder names encode the ~10 generators; bona-fide FMA -> real."""
    spec = REGISTRY["echoes"]
    root = _require(spec.dest, spec.name)
    rows = []
    for p in _walk_audio(root):
        family = "unknown"
        for part in p.relative_to(root).parts[:-1]:
            fam = normalize_family(part)
            if fam not in ("unknown", "tta", "ata"):
                family = fam
                break
        rows.append(_row(spec, p.relative_to(root).with_suffix("").as_posix(),
                         str(p), family=family, is_ai=(family != "human"),
                         extra={"folder": p.parent.name}))
    return _insert(db, rows, spec.name)


def register_aime(db: MetadataDatabase) -> int:
    """AIME: 6k AI (12 models) + 500 real MTG-Jamendo ('model' column)."""
    spec = REGISTRY["aime"]
    root = _require(spec.dest, spec.name)
    rows = []
    for pq_path in sorted(root.rglob("*.parquet")):
        for i, rec in enumerate(_parquet_rows(
                pq_path, ["id", "model", "description"])):
            model = str(rec.get("model") or "unknown")
            native = str(rec.get("id") or f"{pq_path.stem}_{i}")
            family = normalize_family(model)
            rows.append(_row(spec, native, f"{pq_path}#row={i}",
                             family=family, is_ai=(family != "human"),
                             version=None if family == "human" else model,
                             extra={"model": model, "native_id": native}))
    return _insert(db, rows, spec.name)


def register_sonics(db: MetadataDatabase) -> int:
    """SONICS: real/fake CSV manifests; 'algorithm'/'source' -> family/version."""
    spec = REGISTRY["sonics"]
    root = _require(spec.dest, spec.name)
    rows = []
    for csv_path in sorted(root.rglob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            if "filepath" not in cols and "filename" not in cols:
                continue
            fake_manifest = "fake" in csv_path.name.lower()
            for i, rec in enumerate(reader):
                rel = rec.get("filepath") or rec.get("filename") or ""
                native = str(rec.get("id") or Path(rel).stem or
                             f"{csv_path.stem}_{i}")
                algo = rec.get("algorithm") or rec.get("source") or ""
                family = normalize_family(algo) if fake_manifest else "human"
                rows.append(_row(
                    spec, native, str(root / rel),
                    family=family, is_ai=fake_manifest,
                    version=(algo or None) if fake_manifest else None,
                    duration=float(rec["duration"]) if rec.get("duration")
                    else None,
                    extra={"manifest": csv_path.name}))
    return _insert(db, rows, spec.name)


def register_fakemusiccaps(db: MetadataDatabase) -> int:
    """FakeMusicCaps: one folder per TTM model (MusicGen, MusicLDM, ...)."""
    spec = REGISTRY["fakemusiccaps"]
    root = _require(spec.dest, spec.name)
    rows = []
    for p in _walk_audio(root):
        rel = p.relative_to(root)
        family = normalize_family(rel.parts[0]) if len(rel.parts) > 1 \
            else "unknown"
        if family == "human":
            continue  # any bundled reference/real clips are not our pool
        rows.append(_row(spec, rel.with_suffix("").as_posix(), str(p),
                         family=family))
    return _insert(db, rows, spec.name)


def register_sdd(db: MetadataDatabase) -> int:  # noqa: ARG001
    """SDD registers NOTHING: its audio is quarantined end-to-end (§4.1)."""
    _require(SDD_METADATA_CSV, "sdd")
    logger.warning("SDD is metadata-only and QUARANTINED — 0 tracks "
                   "registered by design. quarantine.py consumes %s directly.",
                   SDD_METADATA_CSV)
    return 0


REGISTRARS: dict[str, Callable[[MetadataDatabase], int]] = {
    "mtg_jamendo": register_mtg_jamendo, "fma": register_fma,
    "musicnet": register_musicnet, "muse": register_muse,
    "suno_audio": register_suno_audio, "udio": register_udio,
    "echoes": register_echoes, "aime": register_aime,
    "sonics": register_sonics, "fakemusiccaps": register_fakemusiccaps,
    "sdd": register_sdd,
}


# --- CLI ------------------------------------------------------------------

def fetch_dataset(name: str, metadata_only: bool = False,
                  subset_gb: Optional[float] = None,
                  register_only: bool = False,
                  db: Optional[MetadataDatabase] = None) -> None:
    """Fetch one dataset then register it (plan §14 phase P1)."""
    spec = REGISTRY[name]
    if not register_only:
        if spec.kind == "hf_dataset":
            fetch_hf(spec, metadata_only, subset_gb)
        elif spec.kind == "metadata_only":
            fetch_sdd(spec)
        else:
            fetch_http(spec, metadata_only, subset_gb)
    try:
        REGISTRARS[name](db or MetadataDatabase())
    except FileNotFoundError as exc:
        logger.warning("Registration for %s skipped: %s", name, exc)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MIREX 2026 dataset acquisition + registration (§4.2/4.3)")
    parser.add_argument("--dataset", default="all",
                        choices=sorted(REGISTRY) + ["all"])
    parser.add_argument("--metadata-only", action="store_true",
                        help="fetch only metadata files (CSV/TSV/JSON)")
    parser.add_argument("--subset-gb", type=float, default=None, metavar="N",
                        help="fetch only ~first N GB of audio shards (dev)")
    parser.add_argument("--register-only", action="store_true",
                        help="skip downloads; (re)walk local files into the DB")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    db = MetadataDatabase()
    names = sorted(REGISTRY) if args.dataset == "all" else [args.dataset]
    for name in names:
        logger.info("=== %s (%s) ===", name, REGISTRY[name].kind)
        fetch_dataset(name, args.metadata_only, args.subset_gb,
                      args.register_only, db=db)
    db.census(write=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
