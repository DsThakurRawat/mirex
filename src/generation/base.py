"""Shared infrastructure for the generation campaign (plan P5).

Provides:

- :class:`GenerationJob` — one unit of work (style -> one generated track).
- :class:`JobLedger` — append-only JSONL ledger with resume: finished jobs
  are skipped on rerun, failed/pending jobs are retried. The campaign can be
  killed and restarted at any point (plan §11: "runs unattended").
- :class:`RateLimiter` — thread-safe minimum-interval limiter for API
  backends (Mureka/MiniMax rate limits).
- :func:`with_retries` — exponential backoff with jitter;
  :class:`FatalGenerationError` (auth failures, 4xx, content policy) is
  never retried so credits are not burned on hopeless jobs.
- :func:`register_output` — inserts finished tracks into
  :class:`metadata_db.MetadataDatabase` with ``source_dataset=
  "gen_<backend>"``, ``is_ai=1`` and the full style+settings JSON, which is
  the reproducibility record for the released corpus (plan §13 item 4).
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import config
from metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)

# CLI backend key -> canonical generator_family (must match
# config.TEST_FAMILIES so LOGO folds (§5) line up with self-generated data).
BACKEND_FAMILY = {
    "ace_step": "ace-step",
    "yue": "yue",
    "mureka": "mureka",
    "minimax": "minimax",
}


class GenerationError(Exception):
    """Retryable generation failure (network blips, 5xx, GPU OOM, timeouts)."""


class FatalGenerationError(GenerationError):
    """Non-retryable failure: bad API key, 4xx, invalid params, moderation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def backend_output_dir(backend: str) -> Path:
    """``config.GENERATED_DATA_DIR/<backend>/`` (created on demand)."""
    d = Path(config.GENERATED_DATA_DIR) / backend
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_requests():
    """Lazy import of ``requests`` (kept out of module top-level so the
    taxonomy/ledger/tests work in minimal environments)."""
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise GenerationError(
            "The 'requests' package is required for API generation backends "
            "(pip install requests)") from exc
    return requests


@dataclass
class GenerationJob:
    """One planned generation: deterministic style -> one (or more) tracks."""
    job_id: str
    backend: str
    style: dict
    status: str = "pending"          # pending | done | failed
    output_path: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    extra: dict = field(default_factory=dict)


class JobLedger:
    """Append-only JSONL job ledger with resume semantics.

    Each line is ``{"job_id": ..., "status": ..., "ts": ..., ...}``; the last
    line per job wins. ``pending()`` filters out jobs whose latest status is
    ``done`` — failed jobs are retried on the next campaign run.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Ledger %s: skipping corrupt line",
                                       self.path)
                        continue
                    if "job_id" in rec:
                        self._state[rec["job_id"]] = rec
            logger.info("Ledger %s: loaded %d records (%d done)",
                        self.path, len(self._state), len(self.done_ids()))

    def _append(self, rec: dict) -> None:
        with self._lock:
            self._state[rec["job_id"]] = rec
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def mark_done(self, job_id: str, output_paths: Sequence[str],
                  meta: Optional[dict] = None) -> None:
        self._append({"job_id": job_id, "status": "done",
                      "ts": utc_now_iso(),
                      "output_paths": list(map(str, output_paths)),
                      "meta": meta or {}})

    def mark_failed(self, job_id: str, error: str) -> None:
        self._append({"job_id": job_id, "status": "failed",
                      "ts": utc_now_iso(), "error": str(error)[:2000]})

    def status(self, job_id: str) -> Optional[str]:
        rec = self._state.get(job_id)
        return rec.get("status") if rec else None

    def done_ids(self) -> set[str]:
        return {j for j, r in self._state.items()
                if r.get("status") == "done"}

    def pending(self, jobs: Iterable[GenerationJob]) -> list[GenerationJob]:
        """Jobs not yet successfully finished (the resume filter)."""
        done = self.done_ids()
        return [j for j in jobs if j.job_id not in done]


class RateLimiter:
    """Thread-safe minimum-interval limiter (e.g. 60/rpm -> 1.0 s)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._next_t = 0.0

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_t - now
            self._next_t = max(now, self._next_t) + self.min_interval_s
        if delay > 0:
            time.sleep(delay)


def with_retries(fn: Callable[[], Any], *, attempts: int = 4,
                 base_delay_s: float = 2.0, max_delay_s: float = 120.0,
                 rng: Optional[random.Random] = None) -> Any:
    """Run ``fn`` with exponential backoff + jitter.

    :class:`FatalGenerationError` propagates immediately (no retry);
    any other exception is retried up to ``attempts`` times.
    """
    rng = rng or random
    last: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            return fn()
        except FatalGenerationError:
            raise
        except Exception as exc:  # noqa: BLE001 — deliberate catch-all
            last = exc
            if attempt == attempts - 1:
                break
            delay = min(max_delay_s, base_delay_s * (2 ** attempt))
            delay *= 0.5 + rng.random()          # jitter in [0.5x, 1.5x]
            logger.warning("Attempt %d/%d failed (%s); retrying in %.1fs",
                           attempt + 1, attempts, exc, delay)
            time.sleep(delay)
    raise GenerationError(
        f"All {attempts} attempts failed; last error: {last}") from last


def register_output(db: MetadataDatabase, job: GenerationJob,
                    output_paths: Sequence[Path | str], settings: dict,
                    generator_version: Optional[str] = None) -> list[str]:
    """Insert finished track(s) into the metadata DB (P1 contract).

    ``extra_json`` carries the *complete* style dict and sampler/API settings
    so every released track is exactly reproducible (plan §13, contribution
    4). ``duration_s`` is the requested duration; the P5 spectrogram-QA sweep
    re-probes real durations/sample rates before training.

    Returns the inserted track_ids. Multi-output APIs (Mureka returns
    multiple choices per task) get ``#<k>`` suffixed track_ids.
    """
    family = BACKEND_FAMILY[job.backend]
    rows, ids = [], []
    for k, p in enumerate(output_paths):
        track_id = f"gen_{job.backend}:{job.job_id}"
        if len(output_paths) > 1:
            track_id += f"#{k}"
        rows.append({
            "track_id": track_id,
            "source_dataset": f"gen_{job.backend}",
            "generator_family": family,
            "generator_version": generator_version,
            "is_ai": config.AI_LABEL,
            "license": "self-generated (research disclosure per MIREX rules)",
            "file_path": str(p),
            "duration_s": float(job.style.get("duration_s") or 0) or None,
            "extra_json": json.dumps(
                {"style": job.style, "settings": settings,
                 "campaign_seed": config.SEED, "generated_at": utc_now_iso()},
                ensure_ascii=False, sort_keys=True),
        })
        ids.append(track_id)
    db.insert_tracks(rows)
    logger.debug("Registered %d track(s) for job %s", len(rows), job.job_id)
    return ids
