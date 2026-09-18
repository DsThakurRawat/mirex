"""Self-generation campaign CLI (plan §4.2 prompt strategy, phase P5).

Builds a deterministic job list from ``PromptTaxonomy(seed=config.SEED)``,
runs the chosen backend with resume (finished jobs are skipped via the JSONL
ledger), retries with exponential backoff, registers every finished track in
the metadata DB, and prints a final census.

Usage (run from ``src/``)::

    python -m generation.campaign --backend ace_step --count 20000 --workers 1
    python -m generation.campaign --backend yue      --count 5000
    python -m generation.campaign --backend mureka   --count 2000 --workers 4 \
        --requests-per-minute 30
    python -m generation.campaign --backend minimax  --count 2000 --workers 2 \
        --requests-per-minute 60
    python -m generation.campaign --backend mureka --count 10 --dry-run

Job IDs are ``<backend>-<index:06d>`` where index is the taxonomy style
index, so re-running with a larger ``--count`` extends the campaign without
regenerating anything (P5 gate: per-generator spectrogram QA runs on the
registered outputs afterwards).
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import config
from metadata_db import MetadataDatabase

from generation.base import (BACKEND_FAMILY, FatalGenerationError,
                             GenerationError, GenerationJob, JobLedger,
                             RateLimiter, backend_output_dir, register_output,
                             with_retries)
from generation.prompts import PromptTaxonomy

logger = logging.getLogger(__name__)

BACKENDS = tuple(BACKEND_FAMILY)         # ace_step, yue, mureka, minimax


def build_runner(backend: str, rate_limiter: Optional[RateLimiter] = None):
    """Lazy backend construction so torch/GPU deps load only when needed."""
    if backend == "ace_step":
        from generation.ace_step_runner import AceStepRunner
        return AceStepRunner()
    if backend == "yue":
        from generation.yue_runner import YueRunner
        return YueRunner()
    if backend == "mureka":
        from generation.mureka_client import MurekaClient
        return MurekaClient(rate_limiter=rate_limiter)
    if backend == "minimax":
        from generation.minimax_client import MiniMaxClient
        return MiniMaxClient(rate_limiter=rate_limiter)
    raise ValueError(f"unknown backend {backend!r}")


def build_jobs(taxonomy: PromptTaxonomy, backend: str, count: int,
               start: int = 0) -> list[GenerationJob]:
    """Deterministic job list: job i <-> taxonomy style i (resume-stable)."""
    return [GenerationJob(job_id=f"{backend}-{i:06d}", backend=backend,
                          style=taxonomy.style(i))
            for i in range(start, start + count)]


def run_campaign(backend: str, count: int, workers: int = 1,
                 start: int = 0, seed: Optional[int] = None,
                 requests_per_minute: Optional[float] = None,
                 ledger_path: Optional[Path] = None, log_every: int = 25,
                 retry_attempts: int = 4, dry_run: bool = False) -> dict:
    """Run (or resume) the campaign; returns a summary dict."""
    taxonomy = PromptTaxonomy(seed=config.SEED if seed is None else seed)
    jobs = build_jobs(taxonomy, backend, count, start)
    out_dir = backend_output_dir(backend)
    ledger = JobLedger(ledger_path or out_dir / "jobs.jsonl")
    pending = ledger.pending(jobs)
    logger.info("Campaign %s: %d jobs planned, %d already done, %d to run",
                backend, len(jobs), len(jobs) - len(pending), len(pending))

    if dry_run:
        for job in pending[:10]:
            rendered = taxonomy.render(job.style, backend)
            print(f"--- {job.job_id} ---")
            print(json.dumps({"style": job.style, "rendered": rendered},
                             ensure_ascii=False, indent=2))
        return {"backend": backend, "planned": len(jobs),
                "done": len(jobs) - len(pending), "pending": len(pending),
                "dry_run": True}

    limiter = (RateLimiter(60.0 / requests_per_minute)
               if requests_per_minute else None)
    runner = build_runner(backend, rate_limiter=limiter)
    db = MetadataDatabase()
    db_lock = threading.Lock()           # sqlite: serialize inserts
    counters = {"ok": 0, "failed": 0}
    counters_lock = threading.Lock()

    def process(job: GenerationJob) -> None:
        rendered = taxonomy.render(job.style, backend)
        try:
            paths, settings = with_retries(
                lambda: runner.run(job, rendered), attempts=retry_attempts)
            ledger.mark_done(job.job_id, [str(p) for p in paths],
                             meta={"settings_keys": sorted(settings)})
            with db_lock:
                register_output(db, job, paths, settings,
                                generator_version=getattr(
                                    runner, "generator_version", None))
            with counters_lock:
                counters["ok"] += 1
        except FatalGenerationError as exc:
            ledger.mark_failed(job.job_id, f"FATAL: {exc}")
            with counters_lock:
                counters["failed"] += 1
            logger.error("Job %s fatally failed: %s", job.job_id, exc)
            raise                        # fatal (bad key etc.) -> stop early
        except GenerationError as exc:
            ledger.mark_failed(job.job_id, str(exc))
            with counters_lock:
                counters["failed"] += 1
            logger.error("Job %s failed after retries: %s", job.job_id, exc)

    processed = 0
    if workers <= 1:
        for job in pending:
            process(job)
            processed += 1
            if processed % log_every == 0:
                logger.info("[%s] %d/%d processed (ok=%d failed=%d)",
                            backend, processed, len(pending),
                            counters["ok"], counters["failed"])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, job): job for job in pending}
            for fut in as_completed(futures):
                fut.result()             # re-raise FatalGenerationError
                processed += 1
                if processed % log_every == 0:
                    logger.info("[%s] %d/%d processed (ok=%d failed=%d)",
                                backend, processed, len(pending),
                                counters["ok"], counters["failed"])

    # Final census (P1 contract: the census is the acceptance artifact).
    census = db.census(write=False)
    gen_rows = [r for r in census["by_source"]
                if r["source_dataset"] == f"gen_{backend}"]
    print(f"\n=== Campaign census: gen_{backend} "
          f"(family={BACKEND_FAMILY[backend]}) ===")
    for r in gen_rows:
        print(f"  tracks={r['n']}  hours={r['hours']:.1f}  "
              f"quarantined={r['quarantined']}")
    print(f"  this run: ok={counters['ok']} failed={counters['failed']} "
          f"skipped(done)={len(jobs) - len(pending)}")
    return {"backend": backend, "planned": len(jobs), **counters,
            "skipped": len(jobs) - len(pending), "census_rows": gen_rows}


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m generation.campaign",
        description="MIREX 2026 self-generation campaign (plan P5/§4.2)")
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--count", type=int, required=True,
                    help="number of taxonomy styles (jobs) in the campaign")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel workers (keep 1 for local GPU backends)")
    ap.add_argument("--start", type=int, default=0,
                    help="first taxonomy style index (default 0)")
    ap.add_argument("--seed", type=int, default=None,
                    help=f"taxonomy seed (default config.SEED={config.SEED})")
    ap.add_argument("--requests-per-minute", type=float, default=None,
                    help="rate limit for API backends (mureka/minimax)")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="ledger path (default <out_dir>/jobs.jsonl)")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--retry-attempts", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="print first pending styles/prompts and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_dirs()
    run_campaign(backend=args.backend, count=args.count,
                 workers=args.workers, start=args.start, seed=args.seed,
                 requests_per_minute=args.requests_per_minute,
                 ledger_path=args.ledger, log_every=args.log_every,
                 retry_attempts=args.retry_attempts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
