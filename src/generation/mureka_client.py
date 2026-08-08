"""Mureka official platform API client (plan §4.2: ~2k songs, P5).

Verified against platform.mureka.ai docs (fetched 2026-08-08):
- Base URL ``https://api.mureka.ai``; auth ``Authorization: Bearer
  $MUREKA_API_KEY`` (https://platform.mureka.ai/docs/en/quickstart.html).
- Vocal songs: ``POST /v1/song/generate`` with JSON ``{"lyrics": "[Verse]\\n
  ...", "model": "auto"|"mureka-6", "prompt": "r&b, slow, passionate, male
  vocal"}`` -> asynchronous task ``{"id": ..., "status": "preparing", ...}``
  (https://platform.mureka.ai/docs/api/operations/post-v1-song-generate.html).
- Polling: ``GET /v1/song/query/{task_id}``
  (https://platform.mureka.ai/docs/api/operations/get-v1-song-query-{task_id}.html).

WARNING (unverified specifics): the docs pages are JS-rendered and the full
task-response schema could not be captured this session. This client assumes
the commonly documented shape — terminal statuses {"succeeded", "failed",
"timeouted", "cancelled"} and finished audio under ``choices[]`` with
``url``/``flac_url``/``mp3_url`` — and additionally falls back to a
``songs[].mp3_url`` array. The instrumental endpoints
``POST /v1/instrumental/generate`` / ``GET /v1/instrumental/query/{id}``
follow the platform's naming convention but were NOT directly verified;
re-check https://platform.mureka.ai/docs/ before burning credits, and note
that all HTTP errors surface the raw response body (no silent fallbacks).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import config
from generation.base import (FatalGenerationError, GenerationError,
                             GenerationJob, RateLimiter, backend_output_dir,
                             require_requests)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.mureka.ai"
TERMINAL_FAIL = {"failed", "timeouted", "cancelled"}


class MurekaError(GenerationError):
    """Mureka API failure with HTTP status and response body."""


class MurekaClient:
    """Submit -> poll -> download client for Mureka song generation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "auto",
                 base_url: str = BASE_URL, poll_interval_s: float = 10.0,
                 poll_timeout_s: float = 1800.0,
                 rate_limiter: Optional[RateLimiter] = None,
                 http_timeout_s: float = 60.0):
        self.api_key = (api_key if api_key is not None
                        else config.MUREKA_API_KEY)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self.rate_limiter = rate_limiter or RateLimiter(0.0)
        self.http_timeout_s = http_timeout_s

    # --- plumbing ---------------------------------------------------------
    def _require_key(self) -> None:
        if not self.api_key:
            raise FatalGenerationError(
                "MUREKA_API_KEY is not set (export it or pass api_key=)")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def _request(self, method: str, path: str,
                 json_body: Optional[dict] = None) -> dict:
        self._require_key()
        requests = require_requests()
        self.rate_limiter.wait()
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self._headers(),
                                json=json_body, timeout=self.http_timeout_s)
        if 400 <= resp.status_code < 500:
            # Honest error surface: raise with the body, never swallow.
            raise FatalGenerationError(
                f"Mureka {method} {path} -> HTTP {resp.status_code}: "
                f"{resp.text[:2000]}")
        if resp.status_code >= 500:
            raise MurekaError(
                f"Mureka {method} {path} -> HTTP {resp.status_code}: "
                f"{resp.text[:2000]}")
        return resp.json()

    # --- API operations ---------------------------------------------------
    def submit(self, prompt: str, lyrics: str, vocal: bool) -> tuple[str, str]:
        """Start a generation task; returns (task_id, query_path_prefix)."""
        if vocal:
            body = {"lyrics": lyrics, "model": self.model, "prompt": prompt}
            data = self._request("POST", "/v1/song/generate", body)
            prefix = "/v1/song/query"
        else:
            # WARNING: unverified endpoint (see module docstring).
            body = {"model": self.model, "prompt": prompt}
            data = self._request("POST", "/v1/instrumental/generate", body)
            prefix = "/v1/instrumental/query"
        task_id = data.get("id")
        if not task_id:
            raise MurekaError(f"Mureka response has no task id: {data}")
        logger.info("Mureka task %s submitted (status=%s)", task_id,
                    data.get("status"))
        return str(task_id), prefix

    def poll(self, task_id: str, query_prefix: str) -> dict:
        """Poll until the task reaches a terminal status."""
        deadline = time.monotonic() + self.poll_timeout_s
        while True:
            data = self._request("GET", f"{query_prefix}/{task_id}")
            status = str(data.get("status", "")).lower()
            if status == "succeeded":
                return data
            if status in TERMINAL_FAIL:
                raise MurekaError(
                    f"Mureka task {task_id} ended with status={status}: "
                    f"{data.get('failed_reason', data)}")
            if time.monotonic() > deadline:
                raise MurekaError(
                    f"Mureka task {task_id} still '{status}' after "
                    f"{self.poll_timeout_s}s")
            time.sleep(self.poll_interval_s)

    @staticmethod
    def _extract_audio_urls(payload: dict) -> list[str]:
        urls: list[str] = []
        for choice in payload.get("choices") or []:
            u = (choice.get("url") or choice.get("flac_url")
                 or choice.get("mp3_url"))
            if u:
                urls.append(u)
        if not urls:
            for song in payload.get("songs") or []:
                if song.get("mp3_url"):
                    urls.append(song["mp3_url"])
        return urls

    def download(self, url: str, dest: Path) -> Path:
        requests = require_requests()
        resp = requests.get(url, timeout=300)
        if resp.status_code != 200:
            raise MurekaError(f"Download {url} -> HTTP {resp.status_code}")
        dest.write_bytes(resp.content)
        return dest

    # --- campaign entry point ---------------------------------------------
    def run(self, job: GenerationJob, rendered: dict) -> tuple[list[Path], dict]:
        """Full submit->poll->download cycle; returns (paths, settings).

        Mureka typically returns several ``choices`` per task; all are
        downloaded (more exact-decoder data per credit).
        """
        self._require_key()
        vocal = bool(job.style["vocal"])
        task_id, prefix = self.submit(rendered["prompt"],
                                      rendered.get("lyrics", ""), vocal)
        result = self.poll(task_id, prefix)
        urls = self._extract_audio_urls(result)
        if not urls:
            raise MurekaError(
                f"Mureka task {task_id} succeeded but no audio URLs found "
                f"in response: {str(result)[:2000]}")
        out_dir = backend_output_dir("mureka")
        paths = []
        for k, url in enumerate(urls):
            ext = ".flac" if ".flac" in url.split("?")[0].lower() else ".mp3"
            suffix = f"_c{k}" if len(urls) > 1 else ""
            paths.append(self.download(
                url, out_dir / f"{job.job_id}{suffix}{ext}"))
        settings = {"model": result.get("model", self.model),
                    "endpoint": ("/v1/song/generate" if vocal
                                 else "/v1/instrumental/generate"),
                    "task_id": task_id, "prompt": rendered["prompt"],
                    "n_choices": len(urls)}
        return paths, settings

    @property
    def generator_version(self) -> str:
        return self.model
