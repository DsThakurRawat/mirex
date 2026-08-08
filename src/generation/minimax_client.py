"""MiniMax Music generation API client (plan §4.2: ~2k songs, P5).

Verified against https://platform.minimax.io/docs/api-reference/music-generation
(fetched 2026-08-08):
- ``POST https://api.minimax.io/v1/music_generation`` (Mainland-China host:
  ``https://api.minimaxi.com`` — host and key region must match).
- Auth: ``Authorization: Bearer $MINIMAX_API_KEY``.
- Body: ``model`` in {"music-3.0", "music-2.6", "music-cover"} (+ ``-free``
  tiers); ``prompt`` (style/mood; required 1-2000 chars for instrumental);
  ``lyrics`` (required 1-3500 chars for vocal tracks, supports [Intro]/
  [Verse]/[Chorus]/[Bridge]/[Outro] tags); ``is_instrumental`` bool;
  ``output_format`` "url"|"hex" (default hex);
  ``audio_setting`` {"sample_rate": 16000|24000|32000|44100,
  "bitrate": 32000|64000|128000|256000, "format": "mp3"|"wav"|"pcm"}.
- Response: ``data.audio`` (hex-encoded audio, or a URL when
  ``output_format="url"``), ``data.status`` (2 = completed), ``base_resp``
  {status_code: 0 success, 1002 rate limit, 1004 auth, 1008 balance,
  1026 sensitive content, 2013 invalid params, 2049 invalid key},
  ``extra_info`` (duration, sample rate, file size).

NOTE: the plan (§4.2) names "MiniMax Music 1.5/2.5"; the current official
docs list music-3.0/music-2.6 as the available generation models (music-01
and music-1.5 are the legacy names still served by third-party resellers).
Default model here is "music-3.0"; override with the MINIMAX_MUSIC_MODEL
environment variable if your account exposes different model names.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import config
from generation.base import (FatalGenerationError, GenerationError,
                             GenerationJob, RateLimiter, backend_output_dir,
                             require_requests)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.minimax.io"
ENDPOINT = "/v1/music_generation"
DEFAULT_MODEL = "music-3.0"

# base_resp.status_code values that are permanent for this request.
FATAL_STATUS_CODES = {1004, 1008, 1026, 2013, 2049}
RETRYABLE_STATUS_CODES = {1002}          # rate limit

AUDIO_SETTING = {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}


class MiniMaxError(GenerationError):
    """MiniMax API failure with status code and message."""


class MiniMaxClient:
    """Single-call client: MiniMax music_generation is synchronous (the
    response carries the finished audio; no task polling)."""

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None, base_url: str = BASE_URL,
                 audio_setting: Optional[dict] = None,
                 output_format: str = "url",
                 rate_limiter: Optional[RateLimiter] = None,
                 http_timeout_s: float = 600.0):
        self.api_key = (api_key if api_key is not None
                        else config.MINIMAX_API_KEY)
        self.model = model or os.environ.get("MINIMAX_MUSIC_MODEL",
                                             DEFAULT_MODEL)
        self.base_url = base_url.rstrip("/")
        self.audio_setting = dict(audio_setting or AUDIO_SETTING)
        self.output_format = output_format
        self.rate_limiter = rate_limiter or RateLimiter(0.0)
        self.http_timeout_s = http_timeout_s

    def _require_key(self) -> None:
        if not self.api_key:
            raise FatalGenerationError(
                "MINIMAX_API_KEY is not set (export it or pass api_key=)")

    def _build_payload(self, rendered: dict, vocal: bool) -> dict:
        payload = {"model": self.model,
                   "prompt": rendered["prompt"][:2000],
                   "audio_setting": self.audio_setting,
                   "output_format": self.output_format}
        if vocal:
            payload["lyrics"] = rendered["lyrics"][:3500]
        else:
            payload["is_instrumental"] = True
        return payload

    def generate(self, rendered: dict, vocal: bool) -> tuple[bytes | str, dict]:
        """Call the API; returns (audio_hex_bytes_or_url, response_meta).

        Raises with the raw response body on any HTTP or base_resp error —
        no silent fallbacks (honest error surface per P5 spec).
        """
        self._require_key()
        requests = require_requests()
        self.rate_limiter.wait()
        resp = requests.post(
            f"{self.base_url}{ENDPOINT}",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=self._build_payload(rendered, vocal),
            timeout=self.http_timeout_s)
        if 400 <= resp.status_code < 500:
            raise FatalGenerationError(
                f"MiniMax POST {ENDPOINT} -> HTTP {resp.status_code}: "
                f"{resp.text[:2000]}")
        if resp.status_code >= 500:
            raise MiniMaxError(
                f"MiniMax POST {ENDPOINT} -> HTTP {resp.status_code}: "
                f"{resp.text[:2000]}")
        body = resp.json()
        base = body.get("base_resp") or {}
        code = base.get("status_code", 0)
        if code != 0:
            msg = (f"MiniMax base_resp {code}: "
                   f"{base.get('status_msg', '')} | body={str(body)[:1000]}")
            if code in FATAL_STATUS_CODES:
                raise FatalGenerationError(msg)
            raise MiniMaxError(msg)          # retryable (e.g. 1002 rate limit)
        data = body.get("data") or {}
        audio = data.get("audio")
        if not audio:
            raise MiniMaxError(
                f"MiniMax returned success but no data.audio: "
                f"{str(body)[:1000]}")
        meta = {"extra_info": body.get("extra_info"),
                "data_status": data.get("status")}
        return audio, meta

    def run(self, job: GenerationJob, rendered: dict) -> tuple[list[Path], dict]:
        """Generate one track and write it to GENERATED_DATA_DIR/minimax/."""
        vocal = bool(job.style["vocal"])
        audio, meta = self.generate(rendered, vocal)
        out_dir = backend_output_dir("minimax")
        ext = "." + str(self.audio_setting.get("format", "mp3"))
        dest = out_dir / f"{job.job_id}{ext}"
        if isinstance(audio, str) and audio.startswith(("http://", "https://")):
            requests = require_requests()
            dl = requests.get(audio, timeout=300)
            if dl.status_code != 200:
                raise MiniMaxError(
                    f"MiniMax audio download -> HTTP {dl.status_code}")
            dest.write_bytes(dl.content)
        else:
            try:
                dest.write_bytes(bytes.fromhex(audio))
            except ValueError as exc:
                raise MiniMaxError(
                    f"MiniMax data.audio is neither URL nor valid hex "
                    f"(first 80 chars: {str(audio)[:80]!r})") from exc
        settings = {"model": self.model, "audio_setting": self.audio_setting,
                    "output_format": self.output_format,
                    "is_instrumental": not vocal,
                    "prompt": rendered["prompt"], **meta}
        return [dest], settings

    @property
    def generator_version(self) -> str:
        return self.model
