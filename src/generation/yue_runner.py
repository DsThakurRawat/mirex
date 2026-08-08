"""Local YuE-7B two-stage inference wrapper (plan §4.2: 5k songs, P5).

Verified against github.com/multimodal-art-projection/YuE (fetched
2026-08-08). Inference is a script, not a package::

    cd $YUE_REPO/inference
    python infer.py --cuda_idx 0 \
        --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
        --stage2_model m-a-p/YuE-s2-1B-general \
        --genre_txt genre.txt --lyrics_txt lyrics.txt \
        --run_n_segments 2 --stage2_batch_size 4 \
        --output_dir out --max_new_tokens 3000 --repetition_penalty 1.1

File formats (verified from repo README/prompt_egs):
- ``genre.txt``: one line of space-separated genre/instrument/mood/gender/
  timbre tokens, e.g. ``inspiring female uplifting pop airy vocal``.
- ``lyrics.txt``: sections labeled ``[verse]``/``[chorus]``/... separated by
  two newlines; each section covers roughly 30 s of audio.

Notes:
- YuE is lyrics-to-song; every job renders lyrics (instrumental strata are
  covered by ACE-Step/MiniMax). Stage-1 checkpoints are language-annealed;
  we map style language -> en/zh/jp-kr checkpoints (romance languages fall
  back to the en checkpoint).
- Runtimes are long (minutes per segment on an L40S/H100); each job runs
  under a hard timeout (``YUE_TIMEOUT_S``, default 2 h) so a hung job cannot
  stall the campaign (plan §11 unattended operation).

Environment: ``YUE_REPO`` (required), ``YUE_STAGE1_MODEL``/
``YUE_STAGE2_MODEL`` (optional overrides), ``YUE_PYTHON``, ``YUE_CUDA_IDX``,
``YUE_TIMEOUT_S``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from generation.base import (FatalGenerationError, GenerationError,
                             GenerationJob, backend_output_dir)

logger = logging.getLogger(__name__)

STAGE2_DEFAULT = "m-a-p/YuE-s2-1B-general"
STAGE1_BY_LANGUAGE = {
    "en": "m-a-p/YuE-s1-7B-anneal-en-cot",
    "zh": "m-a-p/YuE-s1-7B-anneal-zh-cot",
    "ja": "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
}
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac")
SECONDS_PER_SEGMENT = 30                 # per YuE docs: ~30 s per section


class YueRunner:
    """Subprocess wrapper around YuE's ``inference/infer.py``."""

    def __init__(self, repo: Optional[str] = None, cuda_idx: Optional[int] = None,
                 timeout_s: Optional[float] = None,
                 stage2_batch_size: int = 4, max_new_tokens: int = 3000,
                 repetition_penalty: float = 1.1):
        self.repo = Path(repo or os.environ.get("YUE_REPO", ""))
        self.cuda_idx = (cuda_idx if cuda_idx is not None
                         else int(os.environ.get("YUE_CUDA_IDX", "0")))
        self.timeout_s = timeout_s if timeout_s is not None else float(
            os.environ.get("YUE_TIMEOUT_S", "7200"))
        self.stage2_batch_size = stage2_batch_size
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.stage2_model = os.environ.get("YUE_STAGE2_MODEL", STAGE2_DEFAULT)

    def _stage1_model(self, language: str) -> str:
        override = os.environ.get("YUE_STAGE1_MODEL")
        if override:
            return override
        return STAGE1_BY_LANGUAGE.get(language, STAGE1_BY_LANGUAGE["en"])

    def _check_repo(self) -> Path:
        infer = self.repo / "inference" / "infer.py"
        if not self.repo.is_dir() or not infer.exists():
            raise FatalGenerationError(
                "YUE_REPO is not set to a checkout of "
                "github.com/multimodal-art-projection/YuE "
                f"(looked for {infer})")
        return infer

    @staticmethod
    def _newest_audio(root: Path) -> Optional[Path]:
        candidates = [p for p in root.rglob("*")
                      if p.suffix.lower() in AUDIO_SUFFIXES and p.is_file()]
        if not candidates:
            return None
        # YuE writes stage outputs too; prefer files that look final (mixed),
        # falling back to the newest audio file of any kind.
        mixed = [p for p in candidates if "mix" in p.name.lower()]
        pool = mixed or candidates
        return max(pool, key=lambda p: p.stat().st_mtime)

    def run(self, job: GenerationJob, rendered: dict) -> tuple[list[Path], dict]:
        """Generate one track; returns ([final_audio_path], settings)."""
        infer = self._check_repo()
        out_dir = backend_output_dir("yue")
        workdir = out_dir / "work" / job.job_id
        job_out = workdir / "out"
        job_out.mkdir(parents=True, exist_ok=True)

        genre_txt = workdir / "genre.txt"
        lyrics_txt = workdir / "lyrics.txt"
        genre_txt.write_text(rendered["genre"] + "\n", encoding="utf-8")
        lyrics_txt.write_text(rendered["lyrics"] + "\n", encoding="utf-8")

        n_segments = max(2, min(6, int(job.style["duration_s"])
                                // SECONDS_PER_SEGMENT))
        stage1 = self._stage1_model(job.style.get("language", "en"))
        python = os.environ.get("YUE_PYTHON", sys.executable)
        cmd = [python, str(infer),
               "--cuda_idx", str(self.cuda_idx),
               "--stage1_model", stage1,
               "--stage2_model", self.stage2_model,
               "--genre_txt", str(genre_txt),
               "--lyrics_txt", str(lyrics_txt),
               "--run_n_segments", str(n_segments),
               "--stage2_batch_size", str(self.stage2_batch_size),
               "--output_dir", str(job_out),
               "--max_new_tokens", str(self.max_new_tokens),
               "--repetition_penalty", str(self.repetition_penalty)]
        logger.info("YuE job %s: %d segments, stage1=%s", job.job_id,
                    n_segments, stage1)
        try:
            proc = subprocess.run(cmd, cwd=infer.parent, capture_output=True,
                                  text=True, timeout=self.timeout_s,
                                  check=False)
        except subprocess.TimeoutExpired as exc:
            raise GenerationError(
                f"YuE job {job.job_id} timed out after {self.timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            raise GenerationError(
                f"YuE infer.py failed (rc={proc.returncode}): "
                f"{proc.stderr[-2000:]}")

        audio = self._newest_audio(job_out)
        if audio is None:
            raise GenerationError(
                f"YuE job {job.job_id}: no audio produced under {job_out}")
        final = out_dir / f"{job.job_id}{audio.suffix.lower()}"
        shutil.move(str(audio), final)
        shutil.rmtree(workdir, ignore_errors=True)

        settings = {"stage1_model": stage1, "stage2_model": self.stage2_model,
                    "run_n_segments": n_segments,
                    "stage2_batch_size": self.stage2_batch_size,
                    "max_new_tokens": self.max_new_tokens,
                    "repetition_penalty": self.repetition_penalty,
                    "genre_line": rendered["genre"]}
        return [final], settings

    @property
    def generator_version(self) -> str:
        return f"7B ({self.stage2_model})"
