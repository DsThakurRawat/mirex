"""Local ACE-Step v1-3.5B inference driver (plan §4.2: 20k songs, P5).

Verified against github.com/ace-step/ACE-Step (fetched 2026-08-08):
- pip package ``acestep``; pipeline class
  ``acestep.pipeline_ace_step.ACEStepPipeline(checkpoint_dir=..., dtype=
  "bfloat16"|"float32", torch_compile=..., cpu_offload=...,
  overlapped_decode=...)`` (auto-downloads weights when ``checkpoint_dir``
  is None/empty).
- Call signature (repo ``infer.py``): ``pipeline(prompt=..., lyrics=...,
  audio_duration=..., infer_step=..., guidance_scale=..., scheduler_type=...,
  cfg_type=..., omega_scale=..., manual_seeds=..., save_path=...)``.

Two execution modes:
1. In-process (preferred): ``import acestep`` if installed; the pipeline is
   constructed once and reused across jobs (model init dominates otherwise).
2. Subprocess fallback: set ``ACE_STEP_REPO`` to a clone of the repo; a tiny
   driver script is executed with ``PYTHONPATH=$ACE_STEP_REPO`` per job.

Sampler-setting variation (plan §4.2: "artifact intensity varies with
sampler settings, and the eval 'difficulty' stratum likely encodes exactly
this"): ``infer_step`` in {27, 60}, ``guidance_scale`` in {7.5, 10.0, 15.0},
scheduler euler/heun — all drawn deterministically from the style seed.

dtype: defaults to ``"auto"`` — bfloat16 on Ampere or newer, float32
otherwise. ACE-Step accepts no fp16, and Volta/Pascal (every DGX-1) have no
bf16 units, so a hardcoded "bfloat16" fails there. See resolve_dtype().

Environment: ``ACE_STEP_REPO``, ``ACE_STEP_CHECKPOINT``, ``ACE_STEP_PYTHON``
(interpreter for subprocess mode), ``ACE_STEP_TIMEOUT_S``, ``ACE_STEP_DTYPE``
(forces the dtype, bypassing autodetection).
"""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

from generation.base import (FatalGenerationError, GenerationError,
                             GenerationJob, backend_output_dir)

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "v1-3.5B"

STEPS_CHOICES = (27, 60)                 # turbo vs high-quality regime
GUIDANCE_CHOICES = (7.5, 10.0, 15.0)
SCHEDULER_CHOICES = ("euler", "heun")

# Driver executed via `python -c` in subprocess mode; argv[1] = JSON args.
_SUBPROCESS_DRIVER = r"""
import json, sys
args = json.load(open(sys.argv[1], encoding="utf-8"))
from acestep.pipeline_ace_step import ACEStepPipeline
pipe = ACEStepPipeline(checkpoint_dir=args["checkpoint_dir"],
                       dtype=args["dtype"], torch_compile=False)
pipe(prompt=args["prompt"], lyrics=args["lyrics"],
     audio_duration=args["audio_duration"], infer_step=args["infer_step"],
     guidance_scale=args["guidance_scale"],
     scheduler_type=args["scheduler_type"], cfg_type=args["cfg_type"],
     omega_scale=args["omega_scale"], manual_seeds=args["manual_seeds"],
     save_path=args["save_path"])
"""


def sample_settings(style: dict) -> dict:
    """Deterministic sampler settings for a style (varied per plan §4.2)."""
    rng = random.Random(f"ace_step:{style['seed']}")
    return {
        "infer_step": rng.choice(STEPS_CHOICES),
        "guidance_scale": rng.choice(GUIDANCE_CHOICES),
        "scheduler_type": rng.choices(SCHEDULER_CHOICES, [0.8, 0.2])[0],
        "cfg_type": "apg",               # repo default
        "omega_scale": 10.0,             # repo default
        "manual_seeds": str(style["seed"]),
    }


def resolve_dtype(requested: str = "auto") -> str:
    """Pick a dtype ACE-Step will actually accept on this GPU.

    The pipeline takes ``"bfloat16"`` or ``"float32"`` only — there is no fp16
    path, so the fallback for a card without bf16 is fp32, not half.

    bf16 is native from Ampere (sm_80) onward. Volta (V100, sm_70) and Pascal
    (P100, sm_60) — i.e. every DGX-1 — have no bf16 units. We test the compute
    capability directly rather than ``torch.cuda.is_bf16_supported()``, which
    in torch >= 2.6 reports True on older cards via slow emulation.
    """
    if requested != "auto":
        return requested
    override = os.environ.get("ACE_STEP_DTYPE")
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "float32"
    if not torch.cuda.is_available():
        return "float32"
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        return "bfloat16"
    logger.warning(
        "GPU %s is compute capability %d.%d — no native bf16; falling back to "
        "float32. Expect roughly 2x the memory and a slower campaign. Set "
        "ACE_STEP_DTYPE to override.",
        torch.cuda.get_device_name(0), major, minor)
    return "float32"


class AceStepRunner:
    """Drives ACE-Step for one job at a time (GPU-bound; use --workers 1)."""

    def __init__(self, checkpoint_dir: Optional[str] = None,
                 dtype: str = "auto",
                 timeout_s: Optional[float] = None):
        self.checkpoint_dir = (checkpoint_dir
                               or os.environ.get("ACE_STEP_CHECKPOINT") or "")
        self.dtype = resolve_dtype(dtype)
        self.timeout_s = timeout_s if timeout_s is not None else float(
            os.environ.get("ACE_STEP_TIMEOUT_S", "1800"))
        self._pipeline = None            # lazy; heavy (torch) import

    # --- execution modes --------------------------------------------------
    def _get_pipeline(self):
        """In-process pipeline if the ``acestep`` package is importable."""
        if self._pipeline is None:
            try:
                from acestep.pipeline_ace_step import ACEStepPipeline
            except ImportError:
                return None
            logger.info("Initializing in-process ACEStepPipeline "
                        "(checkpoint=%s dtype=%s)",
                        self.checkpoint_dir or "<auto-download>", self.dtype)
            self._pipeline = ACEStepPipeline(
                checkpoint_dir=self.checkpoint_dir, dtype=self.dtype,
                torch_compile=False)
        return self._pipeline

    def _run_subprocess(self, args: dict, workdir: Path) -> None:
        repo = os.environ.get("ACE_STEP_REPO", "")
        if not repo or not Path(repo).is_dir():
            raise FatalGenerationError(
                "ACE-Step not importable and ACE_STEP_REPO is not set to a "
                "checkout of github.com/ace-step/ACE-Step")
        args_file = workdir / "args.json"
        args_file.write_text(json.dumps(args, ensure_ascii=False),
                             encoding="utf-8")
        python = os.environ.get("ACE_STEP_PYTHON", sys.executable)
        env = dict(os.environ)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [python, "-c", _SUBPROCESS_DRIVER, str(args_file)],
            cwd=repo, env=env, capture_output=True, text=True,
            timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            raise GenerationError(
                f"ACE-Step subprocess failed (rc={proc.returncode}): "
                f"{proc.stderr[-2000:]}")

    # --- public API -------------------------------------------------------
    def run(self, job: GenerationJob, rendered: dict) -> tuple[list[Path], dict]:
        """Generate one track; returns ([output_path], settings)."""
        settings = sample_settings(job.style)
        out_dir = backend_output_dir("ace_step")
        out_path = out_dir / f"{job.job_id}.wav"
        call_args = {
            "checkpoint_dir": self.checkpoint_dir,
            "dtype": self.dtype,
            "prompt": rendered["prompt"],
            "lyrics": rendered["lyrics"],
            "audio_duration": float(job.style["duration_s"]),
            "save_path": str(out_path),
            **settings,
        }
        pipeline = self._get_pipeline()
        if pipeline is not None:
            pipeline(prompt=call_args["prompt"], lyrics=call_args["lyrics"],
                     audio_duration=call_args["audio_duration"],
                     infer_step=settings["infer_step"],
                     guidance_scale=settings["guidance_scale"],
                     scheduler_type=settings["scheduler_type"],
                     cfg_type=settings["cfg_type"],
                     omega_scale=settings["omega_scale"],
                     manual_seeds=settings["manual_seeds"],
                     save_path=str(out_path))
        else:
            self._run_subprocess(call_args, out_dir)
        if not out_path.exists():
            # Some acestep versions append suffixes; accept any match.
            matches = sorted(out_dir.glob(f"{job.job_id}*"),
                             key=lambda p: p.stat().st_mtime)
            audio = [p for p in matches
                     if p.suffix.lower() in (".wav", ".mp3", ".flac")]
            if not audio:
                raise GenerationError(
                    f"ACE-Step produced no audio for {job.job_id}")
            out_path = audio[-1]
        settings_out = {**settings, "checkpoint_dir": self.checkpoint_dir,
                        "dtype": self.dtype,
                        "audio_duration": call_args["audio_duration"],
                        "prompt": rendered["prompt"]}
        return [out_path], settings_out

    @property
    def generator_version(self) -> str:
        return GENERATOR_VERSION
