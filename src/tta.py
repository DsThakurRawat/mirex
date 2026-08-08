"""Deterministic test-time augmentation (plan §9, submission slot 4).

Fixed transform set — NOT the random simulator: {identity, MP3-128 round-trip,
32 kHz resample round-trip, +0.5 st pitch, -0.5 st pitch}. Scores are averaged
over the group, stabilizing rankings under the hidden robustness strata.
"""
from __future__ import annotations

import torch

from simulator import _pitch_shift, _resample, codec_roundtrip

TTA_TRANSFORMS = ("identity", "mp3_128", "rs_32k", "pitch_up", "pitch_down")


def apply_tta_transform(wave: torch.Tensor, sr: int, name: str) -> torch.Tensor:
    if name == "identity":
        return wave
    if name == "mp3_128":
        return codec_roundtrip(wave, sr, "mp3", 128)
    if name == "rs_32k":
        return _resample(_resample(wave, sr, 32000), 32000, sr)
    if name == "pitch_up":
        return _pitch_shift(wave, sr, 0.5)
    if name == "pitch_down":
        return _pitch_shift(wave, sr, -0.5)
    raise ValueError(f"unknown TTA transform {name!r}")


def tta_score(score_fn, wave: torch.Tensor, sr: int,
              transforms=TTA_TRANSFORMS) -> float:
    """score_fn(wave, sr) -> float. Mean over the deterministic group."""
    vals = [score_fn(apply_tta_transform(wave, sr, t), sr) for t in transforms]
    return float(sum(vals) / len(vals))
