"""Metadata-DB-driven datasets and samplers (plan §4.3).

Flow per item: load full track -> delivery-chain simulator (§4.4) -> resample
to the branch's input rate -> mono -> random fixed-length chunk. Class-balanced
sampling with per-source quota caps so no single dataset dominates its class.
"""
from __future__ import annotations

import logging
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import config
from simulator import DeliveryChainSimulator, _resample

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, max_s: float | None = None
               ) -> tuple[torch.Tensor, int]:
    """Robust loader: soundfile first, ffmpeg-decode fallback (odd codecs).
    Returns (channels, time) float32 and sample rate."""
    path = str(path)
    try:
        if max_s is not None:
            info = sf.info(path)
            frames = int(max_s * info.samplerate)
            data, sr = sf.read(path, frames=frames, dtype="float32",
                               always_2d=True)
        else:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    except Exception:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path]
            if max_s is not None:
                cmd += ["-t", str(max_s)]
            cmd += [tmp.name]
            subprocess.run(cmd, check=True, capture_output=True)
            data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr


class TrackChunkDataset(Dataset):
    """rows: metadata-DB dicts (track_id, file_path, is_ai, generator_family,
    source_dataset). One random simulated chunk per __getitem__."""

    def __init__(self, rows: list[dict], branch: str,
                 augment: bool = True, epoch_seed: int = 0,
                 max_load_s: float = 300.0):
        self.rows = rows
        cfg = config.BRANCHES[branch]
        self.input_sr = cfg["input_sr"]
        self.chunk_s = cfg["chunk_s"]
        self.augment = augment
        self.epoch_seed = epoch_seed
        self.max_load_s = max_load_s
        self.sim = DeliveryChainSimulator()

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int):
        self.epoch_seed = epoch

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wave, sr = load_audio(row["file_path"], max_s=self.max_load_s)
        key = f"{row['track_id']}#{self.epoch_seed}"
        if self.augment:
            wave = self.sim.random_chain(wave, sr, item_key=key, excerpt=False)
            sr = self.sim.sr
        wave = _resample(wave, sr, self.input_sr).mean(dim=0)   # mono (T,)
        n = int(self.chunk_s * self.input_sr)
        if wave.shape[0] < n:
            wave = torch.nn.functional.pad(wave, (0, n - wave.shape[0]))
        else:
            rng = np.random.RandomState(abs(hash(key)) % (2 ** 31))
            start = rng.randint(0, wave.shape[0] - n + 1)
            wave = wave[start:start + n]
        return wave, torch.tensor(float(row["is_ai"])), row["track_id"]


def logo_split(rows: list[dict], holdout_family: str | None
               ) -> tuple[list[dict], list[dict]]:
    """Leave-one-generator-out (plan §5): train excludes the held-out AI
    family; val = held-out family AI + a slice of real."""
    if holdout_family is None:
        return rows, []
    train = [r for r in rows if r["generator_family"] != holdout_family]
    held = [r for r in rows if r["generator_family"] == holdout_family]
    real = [r for r in rows if not r["is_ai"]]
    val_real = real[:: max(1, len(real) // max(len(held), 1))][:len(held)]
    return train, held + val_real


def _capped_source_masses(counts: Counter, cap: float) -> dict[str, float]:
    """Water-filling: source masses proportional to counts, but no source may
    exceed `cap` of the class after normalization. Excess mass redistributes
    to uncapped sources; if every source hits the cap, fall back to equal."""
    total = sum(counts.values())
    masses = {s: n / total for s, n in counts.items()}
    if len(masses) == 1:
        return {next(iter(masses)): 1.0}          # single source: no cap useful
    for _ in range(len(masses) + 1):
        over = {s for s, m in masses.items() if m > cap + 1e-12}
        if not over:
            return masses
        free = {s: counts[s] for s in masses if s not in over}
        fixed_mass = cap * len(over)
        if not free or fixed_mass >= 1.0:
            return {s: 1.0 / len(masses) for s in masses}   # infeasible cap
        free_total = sum(free.values())
        masses = {s: cap if s in over else
                  (1.0 - fixed_mass) * counts[s] / free_total
                  for s in masses}
    return masses


def quota_sampler(rows: list[dict], num_samples: int | None = None
                  ) -> WeightedRandomSampler:
    """Class-balanced exactly 1:1 in expectation, with per-source share inside
    each class capped at config.SOURCE_QUOTA_CAP (plan §4.3, water-filled)."""
    labels = np.array([r["is_ai"] for r in rows])
    sources = [r["source_dataset"] for r in rows]
    weights = np.zeros(len(rows))
    for cls in (0, 1):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        counts = Counter(sources[i] for i in cls_idx)
        masses = _capped_source_masses(counts, config.SOURCE_QUOTA_CAP)
        for i in cls_idx:
            src = sources[i]
            weights[i] = 0.5 * masses[src] / counts[src]   # class mass = 0.5
    weights /= weights.sum()
    return WeightedRandomSampler(torch.from_numpy(weights),
                                 num_samples or len(rows), replacement=True)
