"""MusicScope-CL — Dataset & Augmentation pipeline.

Provides:
  - SpectrogramAugmentation: time/freq masking, pitch shift, noise injection
  - SimCLRDataset: yields paired augmented views (no labels) for Stage 1
  - LabeledDataset: yields (spectrogram, narrative_vec, label) for Stage 3
  - Helper functions to build DataLoaders

Audio files are discovered from the shared data directory and the metadata.db
built by Architecture-2's data_fetch pipeline.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import os
import sys
from contextlib import contextmanager

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


@contextmanager
def suppress_stderr():
    """Context manager to suppress C-level stderr output (like mpg123 warnings)."""
    try:
        fd = sys.stderr.fileno()
        old_stderr = os.dup(fd)
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)
            try:
                yield
            finally:
                os.dup2(old_stderr, fd)
                os.close(old_stderr)
    except Exception:
        yield



# ---------------------------------------------------------------------------
# Augmentations for log-Mel spectrograms (SimCLR paired views)
# ---------------------------------------------------------------------------

class SpectrogramAugmentation:
    """Stochastic augmentation of a log-Mel spectrogram tensor.

    Each call applies a random subset of:
      - Time masking (SpecAugment-style)
      - Frequency masking (SpecAugment-style)
      - Additive Gaussian noise
      - Pitch shift (via frequency-axis roll)
    """

    def __init__(
        self,
        time_mask_max: int = 40,
        freq_mask_max: int = 15,
        noise_std: float = 0.02,
        pitch_shift_max: int = 4,
    ):
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.noise_std = noise_std
        self.pitch_shift_max = pitch_shift_max

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """spec: (n_mels, T) float32 array, values in [0, 1]."""
        s = spec.copy()
        n_mels, T = s.shape

        # Time mask
        if random.random() < 0.8:
            t_width = random.randint(1, min(self.time_mask_max, T // 4))
            t_start = random.randint(0, max(0, T - t_width))
            s[:, t_start:t_start + t_width] = 0.0

        # Frequency mask
        if random.random() < 0.8:
            f_width = random.randint(1, min(self.freq_mask_max, n_mels // 4))
            f_start = random.randint(0, max(0, n_mels - f_width))
            s[f_start:f_start + f_width, :] = 0.0

        # Additive noise
        if random.random() < 0.5:
            s = s + np.random.randn(*s.shape).astype(np.float32) * self.noise_std

        # Pitch shift (roll along frequency axis)
        if random.random() < 0.5:
            shift = random.randint(-self.pitch_shift_max, self.pitch_shift_max)
            s = np.roll(s, shift, axis=0)

        return np.clip(s, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------

def load_audio_chunk(path: str | Path, sr: int = config.SAMPLE_RATE,
                     duration: float = config.CHUNK_SECONDS,
                     offset: float = 0.0) -> np.ndarray:
    """Load a mono audio chunk, zero-padded if shorter than duration."""
    with suppress_stderr():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=sr, mono=True,
                                duration=duration, offset=offset)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype(np.float32)


def audio_to_logmel(y: np.ndarray, sr: int = config.SAMPLE_RATE,
                    n_mels: int = config.N_MELS) -> np.ndarray:
    """Convert waveform to normalized log-Mel spectrogram in [0, 1]."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    rng = float(log_mel.max() - log_mel.min())
    if rng > 1e-6:
        log_mel = (log_mel - log_mel.min()) / rng
    else:
        log_mel = np.zeros_like(log_mel)
    return np.nan_to_num(log_mel, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def discover_audio_files(root: Path) -> list[Path]:
    """Recursively find all audio files under root."""
    files = []
    for ext in config.AUDIO_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


# ---------------------------------------------------------------------------
# SimCLR Dataset (Stage 1 — unsupervised, paired augmented views)
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """Yields two differently-augmented log-Mel views of the same audio chunk.

    No labels are used — this is for unsupervised contrastive pretraining.
    """

    def __init__(self, audio_paths: list[Path],
                 augmentation: Optional[SpectrogramAugmentation] = None):
        self.paths = audio_paths
        self.aug = augmentation or SpectrogramAugmentation()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(10):
            path = self.paths[idx]
            try:
                with suppress_stderr():
                    info = librosa.get_duration(path=str(path))
                max_offset = max(0.0, info - config.CHUNK_SECONDS)
            except Exception:
                max_offset = 0.0
            offset = random.uniform(0, max_offset) if max_offset > 0 else 0.0

            try:
                y = load_audio_chunk(path, offset=offset)
                log_mel = audio_to_logmel(y)
                view1 = self.aug(log_mel)
                view2 = self.aug(log_mel)
                return (torch.from_numpy(view1).unsqueeze(0),
                        torch.from_numpy(view2).unsqueeze(0))
            except Exception:
                idx = random.randint(0, len(self.paths) - 1)

        # Fallback to zero-spectrogram if 10 files fail in a row
        blank = np.zeros((config.N_MELS, 1293), dtype=np.float32)
        return (torch.from_numpy(blank).unsqueeze(0),
                torch.from_numpy(blank).unsqueeze(0))


# ---------------------------------------------------------------------------
# Labeled Dataset (Stage 3 — supervised, for SupCon + classification)
# ---------------------------------------------------------------------------

class LabeledDataset(Dataset):
    """Yields (log_mel_tensor, label) for supervised training.

    `entries` is a list of (path, label) tuples where label is 0 (human)
    or 1 (AI-generated).
    """

    def __init__(self, entries: list[tuple[Path, int]],
                 augmentation: Optional[SpectrogramAugmentation] = None):
        self.entries = entries
        self.aug = augmentation

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        for _ in range(10):
            path, label = self.entries[idx]
            try:
                y = load_audio_chunk(path)
                log_mel = audio_to_logmel(y)
                if self.aug is not None:
                    log_mel = self.aug(log_mel)
                return torch.from_numpy(log_mel).unsqueeze(0), label
            except Exception:
                idx = random.randint(0, len(self.entries) - 1)

        blank = np.zeros((config.N_MELS, 1293), dtype=np.float32)
        return torch.from_numpy(blank).unsqueeze(0), self.entries[idx][1]


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_simclr_loader(data_dir: Optional[Path] = None,
                        batch_size: int = config.SIMCLR_BATCH_SIZE,
                        num_workers: int = 4) -> DataLoader:
    """Build a DataLoader for SimCLR pretraining over all audio in data_dir."""
    root = data_dir or config.RAW_DATA_DIR
    paths = discover_audio_files(root)
    if not paths:
        raise FileNotFoundError(
            f"No audio files found under {root}. Run data_fetch first.")
    print(f"[SimCLR DataLoader] Found {len(paths)} audio files under {root}")
    ds = SimCLRDataset(paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True)


def build_labeled_loader(entries: list[tuple[Path, int]],
                         batch_size: int = config.SUPCON_BATCH_SIZE,
                         shuffle: bool = True,
                         augment: bool = True,
                         num_workers: int = 4) -> DataLoader:
    """Build a DataLoader for supervised training (SupCon or classification)."""
    aug = SpectrogramAugmentation() if augment else None
    ds = LabeledDataset(entries, augmentation=aug)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=False)


def load_labeled_entries_from_db(db_path: Optional[Path] = None
                                 ) -> list[tuple[Path, int]]:
    """Read track entries from metadata.db (built by Architecture-2's pipeline).

    Returns list of (audio_path, label) where label = is_ai (0 or 1).
    Only includes tracks whose audio file actually exists on disk.
    """
    import sqlite3
    db = db_path or config.PROCESSED_DATA_DIR / "metadata.db"
    if not db.exists():
        raise FileNotFoundError(
            f"{db} not found. Run Architecture-2's data_fetch to build it.")
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT file_path, is_ai FROM tracks WHERE file_path IS NOT NULL"
    ).fetchall()
    conn.close()

    entries = []
    for fp, is_ai in rows:
        p = Path(fp)
        if p.exists():
            entries.append((p, int(is_ai)))
    print(f"[LabeledLoader] {len(entries)} tracks with existing audio "
          f"(of {len(rows)} in DB)")
    return entries
