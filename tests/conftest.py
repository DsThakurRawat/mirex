import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import soundfile as sf
import torch


@pytest.fixture
def tone_stereo():
    """5 s stereo 440/554 Hz tones + noise at 44.1 kHz."""
    sr = 44100
    t = np.arange(5 * sr) / sr
    left = 0.4 * np.sin(2 * np.pi * 440 * t)
    right = 0.4 * np.sin(2 * np.pi * 554 * t)
    rng = np.random.RandomState(0)
    wave = np.stack([left, right]) + 0.01 * rng.randn(2, len(t))
    return torch.from_numpy(wave.astype(np.float32)), sr


def make_noise_wave(seconds: float, sr: int = 44100, channels: int = 2,
                    seed: int = 0, cutoff_hz: float | None = None
                    ) -> torch.Tensor:
    """Broadband noise, optionally band-limited below cutoff_hz (steep FFT
    brick-wall) — used to plant detectable band-limit artifacts."""
    rng = np.random.RandomState(seed)
    n = int(seconds * sr)
    wave = rng.randn(channels, n).astype(np.float32) * 0.2
    if cutoff_hz is not None:
        spec = np.fft.rfft(wave, axis=1)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        spec[:, freqs > cutoff_hz] *= 1e-6
        wave = np.fft.irfft(spec, n=n, axis=1).astype(np.float32)
    return torch.from_numpy(wave)


@pytest.fixture
def wav_dir(tmp_path):
    """Directory of 6 small real WAV files (varied sr/channels/lengths)."""
    specs = [("t0", 44100, 2, 4.0), ("t1", 48000, 1, 3.0),
             ("t2", 44100, 2, 6.0), ("t3", 44100, 1, 2.5),
             ("t4", 48000, 2, 5.0), ("t5", 44100, 2, 3.5)]
    for i, (name, sr, ch, sec) in enumerate(specs):
        w = make_noise_wave(sec, sr, ch, seed=i)
        sf.write(tmp_path / f"{name}.wav", w.T.numpy(), sr)
    return tmp_path
