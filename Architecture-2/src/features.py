"""Signal-level feature extractors shared by the physics branch, the fakeprint
baseline, and the confound audit.

Fakeprint (plan §6.C.1, after Afchar et al. ISMIR 2025): time-averaged
log-magnitude spectrum with the smooth lower spectral envelope removed, leaving
the deconvolution peak comb. Peaks appear at f = k * (f_out / prod(strides
above layer l)) — architecture-determined, weight-independent.
"""
from __future__ import annotations

import numpy as np
import torch

import config

# Candidate intermediate rates of known decoders (Hz). EnCodec strides
# {8,5,4,2} @24k, DAC @44.1k, X-Codec 50 Hz frame @16k, ADaMoSHiFiGAN 512x
# @44.1k, mel-hop grids. We take the union of implied comb fundamentals.
COMB_FUNDAMENTALS_HZ = [
    75.0, 150.0, 300.0, 600.0,          # 2^k ladders of 512x/1024x upsamplers
    86.13, 172.27, 344.53,               # 44100/512, /256, /128 (mel-hop grids)
    50.0, 100.0, 200.0, 400.0,           # X-Codec 50 Hz token grid images
    93.75, 187.5, 375.0, 750.0,          # 24000/256 EnCodec-family ladders
]


def average_log_spectrum(wave: torch.Tensor, sr: int,
                         n_fft: int = 4096, hop: int = 1024) -> np.ndarray:
    """Time-averaged log-magnitude spectrum (the raw 'fakeprint' carrier)."""
    mono = wave.mean(dim=0) if wave.dim() == 2 else wave
    window = torch.hann_window(n_fft)
    spec = torch.stft(mono, n_fft=n_fft, hop_length=hop, window=window,
                      return_complex=True)
    mag = spec.abs().mean(dim=-1).numpy()
    return np.log1p(mag)


def remove_lower_envelope(log_spec: np.ndarray, kernel: int = 51) -> np.ndarray:
    """Subtract the running-minimum envelope (grey-opening) so only narrow
    peaks — the deconvolution comb — survive. kernel is in FFT bins."""
    from scipy.ndimage import grey_opening
    envelope = grey_opening(log_spec, size=kernel)
    return log_spec - envelope


def fakeprint(wave: torch.Tensor, sr: int, n_fft: int = 4096,
              hop: int = 1024) -> np.ndarray:
    """Peak-residual spectrum restricted to the informative 1-8 kHz band
    plus the 8-24 kHz band (band-limit forensics), concatenated."""
    residual = remove_lower_envelope(average_log_spectrum(wave, sr, n_fft, hop))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    low = residual[(freqs >= 1000) & (freqs < 8000)]
    high = residual[(freqs >= 8000) & (freqs < 24000)]
    return np.concatenate([low, high]).astype(np.float32)


def comb_peak_energies(wave: torch.Tensor, sr: int, n_fft: int = 8192,
                       max_harmonic: int = 24) -> np.ndarray:
    """Residual energy sampled at each candidate comb's harmonics, relative to
    local background — one scalar per fundamental in COMB_FUNDAMENTALS_HZ."""
    residual = remove_lower_envelope(
        average_log_spectrum(wave, sr, n_fft, n_fft // 4))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bin_hz = freqs[1]
    out = []
    for f0 in COMB_FUNDAMENTALS_HZ:
        peaks, background = [], []
        for k in range(1, max_harmonic + 1):
            f = f0 * k
            if f >= sr / 2 - bin_hz * 4:
                break
            idx = int(round(f / bin_hz))
            peaks.append(residual[idx - 1:idx + 2].max())
            background.append(np.median(residual[max(0, idx - 20):idx + 20]))
        out.append(float(np.mean(peaks) - np.mean(background)) if peaks else 0.0)
    return np.array(out, dtype=np.float32)


def estimate_cutoff_hz(wave: torch.Tensor, sr: int, n_fft: int = 4096,
                       rel_db: float = -50.0) -> float:
    """Highest frequency whose average power is above (peak + rel_db) —
    detects band-limits (YuE 16 kHz codec, Suno 12-16 kHz drift, MP3 legacy)."""
    mono = wave.mean(dim=0) if wave.dim() == 2 else wave
    window = torch.hann_window(n_fft)
    spec = torch.stft(mono, n_fft=n_fft, hop_length=n_fft // 4, window=window,
                      return_complex=True)
    power = (spec.abs() ** 2).mean(dim=-1).numpy()
    db = 10 * np.log10(power + 1e-20)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    above = np.where(db > db.max() + rel_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def confound_features(wave: torch.Tensor, sr: int) -> dict[str, float]:
    """Non-content features the CLASS LABEL must NOT be predictable from
    after the delivery-chain simulator (plan §4.5 gate)."""
    import librosa
    from simulator import measure_lufs
    mono = wave.mean(dim=0).numpy() if wave.dim() == 2 else wave.numpy()
    feats: dict[str, float] = {}
    feats["duration_s"] = len(mono) / sr
    feats["lufs"] = measure_lufs(wave if wave.dim() == 2 else wave[None], sr)
    rms = np.sqrt(np.mean(mono ** 2) + 1e-12)
    feats["rms"] = float(rms)
    feats["crest_db"] = float(20 * np.log10(np.abs(mono).max() / (rms + 1e-12)
                                            + 1e-12))
    feats["silence_ratio"] = float(np.mean(np.abs(mono) < 1e-4))
    feats["zcr"] = float(np.mean(librosa.zero_crossings(mono)))
    S = np.abs(librosa.stft(mono, n_fft=2048))
    feats["rolloff85_hz"] = float(np.mean(
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)))
    feats["rolloff99_hz"] = float(np.mean(
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.99)))
    feats["cutoff_hz"] = estimate_cutoff_hz(
        wave if wave.dim() == 2 else wave[None], sr)
    total = float(np.sum(S ** 2) + 1e-12)
    freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
    feats["hf16k_ratio"] = float(np.sum(S[freqs >= 16000] ** 2) / total)
    if wave.dim() == 2 and wave.shape[0] == 2:
        l, r = wave[0].numpy(), wave[1].numpy()
        feats["stereo_width"] = float(np.mean((l - r) ** 2) /
                                      (np.mean((l + r) ** 2) + 1e-12))
    else:
        feats["stereo_width"] = 0.0
    return feats


CONFOUND_FEATURE_NAMES = ["duration_s", "lufs", "rms", "crest_db",
                          "silence_ratio", "zcr", "rolloff85_hz",
                          "rolloff99_hz", "cutoff_hz", "hf16k_ratio",
                          "stereo_width"]
