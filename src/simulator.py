"""Delivery-chain simulator (plan §4.4) — the anti-shortcut component.

One randomized augmentation chain applied IDENTICALLY to both classes:

  [pitch|stretch] -> [EQ tilt] -> [noise floor] -> [resample round-trip]
  -> [codec round-trip] -> [loudness normalize] -> [mono fold] -> [excerpt]

Two APIs:
  * DeliveryChainSimulator(seed).random_chain(wave, sr)     — training-time
  * DeliveryChainSimulator.apply_condition(wave, sr, cond)  — deterministic,
    used by the harness to materialize eval strata (§5).

All randomness flows through a per-call ``random.Random`` seeded from
(base_seed, item_key), so the pipeline is bit-reproducible given the same
inputs — required by the P2 unit-test gate. Codec round-trips use the ffmpeg
CLI via temp files (robust across torchaudio backend builds).
"""
from __future__ import annotations

import io
import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import config

logger = logging.getLogger(__name__)

_FFMPEG = shutil.which("ffmpeg")

_CODEC_ARGS = {
    "mp3":    (["-c:a", "libmp3lame"], ".mp3"),
    "aac":    (["-c:a", "aac"], ".m4a"),
    "vorbis": (["-c:a", "libvorbis"], ".ogg"),
    "opus":   (["-c:a", "libopus"], ".opus"),
}


def _resample(wave: torch.Tensor, sr_from: int, sr_to: int) -> torch.Tensor:
    if sr_from == sr_to:
        return wave
    # Kaiser-windowed sinc with a wide filter: deep stopband attenuation, so a
    # 22.05 kHz round-trip really is band-limited (defaults leak above -50 dB).
    return torchaudio.functional.resample(
        wave, sr_from, sr_to, resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64, rolloff=0.9475, beta=14.77)


def codec_roundtrip(wave: torch.Tensor, sr: int, codec: str,
                    bitrate_kbps: int) -> torch.Tensor:
    """Encode+decode through a lossy codec via ffmpeg. Returns same sr/shape
    (length may change by a few frames at codec boundaries — trimmed/padded)."""
    if _FFMPEG is None:
        logger.warning("ffmpeg not found; codec round-trip skipped")
        return wave
    args, ext = _CODEC_ARGS[codec]
    n = wave.shape[1]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        dst = Path(td) / f"out{ext}"
        # opus only supports 48 kHz families — ffmpeg resamples internally.
        sf.write(src, wave.T.numpy(), sr)
        cmd = [_FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
               *args, "-b:a", f"{bitrate_kbps}k", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        cmd = [_FFMPEG, "-y", "-loglevel", "error", "-i", str(dst),
               "-ar", str(sr), "-f", "wav", str(src)]
        subprocess.run(cmd, check=True, capture_output=True)
        out, _ = sf.read(src, dtype="float32", always_2d=True)
    out = torch.from_numpy(out.T)
    if out.shape[1] >= n:
        out = out[:, :n]
    else:
        out = torch.nn.functional.pad(out, (0, n - out.shape[1]))
    return out


def measure_lufs(wave: torch.Tensor, sr: int) -> float:
    """Integrated loudness (ITU-R BS.1770) via pyloudnorm; RMS-dB fallback."""
    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(sr)
        return float(meter.integrated_loudness(wave.T.numpy()))
    except Exception:
        rms = torch.sqrt(torch.mean(wave ** 2) + 1e-12)
        return float(20 * torch.log10(rms + 1e-12))


def normalize_lufs(wave: torch.Tensor, sr: int, target_lufs: float) -> torch.Tensor:
    current = measure_lufs(wave, sr)
    if not np.isfinite(current):
        return wave
    gain_db = target_lufs - current
    out = wave * (10 ** (gain_db / 20))
    peak = out.abs().max()
    if peak > 0.999:                       # guard clipping
        out = out * (0.999 / peak)
    return out


def _pitch_shift(wave: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    import librosa
    out = np.stack([librosa.effects.pitch_shift(ch, sr=sr, n_steps=semitones)
                    for ch in wave.numpy()])
    return torch.from_numpy(out.astype(np.float32))


def _time_stretch(wave: torch.Tensor, rate: float) -> torch.Tensor:
    import librosa
    out = [librosa.effects.time_stretch(ch, rate=rate) for ch in wave.numpy()]
    return torch.from_numpy(np.stack(out).astype(np.float32))


def _eq_tilt(wave: torch.Tensor, sr: int, rng: random.Random) -> torch.Tensor:
    """Random gentle EQ: 2-4 peaking biquads, +/-4 dB."""
    out = wave
    for _ in range(rng.randint(2, 4)):
        fc = rng.uniform(80, min(16000, sr / 2 * 0.9))
        gain = rng.uniform(-4, 4)
        q = rng.uniform(0.5, 2.0)
        out = torchaudio.functional.equalizer_biquad(out, sr, fc, gain, q)
    return out


def _noise_floor(wave: torch.Tensor, rng: random.Random) -> torch.Tensor:
    snr_db = rng.uniform(25, 45)
    sig_pow = torch.mean(wave ** 2) + 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    return wave + torch.randn_like(wave) * torch.sqrt(noise_pow)


class DeliveryChainSimulator:
    def __init__(self, sample_rate: int = config.SAMPLE_RATE,
                 base_seed: int = config.SEED):
        self.sr = sample_rate
        self.base_seed = base_seed

    # ------------------------------------------------------------------
    def random_chain(self, wave: torch.Tensor, sr: int,
                     item_key: str = "", excerpt: bool = True) -> torch.Tensor:
        """Full randomized chain. wave: (channels, time) float32."""
        rng = random.Random(f"{self.base_seed}|{item_key}")
        wave = _resample(wave, sr, self.sr)

        if rng.random() >= config.SIM_CLEAN_PROB:
            if rng.random() < 0.25:
                wave = _pitch_shift(wave, self.sr, rng.uniform(-2.0, 2.0))
            elif rng.random() < 0.25:
                wave = _time_stretch(wave, rng.uniform(0.9, 1.1))
            if rng.random() < 0.5:
                wave = _eq_tilt(wave, self.sr, rng)
            if rng.random() < 0.3:
                wave = _noise_floor(wave, rng)
            if rng.random() < 0.4:
                inter = rng.choice([r for r in config.SIM_RESAMPLE_RATES
                                    if r != self.sr])
                wave = _resample(_resample(wave, self.sr, inter), inter, self.sr)
            if rng.random() < 0.5:
                codec, (lo, hi) = rng.choice(config.SIM_CODECS)
                wave = codec_roundtrip(wave, self.sr, codec, rng.randint(lo, hi))

        wave = normalize_lufs(wave, self.sr,
                              rng.uniform(*config.SIM_LUFS_RANGE))
        # Mono-fold with p=0.15; otherwise PRESERVE stereo (ACE-Step decodes
        # channels sequentially — stereo incoherence is signal, plan §6.C).
        if wave.shape[0] > 1 and rng.random() < config.SIM_MONO_FOLD_PROB:
            wave = wave.mean(dim=0, keepdim=True)
        if excerpt:
            wave = self.random_excerpt(wave, rng)
        return wave

    def random_excerpt(self, wave: torch.Tensor,
                       rng: random.Random) -> torch.Tensor:
        total_s = wave.shape[1] / self.sr
        if total_s <= config.SIM_EXCERPT_MIN_S:
            return wave
        target_s = rng.uniform(config.SIM_EXCERPT_MIN_S, total_s)
        n = int(target_s * self.sr)
        start = rng.randint(0, wave.shape[1] - n)
        return wave[:, start:start + n]

    # ------------------------------------------------------------------
    def apply_condition(self, wave: torch.Tensor, sr: int,
                        condition: dict, excerpt_s: float | None = None
                        ) -> torch.Tensor:
        """Deterministic single-condition application for harness strata.
        condition keys (see config.HARNESS_CONDITIONS): codec=(name,kbps),
        resample=rate, pitch_semitones=float."""
        wave = _resample(wave, sr, self.sr)
        if "pitch_semitones" in condition:
            wave = _pitch_shift(wave, self.sr, condition["pitch_semitones"])
        if "resample" in condition:
            r = condition["resample"]
            wave = _resample(_resample(wave, self.sr, r), r, self.sr)
        if "codec" in condition:
            codec, kbps = condition["codec"]
            wave = codec_roundtrip(wave, self.sr, codec, kbps)
        if excerpt_s is not None:
            n = int(excerpt_s * self.sr)
            if wave.shape[1] > n:                 # center crop (deterministic)
                start = (wave.shape[1] - n) // 2
                wave = wave[:, start:start + n]
        return wave
