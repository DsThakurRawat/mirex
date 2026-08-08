"""Property-based physics tests: plant a known artifact, require the feature
extractors to find it — and to NOT find it in matched clean audio."""
import numpy as np
import torch

from conftest import make_noise_wave
from features import (average_log_spectrum, comb_peak_energies,
                      confound_features, estimate_cutoff_hz, fakeprint,
                      remove_lower_envelope)

SR = 44100


def _planted_comb_wave(f0: float, seconds: float = 5.0, seed: int = 0,
                       n_harmonics: int = 30) -> torch.Tensor:
    """Noise + stationary sinusoids at k*f0 — mimics a deconvolution comb."""
    rng = np.random.RandomState(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    wave = rng.randn(n) * 0.15
    for k in range(1, n_harmonics + 1):
        f = f0 * k
        if f >= SR / 2:
            break
        wave += 0.02 * np.sin(2 * np.pi * f * t + rng.rand() * 6.28)
    return torch.from_numpy(np.stack([wave, wave]).astype(np.float32))


def test_cutoff_detection_planted_16k():
    """YuE-style 16 kHz brick wall must be measured within 1 kHz."""
    wave = make_noise_wave(5.0, SR, cutoff_hz=16000)
    cut = estimate_cutoff_hz(wave, SR)
    assert 15000 < cut < 17000, f"cutoff {cut} not near 16 kHz"


def test_cutoff_detection_planted_12k_vs_fullband():
    limited = make_noise_wave(5.0, SR, cutoff_hz=12000)
    fullband = make_noise_wave(5.0, SR)
    assert estimate_cutoff_hz(limited, SR) < 13500
    assert estimate_cutoff_hz(fullband, SR) > 18000


def test_comb_peaks_fire_on_matching_fundamental_only():
    """Planting a 300 Hz comb must raise the 300 Hz comb energy well above
    the same statistic on matched clean noise, and above unrelated combs."""
    from features import COMB_FUNDAMENTALS_HZ
    idx300 = COMB_FUNDAMENTALS_HZ.index(300.0)
    planted = comb_peak_energies(_planted_comb_wave(300.0), SR)
    clean = comb_peak_energies(make_noise_wave(5.0, SR, seed=1), SR)
    assert planted[idx300] > clean[idx300] + 0.5, \
        f"planted comb not detected: {planted[idx300]} vs {clean[idx300]}"
    # The planted comb's energy must dominate most non-harmonic fundamentals.
    others = [planted[i] for i, f in enumerate(COMB_FUNDAMENTALS_HZ)
              if f not in (300.0, 150.0, 75.0, 600.0, 100.0, 50.0, 200.0)]
    assert planted[idx300] > np.median(others) + 0.3


def test_envelope_removal_kills_smooth_spectrum_keeps_peaks():
    smooth = np.linspace(5, 1, 500)                     # smooth downward slope
    residual_smooth = remove_lower_envelope(smooth, kernel=51)
    assert np.abs(residual_smooth[60:-60]).max() < 0.6  # slope mostly removed
    peaky = smooth.copy()
    peaky[100] += 3.0
    residual_peaky = remove_lower_envelope(peaky, kernel=51)
    assert residual_peaky[100] > 2.5                    # peak survives


def test_fakeprint_gain_invariance_direction():
    """Fakeprint uses log magnitude — a 6 dB gain must shift it far less than
    the artifact contrast it is meant to capture."""
    wave = _planted_comb_wave(300.0)
    fp1 = fakeprint(wave, SR)
    fp2 = fakeprint(wave * 2.0, SR)
    clean_fp = fakeprint(make_noise_wave(5.0, SR, seed=2), SR)
    gain_dist = np.abs(fp1 - fp2).mean()
    artifact_dist = np.abs(fp1 - clean_fp).mean()
    assert gain_dist < artifact_dist


def test_fakeprint_deterministic():
    wave = make_noise_wave(4.0, SR, seed=3)
    assert np.array_equal(fakeprint(wave, SR), fakeprint(wave, SR))


def test_average_log_spectrum_shape_and_finite():
    wave = make_noise_wave(3.0, SR)
    spec = average_log_spectrum(wave, SR, n_fft=4096)
    assert spec.shape == (2049,)
    assert np.isfinite(spec).all()


def test_confound_features_complete_and_sane():
    from features import CONFOUND_FEATURE_NAMES
    wave = make_noise_wave(4.0, SR, channels=2)
    feats = confound_features(wave, SR)
    assert set(feats) == set(CONFOUND_FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())
    assert abs(feats["duration_s"] - 4.0) < 0.01
    assert 0 <= feats["silence_ratio"] <= 1
    # Uncorrelated stereo noise -> substantial width; duplicated mono -> ~0.
    assert feats["stereo_width"] > 0.1
    mono_dup = wave[0:1].repeat(2, 1)
    assert confound_features(mono_dup, SR)["stereo_width"] < 1e-6


def test_confound_features_expose_codec_bandlimit():
    """A 64 kbps-style band-limit must surface in cutoff/rolloff features —
    this is exactly the shortcut the audit gate exists to catch."""
    clean = make_noise_wave(4.0, SR, seed=4)
    limited = make_noise_wave(4.0, SR, seed=4, cutoff_hz=11000)
    f_clean = confound_features(clean, SR)
    f_lim = confound_features(limited, SR)
    assert f_lim["cutoff_hz"] < f_clean["cutoff_hz"] - 4000
    assert f_lim["rolloff99_hz"] < f_clean["rolloff99_hz"]
    assert f_lim["hf16k_ratio"] < f_clean["hf16k_ratio"]
