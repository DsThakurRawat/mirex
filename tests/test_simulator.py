import numpy as np
import pytest
import torch

import config
from conftest import make_noise_wave
from simulator import (DeliveryChainSimulator, codec_roundtrip, measure_lufs,
                       normalize_lufs)


# ------------------------------------------------------------ determinism --
def test_random_chain_deterministic_given_key(tone_stereo):
    wave, sr = tone_stereo
    sim = DeliveryChainSimulator(base_seed=7)
    out1 = sim.random_chain(wave.clone(), sr, item_key="trk1")
    out2 = sim.random_chain(wave.clone(), sr, item_key="trk1")
    assert out1.shape == out2.shape
    assert torch.allclose(out1, out2, atol=1e-5)


def test_random_chain_varies_across_keys_and_seeds(tone_stereo):
    wave, sr = tone_stereo
    sim = DeliveryChainSimulator(base_seed=7)
    outs = [sim.random_chain(wave.clone(), sr, item_key=f"trk{i}")
            for i in range(4)]
    shapes = {o.shape for o in outs}
    assert len(shapes) > 1 or not all(
        torch.allclose(outs[0], o) for o in outs[1:] if o.shape == outs[0].shape)
    sim2 = DeliveryChainSimulator(base_seed=8)
    alt = sim2.random_chain(wave.clone(), sr, item_key="trk0")
    ref = outs[0]
    assert alt.shape != ref.shape or not torch.allclose(alt, ref)


# ------------------------------------------------------- class symmetry ----
def test_simulator_is_label_blind():
    """The API must not accept any label/class argument — symmetry across
    classes (plan §4.4) is enforced structurally."""
    import inspect
    params = inspect.signature(DeliveryChainSimulator.random_chain).parameters
    assert not {"label", "is_ai", "cls", "y"} & set(params)


def test_clean_passthrough_fraction_statistical():
    """~SIM_CLEAN_PROB of items must skip the heavy chain. Detect via exact
    length preservation + near-equality after loudness-normalizing input."""
    sim = DeliveryChainSimulator(base_seed=3)
    wave = make_noise_wave(3.0, 44100)
    clean = 0
    n = 60
    for i in range(n):
        out = sim.random_chain(wave.clone(), 44100, item_key=f"k{i}",
                               excerpt=False)
        if out.shape == wave.shape:
            ref = normalize_lufs(wave, 44100, measure_lufs(out, 44100))
            if torch.allclose(out, ref, atol=5e-3):
                clean += 1
    expected = config.SIM_CLEAN_PROB * n
    assert clean >= expected * 0.3, f"only {clean}/{n} clean passthroughs"
    assert clean <= expected * 2.5, f"{clean}/{n} clean — chain rarely fires"


def test_mono_fold_probability_statistical():
    sim = DeliveryChainSimulator(base_seed=11)
    wave = make_noise_wave(2.0, 44100, channels=2)
    n = 80
    mono = sum(
        1 for i in range(n)
        if sim.random_chain(wave.clone(), 44100, item_key=f"m{i}",
                            excerpt=False).shape[0] == 1)
    # p = 0.15 -> Binomial(80, .15): mean 12, sd ~3.2; allow 4 sd.
    assert 1 <= mono <= 26, f"mono folds {mono}/80 vs p=0.15"


# ------------------------------------------------------------- codecs ------
@pytest.mark.parametrize("codec,kbps", [("mp3", 128), ("aac", 96),
                                        ("vorbis", 96), ("opus", 64)])
def test_codec_roundtrips_all_formats(tone_stereo, codec, kbps):
    wave, sr = tone_stereo
    out = codec_roundtrip(wave, sr, codec, kbps)
    assert out.shape == wave.shape
    assert torch.isfinite(out).all()
    # Lossy: must change the signal, but keep it correlated with the input.
    assert not torch.allclose(out, wave, atol=1e-6)
    a = (wave - wave.mean()).flatten()
    b = (out - out.mean()).flatten()
    corr = float((a @ b) / (a.norm() * b.norm() + 1e-9))
    assert corr > 0.7, f"{codec} decorrelated the signal (corr={corr:.2f})"


def test_low_bitrate_mp3_bandlimits():
    from features import estimate_cutoff_hz
    wave = make_noise_wave(3.0, 44100)
    out = codec_roundtrip(wave, 44100, "mp3", 64)
    assert estimate_cutoff_hz(out, 44100) < estimate_cutoff_hz(wave, 44100)


# ------------------------------------------------------------- loudness ----
def test_lufs_normalization_hits_target(tone_stereo):
    wave, sr = tone_stereo
    for target in (-18.0, -10.0):
        out = normalize_lufs(wave.clone(), sr, target)
        measured = measure_lufs(out, sr)
        assert abs(measured - target) < 1.5, \
            f"target {target}, measured {measured:.2f}"


def test_lufs_normalization_no_clipping(tone_stereo):
    wave, sr = tone_stereo
    out = normalize_lufs(wave * 3.0, sr, -7.0)
    assert out.abs().max() <= 1.0


# --------------------------------------------------- deterministic API -----
def test_apply_condition_deterministic_and_excerpts(tone_stereo):
    wave, sr = tone_stereo
    sim = DeliveryChainSimulator()
    a = sim.apply_condition(wave.clone(), sr, {"codec": ("mp3", 64)},
                            excerpt_s=2.0)
    b = sim.apply_condition(wave.clone(), sr, {"codec": ("mp3", 64)},
                            excerpt_s=2.0)
    assert torch.allclose(a, b)
    assert abs(a.shape[1] - 2 * sim.sr) <= 1


def test_all_configured_harness_conditions_run(tone_stereo):
    """Every condition in config.HARNESS_CONDITIONS must be applicable —
    a typo'd key must fail HERE, not mid-materialization."""
    wave, sr = tone_stereo
    sim = DeliveryChainSimulator()
    for name, cond in config.HARNESS_CONDITIONS.items():
        out = sim.apply_condition(wave.clone(), sr, cond, excerpt_s=1.0)
        assert torch.isfinite(out).all(), f"condition {name} produced NaNs"
        assert out.shape[1] > 0


def test_resample_condition_bandlimits(tone_stereo):
    from features import estimate_cutoff_hz
    sim = DeliveryChainSimulator()
    wave = make_noise_wave(3.0, 44100)
    out = sim.apply_condition(wave, 44100, {"resample": 22050})
    assert out.shape[1] == wave.shape[1]
    assert estimate_cutoff_hz(out, 44100) < 12500     # Nyquist of 22.05k


def test_pitch_condition_moves_tone_frequency():
    sr = 44100
    t = np.arange(3 * sr) / sr
    tone = torch.from_numpy(
        (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))[None]
    sim = DeliveryChainSimulator()
    out = sim.apply_condition(tone, sr, {"pitch_semitones": 1.0})
    spec_in = np.abs(np.fft.rfft(tone[0].numpy()))
    spec_out = np.abs(np.fft.rfft(out[0].numpy()))
    f_in = np.argmax(spec_in) / len(tone[0]) * sr
    f_out = np.argmax(spec_out) / len(out[0]) * sr
    assert 1030 < f_out < 1090, f"pitch shift moved 1 kHz to {f_out:.0f} Hz " \
                                f"(expected ~1059; input peak {f_in:.0f})"


def test_excerpt_bounds_respected():
    sim = DeliveryChainSimulator(base_seed=5)
    wave = make_noise_wave(30.0, 44100)
    import random
    for i in range(10):
        out = sim.random_excerpt(wave, random.Random(i))
        dur = out.shape[1] / 44100
        assert config.SIM_EXCERPT_MIN_S - 0.01 <= dur <= 30.01
    short = make_noise_wave(5.0, 44100)
    assert sim.random_excerpt(short, random.Random(0)).shape == short.shape
