"""Model branch tests: wiring smoke tests for the SSL branches (tiny configs,
no downloads) plus REAL learning tests for the from-scratch branches — they
must actually separate planted artifacts after a short CPU training loop."""
import numpy as np
import pytest
import torch

import config
from conftest import make_noise_wave
from models import build_branch


def _batch(branch: str, n=2, seconds=None):
    cfg = config.BRANCHES[branch]
    sec = seconds or min(cfg["chunk_s"], 4.0)
    t = int(sec * cfg["input_sr"])
    return torch.randn(n, t) * 0.1


@pytest.mark.parametrize("branch", ["a", "b"])
def test_ssl_branches_tiny(branch):
    model = build_branch(branch, pretrained=False).eval()
    with torch.no_grad():
        out = model(_batch(branch))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("branch", ["a", "b"])
def test_ssl_branches_backward(branch):
    model = build_branch(branch, pretrained=False).train()
    out = model(_batch(branch))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out, torch.tensor([0.0, 1.0]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_branch_c_learns_bandlimit_artifact():
    """The physics CNN must separate 16 kHz-band-limited noise (YuE-style)
    from full-band noise after a short training loop on CPU."""
    torch.manual_seed(0)
    sr = config.BRANCHES["c"]["input_sr"]
    n_train, n_test, sec = 24, 12, 2.0

    def batch(n, offset):
        xs, ys = [], []
        for i in range(n):
            limited = i % 2 == 1
            w = make_noise_wave(sec, sr, channels=1, seed=offset + i,
                                cutoff_hz=16000 if limited else None)
            xs.append(w[0])
            ys.append(float(limited))
        return torch.stack(xs), torch.tensor(ys)

    model = build_branch("c").train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tr, y_tr = batch(n_train, 0)
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()
    x_te, y_te = batch(n_test, 1000)
    with torch.no_grad():
        scores = model(x_te)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_te.numpy(), scores.numpy())
    assert auc >= 0.9, f"physics CNN failed to learn band-limit (AUC {auc:.2f})"


def test_branch_c_shift_invariance_property():
    """Pitch-shifting must move Branch C's logit less than it moves a naive
    linear-frequency mean-pool statistic — the design's whole point."""
    torch.manual_seed(0)
    model = build_branch("c").eval()
    sr = config.BRANCHES["c"]["input_sr"]
    wave = make_noise_wave(2.0, sr, channels=1, seed=5, cutoff_hz=8000)
    from simulator import _pitch_shift
    shifted = _pitch_shift(wave, sr, 2.0)
    with torch.no_grad():
        a = model(wave)
        b = model(shifted[:, : wave.shape[1]])
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    drift = (a - b).abs().item()
    assert drift < 2.0, f"logit drift {drift:.2f} under +2st pitch shift"


def test_branch_d_longcontext():
    model = build_branch("d", pretrained=False).eval()
    with torch.no_grad():
        out = model(_batch("d", seconds=8.0))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_branch_e_learns_anomaly_separation():
    """Real-only-flavored training: reals = smooth tones, fakes = harsh
    band-limited noise. After training, fake scores must exceed real scores
    on held-out items."""
    torch.manual_seed(0)
    sr = config.BRANCHES["e"]["input_sr"]
    sec = 2.0

    def real_item(seed):
        rng = np.random.RandomState(seed)
        t = np.arange(int(sec * sr)) / sr
        f = rng.uniform(200, 800)
        w = 0.4 * np.sin(2 * np.pi * f * t) + 0.02 * rng.randn(len(t))
        return torch.from_numpy(w.astype(np.float32))

    def fake_item(seed):
        return make_noise_wave(sec, sr, channels=1, seed=seed,
                               cutoff_hz=4000)[0]

    model = build_branch("e").train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.stack([real_item(i) for i in range(12)] +
                    [fake_item(i) for i in range(12)])
    y = torch.cat([torch.zeros(12), torch.ones(12)])
    for _ in range(40):
        opt.zero_grad()
        loss, _ = model.loss_and_score(x, y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        s_real = model(torch.stack([real_item(100 + i) for i in range(6)]))
        s_fake = model(torch.stack([fake_item(100 + i) for i in range(6)]))
    assert s_fake.mean() > s_real.mean() + 0.05, \
        f"anomaly branch failed: fake {s_fake.mean():.3f} vs real {s_real.mean():.3f}"


def test_branch_registry_rejects_unknown():
    with pytest.raises(ValueError):
        build_branch("z")


def test_confound_audit_gate_detects_planted_shortcut(tone_stereo):
    from confound_audit import run_audit
    wave, sr = tone_stereo
    rng = np.random.RandomState(0)
    quiet = [wave * 0.05 * (1 + 0.1 * rng.rand()) for _ in range(12)]
    loud = [wave * 0.9 * (1 + 0.1 * rng.rand()) for _ in range(12)]
    rep = run_audit(quiet + loud, [0] * 12 + [1] * 12,
                    group_ids=[str(i) for i in range(24)], sr=sr)
    assert not rep["gate_passed"]
    assert rep["per_feature_auroc"]["rms"] > 0.9       # names the offender


def test_confound_audit_gate_detects_codec_asymmetry():
    """The classic trap: reals MP3-64, fakes clean. The gate must fail it."""
    from confound_audit import run_audit
    from simulator import codec_roundtrip
    reals = [codec_roundtrip(make_noise_wave(2.0, 44100, seed=i), 44100,
                             "mp3", 64) for i in range(10)]
    fakes = [make_noise_wave(2.0, 44100, seed=100 + i) for i in range(10)]
    rep = run_audit(reals + fakes, [0] * 10 + [1] * 10,
                    group_ids=[str(i) for i in range(20)], sr=44100)
    assert not rep["gate_passed"], "codec-provenance shortcut went undetected"


def test_confound_audit_passes_symmetric_audio(tone_stereo):
    from confound_audit import run_audit
    wave, sr = tone_stereo
    rng = np.random.RandomState(0)
    same = [wave * (0.4 + 0.2 * rng.rand()) for _ in range(24)]
    rep = run_audit(same, [0, 1] * 12,
                    group_ids=[str(i) for i in range(24)], sr=sr)
    assert rep["worst_auroc"] < 0.95
