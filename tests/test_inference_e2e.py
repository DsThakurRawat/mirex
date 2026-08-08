"""End-to-end integration: real checkpoints -> inference.run -> submission
CSV. Exercises checkpoint loading, chunking, aggregation, fusion fallback,
corrupt-input fallback scores, and output-format compliance."""
import csv

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

import config


@pytest.fixture
def fake_checkpoints(tmp_path, monkeypatch):
    """Save real BranchModule checkpoints for the cheap branches (c, e) into
    a temp checkpoint dir wired into config."""
    from train import BranchModule
    ckpt_root = tmp_path / "checkpoints"
    monkeypatch.setattr(config, "CHECKPOINT_DIR", ckpt_root)
    monkeypatch.setattr(config, "FUSION_DIR", ckpt_root / "fusion")
    for branch in ("c", "e"):
        module = BranchModule(branch, pretrained=False)
        d = ckpt_root / branch / "full"
        d.mkdir(parents=True)
        ckpt = {"state_dict": module.state_dict(),
                "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": pl.__version__,
                "epoch": 0, "global_step": 0}
        torch.save(ckpt, d / "best-epoch=0-val_auroc=0.5000.ckpt")
    return ckpt_root


def _read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["track_id", "ai_generated_score"]
    return {r[0]: float(r[1]) for r in rows[1:]}


@pytest.mark.parametrize("mode", ["rank_average", "single:c"])
def test_e2e_scores_all_tracks(wav_dir, fake_checkpoints, mode):
    from inference import run
    out_csv = wav_dir / "out" / "scores.csv"
    run(wav_dir, out_csv, mode=mode, device="cpu", timeout_s=300)
    scores = _read_csv(out_csv)
    assert set(scores) == {f"t{i}" for i in range(6)}
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    assert all(np.isfinite(s) for s in scores.values())


def test_e2e_corrupt_file_gets_fallback_not_crash(wav_dir, fake_checkpoints):
    from inference import run
    (wav_dir / "corrupt.wav").write_bytes(b"RIFFgarbage_not_audio" * 10)
    out_csv = wav_dir / "out" / "scores.csv"
    run(wav_dir, out_csv, mode="single:c", device="cpu", timeout_s=300)
    scores = _read_csv(out_csv)
    assert scores["corrupt"] == config.FALLBACK_SCORE
    assert len(scores) == 7                      # all tracks present anyway


def test_e2e_ensemble_falls_back_to_rank_average(wav_dir, fake_checkpoints):
    """No fitted fusion on disk -> ensemble mode must degrade gracefully."""
    from inference import run
    out_csv = wav_dir / "out" / "scores.csv"
    run(wav_dir, out_csv, mode="ensemble", device="cpu", timeout_s=300)
    assert len(_read_csv(out_csv)) == 6


def test_e2e_deterministic_output(wav_dir, fake_checkpoints):
    from inference import run
    a_csv = wav_dir / "a.csv"
    b_csv = wav_dir / "b.csv"
    run(wav_dir, a_csv, mode="single:c", device="cpu", timeout_s=300)
    run(wav_dir, b_csv, mode="single:c", device="cpu", timeout_s=300)
    assert _read_csv(a_csv) == _read_csv(b_csv)


def test_e2e_no_checkpoints_fails_loudly(wav_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "empty")
    from inference import run
    with pytest.raises(SystemExit):
        run(wav_dir, wav_dir / "x.csv", mode="ensemble", device="cpu")


def test_e2e_empty_input_dir_fails_loudly(tmp_path, fake_checkpoints):
    from inference import run
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(SystemExit):
        run(empty, tmp_path / "x.csv", mode="single:c", device="cpu")


def test_tta_transforms_deterministic(tone_stereo):
    from tta import TTA_TRANSFORMS, apply_tta_transform
    wave, sr = tone_stereo
    for name in TTA_TRANSFORMS:
        a = apply_tta_transform(wave.clone(), sr, name)
        b = apply_tta_transform(wave.clone(), sr, name)
        assert torch.allclose(a, b), f"TTA transform {name} not deterministic"
    assert torch.allclose(
        apply_tta_transform(wave, sr, "identity"), wave)
