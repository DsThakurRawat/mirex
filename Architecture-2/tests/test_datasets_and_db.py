"""Dataset plumbing: metadata DB semantics, LOGO splits, quota sampler
statistics, and TrackChunkDataset on real temp WAV files."""
import numpy as np
import soundfile as sf
import torch

import config
from conftest import make_noise_wave
from datasets import TrackChunkDataset, load_audio, logo_split, quota_sampler
from metadata_db import MetadataDatabase


def _db(tmp_path):
    return MetadataDatabase(tmp_path / "meta.db")


def _rows(n_per, sources_ai, sources_real):
    rows = []
    for src, fam in sources_ai:
        for i in range(n_per):
            rows.append({"track_id": f"{src}:{i}", "source_dataset": src,
                         "generator_family": fam, "is_ai": 1,
                         "file_path": f"/x/{src}/{i}.wav"})
    for src in sources_real:
        for i in range(n_per):
            rows.append({"track_id": f"{src}:{i}", "source_dataset": src,
                         "generator_family": "human", "is_ai": 0,
                         "file_path": f"/x/{src}/{i}.wav"})
    return rows


# ------------------------------------------------------------ metadata DB --
def test_db_roundtrip_quarantine_and_splits(tmp_path):
    db = _db(tmp_path)
    db.insert_tracks(_rows(5, [("muse", "suno")], ["fma"]))
    assert len(db.trainable()) == 10
    db.mark_quarantined(["muse:0", "fma:1"])
    assert len(db.trainable()) == 8
    assert all(r["track_id"] not in ("muse:0", "fma:1")
               for r in db.trainable())
    db.assign_split(["muse:1", "fma:0"], "dev_frozen")
    tr = db.trainable()
    assert len(tr) == 6
    assert all(r["split"] != "dev_frozen" for r in tr)
    census = db.census(write=False)
    assert census["total_tracks"] == 10
    assert census["total_quarantined"] == 2
    assert census["ai_tracks"] + census["real_tracks"] == 8


def test_db_artist_quarantine(tmp_path):
    db = _db(tmp_path)
    rows = _rows(3, [], ["jamendo"])
    for i, r in enumerate(rows):
        r["artist_id"] = f"artist{i % 2}"
    db.insert_tracks(rows)
    db.quarantine_by_artists({"artist0"})
    remaining = db.trainable()
    assert all(r["artist_id"] != "artist0" for r in remaining)


def test_db_insert_idempotent(tmp_path):
    db = _db(tmp_path)
    rows = _rows(3, [("muse", "suno")], [])
    db.insert_tracks(rows)
    db.insert_tracks(rows)
    assert len(db.fetch()) == 3


# ------------------------------------------------------------- LOGO split --
def test_logo_split_holdout_isolation():
    rows = _rows(10, [("muse", "suno"), ("udio_ds", "udio"),
                      ("gen_yue", "yue")], ["fma"])
    train, val = logo_split(rows, "yue")
    assert all(r["generator_family"] != "yue" for r in train)
    val_ai = [r for r in val if r["is_ai"]]
    assert val_ai and all(r["generator_family"] == "yue" for r in val_ai)
    assert any(not r["is_ai"] for r in val)            # reals present in val
    assert {r["generator_family"] for r in train} >= {"suno", "udio", "human"}


def test_logo_split_none_returns_all():
    rows = _rows(4, [("muse", "suno")], ["fma"])
    train, val = logo_split(rows, None)
    assert train == rows and val == []


# ----------------------------------------------------------- quota sampler --
def test_quota_sampler_class_balance_and_source_cap():
    """900 AI from one dump vs 100 from another + 500 real: sampled batches
    must be ~50/50 by class and the mega-source capped near SOURCE_QUOTA_CAP
    within the AI class."""
    rows = (_rows(900, [("mega", "suno")], []) +
            _rows(100, [("small", "udio")], []) +
            _rows(500, [], ["fma"]))
    sampler = quota_sampler(rows, num_samples=6000)
    idx = list(iter(sampler))
    labels = np.array([rows[i]["is_ai"] for i in idx])
    assert 0.44 < labels.mean() < 0.56, f"class balance {labels.mean():.3f}"
    ai_sources = [rows[i]["source_dataset"] for i in idx if rows[i]["is_ai"]]
    mega_share = ai_sources.count("mega") / max(len(ai_sources), 1)
    # cap=0.35 vs raw share 0.9; sampling noise tolerance.
    assert mega_share < 0.55, f"mega source share {mega_share:.2f} not capped"
    small_share = ai_sources.count("small") / max(len(ai_sources), 1)
    assert small_share > 0.2, "minority source starved"


# ------------------------------------------------------ TrackChunkDataset --
def _write_tracks(tmp_path, n=4, seconds=3.0):
    rows = []
    for i in range(n):
        p = tmp_path / f"t{i}.wav"
        sf.write(p, make_noise_wave(seconds, 44100, seed=i).T.numpy(), 44100)
        rows.append({"track_id": f"loc:{i}", "source_dataset": "loc",
                     "generator_family": "suno" if i % 2 else "human",
                     "is_ai": i % 2, "file_path": str(p)})
    return rows


def test_chunk_dataset_shapes_labels_padding(tmp_path):
    rows = _write_tracks(tmp_path, n=4, seconds=3.0)
    ds = TrackChunkDataset(rows, branch="e", augment=False)   # 10 s chunks
    cfg = config.BRANCHES["e"]
    n_expected = int(cfg["chunk_s"] * cfg["input_sr"])
    for i in range(len(ds)):
        x, y, tid = ds[i]
        assert x.shape == (n_expected,)          # 3 s file -> padded to 10 s
        assert float(y) == rows[i]["is_ai"]
        assert tid == rows[i]["track_id"]
        assert torch.isfinite(x).all()


def test_chunk_dataset_epoch_changes_chunks(tmp_path):
    rows = _write_tracks(tmp_path, n=1, seconds=30.0)
    ds = TrackChunkDataset(rows, branch="e", augment=False)
    ds.set_epoch(0)
    x0, _, _ = ds[0]
    ds.set_epoch(1)
    x1, _, _ = ds[0]
    assert not torch.allclose(x0, x1)            # different crop per epoch
    ds.set_epoch(0)
    x0b, _, _ = ds[0]
    assert torch.allclose(x0, x0b)               # ...but deterministic per epoch


def test_load_audio_robust_formats(tmp_path):
    w = make_noise_wave(2.0, 48000, channels=1, seed=9)
    p = tmp_path / "x.wav"
    sf.write(p, w.T.numpy(), 48000)
    wave, sr = load_audio(p)
    assert sr == 48000 and wave.shape[0] == 1
    wave2, _ = load_audio(p, max_s=1.0)
    assert abs(wave2.shape[1] - 48000) <= 1
