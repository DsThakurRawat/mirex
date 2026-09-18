import numpy as np
import pytest

from aggregate import aggregate_chunks, tune_aggregation
from fusion import StackedFusion
from harness import eer, full_report, logo_folds, macro_auroc


def _fake_eval(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    fams = np.where(y == 1, rng.choice(["suno", "yue"], n), "human")
    strat = [f"{f}|clean_full" if yy else "human|clean_full"
             for f, yy in zip(fams, y)]
    scores = np.clip(y * 0.6 + rng.rand(n) * 0.5, 0, 1)
    return y, scores, strat, fams


# ------------------------------------------------------- metric exactness --
def test_auroc_hand_computed():
    """AUROC = P(fake > real) with hand-countable pairs.
    reals: .2 .4 | fakes: .3 .5 -> pairs (fake>real): (.3>.2)=1,(.3>.4)=0,
    (.5>.2)=1,(.5>.4)=1 -> 3/4 = .75"""
    y = [0, 0, 1, 1]
    s = [0.2, 0.4, 0.3, 0.5]
    strat = ["human|c", "human|c", "suno|c", "suno|c"]
    m = macro_auroc(y, s, strat)
    assert m["per_stratum"]["suno|c"] == pytest.approx(0.75)
    assert m["macro_auroc"] == pytest.approx(0.75)


def test_macro_is_unweighted_mean_over_strata():
    """3 suno pairs perfect (AUROC 1), 1-of-4 yue ordering (0.25 by hand):
    macro must be (1 + 0.25)/2 regardless of strata sizes."""
    y = [0, 0, 1, 1, 1, 1, 0, 0, 1]
    s = [0.1, 0.2, 0.9, 0.8, 0.95, 0.85, 0.5, 0.6, 0.55]
    strat = ["human|c"] * 2 + ["suno|c"] * 4 + ["human|c"] * 2 + ["yue|c"]
    m = macro_auroc(y, s, strat)
    # yue fake .55 vs reals [.1,.2,.5,.6]: beats 3 of 4 -> 0.75
    assert m["per_stratum"]["yue|c"] == pytest.approx(0.75)
    assert m["per_stratum"]["suno|c"] == pytest.approx(1.0)
    assert m["macro_auroc"] == pytest.approx((1.0 + 0.75) / 2)
    assert m["min_auroc"] == pytest.approx(0.75)


def test_condition_matched_reals_are_used():
    """Fakes in the mp3 condition must be ranked against mp3-condition reals,
    not clean reals. Construct scores where the pairing changes the answer."""
    y = [0, 0, 1]
    s = [0.9, 0.1, 0.5]                       # clean real scores HIGH (0.9)
    strat = ["human|clean_full", "human|mp3_64_full", "suno|mp3_64_full"]
    m = macro_auroc(y, s, strat)
    # vs mp3 real (0.1) only: AUROC 1.0. If clean real leaked in: 0.5.
    assert m["per_stratum"]["suno|mp3_64_full"] == pytest.approx(1.0)


def test_macro_auroc_penalizes_weak_family():
    y, scores, strat, fams = _fake_eval()
    scores2 = scores.copy()
    yue = np.array([s.startswith("yue") for s in strat])
    scores2[yue] = np.random.RandomState(1).rand(yue.sum())
    m1, m2 = macro_auroc(y, scores, strat), macro_auroc(y, scores2, strat)
    assert m2["min_auroc"] < m1["min_auroc"]
    assert m2["macro_auroc"] < m1["macro_auroc"]


def test_monotone_transform_invariance():
    """AUROC metrics must be invariant under strictly monotone score maps."""
    y, scores, strat, fams = _fake_eval()
    warped = 1 / (1 + np.exp(-(scores * 7 - 2)))
    m1, m2 = macro_auroc(y, scores, strat), macro_auroc(y, warped, strat)
    assert m1["macro_auroc"] == pytest.approx(m2["macro_auroc"], abs=1e-9)


def test_eer_hand_computed():
    assert eer(np.array([0, 0, 0, 1, 1, 1]),
               np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 0.0
    # One inversion in 3v3: EER = 1/3.
    v = eer(np.array([0, 0, 0, 1, 1, 1]),
            np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9]))
    assert v == pytest.approx(1 / 3, abs=0.01)


def test_full_report_keys_and_ranges():
    y, scores, strat, fams = _fake_eval()
    rep = full_report(y, scores, strat, fams)
    for k in ["macro_auroc", "min_auroc", "pooled_auroc", "auprc", "eer",
              "fpr_human", "fnr_by_family", "balanced_acc@median",
              "f1@median"]:
        assert k in rep
    assert 0 <= rep["eer"] <= 1
    assert set(rep["fnr_by_family"]) == {"suno", "yue"}
    assert all(0 <= v <= 1 for v in rep["fnr_by_family"].values())


def test_full_report_single_class_does_not_crash():
    rep = full_report([1, 1, 1], [0.5, 0.6, 0.7],
                      ["suno|c"] * 3, ["suno"] * 3)
    assert np.isnan(rep["pooled_auroc"])


def test_logo_folds_cover_all_families():
    folds = logo_folds()
    assert len(folds) == 6
    assert {f["holdout_family"] for f in folds} == {
        "suno", "udio", "mureka", "minimax", "yue", "ace-step"}


# ---------------------------------------------------------- aggregation ----
def test_aggregate_edge_cases_and_bounds():
    assert aggregate_chunks(np.array([])) == 0.5
    assert aggregate_chunks(np.array([0.9])) == pytest.approx(0.9)
    z = np.random.RandomState(0).rand(50)
    for lam in (0.0, 0.3, 1.0):
        v = aggregate_chunks(z, lam=lam, topk_frac=0.2)
        assert z.min() - 1e-9 <= v <= z.max() + 1e-9
    assert aggregate_chunks(z, lam=0.0) == pytest.approx(z.mean())


def test_aggregate_hand_computed_topk():
    z = np.array([0.0, 0.2, 0.4, 1.0])
    # k = ceil(.25*4) = 1 -> top1 = 1.0 ; mean = .4
    assert aggregate_chunks(z, lam=0.5, topk_frac=0.25) == \
        pytest.approx(0.5 * 0.4 + 0.5 * 1.0)


def test_aggregate_monotonicity():
    """Raising any chunk score must never lower the track score."""
    rng = np.random.RandomState(3)
    z = rng.rand(20)
    base = aggregate_chunks(z, lam=0.3, topk_frac=0.25)
    for i in range(20):
        z2 = z.copy()
        z2[i] = min(1.0, z2[i] + 0.2)
        assert aggregate_chunks(z2, lam=0.3, topk_frac=0.25) >= base - 1e-12


def test_tune_aggregation_prefers_topk_for_localized_artifacts():
    """Fakes with artifacts in 2 of 16 chunks: top-k must beat plain mean,
    so the tuner must return lam > 0."""
    rng = np.random.RandomState(0)
    tracks, labels = {}, {}
    for i in range(120):
        label = i % 2
        base = rng.rand(16) * 0.35 + 0.2          # shared background noise
        if label:
            hot = rng.choice(16, 2, replace=False)
            base[hot] = 0.9 + 0.1 * rng.rand(2)   # localized artifact bursts
        tracks[f"t{i}"] = base
        labels[f"t{i}"] = label
    lam, kf, score = tune_aggregation(tracks, labels)
    assert lam > 0.0
    assert score > 0.9


# --------------------------------------------------------------- fusion ----
def _make_oof(n=300, seed=0, weak_branch=None):
    rng = np.random.RandomState(seed)
    oof = []
    for i in range(n):
        label = i % 2
        scores = {}
        for b in "abcde":
            if b == weak_branch:
                scores[b] = rng.rand()                     # pure noise branch
            else:
                scores[b] = np.clip(label * 0.5 + rng.rand() * 0.6, 0, 1)
        oof.append({"label": label, "fold": "logo_suno", "scores": scores})
    return oof


def test_stacker_learns_and_calibrates():
    from sklearn.metrics import roc_auc_score
    oof = _make_oof()
    fus = StackedFusion()
    fus.fit_from_logo(oof)
    p = fus.predict(oof)
    assert (p >= 0).all() and (p <= 1).all()
    assert roc_auc_score([r["label"] for r in oof], p) > 0.85


def test_stacker_downweights_noise_branch():
    """A pure-noise branch must get |weight| well below informative ones."""
    fus = StackedFusion()
    fus.fit_from_logo(_make_oof(n=600, weak_branch="c"))
    w = np.abs(fus.stacker.coef_[0])
    c_idx = list("abcde").index("c")
    informative = [w[i] for i in range(5) if i != c_idx]
    assert w[c_idx] < np.mean(informative)


def test_fusion_handles_missing_branches_and_empty():
    fus = StackedFusion()
    fus.fit_from_logo(_make_oof())
    p = fus.predict([{"scores": {"a": 0.9}}, {"scores": {}}])
    assert p.shape == (2,) and np.isfinite(p).all()


def test_fusion_save_load_roundtrip(tmp_path):
    fus = StackedFusion()
    oof = _make_oof()
    fus.fit_from_logo(oof)
    p1 = fus.predict(oof)
    fus.save(tmp_path)
    p2 = StackedFusion.load(tmp_path).predict(oof)
    assert np.allclose(p1, p2)


def test_rank_average_invariant_to_monotone_branch_warp():
    """Rank averaging must give identical output if one branch's scores are
    warped monotonically — the property that makes it the safe fallback."""
    rng = np.random.RandomState(0)
    rows = [{"scores": {"a": float(rng.rand()), "b": float(rng.rand())}}
            for _ in range(50)]
    ra1 = StackedFusion.rank_average(rows)
    warped = [{"scores": {"a": r["scores"]["a"] ** 3,
                          "b": r["scores"]["b"]}} for r in rows]
    ra2 = StackedFusion.rank_average(warped)
    assert np.allclose(ra1, ra2)


def test_rank_average_ordering():
    ra = StackedFusion.rank_average([{"scores": {"a": 0.9, "b": 0.8}},
                                     {"scores": {"a": 0.1, "b": 0.2}}])
    assert ra[0] > ra[1]


def test_calibrator_is_monotone():
    fus = StackedFusion()
    fus.fit_from_logo(_make_oof())
    grid = np.linspace(0, 1, 101)
    out = fus.calibrator.predict(grid)
    assert (np.diff(out) >= -1e-12).all()
