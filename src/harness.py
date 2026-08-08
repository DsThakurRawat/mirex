"""Internal evaluation harness (plan §5) — our replica of the hidden eval and
the sole arbiter of every modeling decision.

Pieces:
  * freeze_dev_set()    — carve out & lock the frozen dev tracks (never train)
  * materialize_strata()— apply the deterministic condition x excerpt grid to
                          the frozen dev set, caching WAVs + a manifest
  * macro_auroc()       — stratum-wise AUROC -> macro / min (primary metric)
  * full_report()       — everything MIREX reports: macro/pooled AUROC, AUPRC,
                          EER, balanced acc, F1, FPR@human, FNR@family
  * logo_folds()        — leave-one-generator-out fold definitions (§5)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from metadata_db import MetadataDatabase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- metrics --
def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def macro_auroc(y_true, scores, stratum_ids) -> dict:
    """AUROC per stratum (fakes of that stratum vs ALL real items sharing the
    stratum's condition — real items carry stratum 'real:<condition>'; if none
    match, all real items are used), then macro/min over fake strata."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    strat = np.asarray(stratum_ids)
    real_mask = y == 0
    per: dict[str, float] = {}
    for st in sorted(set(strat[(y == 1)])):
        fake_mask = (strat == st) & (y == 1)
        cond = st.split("|", 1)[1] if "|" in st else ""
        real_same = real_mask & np.char.endswith(strat.astype(str), cond) \
            if cond else real_mask
        if real_same.sum() == 0:
            real_same = real_mask
        m = fake_mask | real_same
        per[st] = _auroc(y[m], s[m])
    vals = [v for v in per.values() if np.isfinite(v)]
    return {"per_stratum": per,
            "macro_auroc": float(np.mean(vals)) if vals else float("nan"),
            "min_auroc": float(np.min(vals)) if vals else float("nan")}


def eer(y_true, scores) -> float:
    from sklearn.metrics import roc_curve
    # drop_intermediate would discard the exact FPR==FNR crossing point.
    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def full_report(y_true, scores, stratum_ids, families) -> dict:
    """families: generator_family per item ('human' for real)."""
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 f1_score)
    y = np.asarray(y_true)
    s = np.asarray(scores)
    fam = np.asarray(families)
    rep = macro_auroc(y, s, stratum_ids)
    rep["pooled_auroc"] = _auroc(y, s)
    rep["auprc"] = float(average_precision_score(y, s)) \
        if len(np.unique(y)) > 1 else float("nan")
    rep["eer"] = eer(y, s) if len(np.unique(y)) > 1 else float("nan")
    thr = float(np.median(s))                      # EER-ish operating point
    pred = (s >= thr).astype(int)
    rep["balanced_acc@median"] = float(balanced_accuracy_score(y, pred))
    rep["f1@median"] = float(f1_score(y, pred))
    rep["fpr_human"] = float(np.mean(pred[y == 0] == 1)) if (y == 0).any() else float("nan")
    rep["fnr_by_family"] = {
        f: float(np.mean(pred[(y == 1) & (fam == f)] == 0))
        for f in sorted(set(fam[y == 1]))}
    return rep


# ------------------------------------------------------------- LOGO folds --
def logo_folds(families: list[str] | None = None) -> list[dict]:
    fams = families or config.TEST_FAMILIES
    return [{"fold": f"logo_{f}", "holdout_family": f} for f in fams]


# ------------------------------------------------- frozen dev set + strata --
def freeze_dev_set(per_family: int = 150, real_n: int = 1500,
                   db: MetadataDatabase | None = None) -> int:
    """Assign a stratified, locked dev split in the metadata DB (never
    trained on; all reported numbers come from it)."""
    db = db or MetadataDatabase()
    rng = np.random.RandomState(config.SEED)
    chosen: list[str] = []
    for fam in config.TEST_FAMILIES + config.FILLER_FAMILIES:
        rows = db.fetch("quarantined=0 AND generator_family=? AND is_ai=1",
                        (fam,))
        ids = [r["track_id"] for r in rows]
        rng.shuffle(ids)
        chosen += ids[:per_family]
    real_rows = db.fetch("quarantined=0 AND is_ai=0")
    real_ids = [r["track_id"] for r in real_rows]
    rng.shuffle(real_ids)
    chosen += real_ids[:real_n]
    db.assign_split(chosen, "dev_frozen")
    logger.info("Frozen dev set: %d tracks", len(chosen))
    return len(chosen)


def materialize_strata(out_dir: Path = config.HARNESS_CACHE_DIR,
                       max_per_cell: int = 40,
                       db: MetadataDatabase | None = None) -> Path:
    """Apply config.HARNESS_CONDITIONS x HARNESS_EXCERPT_SECONDS to the frozen
    dev set, writing WAVs + manifest.jsonl. Stratum id format:
    '<family>|<condition>_<excerpt>'. Real tracks get family 'human' (their
    stratum tail is matched by macro_auroc for condition-paired reals)."""
    from datasets import load_audio
    from simulator import DeliveryChainSimulator
    db = db or MetadataDatabase()
    sim = DeliveryChainSimulator()
    rows = db.fetch("split='dev_frozen'")
    if not rows:
        raise RuntimeError("No frozen dev set — run freeze_dev_set() first")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    rng = np.random.RandomState(config.SEED)
    with open(manifest, "w") as mf:
        for cond_name, cond in config.HARNESS_CONDITIONS.items():
            for exc in config.HARNESS_EXCERPT_SECONDS:
                tail = f"{cond_name}_{exc or 'full'}"
                cell_dir = out_dir / tail
                cell_dir.mkdir(exist_ok=True)
                by_fam: dict[str, list[dict]] = {}
                for r in rows:
                    fam = r["generator_family"] if r["is_ai"] else "human"
                    by_fam.setdefault(fam, []).append(r)
                for fam, fam_rows in by_fam.items():
                    idx = rng.permutation(len(fam_rows))[:max_per_cell]
                    for i in idx:
                        r = fam_rows[int(i)]
                        try:
                            wave, sr = load_audio(r["file_path"], max_s=300)
                            wave = sim.apply_condition(wave, sr, cond,
                                                       excerpt_s=exc)
                        except Exception as exc_e:   # noqa: F841
                            logger.warning("skip %s: %s", r["track_id"], exc_e)
                            continue
                        wav_path = cell_dir / f"{r['track_id'].replace('/', '_').replace(':', '_')}.wav"
                        sf.write(wav_path, wave.T.numpy(), sim.sr)
                        mf.write(json.dumps({
                            "path": str(wav_path),
                            "track_id": r["track_id"],
                            "is_ai": r["is_ai"],
                            "family": fam,
                            "stratum": f"{fam}|{tail}",
                        }) + "\n")
    logger.info("Strata materialized under %s", out_dir)
    return manifest


def evaluate_manifest(score_fn, manifest: Path | None = None) -> dict:
    """score_fn(path)->float in [0,1]. Runs the full report over the
    materialized strata."""
    manifest = manifest or (config.HARNESS_CACHE_DIR / "manifest.jsonl")
    items = [json.loads(l) for l in open(manifest)]
    scores = [score_fn(it["path"]) for it in items]
    return full_report([it["is_ai"] for it in items], scores,
                       [it["stratum"] for it in items],
                       [it["family"] for it in items])
