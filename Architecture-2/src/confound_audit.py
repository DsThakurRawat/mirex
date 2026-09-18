"""Confound audit gate (plan §4.5).

Trains deliberately dumb probes to predict the CLASS LABEL from non-content
features of *post-simulator* audio. If any probe reaches AUROC >= 0.60 on a
held-out split, a delivery-pipeline shortcut survives the simulator and MUST
be fixed before model training (P2 gate).

Fixes over the v1 implementation: proper train/test split (the old version
scored the probe on its own training data), full feature set (11 features vs
4), and both a linear and a boosted probe (a shortcut can be non-linear).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import config
from features import CONFOUND_FEATURE_NAMES, confound_features

logger = logging.getLogger(__name__)


def run_audit(waves: list[torch.Tensor], labels: list[int],
              group_ids: list[str] | None = None,
              sr: int = config.SAMPLE_RATE,
              report_path: Path | None = None) -> dict:
    """waves: post-simulator audio tensors (C, T). labels: 0 real / 1 AI.
    group_ids: source-dataset per item, so the split never puts one source on
    both sides (that would leak the very confound we're probing for).
    Returns report dict with 'gate_passed'."""
    X = np.array([[confound_features(w, sr)[k] for k in CONFOUND_FEATURE_NAMES]
                  for w in waves])
    y = np.asarray(labels)
    groups = np.asarray(group_ids if group_ids is not None
                        else np.arange(len(y)).astype(str))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3,
                                 random_state=config.SEED)
    try:
        train_idx, test_idx = next(splitter.split(X, y, groups))
    except ValueError:            # single group — fall back to random split
        rng = np.random.RandomState(config.SEED)
        perm = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        train_idx, test_idx = perm[:cut], perm[cut:]
    if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
        raise ValueError("Audit split has a single class; supply more data")

    scaler = StandardScaler().fit(X[train_idx])
    results = {}
    probes = {
        "linear": LogisticRegression(max_iter=2000),
        "boosted": HistGradientBoostingClassifier(random_state=config.SEED),
    }
    for name, probe in probes.items():
        Xt = scaler.transform(X) if name == "linear" else X
        probe.fit(Xt[train_idx], y[train_idx])
        auc = roc_auc_score(y[test_idx],
                            probe.predict_proba(Xt[test_idx])[:, 1])
        results[name] = float(auc)
        logger.info("Confound probe [%s] held-out AUROC = %.4f", name, auc)

    # Per-feature single-variable AUROCs: pinpoints WHICH confound leaks.
    per_feature = {}
    for j, fname in enumerate(CONFOUND_FEATURE_NAMES):
        try:
            auc = roc_auc_score(y[test_idx], X[test_idx, j])
            per_feature[fname] = float(max(auc, 1 - auc))
        except ValueError:
            per_feature[fname] = float("nan")

    worst = max(results.values())
    report = {
        "n_items": int(len(y)),
        "probe_auroc": results,
        "per_feature_auroc": per_feature,
        "worst_auroc": float(worst),
        "gate_threshold": config.CONFOUND_GATE_AUROC,
        "gate_passed": bool(worst < config.CONFOUND_GATE_AUROC),
    }
    if report["gate_passed"]:
        logger.info("CONFOUND GATE PASSED (worst AUROC %.3f < %.2f)",
                    worst, config.CONFOUND_GATE_AUROC)
    else:
        offenders = sorted(per_feature.items(), key=lambda kv: -kv[1])[:3]
        logger.error("CONFOUND GATE FAILED (worst AUROC %.3f). Top leaking "
                     "features: %s — fix the simulator/data mix before "
                     "training.", worst, offenders)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
    return report
