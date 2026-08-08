"""Score fusion + calibration (plan §7).

Stacked logistic regression trained ONLY on LOGO out-of-fold predictions:
for fold g the stack rows are (branch scores from models trained WITHOUT
family g) on family-g fakes + reals. Fusion weights therefore encode "how much
to trust each branch on a generator it never saw" — the hidden-test question.
Ridge-regularized toward equal weights; rank-average fallback is monotone-safe
for AUROC (submission slot 2). Final isotonic calibration is monotone, so the
primary metric is untouched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)

BRANCH_ORDER = list("abcde")


class StackedFusion:
    def __init__(self, c_reg: float = 0.5):
        from sklearn.linear_model import LogisticRegression
        self.stacker = LogisticRegression(C=c_reg, max_iter=2000)  # L2 default
        self.mu = None
        self.sigma = None
        self.calibrator = None

    # -------------------------------------------------------------- fit ----
    def fit_from_logo(self, oof: list[dict]):
        """oof rows: {"scores": {branch: float}, "label": 0/1, "fold": str}.
        Missing branches are imputed with the row mean after z-norm."""
        X, y = self._matrix(oof, fit_norm=True)
        self.stacker.fit(X, np.array([r["label"] for r in oof]))
        logger.info("Fusion weights (a..e): %s  intercept %.3f",
                    np.round(self.stacker.coef_[0], 3),
                    self.stacker.intercept_[0])
        # Isotonic calibration on the same OOF predictions (monotone).
        from sklearn.isotonic import IsotonicRegression
        p = self.stacker.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0,
                                             out_of_bounds="clip")
        self.calibrator.fit(p, np.array([r["label"] for r in oof]))

    def _matrix(self, rows: list[dict], fit_norm: bool = False) -> tuple:
        raw = np.full((len(rows), len(BRANCH_ORDER)), np.nan)
        for i, r in enumerate(rows):
            for j, b in enumerate(BRANCH_ORDER):
                if b in r["scores"]:
                    raw[i, j] = r["scores"][b]
        if fit_norm:
            self.mu = np.nanmean(raw, axis=0)
            self.sigma = np.nanstd(raw, axis=0) + 1e-9
        z = (raw - self.mu) / self.sigma
        row_mean = np.nanmean(np.where(np.isnan(z), np.nan, z), axis=1,
                              keepdims=True)
        z = np.where(np.isnan(z), np.broadcast_to(row_mean, z.shape), z)
        z = np.nan_to_num(z)
        y = np.array([r.get("label", -1) for r in rows])
        return z, y

    # ---------------------------------------------------------- predict ----
    def predict(self, rows: list[dict], calibrated: bool = True) -> np.ndarray:
        X, _ = self._matrix(rows)
        p = self.stacker.predict_proba(X)[:, 1]
        if calibrated and self.calibrator is not None:
            p = self.calibrator.predict(p)
        return p

    @staticmethod
    def rank_average(rows: list[dict]) -> np.ndarray:
        """Fallback (submission slot 2): per-branch rank -> mean -> [0,1]."""
        from scipy.stats import rankdata
        n = len(rows)
        cols = []
        for b in BRANCH_ORDER:
            vals = np.array([r["scores"].get(b, np.nan) for r in rows])
            if np.isnan(vals).all():
                continue
            med = np.nanmedian(vals)
            vals = np.where(np.isnan(vals), med, vals)
            cols.append(rankdata(vals) / n)
        if not cols:
            return np.full(n, config.FALLBACK_SCORE)
        return np.mean(cols, axis=0)

    # ------------------------------------------------------------- io ------
    def save(self, dir_path: Path = config.FUSION_DIR):
        import joblib
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"stacker": self.stacker, "calibrator": self.calibrator},
                    dir_path / "fusion.joblib")
        (dir_path / "norm.json").write_text(json.dumps(
            {"mu": self.mu.tolist(), "sigma": self.sigma.tolist()}))

    @classmethod
    def load(cls, dir_path: Path = config.FUSION_DIR) -> "StackedFusion":
        import joblib
        obj = cls()
        blob = joblib.load(dir_path / "fusion.joblib")
        obj.stacker = blob["stacker"]
        obj.calibrator = blob["calibrator"]
        norm = json.loads((dir_path / "norm.json").read_text())
        obj.mu = np.array(norm["mu"])
        obj.sigma = np.array(norm["sigma"])
        return obj
