"""P4 baselines — the floor every branch must beat on the harness.

1. FakeprintLR: 10-ish-parameter logistic regression on comb-peak energies
   (Afchar et al. ISMIR 2025 — ~100% on Suno v3.5/Udio in-domain, brittle to
   resampling/pitch; its collapse pattern also validates the harness).
2. FakeprintGBDT (models.branch_c_physics) on the fuller feature set.
3. Branch A itself trained on SONICS-only replicates the MIREX-2025
   wav2vec2+AASIST baseline setting (run via train.py, not here).
"""
from __future__ import annotations

import numpy as np
import torch

import config
from features import comb_peak_energies, estimate_cutoff_hz


class FakeprintLR:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000)

    @staticmethod
    def extract(wave: torch.Tensor, sr: int) -> np.ndarray:
        return np.concatenate([comb_peak_energies(wave, sr),
                               [estimate_cutoff_hz(wave, sr) / (sr / 2)]])

    def fit(self, waves: list[torch.Tensor], labels: list[int],
            sr: int = config.SAMPLE_RATE):
        X = self.scaler.fit_transform(
            np.stack([self.extract(w, sr) for w in waves]))
        self.model.fit(X, np.asarray(labels))
        return self

    def predict_proba(self, waves: list[torch.Tensor],
                      sr: int = config.SAMPLE_RATE) -> np.ndarray:
        X = self.scaler.transform(
            np.stack([self.extract(w, sr) for w in waves]))
        return self.model.predict_proba(X)[:, 1]
