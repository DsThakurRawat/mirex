"""P4 baselines — the floor every branch must beat on the harness.

1. FakeprintLR: 10-ish-parameter logistic regression on comb-peak energies
   (Afchar et al. ISMIR 2025 — ~100% on Suno v3.5/Udio in-domain, brittle to
   resampling/pitch; its collapse pattern also validates the harness).
2. FakeprintGBDT (models.branch_c_physics) on the fuller feature set.
3. Branch A itself trained on SONICS-only replicates the MIREX-2025
   wav2vec2+AASIST baseline setting (run via train.py, not here).
4. MirexProvidedBaseline: the organizers' own baseline — announced but not
   released as of 2026-08-26 (stub below).
"""
from __future__ import annotations

import numpy as np
import torch

import config
from features import comb_peak_energies, estimate_cutoff_hz


class MirexProvidedBaseline:
    """The organizers' baseline model — NOT RELEASED as of 2026-08-26.

    Task wiki: "We plan to provide a baseline model and checkpoint to help
    participants get started." The page names no architecture and gives no
    download link; it lists only intended components (a standard audio
    classifier, or an audio foundation model with a binary head, plus a
    reproducible inference pipeline and example scripts).

    This is the number the organizers will quote, so when it lands it becomes
    the floor every branch must clear *per stratum* on the harness, not just
    pooled — see plan §1.1 and the `mirex_baseline` stub in data_fetch.py.
    """

    NOT_RELEASED = (
        "MIREX 2026 baseline model not released as of 2026-08-26. See "
        "https://music-ir.org/mirex/wiki/2026:AI-Generated_Music_Detection")

    def __init__(self, checkpoint_path=None):     # noqa: ARG002
        raise NotImplementedError(self.NOT_RELEASED)


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
