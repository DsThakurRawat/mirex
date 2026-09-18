"""Branch C (plan §6.C): the physics/artifact branch. Two detectors:

1. LogFreqShiftInvariantCNN — magnitude STFT remapped to a log-frequency
   axis, so pitch-shift/resampling become translations; convolutions + global
   max-pool over the log-frequency axis give transposition invariance
   (Deezer ISMIR-2026 design). Input: mono 44.1 kHz chunks (B, T).

2. FakeprintGBDT — gradient-boosted trees on the interpretable fakeprint
   features (comb-peak energies + cutoff + band statistics). Near-free at
   inference; also serves as the P4 baseline. sklearn, trained separately.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import config
from features import comb_peak_energies, estimate_cutoff_hz, fakeprint


def _log_freq_matrix(n_fft: int, sr: int, n_bins: int = 256,
                     f_min: float = 40.0) -> torch.Tensor:
    """Sparse interpolation matrix (n_bins, n_fft//2+1) mapping a linear STFT
    magnitude to a log-spaced frequency grid (triangular interpolation)."""
    lin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    log_centers = np.geomspace(f_min, sr / 2 * 0.999, n_bins)
    mat = np.zeros((n_bins, len(lin_freqs)), dtype=np.float32)
    edges = np.concatenate([[f_min * (f_min / log_centers[1])], log_centers,
                            [sr / 2]])
    for i in range(n_bins):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (lin_freqs - lo) / max(c - lo, 1e-9)
        down = (hi - lin_freqs) / max(hi - c, 1e-9)
        mat[i] = np.clip(np.minimum(up, down), 0, None)
        s = mat[i].sum()
        if s > 0:
            mat[i] /= s
    return torch.from_numpy(mat)


class LogFreqShiftInvariantCNN(nn.Module):
    def __init__(self, sr: int | None = None, n_fft: int | None = None,
                 hop: int | None = None, n_bins: int = 256):
        super().__init__()
        cfg = config.BRANCHES["c"]
        self.sr = sr or cfg["input_sr"]
        self.n_fft = n_fft or cfg["n_fft"]
        self.hop = hop or cfg["hop"]
        self.register_buffer("logmap", _log_freq_matrix(self.n_fft, self.sr,
                                                        n_bins))
        self.register_buffer("window", torch.hann_window(self.n_fft))
        ch = [1, 32, 64, 128, 128]
        blocks = []
        for cin, cout in zip(ch[:-1], ch[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1),
                       nn.BatchNorm2d(cout), nn.GELU(),
                       nn.MaxPool2d((2, 2))]
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                          window=self.window, return_complex=True).abs()
        logspec = torch.log1p(self.logmap @ spec)          # (B, bins, T)
        h = self.conv(logspec.unsqueeze(1))                # (B, C, bins', T')
        # Global MAX over the log-frequency axis = shift (transposition)
        # invariance; mean over time.
        h = h.max(dim=2).values.mean(dim=2)
        return self.head(h).squeeze(-1)


# Alias used by the branch registry.
BranchC = LogFreqShiftInvariantCNN


class FakeprintGBDT:
    """Interpretable baseline + ensemble member (plan §6.C.1, P4)."""

    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            random_state=config.SEED, max_iter=400)

    @staticmethod
    def extract(wave: torch.Tensor, sr: int) -> np.ndarray:
        fp = fakeprint(wave, sr)
        # Downsample the raw residual to a fixed 128-dim profile.
        idx = np.linspace(0, len(fp) - 1, 128).astype(int)
        return np.concatenate([
            fp[idx], comb_peak_energies(wave, sr),
            [estimate_cutoff_hz(wave, sr)]]).astype(np.float32)

    def fit(self, waves: list[torch.Tensor], labels: list[int],
            sr: int = 44100):
        X = np.stack([self.extract(w, sr) for w in waves])
        self.model.fit(X, np.asarray(labels))

    def predict_proba(self, waves: list[torch.Tensor],
                      sr: int = 44100) -> np.ndarray:
        X = np.stack([self.extract(w, sr) for w in waves])
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)
        return self
