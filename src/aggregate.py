"""Chunk -> track score aggregation (plan §8).

s_track = (1 - lam) * mean(z) + lam * mean(top-k(z))

Mean is robust on long tracks; the top-k term preserves sensitivity to
localized artifacts (Udio seams, shimmer bursts). For short excerpts n is
small and top-k ~= mean automatically, which behaves well across the
excerpt-length strata. lam and k_frac are tuned per branch on the harness.
"""
from __future__ import annotations

import numpy as np

import config


def aggregate_chunks(chunk_scores: np.ndarray,
                     lam: float = config.AGG_LAMBDA,
                     topk_frac: float = config.AGG_TOPK_FRAC) -> float:
    z = np.asarray(chunk_scores, dtype=np.float64)
    if z.size == 0:
        return float(config.FALLBACK_SCORE)
    k = max(1, int(np.ceil(topk_frac * z.size)))
    topk = np.sort(z)[-k:]
    return float((1 - lam) * z.mean() + lam * topk.mean())


def tune_aggregation(per_track_chunk_scores: dict[str, np.ndarray],
                     labels: dict[str, int],
                     strata: dict[str, str] | None = None,
                     lams=(0.0, 0.15, 0.3, 0.5, 0.7),
                     topk_fracs=(0.1, 0.25, 0.5)) -> tuple[float, float, float]:
    """Grid-search (lam, k_frac) maximizing macro-AUROC over strata (falls
    back to pooled AUROC when strata is None). Returns (lam, k_frac, score)."""
    from harness import macro_auroc
    best = (config.AGG_LAMBDA, config.AGG_TOPK_FRAC, -1.0)
    ids = list(per_track_chunk_scores)
    y = np.array([labels[t] for t in ids])
    s_map = [strata.get(t, "all") if strata else "all" for t in ids]
    for lam in lams:
        for kf in topk_fracs:
            scores = np.array([aggregate_chunks(per_track_chunk_scores[t],
                                                lam, kf) for t in ids])
            m = macro_auroc(y, scores, s_map)["macro_auroc"]
            if m > best[2]:
                best = (lam, kf, m)
    return best
