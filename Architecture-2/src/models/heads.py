"""Shared model components: layer-weighted SSL aggregation (SLS-lite) and
attentive statistical pooling (ASP) — the SVDD-recipe head building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerWeightedSum(nn.Module):
    """Softmax-weighted sum over an SSL model's hidden layers (sensitive-layer
    selection, lite): input (B, L, T, D) -> (B, T, D)."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        return torch.einsum("l,bltd->btd", w, hidden_states)


class AttentiveStatsPooling(nn.Module):
    """Attention-weighted mean + std over time: (B, T, D) -> (B, 2D)."""

    def __init__(self, dim: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(dim, bottleneck, 1), nn.Tanh(),
            nn.Conv1d(bottleneck, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)                      # (B, D, T)
        alpha = torch.softmax(self.attn(h), dim=2)
        mean = (alpha * h).sum(dim=2)
        var = (alpha * h ** 2).sum(dim=2) - mean ** 2
        return torch.cat([mean, var.clamp(min=1e-9).sqrt()], dim=1)


def mlp_head(in_dim: int, hidden: int = 256, out_dim: int = 1) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(hidden, out_dim))
