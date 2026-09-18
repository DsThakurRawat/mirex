"""Branch A (plan §6.A): wav2vec2-XLS-R front-end + layer-weighted aggregation
+ attentive-stats pooling head. The SVDD-2024-winning family and the best
MIREX-2025 baseline. Input: mono 16 kHz chunks (B, T). Output: logits (B,)."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

import config
from models.heads import AttentiveStatsPooling, LayerWeightedSum, mlp_head

logger = logging.getLogger(__name__)


class BranchA(nn.Module):
    def __init__(self, pretrained: bool = True,
                 model_name: str | None = None, freeze_feature_encoder: bool = True):
        super().__init__()
        cfg = config.BRANCHES["a"]
        name = model_name or cfg["model_name"]
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        if pretrained:
            self.ssl = Wav2Vec2Model.from_pretrained(name)
        else:                       # tests / offline scaffolding
            small = Wav2Vec2Config(num_hidden_layers=2, hidden_size=64,
                                   num_attention_heads=2, intermediate_size=128,
                                   conv_dim=(64,) * 7)
            self.ssl = Wav2Vec2Model(small)
            logger.warning("BranchA built UNPRETRAINED (tiny config) — "
                           "smoke-test mode only")
        if freeze_feature_encoder:
            self.ssl.feature_extractor._freeze_parameters()
        dim = self.ssl.config.hidden_size
        n_layers = self.ssl.config.num_hidden_layers + 1
        self.layer_sum = LayerWeightedSum(n_layers)
        self.pool = AttentiveStatsPooling(dim)
        self.head = mlp_head(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ssl(x, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=1)   # (B, L, T, D)
        feat = self.layer_sum(hidden)
        return self.head(self.pool(feat)).squeeze(-1)
