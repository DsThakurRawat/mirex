"""Branch B (plan §6.B): MERT music-SSL front-end + small transformer head.
Catches production/performance unnaturalness (timing quantization, missing
sidechain dynamics, uniform mastering). Input: mono 24 kHz chunks (B, T)."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

import config
from models.heads import AttentiveStatsPooling, LayerWeightedSum, mlp_head

logger = logging.getLogger(__name__)


class BranchB(nn.Module):
    def __init__(self, pretrained: bool = True, model_name: str | None = None):
        super().__init__()
        cfg = config.BRANCHES["b"]
        name = model_name or cfg["model_name"]
        if pretrained:
            from transformers import AutoModel
            self.ssl = AutoModel.from_pretrained(name, trust_remote_code=True)
            dim = self.ssl.config.hidden_size
            n_layers = self.ssl.config.num_hidden_layers + 1
        else:
            from transformers import Wav2Vec2Config, Wav2Vec2Model
            small = Wav2Vec2Config(num_hidden_layers=2, hidden_size=64,
                                   num_attention_heads=2, intermediate_size=128,
                                   conv_dim=(64,) * 7)
            self.ssl = Wav2Vec2Model(small)
            dim, n_layers = 64, 3
            logger.warning("BranchB built UNPRETRAINED (tiny config) — "
                           "smoke-test mode only")
        self.layer_sum = LayerWeightedSum(n_layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8 if dim % 8 == 0 else 4,
            dim_feedforward=2 * dim, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = AttentiveStatsPooling(dim)
        self.head = mlp_head(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ssl(x, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=1)
        feat = self.encoder(self.layer_sum(hidden))
        return self.head(self.pool(feat)).squeeze(-1)
