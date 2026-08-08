"""Branch E (plan §6.E): real-only anomaly detector — mel-CNN encoder trained
with OC-Softmax. Zero-shot insurance: immune to generator-coverage gaps by
construction; fires on unseen generator versions. Can be trained real-only
or real-heavy. Input: mono 24 kHz chunks (B, T). Score: higher = more fake."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

import config
from losses import OCSoftmax


class BranchE(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = config.BRANCHES["e"]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg["input_sr"], n_fft=2048, hop_length=512,
            n_mels=96)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        ch = [1, 32, 64, 128, 256]
        blocks = []
        for cin, cout in zip(ch[:-1], ch[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1),
                       nn.BatchNorm2d(cout), nn.GELU(), nn.MaxPool2d(2)]
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(),
                                     nn.Linear(256, cfg["emb_dim"]))
        self.oc = OCSoftmax(cfg["emb_dim"], cfg["oc_m_real"],
                            cfg["oc_m_fake"], cfg["oc_alpha"])

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        m = self.db(self.mel(x)).unsqueeze(1)
        m = (m - m.mean(dim=(2, 3), keepdim=True)) / \
            (m.std(dim=(2, 3), keepdim=True) + 1e-6)
        return self.encoder(m)

    def loss_and_score(self, x: torch.Tensor, labels: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.oc(self.embed(x), labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: anomaly score, higher = more likely AI."""
        return self.oc.m_real - self.oc.similarity(self.embed(x))
