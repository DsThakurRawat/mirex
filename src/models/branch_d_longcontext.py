"""Branch D (plan §6.D): long-context structure model — ConvNeXt-Tiny over
120 s mel spectrograms. Catches song-level cues: Udio's ~32 s window seams,
loop-heavy structure, outro fades, energy uniformity.
Input: mono 44.1 kHz chunks (B, T ~ 120 s). Output: logits (B,)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

import config


class BranchD(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        cfg = config.BRANCHES["d"]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg["input_sr"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop"], n_mels=cfg["n_mels"])
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        from torchvision.models import convnext_tiny
        weights = "IMAGENET1K_V1" if pretrained else None
        net = convnext_tiny(weights=weights)
        # Adapt the 3-channel stem to 1 channel by summing RGB kernels.
        stem = net.features[0][0]
        new_stem = nn.Conv2d(1, stem.out_channels, stem.kernel_size,
                             stem.stride, stem.padding)
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
            new_stem.bias.copy_(stem.bias)
        net.features[0][0] = new_stem
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, 1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.db(self.mel(x)).unsqueeze(1)          # (B, 1, mels, frames)
        m = (m - m.mean(dim=(2, 3), keepdim=True)) / \
            (m.std(dim=(2, 3), keepdim=True) + 1e-6)
        return self.net(m).squeeze(-1)
