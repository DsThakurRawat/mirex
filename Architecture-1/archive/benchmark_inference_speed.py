"""
Benchmark & Profiler for Maximum Inference Throughput
Profiles:
  1. Audio Loading (SoundFile vs Librosa)
  2. Mel-Spectrogram Generation (CPU Librosa vs GPU TorchAudio)
  3. Neural Network Forward Pass (Batch 1, 16, 64, 128, 256)
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torchaudio

_CUR_DIR = Path(__file__).resolve().parent.parent / "mirex"
if str(_CUR_DIR) not in sys.path:
    sys.path.insert(0, str(_CUR_DIR))

import config
from track_b_style import StyleEncoder
from train_supcon import SupConProjectionHead
from fusion_model import FusionMLP, TemperatureScaler


class UnifiedMusicScopeModel(nn.Module):
    def __init__(self, encoder, supcon, fusion, scaler):
        super().__init__()
        self.encoder = encoder
        self.supcon = supcon
        self.fusion = fusion
        self.scaler = scaler

    def forward(self, spec: torch.Tensor, narr: torch.Tensor) -> torch.Tensor:
        surf = self.encoder(spec)
        fused = torch.cat([surf, narr], dim=1)
        z = self.supcon(fused)
        logits = self.fusion(z)
        scaled = self.scaler(logits)
        return torch.sigmoid(scaled).squeeze(-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  MAXIMUM INFERENCE OPTIMIZATION BENCHMARK")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 70)

    ckpt_dir = config.CHECKPOINT_DIR
    simclr_ckpt = ckpt_dir / "simclr_200k_best.pt"
    supcon_ckpt = ckpt_dir / "supcon_200k_best.pt"
    fusion_ckpt = ckpt_dir / "fusion_200k_best.pt"

    encoder = StyleEncoder(pretrained=False).to(device)
    enc_ck = torch.load(simclr_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(enc_ck["encoder_state"])
    encoder.eval()

    supcon = SupConProjectionHead(in_dim=704, out_dim=128).to(device)
    sup_ck = torch.load(supcon_ckpt, map_location=device, weights_only=False)
    supcon.load_state_dict(sup_ck["proj_head_state"])
    supcon.eval()

    fus_ck = torch.load(fusion_ckpt, map_location=device, weights_only=False)
    fusion = FusionMLP(in_dim=128).to(device)
    fusion.load_state_dict(fus_ck["model_state"])
    fusion.eval()

    scaler = TemperatureScaler().to(device)
    if "scaler_state" in fus_ck:
        scaler.load_state_dict(fus_ck["scaler_state"])
    scaler.eval()

    unified_model = UnifiedMusicScopeModel(encoder, supcon, fusion, scaler).to(device)
    unified_model.eval()

    batch_sizes = [1, 16, 64, 128, 256, 512]
    dummy_spec = torch.zeros((16, 1, 128, 1292), device=device)
    dummy_narr = torch.zeros((16, 128), device=device)
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(10):
            _ = unified_model(dummy_spec, dummy_narr)
    torch.cuda.synchronize()

    for bsz in batch_sizes:
        spec_b = torch.randn((bsz, 1, 128, 1292), device=device)
        narr_b = torch.randn((bsz, 128), device=device)
        n_iters = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            for _ in range(n_iters):
                _ = unified_model(spec_b, narr_b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        lat_per_batch = (elapsed / n_iters) * 1000
        lat_per_track = lat_per_batch / bsz
        throughput = bsz / (elapsed / n_iters)
        print(f"  Batch {bsz:3d}: Batch Latency = {lat_per_batch:6.2f} ms | Per-Track = {lat_per_track*1000:6.1f} µs | Throughput = {throughput:7.1f} trk/s")


if __name__ == "__main__":
    main()
