"""
Track B — Style Branch
Extracts a 576-dim "Surface Texture Embedding" from a log-Mel spectrogram
using a MobileNetV3-Small backbone.

Usage (as a library):
    from track_b_style import StyleEncoder
    encoder = StyleEncoder()
    embedding = encoder.encode(waveform, sr)
"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

SR = 22050
N_MELS = 128
CHUNK_SECONDS = 30


def waveform_to_logmel(y: np.ndarray, sr: int = SR, n_mels: int = N_MELS) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1] for CNN input
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
    return log_mel.astype(np.float32)


class StyleEncoder(nn.Module):
    """MobileNetV3-Small backbone producing a 576-dim embedding from a
    single-channel log-Mel spectrogram (repeated to 3 channels for the
    pretrained backbone's stem)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        # Drop the final classifier; keep feature extractor + pooling.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # MobileNetV3-Small's last conv layer outputs 576 channels.
        self.out_dim = 576

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T) -> repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.features(x)
        pooled = self.pool(feats).flatten(1)
        return pooled  # (B, 576)

    @torch.no_grad()
    def encode(self, y: np.ndarray, sr: int = SR) -> np.ndarray:
        log_mel = waveform_to_logmel(y, sr)
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,T)
        self.eval()
        emb = self.forward(tensor)
        return emb.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    # Smoke test with random noise
    dummy = np.random.randn(SR * CHUNK_SECONDS).astype(np.float32)
    encoder = StyleEncoder(pretrained=False)
    emb = encoder.encode(dummy, SR)
    print("Embedding shape:", emb.shape)  # expect (576,)
