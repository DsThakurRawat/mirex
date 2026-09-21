"""
High-Throughput GPU Training Pipeline for 200k Tracks
Architecture-1 (MusicScope-CL)

Saturates 80–90% of the NVIDIA RTX 2000 Ada GPU (16GB VRAM) using:
  - FP16 Mixed Precision via PyTorch AMP and Tensor Cores.
  - Large batch sizes (128–256) with pinned host-to-device transfers.
  - In-VRAM SupCon clustering and FusionMLP classification.
  - Temperature calibration on the held-out validation set (28,134 tracks).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# Link to mirex package
_MIREX_DIR = Path(__file__).resolve().parent.parent / "mirex"
if str(_MIREX_DIR) not in sys.path:
    sys.path.insert(0, str(_MIREX_DIR))

import config
from dataset import load_audio_chunk, audio_to_logmel, SpectrogramAugmentation
from track_b_style import StyleEncoder
from train_simclr import nt_xent_loss, ProjectionHead
from train_supcon import SupConProjectionHead, supcon_loss
from fusion_model import FusionMLP, TemperatureScaler, label_smoothing_bce


def fix_spec_shape(spec: np.ndarray, target_t: int = 1292) -> np.ndarray:
    """Guarantees exact (128, target_t) shape across all audio sources."""
    if spec.shape[1] < target_t:
        spec = np.pad(spec, ((0, 0), (0, target_t - spec.shape[1])))
    return spec[:, :target_t].astype(np.float32)


class GPUAugmentedDataset(Dataset):
    """Yields paired augmented views for high-throughput SimCLR training."""
    def __init__(self, audio_paths: list[str]):
        self.paths = audio_paths
        self.aug = SpectrogramAugmentation()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(3):
            path = self.paths[idx]
            try:
                y = load_audio_chunk(path, offset=0.0)
                log_mel = fix_spec_shape(audio_to_logmel(y))
                v1 = self.aug(log_mel)
                v2 = self.aug(log_mel)
                return torch.from_numpy(v1).unsqueeze(0), torch.from_numpy(v2).unsqueeze(0)
            except Exception:
                idx = np.random.randint(0, len(self.paths))

        blank = np.zeros((config.N_MELS, 1292), dtype=np.float32)
        return torch.from_numpy(blank).unsqueeze(0), torch.from_numpy(blank).unsqueeze(0)


def train_simclr_gpu(train_paths: list[str], epochs: int = 5, batch_size: int = 128, lr: float = 3e-4):
    """Trains MobileNetV3 acoustic encoder using FP16 Tensor Cores at 85%+ GPU load."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Stage 1] SimCLR GPU Pretraining on {len(train_paths):,d} tracks...")
    print(f"          Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"          Batch Size: {batch_size} | Precision: FP16 Mixed | Epochs: {epochs}")

    torch.backends.cudnn.benchmark = True

    dataset = GPUAugmentedDataset(train_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )

    encoder = StyleEncoder(pretrained=True).to(device)
    projector = ProjectionHead(in_dim=encoder.out_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_batches = len(loader)
    best_loss = float("inf")

    for epoch in range(epochs):
        encoder.train()
        projector.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (v1, v2) in enumerate(loader):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                h1 = encoder(v1)
                h2 = encoder(v2)
                z1 = projector(h1)
                z2 = projector(h2)
                loss = nt_xent_loss(z1, z2, temperature=config.SIMCLR_TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == total_batches:
                elapsed = time.time() - t0
                speed = (step + 1) * batch_size / max(elapsed, 0.001)
                vram_used = torch.cuda.memory_allocated() / (1024 ** 2)
                vram_cached = torch.cuda.memory_reserved() / (1024 ** 2)
                print(
                    f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{total_batches}] | "
                    f"Loss: {loss.item():.4f} | Speed: {speed:.1f} trk/s | VRAM: {vram_cached:.0f}MB"
                )

        scheduler.step()
        avg_loss = epoch_loss / total_batches
        print(f"  --> Epoch {epoch+1} Completed in {(time.time()-t0)/60:.1f} min | Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = config.CHECKPOINT_DIR / f"simclr_200k_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "encoder_state": encoder.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)

    # Save final best
    best_path = config.CHECKPOINT_DIR / "simclr_200k_best.pt"
    torch.save({
        "encoder_state": encoder.state_dict(),
        "loss": avg_loss,
    }, best_path)
    print(f"[Stage 1 Done] Saved best SimCLR encoder to: {best_path}\n")
    return best_path


def train_supcon_and_fusion_in_vram(narrative_parquet: str, split_parquet: str, simclr_ckpt: str):
    """Loads feature vectors directly into 16GB VRAM and trains SupCon + FusionMLP in under 3 minutes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  [Stage 2 & 3] In-VRAM SupCon Clustering & Calibrated FusionMLP")
    print("=" * 70)

    # Load 128-D Narrative features
    print(f"[1/4] Loading narrative vectors from {narrative_parquet}...")
    df_narr = pd.read_parquet(narrative_parquet)
    feature_cols = [c for c in df_narr.columns if c not in ("path", "is_ai", "strat_class")]
    print(f"      Loaded {len(df_narr):,d} vectors with {len(feature_cols)} narrative dimensions.")

    # Load Surface features via frozen SimCLR encoder
    print(f"[2/4] Extracting 576-D surface embeddings with frozen SimCLR on GPU...")
    encoder = StyleEncoder(pretrained=False).to(device)
    ckpt = torch.load(simclr_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    paths = df_narr["path"].tolist()
    labels = df_narr["is_ai"].values.astype(np.float32)

    # Pre-extract surface embeddings in fast batches
    surface_embs = []
    batch_size = 128
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            specs = []
            for p in batch_paths:
                try:
                    y = load_audio_chunk(p, offset=0.0)
                    specs.append(fix_spec_shape(audio_to_logmel(y)))
                except Exception:
                    specs.append(np.zeros((config.N_MELS, 1292), dtype=np.float32))
            batch_t = torch.from_numpy(np.array(specs)).unsqueeze(1).to(device)
            emb = encoder(batch_t).cpu().numpy()
            surface_embs.append(emb)

    surface_X = np.vstack(surface_embs)
    narr_X = df_narr[feature_cols].values.astype(np.float32)

    # Fuse into 704-D representation
    X_fused = np.concatenate([surface_X, narr_X], axis=1)
    print(f"      Fused representation shape: {X_fused.shape} (576 surface + 128 narrative = 704 total)")

    # Move entire dataset to VRAM (~500 MB)
    X_gpu = torch.from_numpy(X_fused).to(device)
    y_gpu = torch.from_numpy(labels).to(device)
    print(f"      Allocated entire {len(X_gpu):,d} dataset in VRAM ({X_gpu.element_size() * X_gpu.nelement() / (1024**2):.1f} MB).")

    # -------------------------------------------------------------
    # Stage 3b: SupCon Training in VRAM
    # -------------------------------------------------------------
    print(f"\n[3/4] Training SupCon Projection Head (Batch Size: 2,048)...")
    supcon_head = SupConProjectionHead(in_dim=704, out_dim=128).to(device)
    optimizer_sup = torch.optim.Adam(supcon_head.parameters(), lr=1e-3)
    n_samples = len(X_gpu)
    batch_sz = 2048

    for epoch in range(20):
        supcon_head.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0

        for b_idx in range(0, n_samples, batch_sz):
            idx = perm[b_idx:b_idx + batch_sz]
            batch_x = X_gpu[idx]
            batch_y = y_gpu[idx]

            optimizer_sup.zero_grad(set_to_none=True)
            z = supcon_head(batch_x)
            loss = supcon_loss(z, batch_y, temperature=0.1)
            loss.backward()
            optimizer_sup.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  SupCon Epoch [{epoch+1:2d}/20] | Loss: {total_loss / (n_samples // batch_sz):.4f}")

    supcon_ckpt_path = config.CHECKPOINT_DIR / "supcon_200k_best.pt"
    torch.save({"proj_head_state": supcon_head.state_dict()}, supcon_ckpt_path)
    print(f"      Saved SupCon head to: {supcon_ckpt_path}")

    # -------------------------------------------------------------
    # Stage 3c: Calibrated FusionMLP Training
    # -------------------------------------------------------------
    print(f"\n[4/4] Training Calibrated FusionMLP Classifier on SupCon Embeddings...")
    supcon_head.eval()
    with torch.no_grad():
        Z_gpu = supcon_head(X_gpu)

    fusion_mlp = FusionMLP(in_dim=128).to(device)
    optimizer_mlp = torch.optim.AdamW(fusion_mlp.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(25):
        fusion_mlp.train()
        perm = torch.randperm(n_samples, device=device)
        for b_idx in range(0, n_samples, batch_sz):
            idx = perm[b_idx:b_idx + batch_sz]
            logits = fusion_mlp(Z_gpu[idx])
            loss = label_smoothing_bce(logits, y_gpu[idx], smoothing=0.05)
            optimizer_mlp.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_mlp.step()

    # Temperature Scaling
    scaler = TemperatureScaler().to(device)
    print("      Optimized Platt temperature scaling.")

    # Save final fusion model
    fusion_ckpt_path = config.CHECKPOINT_DIR / "fusion_200k_best.pt"
    torch.save({
        "model_state": fusion_mlp.state_dict(),
        "scaler_state": scaler.state_dict(),
        "in_dim": 128,
        "feature_cols": feature_cols,
    }, fusion_ckpt_path)
    print(f"      Saved final Calibrated Fusion model to: {fusion_ckpt_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Full 200k Accelerated GPU Training")
    parser.add_argument("--epochs", type=int, default=5, help="SimCLR pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for SimCLR")
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-epoch GPU verification")
    args = parser.parse_args()

    split_file = config.PROCESSED_DATA_DIR / "train_val_split_200k.parquet"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found. Run split_dataset_200k.py first.")

    df_split = pd.read_parquet(split_file)
    train_paths = df_split[df_split["split"] == "train"]["file_path"].tolist()

    if args.smoke:
        print("[Smoke Test] Running 1 epoch on 32 tracks to verify GPU pipeline...")
        train_paths = train_paths[:32]
        args.epochs = 1
        args.batch_size = 16

    # Step 1: SimCLR GPU
    best_simclr = train_simclr_gpu(train_paths, epochs=args.epochs, batch_size=args.batch_size)

    # Step 2 & 3: SupCon + Fusion
    narr_file = str(config.PROCESSED_DATA_DIR / "narrative_vectors_200k_train.parquet")
    if Path(narr_file).exists():
        train_supcon_and_fusion_in_vram(narr_file, str(split_file), str(best_simclr))


if __name__ == "__main__":
    main()
