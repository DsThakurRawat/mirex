"""
Stage 1 — Unsupervised Contrastive Pretraining (SimCLR)

Pretrains the MobileNetV3-Small backbone on log-Mel spectrograms using
contrastive learning. The CNN learns robust audio texture representations
before any label is used. This protects the backbone from noisy labels
in the large scraped dataset.

Usage:
    python train_simclr.py --input_dir ../data/raw --epochs 20
    python train_simclr.py --input_dir ../data/raw --epochs 5 --smoke  # quick test
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from track_b_style import StyleEncoder
from dataset import build_simclr_loader


class ProjectionHead(nn.Module):
    """SimCLR projection head: 576 -> 256 -> 128."""

    def __init__(self, in_dim: int = config.SURFACE_DIM,
                 hidden_dim: int = config.SIMCLR_HIDDEN_DIM,
                 out_dim: int = config.SIMCLR_PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = config.SIMCLR_TEMPERATURE):
    """Normalized Temperature-scaled Cross Entropy loss (SimCLR)."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)

    # Positive pairs: (i, i+B) and (i+B, i)
    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss


def train(input_dir: str, epochs: int = config.SIMCLR_EPOCHS,
          batch_size: int = config.SIMCLR_BATCH_SIZE,
          lr: float = config.SIMCLR_LR, smoke: bool = False,
          num_workers: int = 2, max_batches: int = 0):
    """Full SimCLR training loop with the real DataLoader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SimCLR] Device: {device}")

    # Build DataLoader
    loader = build_simclr_loader(
        data_dir=Path(input_dir),
        batch_size=batch_size,
        num_workers=0 if smoke else num_workers,
    )
    total_batches = len(loader) if not max_batches else min(len(loader), max_batches)
    print(f"[SimCLR] {total_batches} batches/epoch, {epochs} epochs")

    # Model
    encoder = StyleEncoder(pretrained=True).to(device)
    projector = ProjectionHead(in_dim=encoder.out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    config.ensure_dirs()
    best_loss = float("inf")

    for epoch in range(epochs):
        encoder.train()
        projector.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (view1, view2) in enumerate(loader):
            view1 = view1.to(device)  # (B, 1, n_mels, T)
            view2 = view2.to(device)

            h1 = encoder(view1)  # (B, 576)
            h2 = encoder(view2)
            z1 = projector(h1)   # (B, 128)
            z2 = projector(h2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if smoke and batch_idx >= 2:
                break
            if max_batches and n_batches >= max_batches:
                break

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1:3d}/{epochs} | "
              f"loss {avg_loss:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = config.CHECKPOINT_DIR / "simclr_best.pt"
            torch.save({
                "epoch": epoch + 1,
                "encoder_state": encoder.state_dict(),
                "projector_state": projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
            }, ckpt_path)
            print(f"  -> saved best checkpoint to {ckpt_path}")

    # Save final
    final_path = config.CHECKPOINT_DIR / "simclr_final.pt"
    torch.save({
        "epoch": epochs,
        "encoder_state": encoder.state_dict(),
        "projector_state": projector.state_dict(),
        "loss": avg_loss,
    }, final_path)
    print(f"[SimCLR] Training complete. Final: {final_path}")
    return encoder


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: SimCLR unsupervised pretraining (Branch A)")
    parser.add_argument("--input_dir", required=True,
                        help="Directory of audio files")
    parser.add_argument("--epochs", type=int, default=config.SIMCLR_EPOCHS)
    parser.add_argument("--batch_size", type=int,
                        default=config.SIMCLR_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.SIMCLR_LR)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="Max batches per epoch (0 = all)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (2 batches/epoch)")
    args = parser.parse_args()

    train(args.input_dir, args.epochs, args.batch_size, args.lr,
          args.smoke, args.num_workers, args.max_batches)


if __name__ == "__main__":
    main()
