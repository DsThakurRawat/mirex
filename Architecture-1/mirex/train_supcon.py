"""
Stage 3b — Supervised Contrastive Fusion (SupCon)

This is the first stage where labels are introduced. It takes the frozen
704-dim fused vector (576-dim surface + 128-dim structural) and trains a
supervised contrastive projection head that explicitly clusters Human vs. AI
representations, maximizing inter-class margin.

The SupCon head is then frozen, and Stage 3c (fusion_model.py) trains a
lightweight classifier on top of the SupCon-shaped embeddings.

Usage:
    python train_supcon.py --input_dir ../data/raw \\
                           --narrative narrative_vectors.parquet \\
                           --simclr_ckpt checkpoints/simclr_best.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from track_b_style import StyleEncoder
from track_a_narrative import (extract_narrative_vector, get_feature_columns,
                                feats_to_vector)
from dataset import load_labeled_entries_from_db, load_audio_chunk, audio_to_logmel


# ---------------------------------------------------------------------------
# SupCon Projection Head
# ---------------------------------------------------------------------------

class SupConProjectionHead(nn.Module):
    """Projects the 704-dim fused vector into a 128-dim normalized space
    where SupCon loss can cluster Human vs. AI representations."""

    def __init__(self, in_dim: int = config.FUSED_DIM,
                 hidden_dim: int = 256,
                 out_dim: int = config.SUPCON_PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (SupCon)
# ---------------------------------------------------------------------------

def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                temperature: float = config.SUPCON_TEMPERATURE) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., 2020).

    For each anchor, all same-class samples are positives and all
    different-class samples are negatives.

    Args:
        features: (B, D) L2-normalized embeddings
        labels: (B,) integer class labels
        temperature: scaling temperature
    """
    device = features.device
    batch_size = features.shape[0]

    # Similarity matrix
    sim = torch.matmul(features, features.T) / temperature  # (B, B)

    # Mask: same-class pairs (excluding self)
    labels = labels.contiguous().view(-1, 1)
    mask_pos = torch.eq(labels, labels.T).float().to(device)  # (B, B)
    mask_self = torch.eye(batch_size, device=device)
    mask_pos = mask_pos - mask_self  # exclude self from positives

    # For numerical stability
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()

    # Log-sum-exp over all negatives + positives (excluding self)
    exp_logits = torch.exp(logits) * (1 - mask_self)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

    # Mean of log-likelihood over positive pairs
    n_positives = mask_pos.sum(dim=1)
    # Avoid division by zero for classes with single sample
    n_positives = torch.clamp(n_positives, min=1.0)
    mean_log_prob = (mask_pos * log_prob).sum(dim=1) / n_positives

    loss = -mean_log_prob.mean()
    return loss


# ---------------------------------------------------------------------------
# Fused Dataset (precomputed narrative + on-the-fly surface embedding)
# ---------------------------------------------------------------------------

class FusedDataset(Dataset):
    """Yields (fused_704_vector, label) for SupCon training.

    Narrative vectors are precomputed; surface embeddings are extracted
    on-the-fly from the frozen SimCLR encoder.
    """

    def __init__(self, entries: list[tuple[Path, int]],
                 narrative_df: pd.DataFrame,
                 feature_cols: list[str]):
        # Index narrative features by path
        self.narrative_lookup = {}
        for _, row in narrative_df.iterrows():
            self.narrative_lookup[row["path"]] = \
                row[feature_cols].values.astype(np.float32)

        # Filter entries to only those with narrative features
        self.entries = []
        for path, label in entries:
            key = str(path)
            if key in self.narrative_lookup:
                self.entries.append((path, label, self.narrative_lookup[key]))

        self.feature_cols = feature_cols
        print(f"[FusedDataset] {len(self.entries)} entries "
              f"(matched from {len(entries)} audio + "
              f"{len(narrative_df)} narrative rows)")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        for _ in range(10):
            path, label, narr_vec = self.entries[idx]
            try:
                y = load_audio_chunk(path)
                log_mel = audio_to_logmel(y)
                spec_tensor = torch.from_numpy(log_mel).unsqueeze(0)  # (1, n_mels, T)
                narr_tensor = torch.from_numpy(narr_vec)
                return spec_tensor, narr_tensor, label
            except Exception:
                idx = random.randint(0, len(self.entries) - 1)

        blank = np.zeros((config.N_MELS, 1293), dtype=np.float32)
        return torch.from_numpy(blank).unsqueeze(0), torch.from_numpy(self.entries[idx][2]), self.entries[idx][1]


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_supcon(input_dir: str, narrative_path: str, simclr_ckpt: str,
                 epochs: int = config.SUPCON_EPOCHS,
                 batch_size: int = config.SUPCON_BATCH_SIZE,
                 lr: float = config.SUPCON_LR,
                 smoke: bool = False,
                 num_workers: int = 2):
    """Train the SupCon projection head on fused embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SupCon] Device: {device}")

    # 1. Load frozen SimCLR encoder
    encoder = StyleEncoder(pretrained=False).to(device)
    ckpt = torch.load(simclr_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[SupCon] Loaded SimCLR encoder from {simclr_ckpt} "
          f"(epoch {ckpt.get('epoch', '?')})")

    # 2. Load narrative features
    narrative_df = pd.read_parquet(narrative_path)
    feature_cols = [c for c in narrative_df.columns
                    if c not in ("path", "label")]
    print(f"[SupCon] Narrative features: {len(feature_cols)} dims")

    # 3. Load labeled entries
    entries = load_labeled_entries_from_db()
    n_ai = sum(1 for _, l in entries if l == 1)
    n_human = sum(1 for _, l in entries if l == 0)
    print(f"[SupCon] Labels: {n_human} human, {n_ai} AI")

    # 4. Build dataset
    dataset = FusedDataset(entries, narrative_df, feature_cols)
    if len(dataset) == 0:
        raise ValueError("[SupCon] No matching entries between database and narrative parquet!")
    effective_bs = min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=effective_bs, shuffle=True,
                        num_workers=0 if smoke else num_workers, pin_memory=False,
                        drop_last=len(dataset) >= effective_bs * 2)

    # 5. SupCon projection head
    proj_head = SupConProjectionHead(
        in_dim=config.SURFACE_DIM + len(feature_cols)
    ).to(device)
    optimizer = torch.optim.Adam(proj_head.parameters(), lr=lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    config.ensure_dirs()
    best_loss = float("inf")

    for epoch in range(epochs):
        proj_head.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (specs, narrs, labels) in enumerate(loader):
            specs = specs.to(device)
            narrs = narrs.to(device)
            labels = labels.to(device)

            # Frozen surface embedding
            with torch.no_grad():
                surface_emb = encoder(specs)  # (B, 576)

            # Fuse: concat surface + narrative
            fused = torch.cat([surface_emb, narrs], dim=1)  # (B, 704)

            # Project to SupCon space
            z = proj_head(fused)  # (B, 128)

            loss = supcon_loss(z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if smoke and batch_idx >= 2:
                break

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1:3d}/{epochs} | "
              f"loss {avg_loss:.4f} | {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = config.CHECKPOINT_DIR / "supcon_best.pt"
            torch.save({
                "epoch": epoch + 1,
                "proj_head_state": proj_head.state_dict(),
                "loss": best_loss,
            }, ckpt_path)
            print(f"  -> saved best to {ckpt_path}")

    final_path = config.CHECKPOINT_DIR / "supcon_final.pt"
    torch.save({
        "epoch": epochs,
        "proj_head_state": proj_head.state_dict(),
        "loss": avg_loss,
    }, final_path)
    print(f"[SupCon] Training complete. Final: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3b: Supervised Contrastive fusion training")
    parser.add_argument("--input_dir", default=str(config.RAW_DATA_DIR))
    parser.add_argument("--narrative", required=True,
                        help="Parquet of precomputed narrative vectors")
    parser.add_argument("--simclr_ckpt", required=True,
                        help="Path to SimCLR encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=config.SUPCON_EPOCHS)
    parser.add_argument("--batch_size", type=int,
                        default=config.SUPCON_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.SUPCON_LR)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    train_supcon(args.input_dir, args.narrative, args.simclr_ckpt,
                 args.epochs, args.batch_size, args.lr, args.smoke, args.num_workers)


if __name__ == "__main__":
    main()
