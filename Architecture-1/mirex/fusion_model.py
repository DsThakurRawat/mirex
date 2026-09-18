"""
Fusion Layer -- concatenates Track A (narrative, 128-dim) and Track B
(style, 576-dim) embeddings into a 704-dim vector and trains a lightweight
classifier head (MLP) to output a calibrated P(AI-generated) score optimized
for AUROC.

Usage:
    python fusion_model.py --narrative narrative_vectors.parquet --style_ckpt checkpoints/simclr_best.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
from track_b_style import StyleEncoder
from dataset import load_audio_chunk, audio_to_logmel


class FusionMLP(nn.Module):
    """Small MLP fusion head: in_dim -> 256 -> 64 -> 1."""

    def __init__(self, in_dim: int = config.FUSED_DIM, hidden1: int = 256, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logits


class TemperatureScaler(nn.Module):
    """Post-hoc calibration: single learned temperature applied to logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def label_smoothing_bce(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = config.LABEL_SMOOTHING) -> torch.Tensor:
    targets_smoothed = targets * (1 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_smoothed)


def train_fusion_head(X: np.ndarray, y: np.ndarray, epochs: int = config.FUSION_EPOCHS, lr: float = config.FUSION_LR):
    can_stratify = False
    if len(y) >= 8:
        test_count = int(len(y) * 0.25)
        counts = np.bincount(y.astype(int)) if len(y) > 0 else []
        if len(counts) > 1 and test_count >= len(counts) and min(counts) >= 2:
            can_stratify = True

    if len(y) >= 4:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42,
            stratify=y if can_stratify else None
        )
    else:
        X_train, X_val, y_train, y_val = X, X, y, y

    model = FusionMLP(in_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = label_smoothing_bce(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_probs = torch.sigmoid(val_logits).numpy()
            try:
                auroc = roc_auc_score(y_val, val_probs)
                print(f"  epoch {epoch+1:3d}/{epochs} | loss {loss.item():.4f} | val AUROC {auroc:.4f}")
            except Exception:
                print(f"  epoch {epoch+1:3d}/{epochs} | loss {loss.item():.4f}")

    return model, (X_val_t, y_val)


def calibrate_temperature(model: FusionMLP, X_val_t: torch.Tensor, y_val: np.ndarray, epochs: int = 100, lr: float = 0.01):
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=epochs)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_val_t)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            scaler(logits), y_val_t
        )
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        pass
    return scaler


def extract_style_embeddings_from_paths(paths: list[str], simclr_ckpt: str) -> np.ndarray:
    """Extract 576-dim surface embeddings for each audio file using frozen encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = StyleEncoder(pretrained=False).to(device)
    ckpt = torch.load(simclr_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    embs = []
    total = len(paths)
    print(f"[Fusion] Extracting style embeddings for {total} tracks using {device}...")
    for i, p in enumerate(paths):
        try:
            y = load_audio_chunk(p)
            log_mel = audio_to_logmel(y)
            spec = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = encoder(spec).squeeze(0).cpu().numpy()
        except Exception:
            emb = np.zeros(config.SURFACE_DIM, dtype=np.float32)
        embs.append(emb)
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  [Style Extraction] {i + 1}/{total} tracks processed ({(i + 1) / total * 100:.0f}%)")
    return np.array(embs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Classification Head")
    parser.add_argument("--narrative", required=True, help="Parquet of narrative vectors")
    parser.add_argument("--style_ckpt", default="", help="Path to SimCLR checkpoint to extract style embeddings")
    parser.add_argument("--supcon_ckpt", default="", help="Path to SupCon projection head checkpoint")
    parser.add_argument("--style_embeddings", default="", help="Parquet/npy of precomputed Track B embeddings")
    parser.add_argument("--labels_col", default="label", help="Column name for 0/1 labels")
    parser.add_argument("--epochs", type=int, default=config.FUSION_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.FUSION_LR)
    args = parser.parse_args()

    narrative_df = pd.read_parquet(args.narrative)
    feature_cols = [c for c in narrative_df.columns if c not in ("path", args.labels_col)]
    narr_X = narrative_df[feature_cols].values.astype(np.float32)

    # Check for labels
    if args.labels_col in narrative_df.columns:
        y = narrative_df[args.labels_col].values.astype(np.float32)
    else:
        raise KeyError(f"Column '{args.labels_col}' not found in {args.narrative}. "
                       "Ensure narrative vectors were extracted with --db.")

    if args.style_ckpt and Path(args.style_ckpt).exists():
        style_X = extract_style_embeddings_from_paths(narrative_df["path"].tolist(), args.style_ckpt)
        X = np.concatenate([style_X, narr_X], axis=1)
        print(f"[Fusion] Fused representations: shape {X.shape} (576 surface + {narr_X.shape[1]} narrative)")
    elif args.style_embeddings and Path(args.style_embeddings).exists():
        style_df = pd.read_parquet(args.style_embeddings)
        merged = narrative_df.merge(style_df, on="path")
        style_cols = [c for c in style_df.columns if c != "path"]
        X = merged[feature_cols + style_cols].values.astype(np.float32)
        y = merged[args.labels_col].values.astype(np.float32)
    else:
        print("[Fusion] No style embeddings provided -- training on narrative features only.")
        X = narr_X

    # Optional: Apply SupCon projection head if provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.supcon_ckpt and Path(args.supcon_ckpt).exists():
        from train_supcon import SupConProjectionHead
        supcon_head = SupConProjectionHead(in_dim=X.shape[1]).to(device)
        sc = torch.load(args.supcon_ckpt, map_location=device, weights_only=True)
        supcon_head.load_state_dict(sc["proj_head_state"])
        supcon_head.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            X = supcon_head(X_t).cpu().numpy()
        print(f"[Fusion] Applied SupCon projection: transformed to shape {X.shape}")

    print(f"[Fusion] Training fusion head on {len(X)} samples with {X.shape[1]} features...")
    model, (X_val_t, y_val) = train_fusion_head(X, y, epochs=args.epochs, lr=args.lr)
    scaler = calibrate_temperature(model, X_val_t, y_val)
    print(f"[Fusion] Learned calibration temperature: {scaler.temperature.item():.3f}")

    # Save checkpoint
    config.ensure_dirs()
    ckpt_path = config.CHECKPOINT_DIR / "fusion_best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "scaler_state": scaler.state_dict(),
        "in_dim": X.shape[1],
        "feature_cols": feature_cols,
    }, ckpt_path)
    print(f"[Fusion] Saved fusion checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
