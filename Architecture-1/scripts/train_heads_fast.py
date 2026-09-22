"""
Fast In-VRAM SupCon & Calibrated FusionMLP Trainer
Trains Stage 3b & 3c on the cached 155,000 surface embeddings and narrative vectors in ~2 minutes.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

_SCRIPTS_DIR = Path(__file__).resolve().parent
_MIREX_DIR = _SCRIPTS_DIR.parent / "mirex"
for p in (str(_SCRIPTS_DIR), str(_MIREX_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from train_supcon import SupConProjectionHead, supcon_loss
from fusion_model import FusionMLP, TemperatureScaler, label_smoothing_bce


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  High-Performance In-VRAM SupCon Clustering & Calibrated FusionMLP")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 70)

    # 1. Load Surface Embeddings (155,000 x 576)
    surf_path = config.PROCESSED_DATA_DIR / "surface_embeddings_200k_train.npy"
    if not surf_path.exists():
        raise FileNotFoundError(f"Surface embeddings not found at {surf_path}")
    print(f"[1/4] Loading cached surface embeddings from {surf_path.name}...")
    surface_X = np.load(surf_path)
    surface_X = np.nan_to_num(surface_X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"      Surface matrix: {surface_X.shape} (dtype: {surface_X.dtype})")

    # 2. Load Narrative Features (155,000 x 128)
    narr_path = config.PROCESSED_DATA_DIR / "narrative_vectors_200k_train.parquet"
    print(f"[2/4] Loading narrative vectors from {narr_path.name}...")
    df_narr = pd.read_parquet(narr_path)
    feature_cols = [c for c in df_narr.columns if c not in ("path", "is_ai", "strat_class")]
    narr_X = df_narr[feature_cols].values.astype(np.float32)

    # Sanitize NaNs & extreme outliers
    nan_count = np.isnan(narr_X).sum()
    narr_X = np.nan_to_num(narr_X, nan=0.0, posinf=50.0, neginf=-50.0)
    narr_X = np.clip(narr_X, -50.0, 50.0)
    print(f"      Narrative matrix: {narr_X.shape} (cleaned {nan_count} NaNs)")

    # 3. Fuse into 704-D representation
    X_fused = np.concatenate([surface_X, narr_X], axis=1).astype(np.float32)
    labels = df_narr["is_ai"].values.astype(np.float32)
    print(f"      Fused representation shape: {X_fused.shape}")

    # Transfer entire dataset to GPU VRAM (~416 MB)
    X_gpu = torch.from_numpy(X_fused).to(device)
    y_gpu = torch.from_numpy(labels).to(device)
    vram_mb = X_gpu.element_size() * X_gpu.nelement() / (1024 ** 2)
    print(f"      Allocated entire {len(X_gpu):,d} dataset in VRAM ({vram_mb:.1f} MB).\n")

    # -------------------------------------------------------------
    # Stage 3b: SupCon Training in VRAM
    # -------------------------------------------------------------
    print("[3/4] Training SupCon Projection Head (Batch Size: 2,048)...")
    supcon_head = SupConProjectionHead(in_dim=704, out_dim=128).to(device)
    optimizer_sup = torch.optim.Adam(supcon_head.parameters(), lr=1e-3)
    n_samples = len(X_gpu)
    batch_sz = 2048
    epochs_sup = 20

    t0 = time.time()
    for epoch in range(epochs_sup):
        supcon_head.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0

        for b_idx in range(0, n_samples, batch_sz):
            idx = perm[b_idx:b_idx + batch_sz]
            batch_x = X_gpu[idx]
            batch_y = y_gpu[idx]

            optimizer_sup.zero_grad(set_to_none=True)
            z = supcon_head(batch_x)
            loss = supcon_loss(z, batch_y, temperature=config.SUPCON_TEMPERATURE)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(supcon_head.parameters(), max_norm=1.0)
                optimizer_sup.step()
                total_loss += loss.item()
                n_batches += 1

        avg_sup_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs_sup:
            print(f"  SupCon Epoch [{epoch+1:2d}/{epochs_sup}] | Loss: {avg_sup_loss:.4f} | Elapsed: {time.time()-t0:.1f}s")

    supcon_ckpt_path = config.CHECKPOINT_DIR / "supcon_200k_best.pt"
    torch.save({"proj_head_state": supcon_head.state_dict()}, supcon_ckpt_path)
    print(f"      >>> Saved verified SupCon head to: {supcon_ckpt_path}\n")

    # -------------------------------------------------------------
    # Stage 3c: Calibrated FusionMLP Training
    # -------------------------------------------------------------
    print("[4/4] Training Calibrated FusionMLP Classifier on SupCon Embeddings...")
    supcon_head.eval()
    with torch.no_grad():
        Z_gpu = supcon_head(X_gpu)

    # 85% train / 15% validation split in VRAM for calibration
    val_size = int(n_samples * 0.15)
    train_size = n_samples - val_size
    perm_split = torch.randperm(n_samples, device=device)
    train_idx = perm_split[:train_size]
    val_idx = perm_split[train_size:]

    Z_train, y_train = Z_gpu[train_idx], y_gpu[train_idx]
    Z_val, y_val = Z_gpu[val_idx], y_gpu[val_idx]

    fusion_mlp = FusionMLP(in_dim=128).to(device)
    optimizer_mlp = torch.optim.AdamW(fusion_mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs_mlp = 25

    t1 = time.time()
    for epoch in range(epochs_mlp):
        fusion_mlp.train()
        perm = torch.randperm(train_size, device=device)
        total_loss = 0.0
        n_b = 0

        for b_idx in range(0, train_size, batch_sz):
            idx = perm[b_idx:b_idx + batch_sz]
            logits = fusion_mlp(Z_train[idx])
            loss = label_smoothing_bce(logits, y_train[idx], smoothing=0.05)

            optimizer_mlp.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_mlp.step()
            total_loss += loss.item()
            n_b += 1

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs_mlp:
            fusion_mlp.eval()
            with torch.no_grad():
                val_logits = fusion_mlp(Z_val)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_y_np = y_val.cpu().numpy()
                val_auc = roc_auc_score(val_y_np, val_probs)
                val_acc = accuracy_score(val_y_np, (val_probs > 0.5).astype(int))
            print(f"  FusionMLP Epoch [{epoch+1:2d}/{epochs_mlp}] | Loss: {total_loss/n_b:.4f} | Val AUROC: {val_auc:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Fit Platt Temperature Scaler on Val Logits
    fusion_mlp.eval()
    with torch.no_grad():
        val_logits = fusion_mlp(Z_val)

    scaler = TemperatureScaler().to(device)
    scaler_opt = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def eval_loss():
        scaler_opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(scaler(val_logits), y_val)
        loss.backward()
        return loss

    scaler_opt.step(eval_loss)
    T = scaler.temperature.item()
    print(f"      Fitted Platt temperature scaling: T = {T:.4f}")

    # Save final models
    fusion_ckpt_path = config.CHECKPOINT_DIR / "fusion_200k_best.pt"
    torch.save({
        "model_state": fusion_mlp.state_dict(),
        "scaler_state": scaler.state_dict(),
        "in_dim": 128,
        "feature_cols": feature_cols,
    }, fusion_ckpt_path)
    print(f"      >>> Saved final Calibrated Fusion model to: {fusion_ckpt_path}")
    print("=" * 70)
    print("  [SUCCESS] All Stage 3 Models Successfully Trained and Saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
