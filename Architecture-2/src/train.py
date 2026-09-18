"""Branch training with LOGO orchestration (plan §6, §7, P6).

CLI:
  python src/train.py --branch a --holdout suno            # one LOGO fold
  python src/train.py --branch a --holdout none            # full-data model
  python src/train.py --branch a --smoke                   # tiny CPU sanity run

Checkpoints land in config.CHECKPOINT_DIR/<branch>/<fold>/. Selection metric
is harness macro-AUROC on the fold's held-out family (never iid accuracy).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from datasets import TrackChunkDataset, logo_split, quota_sampler
from metadata_db import MetadataDatabase
from models import build_branch

logger = logging.getLogger(__name__)


class BranchModule(pl.LightningModule):
    def __init__(self, branch: str, pretrained: bool = True,
                 lr: float | None = None, head_lr: float | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.branch = branch
        self.model = build_branch(branch, pretrained=pretrained)
        cfg = config.BRANCHES[branch]
        self.lr = lr or cfg["lr"]
        self.head_lr = head_lr or cfg.get("head_lr", self.lr)
        self.bce = nn.BCEWithLogitsLoss()
        self._val_scores: list[tuple[float, float]] = []

    def _loss_scores(self, x, y):
        if self.branch == "e":
            loss, scores = self.model.loss_and_score(x, y)
        else:
            logits = self.model(x)
            loss = self.bce(logits, y)
            scores = torch.sigmoid(logits)
        return loss, scores

    def training_step(self, batch, _):
        x, y, _ids = batch
        loss, _ = self._loss_scores(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y, _ids = batch
        loss, scores = self._loss_scores(x, y)
        self.log("val_loss", loss, prog_bar=True)
        self._val_scores += list(zip(y.cpu().tolist(),
                                     scores.detach().cpu().tolist()))

    def on_validation_epoch_end(self):
        if not self._val_scores:
            return
        ys, ss = zip(*self._val_scores)
        self._val_scores.clear()
        if len(set(ys)) > 1:
            from sklearn.metrics import roc_auc_score
            self.log("val_auroc", float(roc_auc_score(ys, ss)), prog_bar=True)

    def configure_optimizers(self):
        # Two-group LR: tiny for the (pretrained) trunk, larger for heads.
        trunk, heads = [], []
        for n, p in self.model.named_parameters():
            (trunk if n.startswith("ssl") else heads).append(p)
        groups = [{"params": heads, "lr": self.head_lr}]
        if trunk:
            groups.append({"params": trunk, "lr": self.lr})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return {"optimizer": opt, "lr_scheduler": sched}


def make_loaders(branch: str, holdout: str | None, smoke: bool = False,
                 batch_size: int | None = None, workers: int = 4):
    db = MetadataDatabase()
    rows = db.trainable()
    if not rows:
        raise RuntimeError(
            "Metadata DB has no trainable tracks. Run data_fetch + quarantine "
            "first (or --smoke with a synthetic DB for CI).")
    train_rows, val_rows = logo_split(rows, holdout)
    if smoke:
        train_rows, val_rows = train_rows[:16], (val_rows or train_rows)[:8]
    bs = batch_size or config.BRANCHES[branch]["batch_size"]
    train_ds = TrackChunkDataset(train_rows, branch, augment=True)
    val_ds = TrackChunkDataset(val_rows, branch, augment=False) if val_rows \
        else None
    train_dl = DataLoader(train_ds, batch_size=bs,
                          sampler=quota_sampler(train_rows,
                                                num_samples=len(train_rows)),
                          num_workers=workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=workers) \
        if val_ds else None
    return train_dl, val_dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True, choices=list("abcde"))
    ap.add_argument("--holdout", default="none",
                    help="LOGO held-out family, or 'none' for the full model")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny unpretrained CPU run to validate the pipeline")
    args = ap.parse_args()

    holdout = None if args.holdout == "none" else args.holdout
    fold = f"logo_{holdout}" if holdout else "full"
    ckpt_dir = config.CHECKPOINT_DIR / args.branch / fold
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.SEED, workers=True)
    module = BranchModule(args.branch, pretrained=not args.smoke)
    train_dl, val_dl = make_loaders(args.branch, holdout, smoke=args.smoke,
                                    batch_size=args.batch_size,
                                    workers=args.workers)
    callbacks = []
    if val_dl is not None:
        callbacks.append(pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir, monitor="val_auroc", mode="max",
            filename="best-{epoch}-{val_auroc:.4f}", save_top_k=1))
    trainer = pl.Trainer(
        max_epochs=1 if args.smoke else args.epochs,
        accelerator="auto", devices=1, precision="16-mixed"
        if torch.cuda.is_available() else 32,
        callbacks=callbacks, default_root_dir=str(ckpt_dir),
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=2 if args.smoke else 1.0,
        log_every_n_steps=1 if args.smoke else 50)
    trainer.fit(module, train_dl, val_dl)
    if not callbacks:
        trainer.save_checkpoint(ckpt_dir / "last.ckpt")
    logger.info("Done. Checkpoints in %s", ckpt_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
