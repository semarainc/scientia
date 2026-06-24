"""
train.py - Training loop for pneumonia detection model.

Usage:
    python train.py
    python train.py --epochs 30 --lr 0.0001 --batch-size 32
    python train.py --resume checkpoints/best_model.pth

Two-phase fine-tuning:
    Phase 1 (freeze_epochs):  only classifier head is trained  -> fast convergence
    Phase 2 (remaining epochs): full network unfrozen           -> higher accuracy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from dataset import get_datasets, get_dataloaders, get_class_weights_from_dir
from model import (
    build_model,
    freeze_backbone,
    unfreeze_backbone,
    count_trainable_params,
    save_checkpoint,
    load_checkpoint,
    export_model_for_inference,
)
from utils import set_seed, AverageMeter, format_time


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DenseNet for pneumonia detection")
    p.add_argument("--epochs",      type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch-size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--device",      type=str,   default=config.DEVICE)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no-weighted-sampler", action="store_true",
                   help="Disable WeightedRandomSampler (use plain shuffle instead)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    str,
    epoch:     int,
) -> dict:
    """
    Run one training epoch.
    Returns dict with keys: loss, acc, tp, tn, fp, fn
    """
    model.train()
    loss_meter = AverageMeter()
    correct = total = tp = tn = fp = fn = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        # BCEWithLogitsLoss expects float targets of shape [N, 1]
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping helps stability especially on CPU
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- metrics ----
        loss_meter.update(loss.item(), images.size(0))
        preds = (torch.sigmoid(logits) >= 0.5).long()
        gt    = labels.long()
        correct += (preds == gt).sum().item()
        total   += gt.size(0)
        tp += ((preds == 1) & (gt == 1)).sum().item()
        tn += ((preds == 0) & (gt == 0)).sum().item()
        fp += ((preds == 1) & (gt == 0)).sum().item()
        fn += ((preds == 0) & (gt == 1)).sum().item()

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f}  acc={correct/total:.4f}"
            )

    return {
        "loss": loss_meter.avg,
        "acc":  correct / max(total, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


@torch.no_grad()
def evaluate_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    str,
) -> dict:
    """
    Evaluate model on a dataloader (val or test).
    Returns dict with keys: loss, acc, sensitivity, specificity, f1
    """
    model.eval()
    loss_meter = AverageMeter()
    correct = total = tp = tn = fp = fn = 0

    for images, labels in loader:
        images = images.to(device)
        labels_f = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss   = criterion(logits, labels_f)

        loss_meter.update(loss.item(), images.size(0))
        preds = (torch.sigmoid(logits) >= 0.5).long()
        gt    = labels.long().unsqueeze(1).to(device)
        correct += (preds == gt).sum().item()
        total   += gt.size(0)
        tp += ((preds == 1) & (gt == 1)).sum().item()
        tn += ((preds == 0) & (gt == 0)).sum().item()
        fp += ((preds == 1) & (gt == 0)).sum().item()
        fn += ((preds == 0) & (gt == 1)).sum().item()

    sensitivity = tp / max(tp + fn, 1)   # recall for PNEUMONIA
    specificity = tn / max(tn + fp, 1)   # recall for NORMAL
    precision   = tp / max(tp + fp, 1)
    f1          = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)

    return {
        "loss":        loss_meter.avg,
        "acc":         correct / max(total, 1),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
        "f1":          f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace | None = None) -> nn.Module:
    """
    Full training procedure. Returns the best model.
    Can be called programmatically (args=None uses config defaults).
    """
    if args is None:
        args = argparse.Namespace(
            epochs=config.NUM_EPOCHS,
            lr=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            device=config.DEVICE,
            resume=None,
            no_weighted_sampler=False,
        )

    set_seed(config.SEED)
    device = args.device
    print(f"Device: {device}")

    # ---- Data ----
    print("\nLoading datasets ...")
    dsets  = get_datasets()
    loaders = get_dataloaders(
        dsets,
        use_weighted_sampler=not args.no_weighted_sampler,
    )

    for split, ds in dsets.items():
        print(f"  {split:5s}: {len(ds):5d} images")

    # ---- Class-weighted loss ----
    class_weights = get_class_weights_from_dir()
    # For BCEWithLogitsLoss with binary output we pass pos_weight
    # pos_weight = weight[PNEUMONIA] / weight[NORMAL]
    pos_weight = torch.tensor(
        [class_weights[config.POS_LABEL].item()], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\nClass weights (NORMAL={class_weights[0]:.3f}, "
          f"PNEUMONIA={class_weights[1]:.3f})  pos_weight={pos_weight.item():.3f}")

    # ---- Model ----
    model = build_model().to(device)
    start_epoch = 0
    best_val_loss = float("inf")
    no_improve_count = 0

    if args.resume:
        print(f"\nResuming from {args.resume}")
        model, ckpt = load_checkpoint(args.resume, model=model, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))

    # ---- Optimiser ----
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=True,
    )

    # ---- Phase 1: freeze backbone ----
    if config.FINE_TUNE_STRATEGY == "full" and start_epoch == 0:
        freeze_backbone(model)
        print(f"\nPhase 1 – head-only training "
              f"({count_trainable_params(model):,} trainable params)")

    # ---- Training loop ----
    print("\n" + "=" * 60)
    print(" Starting training")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Unfreeze backbone after warm-up
        if (
            config.FINE_TUNE_STRATEGY == "full"
            and epoch == config.FREEZE_EPOCHS
            and start_epoch < config.FREEZE_EPOCHS
        ):
            unfreeze_backbone(model)
            # Rebuild optimiser to include all params
            optimizer = Adam(
                model.parameters(),
                lr=args.lr * 0.1,        # lower LR for backbone
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min",
                factor=config.LR_FACTOR,
                patience=config.LR_PATIENCE,
                min_lr=config.MIN_LR,
                verbose=True,
            )
            print(f"\nPhase 2 – full fine-tuning "
                  f"({count_trainable_params(model):,} trainable params)")

        # --- Train ---
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch + 1
        )

        # --- Validate ---
        val_metrics = evaluate_one_epoch(
            model, loaders["val"], criterion, device
        )

        scheduler.step(val_metrics["loss"])

        elapsed = format_time(time.time() - epoch_start)
        print(
            f"\nEpoch {epoch+1:>3}/{args.epochs} ({elapsed}) | "
            f"train loss={train_metrics['loss']:.4f}  acc={train_metrics['acc']:.4f} | "
            f"val   loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.4f}  "
            f"sens={val_metrics['sensitivity']:.4f}  spec={val_metrics['specificity']:.4f}  "
            f"F1={val_metrics['f1']:.4f}"
        )

        # --- Checkpoint (best model) ---
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve_count = 0
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                config.BEST_MODEL_PATH,
            )
            print(f"  *** New best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count} epoch(s).")

        # --- Early stopping ---
        if no_improve_count >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    # ---- Save final model ----
    save_checkpoint(
        model, optimizer, epoch, val_metrics["loss"],
        config.FINAL_MODEL_PATH,
    )
    print(f"\nFinal model saved to {config.FINAL_MODEL_PATH}")

    # ---- Export inference model ----
    inference_path = config.CHECKPOINT_DIR / "inference_model.pth"
    export_model_for_inference(config.BEST_MODEL_PATH, inference_path)

    print("\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Inference model: {inference_path}")

    return model


if __name__ == "__main__":
    args = parse_args()
    train(args)
