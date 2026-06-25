"""
evaluate.py - Full evaluation of a trained model on the test set.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --split val

Outputs:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix (printed and saved as PNG)
    - ROC-AUC score
    - All metrics saved to checkpoints/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import config
from dataset import get_datasets, get_dataloaders
from model import load_checkpoint


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pneumonia detection model")
    p.add_argument(
        "--checkpoint", type=str,
        default=str(config.BEST_MODEL_PATH),
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    p.add_argument(
        "--device", type=str, default=config.DEVICE,
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for positive (PNEUMONIA) class",
    )
    p.add_argument(
        "--save-plots", action="store_true",
        help="Save confusion matrix and ROC curve plots (requires matplotlib)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    device:    str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference pass.

    Returns:
        all_labels:  ground-truth int labels  shape [N]
        all_probs:   sigmoid probabilities     shape [N]  (prob of PNEUMONIA)
        all_preds:   binary predictions        shape [N]
    """
    model.eval()
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)                      # [B, 1]
        probs  = torch.sigmoid(logits).squeeze(1)   # [B]
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    all_labels_np = np.array(all_labels, dtype=int)
    all_probs_np  = np.array(all_probs,  dtype=float)
    return all_labels_np, all_probs_np


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    labels:    np.ndarray,
    probs:     np.ndarray,
    threshold: float = 0.5,
    class_names: list[str] = config.CLASS_NAMES,
) -> dict:
    """
    Compute and print all classification metrics.

    Returns a dict with all scalar metrics (JSON-serialisable).
    """
    preds = (probs >= threshold).astype(int)

    # ---- Classification report ----
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
    )
    print("\nClassification Report:")
    print(report)

    # ---- Confusion matrix ----
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"           NORMAL  PNEUMONIA")
    print(f"NORMAL      {tn:5d}    {fp:5d}")
    print(f"PNEUMONIA   {fn:5d}    {tp:5d}")

    # ---- ROC-AUC ----
    auc = roc_auc_score(labels, probs)
    print(f"\nROC-AUC: {auc:.4f}")

    # ---- Derived metrics ----
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision   = tp / max(tp + fp, 1)
    f1          = 2 * precision * sensitivity / max(precision + sensitivity, 1e-8)
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  (recall for PNEUMONIA)")
    print(f"Specificity: {specificity:.4f}  (recall for NORMAL)")
    print(f"Precision:   {precision:.4f}")
    print(f"F1 score:    {f1:.4f}")

    return {
        "accuracy":    float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1":          float(f1),
        "auc":         float(auc),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Optional plot helpers
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(
    labels: np.ndarray,
    preds:  np.ndarray,
    out_path: Path,
    class_names: list[str] = config.CLASS_NAMES,
) -> None:
    import matplotlib.pyplot as plt  # lazy import – only needed for plots

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    print(f"Confusion matrix saved to {out_path}")
    plt.close(fig)


def save_roc_curve_plot(
    labels:   np.ndarray,
    probs:    np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, probs)
    auc          = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    print(f"ROC curve saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str | Path | None = None,
    split:           str               = "test",
    device:          str               = config.DEVICE,
    threshold:       float             = 0.5,
    save_plots:      bool              = False,
) -> dict:
    """
    Evaluate a checkpoint on the specified split.
    Returns the metrics dict.
    Can be called programmatically from other modules.
    """
    checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH

    # ---- Load model ----
    print(f"Loading checkpoint: {checkpoint_path}")
    model, _ = load_checkpoint(str(checkpoint_path), device=device)
    model.eval()

    # ---- Load data ----
    dsets   = get_datasets()
    if split not in dsets:
        raise ValueError(f"Split '{split}' not found. Available: {list(dsets.keys())}")

    loaders = get_dataloaders(dsets, use_weighted_sampler=False)
    loader  = loaders[split]
    print(f"Evaluating on {split} set ({len(dsets[split])} images) ...")

    # ---- Inference ----
    labels, probs = run_inference(model, loader, device)

    # ---- Metrics ----
    metrics = compute_metrics(labels, probs, threshold=threshold)
    metrics["split"]     = split
    metrics["threshold"] = threshold
    metrics["checkpoint"] = str(checkpoint_path)

    # ---- Save results ----
    results_path = config.CHECKPOINT_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    # ---- Optional plots ----
    if save_plots:
        preds = (probs >= threshold).astype(int)
        save_confusion_matrix_plot(
            labels, preds,
            config.CHECKPOINT_DIR / "confusion_matrix.png",
        )
        save_roc_curve_plot(
            labels, probs,
            config.CHECKPOINT_DIR / "roc_curve.png",
        )

    return metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        device=args.device,
        threshold=args.threshold,
        save_plots=args.save_plots,
    )
