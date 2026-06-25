"""
model.py - DenseNet transfer-learning model for binary pneumonia classification.

Design notes:
  - Uses torchvision's DenseNet (121 / 169 / 201) pre-trained on ImageNet.
  - Replaces the classifier head with a lightweight binary classifier.
  - Supports a two-phase fine-tuning strategy:
      Phase 1 (freeze_epochs): only the new head is trained.
      Phase 2:                 the full network is unfrozen with a lower LR.
  - Model is exported / loaded in a CPU-friendly way (torch.save / torch.load
    with map_location="cpu").
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models

from models import config


# ---------------------------------------------------------------------------
# Head definition
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Replacement head for DenseNet's classifier.

    Input:  feature vector of size `in_features`  (1024 for DenseNet-121)
    Output: logits of shape [batch, 1]  (binary cross-entropy friendly)
    """

    def __init__(self, in_features: int, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),          # single logit for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    variant:    str  = config.DENSENET_VARIANT,
    pretrained: bool = config.PRETRAINED,
    dropout:    float = 0.4,
) -> nn.Module:
    """
    Build and return a DenseNet model with a custom binary classifier head.

    Args:
        variant:    One of "densenet121", "densenet169", "densenet201".
        pretrained: Load ImageNet weights.
        dropout:    Dropout rate in the classifier head.

    Returns:
        nn.Module ready for training or inference.
    """
    # --- Load backbone ---
    weights_enum = {
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "densenet169": models.DenseNet169_Weights.IMAGENET1K_V1,
        "densenet201": models.DenseNet201_Weights.IMAGENET1K_V1,
    }

    if variant not in weights_enum:
        raise ValueError(
            f"Unknown DenseNet variant '{variant}'. "
            f"Choose from {list(weights_enum.keys())}."
        )

    weights = weights_enum[variant] if pretrained else None
    model_fn = getattr(models, variant)
    backbone: models.DenseNet = model_fn(weights=weights)

    # --- Replace classifier head ---
    in_features = backbone.classifier.in_features   # 1024 / 1664 / 1920
    backbone.classifier = ClassifierHead(in_features, dropout=dropout)

    return backbone


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters except the classifier head.
    Call this at the start of training (Phase 1).
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all parameters for full fine-tuning (Phase 2).
    Typically called after `freeze_epochs` warm-up epochs.
    """
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    path:      Path | str,
) -> None:
    """
    Save model checkpoint.

    Saved dict:
        model_state_dict, optimizer_state_dict, epoch, val_loss,
        densenet_variant, num_classes
    """
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "val_loss":             val_loss,
            "densenet_variant":     config.DENSENET_VARIANT,
            "num_classes":          config.NUM_CLASSES,
        },
        str(path),
    )


def load_checkpoint(
    path:       Path | str,
    model:      Optional[nn.Module] = None,
    optimizer:  Optional[torch.optim.Optimizer] = None,
    device:     str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    Load a checkpoint.

    If `model` is None a fresh model is built from the saved variant.
    Always maps tensors to `device` (defaults to CPU for deployment).

    Returns:
        (model, checkpoint_dict)  – model has weights loaded in-place.
    """
    ckpt = torch.load(str(path), map_location=device)

    if model is None:
        model = build_model(
            variant=ckpt.get("densenet_variant", config.DENSENET_VARIANT),
            pretrained=False,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return model, ckpt


def export_model_for_inference(
    checkpoint_path: Path | str,
    output_path:     Path | str,
    device:          str = "cpu",
) -> None:
    """
    Export a lightweight inference-only model (weights only, no optimizer state).
    This is the file that FastAPI will load at startup.

    Saved dict:
        model_state_dict, densenet_variant, class_names
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model = build_model(
        variant=ckpt.get("densenet_variant", config.DENSENET_VARIANT),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "densenet_variant": ckpt.get("densenet_variant", config.DENSENET_VARIANT),
            "class_names":      config.CLASS_NAMES,
        },
        str(output_path),
    )
    print(f"Inference model saved to: {output_path}")


# ---------------------------------------------------------------------------
# Quick sanity check (run: python model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = build_model()
    print(f"Model: {config.DENSENET_VARIANT}")
    print(f"Total params:     {sum(p.numel() for p in m.parameters()):,}")

    freeze_backbone(m)
    print(f"Trainable (head only): {count_trainable_params(m):,}")

    unfreeze_backbone(m)
    print(f"Trainable (full):      {count_trainable_params(m):,}")

    # Forward pass test
    dummy = torch.zeros(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    out   = m(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expected [2, 1])")
