"""
dataset.py - Data loading and augmentation for chest X-ray pneumonia detection.

Key design decisions for chest X-ray images:
  - Transform pipelines are delegated to preprocess.py (CLAHE, denoise,
    aspect-ratio-preserving resize, centre crop).
  - Validation/test transforms are deterministic (no random augmentation).
  - Class weights are computed and exposed so the trainer can use weighted loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

import config
from preprocess import build_train_transform, build_val_transform


# ---------------------------------------------------------------------------
# Transform pipelines (thin wrappers — logic lives in preprocess.py)
# ---------------------------------------------------------------------------

def get_train_transforms():
    """Return training transform pipeline (augmentation + CXR preprocessing)."""
    return build_train_transform()


def get_val_transforms():
    """Return deterministic val/test/inference transform pipeline."""
    return build_val_transform()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle class imbalance.
    Returns a 1-D tensor of shape [num_classes].
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=config.NUM_CLASSES).astype(float)
    total = class_counts.sum()
    # weight = total / (num_classes * count_of_class)
    weights = total / (config.NUM_CLASSES * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each mini-batch is roughly balanced.
    Useful when the imbalance is severe.
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=config.NUM_CLASSES).astype(float)
    sample_weights = 1.0 / class_counts[targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_datasets(
    train_dir: Path | None = None,
    val_dir:   Path | None = None,
    test_dir:  Path | None = None,
) -> Dict[str, datasets.ImageFolder]:
    """
    Load ImageFolder datasets for train / val / test splits.

    Args:
        train_dir: Path to training folder  (default: config.TRAIN_DIR)
        val_dir:   Path to validation folder (default: config.VAL_DIR)
        test_dir:  Path to test folder       (default: config.TEST_DIR)

    Returns:
        dict with keys "train", "val", "test" (only present if directory exists).
    """
    train_dir = Path(train_dir or config.TRAIN_DIR)
    val_dir   = Path(val_dir   or config.VAL_DIR)
    test_dir  = Path(test_dir  or config.TEST_DIR)

    result: Dict[str, datasets.ImageFolder] = {}

    if train_dir.exists():
        result["train"] = datasets.ImageFolder(
            root=str(train_dir),
            transform=get_train_transforms(),
        )

    if val_dir.exists():
        result["val"] = datasets.ImageFolder(
            root=str(val_dir),
            transform=get_val_transforms(),
        )

    if test_dir.exists():
        result["test"] = datasets.ImageFolder(
            root=str(test_dir),
            transform=get_val_transforms(),
        )

    return result


def get_dataloaders(
    datasets_dict: Dict[str, datasets.ImageFolder],
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Wrap datasets in DataLoaders.

    Args:
        datasets_dict:         Output of get_datasets().
        use_weighted_sampler:  If True, use WeightedRandomSampler for training
                               to mitigate class imbalance.

    Returns:
        dict with the same keys as datasets_dict.
    """
    loaders: Dict[str, DataLoader] = {}

    for split, ds in datasets_dict.items():
        is_train = split == "train"

        if is_train and use_weighted_sampler:
            sampler = make_weighted_sampler(ds)
            loader = DataLoader(
                ds,
                batch_size=config.BATCH_SIZE,
                sampler=sampler,          # mutually exclusive with shuffle
                num_workers=config.NUM_WORKERS,
                pin_memory=False,         # CPU deployment: pin_memory off
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=False,
            )

        loaders[split] = loader

    return loaders


def get_class_weights_from_dir(train_dir: Path | None = None) -> torch.Tensor:
    """
    Convenience function: load train dataset and return class weights tensor.
    Used by the trainer to initialise weighted loss.
    """
    train_dir = Path(train_dir or config.TRAIN_DIR)
    ds = datasets.ImageFolder(root=str(train_dir), transform=get_val_transforms())
    return compute_class_weights(ds)


# ---------------------------------------------------------------------------
# Quick sanity check (run: python dataset.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading datasets ...")
    dsets = get_datasets()
    for split, ds in dsets.items():
        print(f"  {split:5s}: {len(ds):5d} images  classes={ds.classes}")

    print("\nBuilding dataloaders ...")
    loaders = get_dataloaders(dsets)
    for split, dl in loaders.items():
        imgs, labels = next(iter(dl))
        print(f"  {split:5s}: batch shape={tuple(imgs.shape)}  labels={labels[:8].tolist()}")

    if "train" in dsets:
        w = compute_class_weights(dsets["train"])
        print(f"\nClass weights (NORMAL, PNEUMONIA): {w.tolist()}")
