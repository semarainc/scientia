"""
config.py - Central configuration for Pneumonia Detection project.
Edit values here; all other modules import from this file.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Root of the raw kaggle dataset  (train/val/test sub-folders live here) will be use the preprocessed one
DATA_ROOT = Path(os.getenv("DATA_ROOT", "models/data/chest_xray_processed"))

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
TEST_DIR  = DATA_ROOT / "test"

# Where to save checkpoints and the final exported model
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "models/checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH  = CHECKPOINT_DIR / "best_model.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Order matters: index 0 = NORMAL, index 1 = PNEUMONIA
CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES   = 2
POS_LABEL     = 1          # PNEUMONIA is the positive class

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
# DenseNet was pre-trained on ImageNet (224x224, RGB)
# Chest X-rays are grayscale → we replicate the single channel to 3 channels
IMG_SIZE      = 224        # DenseNet input resolution
IMG_CHANNELS  = 3          # converted from grayscale to RGB-like

# ImageNet normalisation statistics (used because we use ImageNet weights)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# --- CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
# Enhances local contrast in X-ray images, making lung opacities more visible.
# Applied before tensor conversion, on the grayscale image.
USE_CLAHE       = True
CLAHE_CLIP_LIMIT  = 2.0   # higher = more contrast, risk of noise amplification
CLAHE_TILE_SIZE   = (8, 8) # grid size for local histogram computation

# --- Lung ROI crop (centre-crop fraction) ---
# Chest X-rays contain peripheral regions (shoulders, abdomen) that add noise.
# A centre-crop removes most of these after initial resize.
# 1.0 = no crop; 0.85 = keep central 85% of the resized image.
CENTRE_CROP_FRACTION = 0.9

# --- Augmentation strengths (training only) ---
AUG_ROTATION_DEGREES = 10
AUG_BRIGHTNESS       = 0.2
AUG_CONTRAST         = 0.2
AUG_HFLIP_PROB       = 0.5

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE      = 32
NUM_EPOCHS      = 20
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4

# Learning-rate scheduler: reduce on plateau
LR_PATIENCE     = 3        # epochs without val-loss improvement before LR drops
LR_FACTOR       = 0.5      # multiply LR by this factor
MIN_LR          = 1e-7

# Early stopping
EARLY_STOP_PATIENCE = 7    # stop if val-loss doesn't improve for this many epochs

# ---------------------------------------------------------------------------
# DenseNet variant
# ---------------------------------------------------------------------------
# Options: "densenet121", "densenet169", "densenet201"
# densenet121 is the best balance of accuracy vs. CPU inference speed
DENSENET_VARIANT = "densenet121"
PRETRAINED       = True     # use ImageNet weights

# Fine-tuning strategy:
#   "head_only"  → freeze backbone, train classifier head only (fast)
#   "full"       → unfreeze everything after warm-up (better accuracy)
FINE_TUNE_STRATEGY = "full"
FREEZE_EPOCHS      = 3     # epochs to train head-only before unfreezing backbone

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
# Force CPU so the exported model runs on any deployment machine
# Set to "cuda" locally if you have a GPU for faster training
DEVICE = os.getenv("DEVICE", "cpu")

# DataLoader workers (set to 0 on Windows to avoid multiprocessing issues)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

# ---------------------------------------------------------------------------
# Logging / reproducibility
# ---------------------------------------------------------------------------
SEED            = 42
LOG_INTERVAL    = 10       # log training loss every N batches
