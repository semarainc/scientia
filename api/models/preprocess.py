"""
preprocess.py - Image preprocessing pipeline untuk chest X-ray pneumonia detection.

Modul ini menangani semua langkah transformasi gambar secara eksplisit dan terpisah
agar mudah di-debug, dimodifikasi, atau dinonaktifkan per-step.

Langkah preprocessing (berurutan):
  1. Load         - buka gambar, pastikan mode grayscale
  2. Denoise      - median filter ringan untuk mengurangi sensor noise
  3. CLAHE        - tingkatkan kontras lokal (opasitas paru lebih terlihat)
  4. Resize       - ke ukuran target dengan padding agar aspect ratio terjaga
  5. Centre crop  - buang tepi (bahu, abdomen) yang tidak relevan
  6. Grayscale→RGB- replikasi channel agar kompatibel dengan ImageNet weights
  7. Normalize    - ImageNet mean/std

Public API:
    build_train_transform() -> transforms.Compose
    build_val_transform()   -> transforms.Compose
    preprocess_image(source) -> torch.Tensor   # untuk inference tunggal
    run_pipeline(src_dir, dst_dir)             # offline preprocessing ke disk

Usage:
    from preprocess import build_train_transform, build_val_transform

    # Gunakan langsung di dataset.py
    train_tf = build_train_transform()
    val_tf   = build_val_transform()

    # Atau jalankan offline:
    python preprocess.py --src data/chest_xray --dst data/chest_xray_processed
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

from models import config


# ---------------------------------------------------------------------------
# Step 1 – Load & validate
# ---------------------------------------------------------------------------

def load_image(source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    """
    Load image from file path, raw bytes, or PIL Image.
    Always returns a PIL Image in 'L' (grayscale) mode.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(str(source))

    # Convert to grayscale (handles RGB/RGBA/L input uniformly)
    return img.convert("L")


# ---------------------------------------------------------------------------
# Step 2 – Denoise (median filter)
# ---------------------------------------------------------------------------

def denoise(img: Image.Image, kernel_size: int = 3) -> Image.Image:
    """
    Apply a mild median filter to reduce salt-and-pepper / sensor noise.
    kernel_size=3 is conservative enough not to blur diagnostically relevant edges.
    Skipped automatically if kernel_size < 2.
    """
    if kernel_size < 2:
        return img
    arr    = np.array(img, dtype=np.uint8)
    denoised = cv2.medianBlur(arr, ksize=kernel_size)
    return Image.fromarray(denoised)


# ---------------------------------------------------------------------------
# Step 3 – CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ---------------------------------------------------------------------------

def apply_clahe(
    img:        Image.Image,
    clip_limit: float       = config.CLAHE_CLIP_LIMIT,
    tile_size:  tuple[int, int] = config.CLAHE_TILE_SIZE,
) -> Image.Image:
    """
    Enhance local contrast using CLAHE.

    Why CLAHE for CXR:
      - Global histogram equalisation over-amplifies noise in uniform regions.
      - CLAHE limits contrast amplification (clip_limit) per tile, preserving
        clinically relevant texture differences between normal and consolidated
        lung parenchyma.

    Args:
        img:        Grayscale PIL Image.
        clip_limit: Threshold for contrast limiting (higher = more contrast).
        tile_size:  Grid size (rows, cols) for local histogram computation.

    Returns:
        Contrast-enhanced grayscale PIL Image.
    """
    arr   = np.array(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(arr)
    return Image.fromarray(enhanced)


# ---------------------------------------------------------------------------
# Step 4 – Aspect-ratio-preserving resize with padding
# ---------------------------------------------------------------------------

def resize_with_padding(
    img:        Image.Image,
    target_size: int = config.IMG_SIZE,
    fill_value:  int = 0,           # black padding
) -> Image.Image:
    """
    Resize image to `target_size x target_size` while preserving aspect ratio.
    Pads the shorter axis with `fill_value` (black by default).

    Rationale: naive resize distorts lung shape/aspect ratio which can mislead
    the model. Padding-based resize keeps proportions intact.
    """
    orig_w, orig_h = img.size
    scale  = target_size / max(orig_w, orig_h)
    new_w  = int(orig_w * scale)
    new_h  = int(orig_h * scale)

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create black canvas and paste centred
    canvas = Image.new("L", (target_size, target_size), fill_value)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


# ---------------------------------------------------------------------------
# Step 5 – Centre crop
# ---------------------------------------------------------------------------

def centre_crop(
    img:      Image.Image,
    fraction: float = config.CENTRE_CROP_FRACTION,
) -> Image.Image:
    """
    Crop the central `fraction` of the image, then resize back to original size.
    Removes peripheral areas (shoulders, abdomen) that don't contribute to
    pneumonia diagnosis.

    Args:
        img:      Input PIL Image (square assumed).
        fraction: Fraction of the image to keep (0 < fraction <= 1.0).

    Returns:
        Cropped and resized PIL Image of the same size as input.
    """
    if fraction >= 1.0:
        return img

    w, h     = img.size
    crop_w   = int(w * fraction)
    crop_h   = int(h * fraction)
    left     = (w - crop_w) // 2
    top      = (h - crop_h) // 2
    cropped  = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Step 6 – Grayscale → 3-channel RGB-like
# ---------------------------------------------------------------------------

def to_rgb(img: Image.Image) -> Image.Image:
    """
    Replicate the single grayscale channel to 3 identical channels.
    Required for DenseNet (and all ImageNet-pretrained models) which expect
    3-channel input.
    """
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Composed preprocessing function (for use in offline pipeline)
# ---------------------------------------------------------------------------

def preprocess_pil(
    source:       Union[str, Path, bytes, Image.Image],
    apply_clahe_flag:  bool  = config.USE_CLAHE,
    apply_denoise_flag: bool = True,
) -> Image.Image:
    """
    Run the full preprocessing pipeline and return a 3-channel PIL Image
    ready to be passed through a torchvision transform (ToTensor + Normalize).

    Steps: load → denoise → CLAHE → resize+pad → centre_crop → to_rgb

    This function is used by:
      - run_pipeline()  for offline preprocessing
      - preprocess_image() for single-tensor inference
    """
    img = load_image(source)

    if apply_denoise_flag:
        img = denoise(img)

    if apply_clahe_flag:
        img = apply_clahe(img)  # type: ignore[assignment]

    img = resize_with_padding(img, target_size=config.IMG_SIZE)
    img = centre_crop(img, fraction=config.CENTRE_CROP_FRACTION)
    img = to_rgb(img)
    return img


# ---------------------------------------------------------------------------
# torchvision Transform wrappers
# ---------------------------------------------------------------------------

class CXRPreprocess(torch.nn.Module):
    """
    Custom torchvision-compatible transform that applies the full
    CXR-specific preprocessing pipeline to a PIL Image.

    Can be inserted into a transforms.Compose pipeline.
    """

    def __init__(
        self,
        apply_clahe_flag:   bool = config.USE_CLAHE,
        apply_denoise_flag: bool = True,
    ):
        super().__init__()
        self.apply_clahe_flag   = apply_clahe_flag
        self.apply_denoise_flag = apply_denoise_flag

    def forward(self, img: Image.Image) -> Image.Image:
        """
        img: PIL Image (any mode; will be converted to grayscale internally)
        Returns: PIL Image in RGB mode, resized and preprocessed.
        """
        return preprocess_pil(
            img,
            apply_clahe_flag=self.apply_clahe_flag,
            apply_denoise_flag=self.apply_denoise_flag,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clahe={self.apply_clahe_flag}, "
            f"denoise={self.apply_denoise_flag})"
        )


def build_train_transform(
    apply_clahe_flag:   bool = config.USE_CLAHE,
    apply_denoise_flag: bool = True,
) -> transforms.Compose:
    """
    Full training transform pipeline:
      CXRPreprocess (CLAHE + resize + crop + RGB)
      → random augmentation
      → ToTensor
      → Normalize
    """
    return transforms.Compose([
        # Custom CXR preprocessing (returns RGB PIL Image)
        CXRPreprocess(
            apply_clahe_flag=apply_clahe_flag,
            apply_denoise_flag=apply_denoise_flag,
        ),

        # Random augmentation for training
        transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB),
        transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def build_val_transform(
    apply_clahe_flag:   bool = config.USE_CLAHE,
    apply_denoise_flag: bool = True,
) -> transforms.Compose:
    """
    Deterministic validation / test / inference transform pipeline:
      CXRPreprocess → ToTensor → Normalize
    """
    return transforms.Compose([
        CXRPreprocess(
            apply_clahe_flag=apply_clahe_flag,
            apply_denoise_flag=apply_denoise_flag,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Single-image inference helper
# ---------------------------------------------------------------------------

def preprocess_image(
    source:             Union[str, Path, bytes, Image.Image],
    apply_clahe_flag:   bool = config.USE_CLAHE,
    apply_denoise_flag: bool = True,
) -> torch.Tensor:
    """
    Preprocess a single image and return a [1, 3, H, W] tensor.
    Convenience wrapper for predict.py.
    """
    tf  = build_val_transform(apply_clahe_flag, apply_denoise_flag)
    img = load_image(source).convert("RGB")  # PIL expects RGB for ColorJitter etc.
    return tf(img).unsqueeze(0)              # [1, 3, H, W]


# ---------------------------------------------------------------------------
# Offline preprocessing pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    src_dir:  Path | str,
    dst_dir:  Path | str,
    apply_clahe_flag:   bool = config.USE_CLAHE,
    apply_denoise_flag: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Preprocess all JPEG images in `src_dir` and save results to `dst_dir`,
    preserving the split/class subdirectory structure:

        src_dir/train/NORMAL/img.jpeg  →  dst_dir/train/NORMAL/img.jpeg

    Args:
        src_dir:   Root of raw dataset (contains train/ val/ test/).
        dst_dir:   Output root for preprocessed images.
        overwrite: If False, skip images that already exist in dst_dir.

    This step is OPTIONAL. If you prefer online preprocessing (default),
    the CXRPreprocess transform runs in the DataLoader worker.
    Offline preprocessing saves CPU time during training at the cost of disk space.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    image_paths = list(src_dir.rglob("*.jpeg")) + list(src_dir.rglob("*.jpg"))
    total = len(image_paths)

    if total == 0:
        print(f"[preprocess] No JPEG images found in {src_dir}")
        return

    print(f"[preprocess] Found {total} images in {src_dir}")
    print(f"[preprocess] Output  -> {dst_dir}")
    print(f"[preprocess] CLAHE={apply_clahe_flag}  Denoise={apply_denoise_flag}")

    processed = skipped = errors = 0

    for idx, src_path in enumerate(image_paths, 1):
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path

        if not overwrite and dst_path.exists():
            skipped += 1
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = preprocess_pil(
                src_path,
                apply_clahe_flag=apply_clahe_flag,
                apply_denoise_flag=apply_denoise_flag,
            )
            img.save(str(dst_path), format="JPEG", quality=95)
            processed += 1
        except Exception as exc:
            print(f"  [ERROR] {src_path}: {exc}")
            errors += 1

        if idx % 500 == 0 or idx == total:
            print(f"  [{idx}/{total}] processed={processed}  "
                  f"skipped={skipped}  errors={errors}")

    print(
        f"\n[preprocess] Done.  "
        f"processed={processed}  skipped={skipped}  errors={errors}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline CXR preprocessing pipeline"
    )
    p.add_argument(
        "--src", type=str, default=str(config.DATA_ROOT),
        help="Source dataset root (default: config.DATA_ROOT)",
    )
    p.add_argument(
        "--dst", type=str, default="data/chest_xray_processed",
        help="Destination root for preprocessed images",
    )
    p.add_argument(
        "--no-clahe", action="store_true",
        help="Disable CLAHE contrast enhancement",
    )
    p.add_argument(
        "--no-denoise", action="store_true",
        help="Disable median filter denoising",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-process images even if output already exists",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        src_dir=args.src,
        dst_dir=args.dst,
        apply_clahe_flag=not args.no_clahe,
        apply_denoise_flag=not args.no_denoise,
        overwrite=args.overwrite,
    )
