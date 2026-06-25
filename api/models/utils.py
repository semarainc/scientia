"""
utils.py - Shared utility functions used across the project.

Keeps train.py / evaluate.py / predict.py thin by centralising:
  - Reproducibility (seed setting)
  - Metric tracking (AverageMeter)
  - Time formatting
  - Logger setup
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility.
    NOTE: Full determinism on GPU also requires setting
          CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CuDNN (slight speed cost; fine for CPU-target deploy)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Keeps a running average of a scalar metric.

    Example::

        meter = AverageMeter()
        for loss in losses:
            meter.update(loss, batch_size)
        print(meter.avg)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, count={self.count})"


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """
    Convert elapsed seconds into a human-readable string.

    Examples:
        3661.0  -> "1h 01m 01s"
        121.4   -> "2m 01s"
        45.2    -> "45s"
    """
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(
    name:     str           = "pneumonia",
    level:    int           = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger with a StreamHandler (and optionally a FileHandler).
    Calling this multiple times with the same `name` returns the same logger.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)
    print("Seed set.")

    m = AverageMeter("loss")
    for v in [0.8, 0.6, 0.4]:
        m.update(v, n=1)
    print(f"AverageMeter avg: {m.avg:.4f}  (expected 0.6000)")

    print(f"format_time(3661): {format_time(3661)}")
    print(f"format_time(121):  {format_time(121)}")
    print(f"format_time(45):   {format_time(45)}")

    log = get_logger()
    log.info("Logger working.")
