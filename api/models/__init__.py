"""
Pneumonia Detection Models Package

This package contains the core model, preprocessing, prediction, and configuration modules
for the Scientia Pneumonia Detection API.

Modules:
    - config: Central configuration for paths, hyperparameters, and constants
    - model: DenseNet model architecture and checkpoint loading
    - preprocess: Image preprocessing pipeline (CLAHE, normalization, augmentation)
    - predict: Inference module with PneumoniaPredictor class
"""

__version__ = "1.0.0"

# Make commonly used classes available at package level
try:
    from .predict import PneumoniaPredictor, get_predictor
    from .model import load_checkpoint
    from . import config
    
    __all__ = [
        "PneumoniaPredictor",
        "get_predictor",
        "load_checkpoint",
        "config",
    ]
except ImportError:
    # Allow package to be imported even if dependencies aren't installed yet
    pass
