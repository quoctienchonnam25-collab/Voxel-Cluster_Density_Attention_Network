"""
Hybrid Saliency V4 - Complete Training Pipeline

This module provides the complete pipeline for:
1. Feature extraction from trained model
2. Regional predictor training
3. Prediction generation with bias correction

Author: Anonymous
Date: 2026-02-17
"""

from .extract_regional_features import RegionalFeatureExtractor
from .train_regional_predictors import RegionalPredictorTrainer
from .generate_predictions import RegionalPredictor

__all__ = [
    'RegionalFeatureExtractor',
    'RegionalPredictorTrainer',
    'RegionalPredictor',
]
