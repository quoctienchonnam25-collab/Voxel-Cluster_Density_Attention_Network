"""
Model Components for Hybrid Saliency Enhanced V4

This package contains modular components for the hybrid saliency map model:
- utils: Helper functions for saliency map processing
- unet_extractor: UNet feature extraction
- downsamplers: Downsampling modules
- encoders: ResNet-based encoders
- gnn: Graph neural network components
- transformers: Transformer aggregation
- fusion: Multi-stream fusion modules
"""

# Utility functions
from .utils import (
    extract_top_k_coordinates,
    compute_gaussian_distance_matrix
)

# UNet
from .unet_extractor import UNetFeatureExtractor

# Downsamplers
from .downsamplers import StridedConvDownsampler

# Encoders
from .encoders import (
    MatrixResNetEncoder,
    ResNetEncoder
)

# GNN
from .gnn import MultiEdgeAttention

# Transformers
from .transformers import TransformerAggregation

# Fusion
from .fusion import (
    ResBlock3D,
    BottleneckStream,
    GatedFusion
)

__all__ = [
    # Utils
    'extract_top_k_coordinates',
    'compute_gaussian_distance_matrix',
    # UNet
    'UNetFeatureExtractor',
    # Downsamplers
    'StridedConvDownsampler',
    # Encoders
    'MatrixResNetEncoder',
    'ResNetEncoder',
    # GNN
    'MultiEdgeAttention',
    # Transformers
    'TransformerAggregation',
    # Fusion
    'ResBlock3D',
    'BottleneckStream',
    'GatedFusion',
]
