"""Model module - Voxel Cluster Density Attention Network (VCDA-Net)

⚠️ Note: Uses activation magnitude-based density attention maps, NOT gradient-based GradCAM
"""

from .vcda_net import (
    VCDANet,
    UNetFeatureExtractor,
    GatedFusion,
    BottleneckStream,
    TransformerAggregation,
    ResNetEncoder,
    MatrixResNetEncoder,
)

__all__ = [
    'VCDANet',
    'UNetFeatureExtractor',
    'GatedFusion',
    'BottleneckStream',
    'TransformerAggregation',
    'ResNetEncoder',
    'MatrixResNetEncoder',
]
