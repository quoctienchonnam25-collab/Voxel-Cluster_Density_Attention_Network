"""Model module - Hybrid Saliency Enhanced V4

⚠️ Note: Uses activation magnitude-based saliency maps, NOT gradient-based GradCAM
"""

from .hybrid_saliency_enhanced_v4 import (
    HybridSaliencyEnhanced,
    UNetFeatureExtractor,
    GatedFusion,
    BottleneckStream,
    TransformerAggregation,
    ResNetEncoder,
    MatrixResNetEncoder,
)

__all__ = [
    'HybridSaliencyEnhanced',
    'UNetFeatureExtractor',
    'GatedFusion',
    'BottleneckStream',
    'TransformerAggregation',
    'ResNetEncoder',
    'MatrixResNetEncoder',
]
