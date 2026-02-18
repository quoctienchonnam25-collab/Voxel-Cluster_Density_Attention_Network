"""
Voxel Cluster Density Attention Network (VCDA-Net) Package
====================================

Brain Age Prediction with Density Attention-Enhanced Features and Gated Fusion

⚠️ IMPORTANT: This package uses SALIENCY MAPS, not GradCAM!
   - Saliency Map: Activation magnitude-based weighting (NO gradients)
   - GradCAM: Gradient-based weighting (requires backward pass)

Our implementation is faster and simpler because we don't need gradients.

Key Features:
- Dual-Stream Architecture (Transformer + Bottleneck)
- Gated Fusion mechanism (V4 innovation)
- Saliency Map-enhanced features (activation magnitude-based)
- Graph Neural Networks
- Transformer aggregation

Quick Start:
-----------
>>> from vcda_net.model import VCDANet
>>> model = VCDANet(
...     unet_checkpoint='checkpoints/IXI_3dunet_best_model.pth',
...     num_regions=32
... )

Author: Anonymous
License: MIT
Version: 4.0.2
"""

__version__ = "4.0.2"
__author__ = "Anonymous"
__license__ = "MIT"

# Import main components
try:
    from .model.vcda_net import (
        VCDANet,
        UNetFeatureExtractor,
        GatedFusion,
        BottleneckStream,
        TransformerAggregation,
    )
    
    __all__ = [
        'VCDANet',
        'UNetFeatureExtractor',
        'GatedFusion',
        'BottleneckStream',
        'TransformerAggregation',
        '__version__',
        '__author__',
        '__license__',
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import model components: {e}\n"
        "Make sure to install all dependencies: pip install -r requirements.txt"
    )
    __all__ = ['__version__', '__author__', '__license__']


def print_info():
    """Print package information"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              Voxel Cluster Density Attention Network (VCDA-Net) - Brain Age Prediction             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Version: {__version__}
Author: {__author__}
License: {__license__}

⚠️  IMPORTANT NAMING CLARIFICATION:
   This package uses SALIENCY MAPS (activation magnitude-based),
   NOT GradCAM (gradient-based)!
   
   Why? Because we don't need gradients for feature weighting.
   This makes training faster and implementation simpler.

Features:
  • Dual-Stream Architecture
  • Gated Fusion (V4 innovation)
  • Saliency Map Enhancement
  • Graph Neural Networks
  • Transformer Aggregation

For more information, see README.md
    """)
