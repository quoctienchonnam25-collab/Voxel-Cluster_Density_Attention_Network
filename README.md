# Voxel Cluster Density Attention Network (VCDA-Net)

**Brain Age Prediction with Density Attention-Enhanced Features and Gated Fusion**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Overview

**VCDA-Net** (Voxel Cluster Density Attention Network) is a novel deep learning architecture for robust brain age prediction from MRI scans. It introduces a **Density Attention Map** mechanism to enhance regional feature extraction without the computational overhead of gradient-based methods (like GradCAM).

Key innovations:
- **Density Attention Map**: Investigates voxel cluster density activation magnitudes to weight regional features effectively.
- **Gated Fusion**: A dynamic mechanism to adaptively combine global features (Transformer stream) and local features (Bottleneck stream).
- **Dual-Stream Architecture**: Synergistically processes spatial and semantic information.
- **Regional Analysis**: Provides interpretability by predicting brain age for 32 distinct brain regions.

## 🚀 Key Features

- **Efficient Feature Weighting**: Uses activation magnitude-based **Density Attention Maps** (no backward pass required).
- **Fast Training**: Significantly faster than gradient-based attention methods.
- **Robustness**: Gated fusion adapts to individual sample characteristics.
- **Interpretability**: Outputs both global brain age and regional age gaps.

## 🛠️ Installation

### Prerequisites
- Linux/MacOS
- Python 3.8+
- CUDA-capable GPU (recommended)

### Install via pip

```bash
# Clone the repository
git clone https://github.com/yourusername/vcda-net.git
cd vcda-net

# Install in editable mode
pip install -e .
```

### Install Dependencies Manually

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Unified Training Pipeline

We provide a single script to handle the entire training loop for VCDA-Net:

```bash
# Run the training wrapper script
bash train_vcda_net.sh
```

You can configure parameters (batch size, epochs, paths) directly inside `train_vcda_net.sh`.

### 2. Python API Usage

You can use VCDA-Net components directly in your Python code:

```python
import torch
from vcda_net.model import VCDANet

# Initialize model
model = VCDANet(
    unet_checkpoint='path/to/unet.pth',
    num_regions=32,
    embedding_dim=256
).cuda()

# Forward pass
# input_tensor: [Batch, 1, 96, 128, 96]
mri_scan = torch.randn(1, 1, 96, 128, 96).cuda()
predicted_age, gate_value = model(mri_scan)

print(f"Predicted Age: {predicted_age.item():.2f} years")
print(f"Gate Value: {gate_value.item():.4f}")
```

## 🧠 Architecture

The VCDA-Net architecture consists of:

1.  **3D UNet Backbone**: Extracts hierarchical features from MRI volumes.
2.  **Density Attention Module**: Computes attention maps based on voxel activation density.
3.  **Dual Streams**:
    *   **Transformer Stream**: Captures long-range dependencies and global context.
    *   **Bottleneck Stream**: Preserves local anatomical details.
4.  **Gated Fusion Layer**: Dynamically balances the contribution of both streams for the final prediction.

## 📂 Directory Structure

```
vcda_net/
├── src/
│   └── vcda_net/
│       ├── model/          # VCDA-Net architecture definition
│       ├── pipeline/       # End-to-end processing pipeline
│       ├── training/       # Training loops and loss functions
│       └── experiments/    # Experiment scripts (e.g., AD prediction)
├── train_vcda_net.sh       # Main training script
├── setup.py                # Package installation script
└── pyproject.toml          # Project configuration
```

## 📝 Naming Clarification

**Why "Density Attention" instead of "Saliency"?**
While technically a form of saliency map, we use the term **Density Attention** to emphasize that our method relies on the *density of voxel activations* in feature clusters, distinct from gradient-based "saliency" methods often used in visualization.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Anonymous Authors**  
*Submitted for Blind Review*
