# Voxel-Cluster Density Attention Network (VCDA-Net)

**Brain Age Prediction with Voxel-Cluster Density Attention and Multi-Edge Graph Fusion**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Abstract

We propose the **Voxel-Cluster Density Attention (VCDA)** mechanism to quantify neurodegeneration through spatial-density variations. Our **adaptive dual-stream system** integrates global UNet features with local VCDA graphs via a **Multi-Edge Attention** mechanism. This mechanism incorporates dot-product, Euclidean distance, and cosine similarity to prioritize diagnostic connectivity. 

By combining a Transformer with **Gated Fusion**, the model effectively captures early micro-scale changes. Furthermore, our **accelerated-aging framework** accurately identifies global and regional Brain Age Gaps (BAGs) across 32 structures, enabling quantitative stratification from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD).

## 🚀 Key Contributions

The main contributions of this work are:

1. **VCDA Mechanism**: A novel mechanism for enhanced sensitivity to early-stage neurodegeneration by analyzing voxel density variations.
2. **Dual-Stream Architecture**: A robust system using **Gated Fusion** and **Multi-Edge Attention** to capture complex inter-regional dependencies.
3. **Clinical Framework**: A comprehensive framework tracking global and regional BAG across 32 structures to characterize pathological shifts in the Alzheimer's spectrum.

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
2.  **VCDA Module**: Computes density attention maps based on voxel activation density.
3.  **Dual Streams**:
    *   **Transformer Stream**: Captures long-range dependencies and global context using Multi-Edge Attention.
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Anonymous Authors**  
*Submitted for Blind Review*
