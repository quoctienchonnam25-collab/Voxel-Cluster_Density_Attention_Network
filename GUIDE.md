# 📖 User Guide — Hybrid Saliency V4

**Brain Age Prediction with Saliency Maps and Gated Fusion**

> Package: `hybrid-saliency-v4` · Version: `4.0.2` · Python ≥ 3.8

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Directory Structure](#3-directory-structure)
4. [Workflow](#4-workflow)
5. [Step 0 — Train the Main Model](#5-step-0--train-the-main-model)
6. [Step 1 — Extract Regional Features](#6-step-1--extract-regional-features)
7. [Step 2 — Train Regional Predictors](#7-step-2--train-regional-predictors)
8. [Step 3 — Generate Predictions on New Data](#8-step-3--generate-predictions-on-new-data)
9. [Run the Full Pipeline with One Command](#9-run-the-full-pipeline-with-one-command)
10. [Model Architecture](#10-model-architecture)
11. [Input Data Format](#11-input-data-format)
12. [Output Format](#12-output-format)
13. [Configuration Parameters](#13-configuration-parameters)
14. [Troubleshooting](#14-troubleshooting)
15. [Integration with Experiments](#15-integration-with-experiments)

---

## 1. Overview

The **Hybrid Saliency V4** package provides a complete end-to-end pipeline for regional brain age prediction from 3D MRI scans.

### Key Features

| Feature | Description |
|---------|-------------|
| **Saliency Map** | Uses activation magnitude (no gradients needed) — faster than GradCAM |
| **Gated Fusion** | Adaptive combination of two feature streams (V4 innovation) |
| **32 Brain Regions** | Independent age prediction per region, then ensemble |
| **Bias Correction** | Linear correction to reduce age prediction bias |
| **Ridge Regression** | Lightweight, fast predictor — no GPU required |

### Saliency Map vs GradCAM

```
Saliency Map (Ours):
  channel_importance = feature_maps.abs().mean(dim=(2,3,4))  ← No backward pass!
  weighted_maps = feature_maps * channel_importance

GradCAM (NOT used):
  gradients = compute_gradients(...)  ← Requires backward pass → Slower
```

---

## 2. Installation

### 2.1 System Requirements

- Python ≥ 3.8
- CUDA (recommended; CPU also supported)
- RAM ≥ 16 GB
- GPU VRAM ≥ 8 GB (for feature extraction)

### 2.2 Install the Package

```bash
# Navigate to the package directory
cd /path/to/hybrid_saliency_v4_package

# Install in development mode (recommended)
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### 2.3 Verify Installation

```python
from hybrid_saliency_v4.model import HybridSaliencyEnhanced
import torch

model = HybridSaliencyEnhanced(
    unet_checkpoint='src/hybrid_saliency_v4/checkpoints/IXI_3dunet_best_model.pth',
    num_regions=32,
    embedding_dim=256
)
print("✓ Import successful!")

# Test inference
model.eval()
with torch.no_grad():
    mri = torch.randn(1, 1, 128, 128, 128)
    predicted_age, gate_mean = model(mri)
    print(f"Predicted age: {predicted_age.item():.1f} years")
```

### 2.4 Set PYTHONPATH

If you did not install via `pip install -e .`, add the source directory manually:

```bash
export PYTHONPATH="/path/to/hybrid_saliency_v4_package/src:$PYTHONPATH"
```

---

## 3. Directory Structure

```
hybrid_saliency_v4_package/
│
├── 📄 README.md                        ← Technical overview (English)
├── 📄 GUIDE.md                         ← This file (comprehensive guide)
├── 📄 CHANGELOG.md                     ← Version history
├── 📄 requirements.txt                 ← Dependencies
├── 📄 pyproject.toml                   ← Package configuration
├── 📄 setup.py                         ← Build script
│
├── 🚀 train_saliency_v4.sh             ← Main model training script
│
├── src/
│   └── hybrid_saliency_v4/
│       ├── __init__.py
│       │
│       ├── model/                      ← Model architecture
│       │   ├── hybrid_saliency_enhanced_v4.py   ← Main model
│       │   └── components/             ← Sub-components (GNN, Transformer, ...)
│       │
│       ├── training/                   ← Training logic
│       │   ├── train.py                ← Training entry point
│       │   └── loss_functions.py       ← Loss functions (Huber, MSE, ...)
│       │
│       ├── pipeline/                   ← 🔑 Regional brain age pipeline
│       │   ├── extract_regional_features.py    ← Step 1: Feature extraction
│       │   ├── train_regional_predictors.py    ← Step 2: Train Ridge models
│       │   ├── generate_predictions.py         ← Step 3: Generate predictions
│       │   ├── run_complete_pipeline.sh        ← Run the full pipeline
│       │   ├── README.md               ← Pipeline documentation
│       │   └── QUICK_START.md          ← Quick start guide
│       │
│       ├── experiments/                ← Application experiments
│       │   └── ad_prediction/          ← Alzheimer's Disease prediction
│       │
│       ├── configs/
│       │   └── default_config.yaml     ← Default configuration
│       │
│       └── checkpoints/
│           └── IXI_3dunet_best_model.pth   ← Pretrained UNet
│
├── saliency_runs/                      ← Training outputs (auto-created)
│   └── saliency_enhanced_YYYYMMDD_HHMMSS/
│       └── checkpoints/
│           └── best_model.pth          ← Best checkpoint
│
└── pipeline_output/                    ← Pipeline outputs (auto-created)
    ├── regional_features/
    └── regional_predictors/
```

---

## 4. Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                       OVERALL WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MRI Data (healthy controls)                                    │
│         │                                                       │
│         ▼                                                       │
│  [Step 0] Train the main model                                  │
│         train_saliency_v4.sh                                    │
│         → saliency_runs/.../best_model.pth                      │
│         │                                                       │
│         ▼                                                       │
│  [Step 1] Extract regional brain features                       │
│         extract_regional_features.py                            │
│         → regional_features.npy  [N, 32, 512]                  │
│         │                                                       │
│         ▼                                                       │
│  [Step 2] Train regional predictors                             │
│         train_regional_predictors.py                            │
│         → 32 × region_XX_model.pkl                             │
│         → bias_correction_params.json                           │
│         │                                                       │
│         ▼                                                       │
│  [Step 3] Generate predictions on new data (e.g. AD patients)  │
│         generate_predictions.py                                 │
│         → predictions.csv                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Step 0 — Train the Main Model

> **Skip this step if you already have a checkpoint** in `saliency_runs/`.

### 5.1 Prepare Data

You need:
- A directory containing MRI files (`.nii` or `.nii.gz`)
- A metadata CSV file with columns: `FILENAME`, `AGE`, `SEX`, `DATASET`, `SUBJECT_ID`

Example metadata CSV:
```csv
FILENAME,AGE,SEX,DATASET,SUBJECT_ID
sub001_T1w.nii.gz,45.3,M,IXI,IXI001
sub002_T1w.nii.gz,62.1,F,ADNI,ADNI002
...
```

### 5.2 Edit the Training Script

Open `train_saliency_v4.sh` and update the paths:

```bash
# Edit these lines:
DATA_DIR="/path/to/mri/directory"
METADATA_CSV="/path/to/metadata.csv"
UNET_CHECKPOINT="${SCRIPT_DIR}/src/hybrid_saliency_v4/checkpoints/IXI_3dunet_best_model.pth"
```

### 5.3 Run Training

```bash
cd /path/to/hybrid_saliency_v4_package
bash train_saliency_v4.sh
```

### 5.4 Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 300 | Number of training epochs |
| `BATCH_SIZE` | 4 | Batch size |
| `LEARNING_RATE` | 1e-3 | Learning rate |
| `LOSS_TYPE` | huber | Loss function (huber/mse/mae) |
| `FREEZE_UNET` | `--unfreeze_unet` | Whether to fine-tune UNet |
| `PATIENCE` | 25 | Early stopping patience |
| `DEVICE` | cuda | Device (cuda/cpu) |

### 5.5 Training Output

```
saliency_runs/
└── saliency_enhanced_20260216_011913/
    ├── checkpoints/
    │   ├── best_model.pth          ← Use this checkpoint for the pipeline
    │   └── last_model.pth
    └── logs/
        └── training.log
```

Monitor training progress:
```bash
tensorboard --logdir=saliency_runs
```

---

## 6. Step 1 — Extract Regional Features

Script: `src/hybrid_saliency_v4/pipeline/extract_regional_features.py`

### 6.1 How It Works

```
MRI Input [1, 1, 128, 128, 128]
    ↓
UNet Feature Extractor
    ↓
Forward Hook at the gated_fusion layer
    ↓
features_concat [1, 32, 512]
    ↓  (256-dim original + 256-dim saliency)
Save to .npy file
```

### 6.2 Run Feature Extraction

```bash
python -m hybrid_saliency_v4.pipeline.extract_regional_features \
    --checkpoint saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth \
    --unet_checkpoint src/hybrid_saliency_v4/checkpoints/IXI_3dunet_best_model.pth \
    --data_dir /path/to/mri/directory \
    --metadata /path/to/metadata.csv \
    --output_dir pipeline_output/regional_features \
    --device cuda
```

### 6.3 Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--checkpoint` | ✅ | Path to the trained model checkpoint |
| `--unet_checkpoint` | ✅ | Path to the pretrained UNet checkpoint |
| `--data_dir` | ✅ | Directory containing MRI files |
| `--metadata` | ✅ | Metadata CSV file |
| `--output_dir` | ✅ | Output directory |
| `--device` | ❌ | `cuda` or `cpu` (default: `cuda`) |

### 6.4 Output

```
pipeline_output/regional_features/
├── regional_features.npy          ← [N_samples, 32, 512] — Main file
├── metadata.csv                   ← Processed sample information
├── extraction_summary.json        ← Summary statistics
└── regions/
    ├── region_00_features.npy     ← [N_samples, 512] — Region 0
    ├── region_01_features.npy     ← [N_samples, 512] — Region 1
    ├── ...
    └── region_31_features.npy     ← [N_samples, 512] — Region 31
```

**Understanding the shape `[N, 32, 512]`:**
- `N` = number of MRI samples
- `32` = number of brain regions
- `512` = 256 (original features) + 256 (saliency features)

---

## 7. Step 2 — Train Regional Predictors

Script: `src/hybrid_saliency_v4/pipeline/train_regional_predictors.py`

### 7.1 How It Works

```
Region features [N, 512]
    ↓
Train/Valid/Test split (70/15/15)
    ↓
Ridge Regression (α=1.0) per region
    ↓
Compute Bias Correction: true_age = β₀ + β₁ × predicted_age
    ↓
Save 32 model .pkl files + bias_params.json
```

### 7.2 Run Training

```bash
python -m hybrid_saliency_v4.pipeline.train_regional_predictors \
    --features_dir pipeline_output/regional_features \
    --output_dir pipeline_output/regional_predictors \
    --alpha 1.0 \
    --random_seed 42
```

### 7.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--features_dir` | — | Features directory from Step 1 |
| `--output_dir` | — | Output directory for models |
| `--alpha` | `1.0` | Ridge regularization strength |
| `--random_seed` | `42` | Random seed for reproducibility |

**Notes on `--alpha`:**
- Smaller values (0.1) → less regularization → risk of overfitting
- Larger values (10.0) → more regularization → risk of underfitting
- Default `1.0` generally works well

### 7.4 Output

```
pipeline_output/regional_predictors/
├── models/
│   ├── region_00_model.pkl        ← Ridge model for region 0
│   ├── region_01_model.pkl
│   ├── ...
│   └── region_31_model.pkl        ← 32 models total
│
├── bias_correction/
│   └── regional_bias_correction_params.json   ← β₀, β₁ per region
│
├── data_split.json                ← Train/valid/test indices
├── regional_predictors_results.csv ← Per-region metrics
└── training_summary.json          ← Aggregate statistics
```

### 7.5 Expected Performance

On healthy controls (n≈1710):

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **MAE** | ~3.5 years | ~4.0 years | ~4.0 years |
| **R²** | ~0.85 | ~0.80 | ~0.80 |

### 7.6 Reading Results

```python
import pandas as pd

# View per-region metrics
results = pd.read_csv('pipeline_output/regional_predictors/regional_predictors_results.csv')
print(results[['region_idx', 'mae_test', 'r2_test']].sort_values('mae_test'))

# Find the best-performing region
best_region = results.loc[results['mae_test'].idxmin()]
print(f"Best region: {best_region['region_idx']} (MAE={best_region['mae_test']:.2f})")
```

---

## 8. Step 3 — Generate Predictions on New Data

Script: `src/hybrid_saliency_v4/pipeline/generate_predictions.py`

### 8.1 Prerequisites

- Features extracted for the new dataset (run Step 1 on the new data)
- Trained models from Step 2

### 8.2 Run Prediction

```bash
# Step 1: Extract features for new data (e.g. AD patients)
python -m hybrid_saliency_v4.pipeline.extract_regional_features \
    --checkpoint saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth \
    --unet_checkpoint src/hybrid_saliency_v4/checkpoints/IXI_3dunet_best_model.pth \
    --data_dir /path/to/new/data \
    --metadata /path/to/new_metadata.csv \
    --output_dir pipeline_output/new_data_features

# Step 2: Generate predictions
python -m hybrid_saliency_v4.pipeline.generate_predictions \
    --features_dir pipeline_output/new_data_features/regions \
    --metadata_file pipeline_output/new_data_features/metadata.csv \
    --models_dir pipeline_output/regional_predictors/models \
    --bias_params_file pipeline_output/regional_predictors/bias_correction/regional_bias_correction_params.json \
    --output_file pipeline_output/predictions.csv \
    --num_regions 32
```

### 8.3 Parameters

| Parameter | Description |
|-----------|-------------|
| `--features_dir` | The `regions/` directory from Step 1 (contains 32 .npy files) |
| `--metadata_file` | Metadata file for the new dataset |
| `--models_dir` | The `models/` directory from Step 2 |
| `--bias_params_file` | Bias correction JSON file from Step 2 |
| `--output_file` | Output CSV file path |
| `--num_regions` | Number of brain regions (default: 32) |

### 8.4 Output

```
pipeline_output/
├── predictions.csv                ← Full predictions for all regions
└── predictions_statistics.json    ← Summary statistics
```

**Structure of `predictions.csv`:**

```
subject_id | age | region_00_raw | region_00_corrected | ... | ensemble_mean | ensemble_median | brain_age_gap
```

- `region_XX_raw`: Raw Ridge regression prediction
- `region_XX_corrected`: After bias correction
- `ensemble_mean`: Mean of 32 regions (final prediction)
- `brain_age_gap`: `ensemble_mean - age` (brain aging index)

---

## 9. Run the Full Pipeline with One Command

Script: `src/hybrid_saliency_v4/pipeline/run_complete_pipeline.sh`

### 9.1 Configure the Script

Open the file and edit the CONFIGURATION section:

```bash
# Open the file
nano src/hybrid_saliency_v4/pipeline/run_complete_pipeline.sh

# Edit these lines:
CHECKPOINT="${PACKAGE_ROOT}/saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth"
UNET_CHECKPOINT="${PACKAGE_ROOT}/unet_checkpoint/IXI_3dunet_best_model.pth"
DATA_DIR="/path/to/mri/directory"
METADATA_CSV="/path/to/metadata.csv"
```

### 9.2 Run the Pipeline

```bash
cd src/hybrid_saliency_v4/pipeline
bash run_complete_pipeline.sh
```

### 9.3 What the Pipeline Does

1. ✅ Creates output directories
2. ✅ Activates the conda environment
3. ✅ **Step 1**: Extracts regional brain features
4. ✅ **Step 2**: Trains 32 regional predictors + bias correction
5. ✅ Prints the output structure and next steps

---

## 10. Model Architecture

### 10.1 Overview

```
Input MRI [B, 1, 128, 128, 128]
    ↓
UNet Feature Extractor
    ├── features: [B, 32, 64, 64, 64]
    └── bottleneck: [B, 128, 16, 16, 16]
    ↓
┌─────────────────────────────────────┬──────────────────────────────────┐
│ Stream 1 (Transformer)              │ Stream 2 (Bottleneck)            │
├─────────────────────────────────────┼──────────────────────────────────┤
│ • Saliency Map Weighting            │ • 1×1×1 Conv (128→256)           │
│ • Gaussian Distance Matrix          │ • 3× ResNet Blocks               │
│ • Dual ResNet Paths:                │ • Global Average Pool            │
│   - Original: 256-dim               │ Output: [B, 256]                 │
│   - Saliency: 256-dim               │                                  │
│ • Graph Neural Network (GNN)        │                                  │
│ • Transformer Aggregation           │                                  │
│ Output: [B, 256]                    │                                  │
└─────────────────────────────────────┴──────────────────────────────────┘
    ↓
Gated Fusion (V4 Innovation)
    gate = sigmoid(W × [stream1, stream2])  ∈ [0, 1]
    fused = gate × transform(stream1) + (1-gate) × transform(stream2)
    Output: [B, 512]
    ↓
Prediction Head: 512 → 256 → 128 → 1
    ↓
Predicted Age (scalar)
```

### 10.2 How Regional Features Are Extracted

```python
# A forward hook is placed at the gated_fusion layer.
# The input to gated_fusion is features_concat [B, 32, 512],
# which holds 512-dim features for each of the 32 brain regions.

hook = model.gated_fusion.register_forward_hook(hook_fn)
_ = model(mri_tensor)
# features_concat: [B, 32, 512]
#   → 256 from the original path
#   → 256 from the saliency path
```

### 10.3 Saliency Map Computation

```python
# Step 1: Compute channel importance from activation magnitude
channel_importance = feature_maps.abs().mean(dim=(2, 3, 4), keepdim=True)

# Step 2: Weight feature maps
weighted_maps = feature_maps * channel_importance

# Step 3: Normalize
weighted_maps = F.normalize(weighted_maps, p=2, dim=(2, 3, 4))

# Step 4: Top-K coordinates + Gaussian distance matrix
# top_k=128, sigma=10.0, matrix_resize=64
```

---

## 11. Input Data Format

### 11.1 MRI Files

- **Format**: NIfTI (`.nii` or `.nii.gz`)
- **Size**: Any (automatically resized to `128×128×128`)
- **Values**: Any range (automatically normalized to `[0, 1]`)

### 11.2 Metadata CSV File

**Required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `FILENAME` | string | MRI filename (relative to `data_dir`) |
| `AGE` | float | Chronological age of the subject |
| `SEX` | string | Sex (`M`/`F`) |

**Optional columns:**

| Column | Type | Description |
|--------|------|-------------|
| `DATASET` | string | Dataset name (IXI, ADNI, ...) |
| `SUBJECT_ID` | string | Subject identifier |

**Example:**
```csv
FILENAME,AGE,SEX,DATASET,SUBJECT_ID
IXI001-Guys-0828-T1.nii.gz,45.3,M,IXI,IXI001
ADNI_002_S_0413_MR.nii.gz,72.1,F,ADNI,ADNI002
```

> **Note**: Column names are case-insensitive — the script automatically converts them to uppercase.

---

## 12. Output Format

### 12.1 regional_features.npy

```python
import numpy as np
features = np.load('pipeline_output/regional_features/regional_features.npy')
# Shape: [N_samples, 32, 512]
# features[i, j, :] = 512-dim feature vector for sample i, region j
```

### 12.2 regional_predictors_results.csv

```
region_idx | mae_train | rmse_train | r2_train | r_train | mae_valid | ... | mae_test | r2_test
0          | 3.21      | 4.15       | 0.87     | 0.93    | 3.89      | ... | 3.95     | 0.82
1          | 3.45      | 4.32       | 0.85     | 0.92    | 4.12      | ... | 4.08     | 0.80
...
```

### 12.3 regional_bias_correction_params.json

```json
[
  {
    "region_idx": 0,
    "beta_0_intercept": 12.45,
    "beta_1_slope": 0.73
  },
  ...
]
```

Correction formula: `corrected_age = β₀ + β₁ × predicted_age`

### 12.4 predictions.csv

```
subject_id | age | region_00_raw | region_00_corrected | ... | ensemble_mean | brain_age_gap
sub001     | 45  | 47.2          | 45.8                | ... | 46.1          | 1.1
sub002     | 72  | 68.5          | 71.2                | ... | 70.8          | -1.2
```

---

## 13. Configuration Parameters

### 13.1 Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_regions` | 32 | Number of brain regions |
| `embedding_dim` | 256 | Embedding dimension |
| `resnet_depth` | resnet18 | ResNet depth |
| `top_k` | 128 | Top-K voxels for saliency |
| `sigma` | 10.0 | Sigma for Gaussian distance |
| `matrix_resize` | 64 | Resized matrix size |

### 13.2 GNN Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_num` | 31 | Number of graph edges |
| `hidden_channels` | 64 | GNN hidden channels |
| `num_gnn_layers` | 3 | Number of GNN layers |
| `use_edge_attention` | True | Use edge attention |

### 13.3 Transformer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `transformer_d_model` | 256 | Model dimension |
| `transformer_nhead` | 8 | Number of attention heads |
| `transformer_num_layers` | 3 | Number of transformer layers |

### 13.4 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 300 | Number of epochs |
| `batch_size` | 4 | Batch size |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 1e-4 | Weight decay |
| `dropout` | 0.3 | Dropout rate |
| `loss_type` | huber | Loss function type |
| `huber_delta` | 1.0 | Huber loss delta |
| `train_ratio` | 0.70 | Training set ratio |
| `val_ratio` | 0.15 | Validation set ratio |
| `patience` | 25 | Early stopping patience |

---

## 14. Troubleshooting

### ❌ CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead of GPU
--device cpu

# Or reduce batch size during training
BATCH_SIZE=2
```

### ❌ Import Error: Module Not Found

```
ModuleNotFoundError: No module named 'hybrid_saliency_v4'
```

**Solution:**
```bash
# Add src to PYTHONPATH
export PYTHONPATH="/path/to/hybrid_saliency_v4_package/src:$PYTHONPATH"

# Or install the package
pip install -e /path/to/hybrid_saliency_v4_package
```

### ❌ Hook Not Capturing Features

```
Warning: features_container is empty
```

**Solution:**
- Verify the model architecture contains a `gated_fusion` layer
- Ensure the checkpoint is the correct version (V4, not V3)

```python
# Check
print([name for name, _ in model.named_modules()])
# 'gated_fusion' must appear in the list
```

### ❌ MRI File Not Found

```
FileNotFoundError: MRI file not found
```

**Solution:**
- Check the `FILENAME` column in the metadata CSV
- Paths in `FILENAME` must be relative to `data_dir`

```python
# Verify
import pandas as pd
from pathlib import Path

df = pd.read_csv('metadata.csv')
data_dir = Path('/path/to/mri')
for _, row in df.iterrows():
    path = data_dir / row['FILENAME']
    if not path.exists():
        print(f"Missing: {path}")
```

### ❌ Incompatible Checkpoint

```
RuntimeError: Error(s) in loading state_dict
```

**Solution:**
- Make sure you are using the correct Hybrid Saliency V4 checkpoint (not V3/GradCAM)
- Inspect the config stored in the checkpoint:

```python
import torch
ckpt = torch.load('best_model.pth', map_location='cpu', weights_only=False)
print(ckpt.get('config', {}))
```

### ❌ Conda Environment Not Activating

```
source: ~/devin/programs/anaconda3/bin/activate: No such file or directory
```

**Solution:**
Update the conda path in the scripts:
```bash
# Find the correct conda path
which conda
conda info --base

# Edit run_complete_pipeline.sh and train_saliency_v4.sh
source /correct/path/to/anaconda3/bin/activate base
```

---

## 15. Integration with Experiments

### 15.1 Use Models in AD Prediction

```bash
# Copy models to the experiments directory
cp -r pipeline_output/regional_predictors/* \
    src/hybrid_saliency_v4/experiments/ad_prediction/data/models/

# Run the AD prediction experiment
cd src/hybrid_saliency_v4/experiments/ad_prediction
bash run_experiment.sh
```

### 15.2 Use in Python

```python
from hybrid_saliency_v4.experiments.ad_prediction.predict_regional_brain_age import (
    RegionalBrainAgePredictor
)

predictor = RegionalBrainAgePredictor(
    models_dir='pipeline_output/regional_predictors/models',
    bias_params_file='pipeline_output/regional_predictors/bias_correction/regional_bias_correction_params.json',
    num_regions=32
)

predictor.load_models()
predictor.load_bias_params()

# Predict
all_raw, all_corrected = predictor.predict_all_regions(
    features_dir='pipeline_output/new_data_features/regions',
    n_samples=100,
    apply_correction=True
)
```

### 15.3 Compute Brain Age Gap (BAG)

```python
import pandas as pd

predictions = pd.read_csv('pipeline_output/predictions.csv')

# Brain Age Gap = Predicted Age - Chronological Age
predictions['BAG'] = predictions['ensemble_mean'] - predictions['age']

# Summary statistics
print(f"Mean BAG: {predictions['BAG'].mean():.2f} years")
print(f"BAG std:  {predictions['BAG'].std():.2f} years")

# Group by dataset
print(predictions.groupby('dataset')['BAG'].agg(['mean', 'std']))
```

---

## 📝 Important Notes

1. **Default checkpoint**: `saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth`
2. **UNet checkpoint**: `src/hybrid_saliency_v4/checkpoints/IXI_3dunet_best_model.pth` (pretrained on the IXI dataset)
3. **Training data for regional predictors**: Use healthy controls (not patients) to ensure accurate bias correction
4. **Reproducibility**: Always use `--random_seed 42` for consistent results
5. **Ensemble**: The final prediction is the mean of all 32 regional predictions, not any single region

---

## 🔗 Related Documentation

- `README.md` — Technical overview (English)
- `src/hybrid_saliency_v4/pipeline/README.md` — Detailed pipeline documentation
- `src/hybrid_saliency_v4/pipeline/QUICK_START.md` — Quick start guide
- `CHANGELOG.md` — Version history
- `TRAINING_SCRIPTS_README.md` — Training script guide

---

**Last updated**: 2026-02-18  
**Package version**: 4.0.2
