# Hybrid Saliency V4 - Complete Training Pipeline

This module provides the complete pipeline for regional brain age prediction using the Hybrid Saliency V4 model.

## Overview

The pipeline consists of 3 main steps:

1. **Extract Regional Features** - Extract 32 regional feature vectors from trained model
2. **Train Regional Predictors** - Train Ridge regression models with bias correction
3. **Generate Predictions** - Apply trained models to new datasets

## Quick Start

### Run Complete Pipeline

```bash
cd src/hybrid_saliency_v4/pipeline
bash run_complete_pipeline.sh
```

This will:
- Extract features from healthy brain dataset
- Train 32 regional predictors
- Calculate bias correction parameters
- Save all outputs to `pipeline_output/`

### Individual Steps

#### Step 1: Extract Features

```bash
python -m hybrid_saliency_v4.pipeline.extract_regional_features \
    --checkpoint path/to/best_model.pth \
    --unet_checkpoint path/to/unet_checkpoint.pth \
    --data_dir path/to/mri/data \
    --metadata path/to/metadata.csv \
    --output_dir output/features \
    --device cuda
```

**Output**:
- `regional_features.npy` - [n_samples, 32, 512]
- `metadata.csv` - Sample information
- `regions/region_XX_features.npy` - Individual region files
- `extraction_summary.json` - Statistics

#### Step 2: Train Predictors

```bash
python -m hybrid_saliency_v4.pipeline.train_regional_predictors \
    --features_dir output/features \
    --output_dir output/models \
    --alpha 1.0 \
    --random_seed 42
```

**Output**:
- `models/region_XX_model.pkl` - 32 trained models
- `bias_correction/regional_bias_correction_params.json` - Bias parameters
- `data_split.json` - Train/valid/test split
- `regional_predictors_results.csv` - Training metrics
- `training_summary.json` - Aggregate statistics

#### Step 3: Generate Predictions

```bash
python -m hybrid_saliency_v4.pipeline.generate_predictions \
    --features_dir path/to/new/features \
    --metadata_file path/to/new/metadata.csv \
    --models_dir output/models/models \
    --bias_params_file output/models/bias_correction/regional_bias_correction_params.json \
    --output_file predictions.csv \
    --num_regions 32
```

**Output**:
- `predictions.csv` - Full predictions with all regions
- `predictions_statistics.json` - Summary statistics

## Configuration

### Model Checkpoint

The pipeline uses the trained Hybrid Saliency V4 model:

```bash
CHECKPOINT="saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth"
```

### Data Requirements

**For Training** (Step 1-2):
- MRI scans (NIfTI format, .nii or .nii.gz)
- Metadata CSV with columns: `FILENAME`, `AGE`, `SEX`, `DATASET`, `SUBJECT_ID`
- Healthy controls dataset (recommended: 1000+ samples)

**For Prediction** (Step 3):
- Regional features extracted from Step 1
- Metadata CSV with `age` column
- Trained models from Step 2

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 1.0 | Ridge regression regularization |
| `--random_seed` | 42 | Random seed for reproducibility |
| `--device` | cuda | Device (cuda or cpu) |
| `--num_regions` | 32 | Number of brain regions |

## Output Structure

```
pipeline_output/
├── regional_features/
│   ├── regional_features.npy          # [n_samples, 32, 512]
│   ├── metadata.csv                   # Sample metadata
│   ├── extraction_summary.json        # Extraction stats
│   └── regions/
│       ├── region_00_features.npy     # [n_samples, 512]
│       ├── region_01_features.npy
│       └── ... (32 files total)
│
└── regional_predictors/
    ├── models/
    │   ├── region_00_model.pkl        # Trained Ridge model
    │   ├── region_01_model.pkl
    │   └── ... (32 files total)
    │
    ├── bias_correction/
    │   └── regional_bias_correction_params.json
    │
    ├── data_split.json                # Train/valid/test indices
    ├── regional_predictors_results.csv # Training metrics
    └── training_summary.json          # Aggregate statistics
```

## Architecture

### Feature Extraction

Features are extracted from the `features_concat` layer of the model:

```
Input MRI [128³]
    ↓
UNet Feature Extraction
    ↓
32 Regional Features
    ├── Original path: 256-dim
    └── Saliency path: 256-dim
    ↓
Concatenate → 512-dim per region
    ↓
Output: [32, 512]
```

### Regional Predictors

Each region has its own Ridge regression model:

```
Region Features [512-dim]
    ↓
Ridge Regression (α=1.0)
    ↓
Predicted Age
    ↓
Bias Correction (β₀ + β₁ × pred)
    ↓
Corrected Age
```

### Ensemble Prediction

Final prediction combines all 32 regions:

```
32 Regional Predictions
    ↓
Ensemble (mean/median)
    ↓
Final Brain Age
```

## Model Architecture Differences

This pipeline is adapted for the **Hybrid Saliency V4** model which differs from the original GradCAM V4:

| Feature | GradCAM V4 | Saliency V4 |
|---------|------------|-------------|
| **Fusion** | Concatenation | Gated Fusion |
| **Architecture** | Dual-stream | Dual-stream + Gated Attention |
| **Feature Dim** | 512 (256+256) | 512 (256+256) |
| **Hook Point** | `feature_proj` | `gated_fusion` |

## Usage Examples

### Example 1: Train on Healthy Controls

```bash
# Extract features
python -m hybrid_saliency_v4.pipeline.extract_regional_features \
    --checkpoint saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth \
    --unet_checkpoint unet_checkpoint/IXI_3dunet_best_model.pth \
    --data_dir /path/to/healthy_controls \
    --metadata /path/to/hc_metadata.csv \
    --output_dir pipeline_output/hc_features

# Train predictors
python -m hybrid_saliency_v4.pipeline.train_regional_predictors \
    --features_dir pipeline_output/hc_features \
    --output_dir pipeline_output/hc_models
```

### Example 2: Predict on AD Patients

```bash
# First extract features for AD patients
python -m hybrid_saliency_v4.pipeline.extract_regional_features \
    --checkpoint saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth \
    --unet_checkpoint unet_checkpoint/IXI_3dunet_best_model.pth \
    --data_dir /path/to/ad_patients \
    --metadata /path/to/ad_metadata.csv \
    --output_dir pipeline_output/ad_features

# Generate predictions
python -m hybrid_saliency_v4.pipeline.generate_predictions \
    --features_dir pipeline_output/ad_features/regions \
    --metadata_file pipeline_output/ad_features/metadata.csv \
    --models_dir pipeline_output/hc_models/models \
    --bias_params_file pipeline_output/hc_models/bias_correction/regional_bias_correction_params.json \
    --output_file pipeline_output/ad_predictions.csv
```

## Integration with Experiments

The trained models can be used directly in experiments:

```bash
# Copy trained models to experiments data directory
cp -r pipeline_output/hc_models/* data/ad_prediction/models/

# Run AD prediction experiment
cd src/hybrid_saliency_v4/experiments/ad_prediction
bash run_experiment.sh
```

## Performance

Expected training performance on healthy controls (n=1710):

| Metric | Value |
|--------|-------|
| **Train MAE** | ~3.5 years |
| **Valid MAE** | ~4.0 years |
| **Test MAE** | ~4.0 years |
| **Train R²** | ~0.85 |
| **Test R²** | ~0.80 |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:
```bash
--device cpu
```

### Import Errors

Ensure PYTHONPATH is set:
```bash
export PYTHONPATH="/path/to/package/src:$PYTHONPATH"
```

### Hook Not Capturing Features

Verify model architecture matches expected structure. Check that `gated_fusion` layer exists.

## References

- Original pipeline: `src/brain_age_prediction/extract_v4_regional_features.py`
- Model architecture: `src/hybrid_saliency_v4/model/hybrid_saliency_enhanced_v4.py`
- Experiments: `src/hybrid_saliency_v4/experiments/ad_prediction/`

## Notes

1. **Feature Extraction**: Uses forward hooks to capture intermediate features
2. **Bias Correction**: Linear correction fitted on training set
3. **Ensemble**: Mean of 32 regional predictions
4. **Reproducibility**: Fixed random seed ensures consistent splits

---

**Created**: 2026-02-17  
**Model**: Hybrid Saliency V4  
**Checkpoint**: `saliency_enhanced_20260216_011913`
