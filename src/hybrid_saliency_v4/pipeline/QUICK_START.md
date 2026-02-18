# Quick Start - Pipeline Module

## 🚀 One-Command Pipeline

```bash
cd src/hybrid_saliency_v4/pipeline
bash run_complete_pipeline.sh
```

This runs:
1. ✅ Extract regional features (Step 1)
2. ✅ Train regional predictors (Step 2)

## 📁 Output

```
pipeline_output/
├── regional_features/      # Features [n_samples, 32, 512]
└── regional_predictors/    # 32 trained models + bias correction
```

## 🔧 Individual Steps

### Extract Features

```bash
python -m hybrid_saliency_v4.pipeline.extract_regional_features \
    --checkpoint saliency_runs/saliency_enhanced_20260216_011913/checkpoints/best_model.pth \
    --unet_checkpoint unet_checkpoint/IXI_3dunet_best_model.pth \
    --data_dir /path/to/data \
    --metadata /path/to/metadata.csv \
    --output_dir output/features
```

### Train Models

```bash
python -m hybrid_saliency_v4.pipeline.train_regional_predictors \
    --features_dir output/features \
    --output_dir output/models
```

### Generate Predictions

```bash
python -m hybrid_saliency_v4.pipeline.generate_predictions \
    --features_dir new_data/features/regions \
    --metadata_file new_data/metadata.csv \
    --models_dir trained_models/models \
    --bias_params_file trained_models/bias_correction/regional_bias_correction_params.json \
    --output_file predictions.csv
```

## 📊 Expected Output

### Training Metrics
- Train MAE: ~3.5 years
- Test MAE: ~4.0 years  
- Test R²: ~0.80

### Files Generated
- 32 trained models (.pkl)
- Bias correction parameters
- Training results CSV
- Summary statistics JSON

## 🔗 Integration

Use trained models in experiments:

```bash
# Copy models to experiments
cp -r pipeline_output/regional_predictors/* data/ad_prediction/models/

# Run AD prediction
cd src/hybrid_saliency_v4/experiments/ad_prediction
bash run_experiment.sh
```

## 📖 Full Documentation

See `README.md` for complete documentation.

---

**Model**: Hybrid Saliency V4  
**Checkpoint**: `saliency_enhanced_20260216_011913`
