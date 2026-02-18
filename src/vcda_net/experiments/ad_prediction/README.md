# AD Brain Age Gap Prediction Experiment

## Overview

This experiment applies the **VCDA-Net** model to predict brain age gap in Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI) patients from the ADNI dataset.

## Methodology

### Pipeline Steps

1. **Feature Extraction**
   - Extract regional features from 32 brain regions using the trained V4 model
   - Features are extracted from preprocessed MRI scans

2. **Regional Prediction**
   - Apply 32 independent regional predictors (trained on healthy controls)
   - Each predictor estimates brain age for its corresponding region

3. **Bias Correction**
   - Apply VNN-style bias correction using parameters derived from healthy controls
   - Correction formula: `corrected_age = β₀ + β₁ × predicted_age`

4. **Ensemble Prediction**
   - Calculate ensemble predictions (mean, median, std) across all regions
   - Compute brain age gap: `delta_age = predicted_age - true_age`

### Dataset

- **Source**: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Cohorts**: AD and MCI patients
- **Regions**: 32 brain regions
- **Features**: 512-dimensional embeddings per region

## Usage

### Quick Start

```bash
cd /media/devin/WORK/devin/tien/src/brain_age_prediction/vcda_net_package/src/vcda_net/experiments/ad_prediction
bash run_experiment.sh
```

### Manual Execution

```bash
python -m vcda_net.experiments.ad_prediction.predict_regional_brain_age \
    --features_dir /path/to/regional_features/regions \
    --metadata_file /path/to/metadata.csv \
    --models_dir /path/to/regional_predictors/models \
    --bias_params_file /path/to/bias_correction_params.json \
    --output_file /path/to/output/predictions.csv \
    --num_regions 32
```

## Dependencies

### Data Requirements

1. **Regional Features** (`regional_features_v4_adni_ad_mci/`)
   - Extracted features for each brain region
   - Format: `.npy` files, one per region

2. **Trained Models** (`regional_predictors_v4_hc1710/models/`)
   - 32 trained regional predictors
   - Format: `.pkl` files (scikit-learn models)

3. **Bias Correction Parameters** (`regional_bias_correction_params.json`)
   - β₀ and β₁ coefficients for each region
   - Derived from healthy control validation set

4. **Metadata** (`metadata.csv`)
   - Subject information (age, sex, diagnosis, etc.)

### Python Packages

- numpy
- pandas
- scipy
- scikit-learn
- tqdm

## Output

### Files Generated

1. **`predictions_adni_ad_mci.csv`**
   - Comprehensive predictions for all samples
   - Columns:
     - Metadata: `subject_id`, `true_age`, `sex`, etc.
     - Per-region raw predictions: `region_XX_raw`
     - Per-region corrected predictions: `region_XX_corrected`
     - Per-region delta-age: `region_XX_delta_raw`, `region_XX_delta_corrected`
     - Ensemble predictions: `ensemble_raw_mean`, `ensemble_corrected_mean`
     - Ensemble delta-age: `ensemble_corrected_delta`

2. **`predictions_adni_ad_mci_statistics.json`**
   - Summary statistics
   - Ensemble performance metrics (MAE, RMSE, R, R²)
   - Per-region statistics

### Interpretation

- **Positive delta-age** (brain age > chronological age): Accelerated brain aging
- **Negative delta-age** (brain age < chronological age): Younger-appearing brain
- **Expected pattern**: AD patients show higher delta-age than MCI patients

## Results Location

All results are saved to:
```
vcda_net_package/experiments/ad_prediction/results/
```

## Related Scripts

- **Original script**: `/media/devin/WORK/devin/tien/src/brain_age_prediction/predict_all_regions_with_bias_correction_adni.py`
- **Feature extraction**: Scripts in `vcda_net/training/`
- **Model training**: `train_v4_regional_predictors.py`

## Citation

If you use this experiment pipeline, please cite:
- VCDA-Net Model
- ADNI Dataset (adni.loni.usc.edu)

## Notes

- Bias correction parameters are derived from healthy controls (1710 samples)
- The model was trained on multi-site data (ABIDE, ADNI, IXI, OASIS, PPMI)
- Regional analysis provides interpretability beyond global brain age prediction
