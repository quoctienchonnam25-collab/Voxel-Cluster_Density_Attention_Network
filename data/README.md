# Package Data Directory

This directory contains data files used by the Hybrid Saliency V4 package experiments.

## Structure

```
data/
└── ad_prediction/
    ├── features/              # Regional features for AD/MCI prediction
    │   ├── regions/           # 32 regional feature files (.npy)
    │   ├── metadata.csv       # Sample metadata
    │   ├── regional_features.npy  # Combined features
    │   └── extraction_summary.json
    │
    └── models/                # Trained regional predictors
        ├── models/            # 32 regional model files (.pkl)
        ├── bias_correction/   # Bias correction parameters
        ├── predictions/       # Training predictions
        ├── data_split.json    # Train/val/test split
        └── training_summary.json
```

## Contents

### Features Directory (~74MB)

**Source**: Extracted from ADNI AD/MCI cohort using Hybrid Saliency V4 model

**Files**:
- `regions/region_XX_features.npy`: Features for each of 32 brain regions
- `metadata.csv`: Subject information (age, sex, diagnosis, etc.)
- `regional_features.npy`: Combined features [n_samples, 32, 512]
- `extraction_summary.json`: Extraction statistics

**Samples**: 584 ADNI AD/MCI patients

### Models Directory (~3.6MB)

**Source**: Trained on healthy controls (1710 samples)

**Files**:
- `models/region_XX_model.pkl`: Trained predictor for each region (scikit-learn)
- `bias_correction/regional_bias_correction_params.json`: β₀ and β₁ for each region
- `predictions/`: Training/validation/test predictions
- `data_split.json`: Train/val/test indices
- `training_summary.json`: Training statistics

**Model Type**: Ridge Regression (scikit-learn)

## Usage

### In Scripts

The experiment scripts automatically use these paths:

```bash
cd src/hybrid_saliency_v4/experiments/ad_prediction
bash run_experiment.sh
```

### Manual Access

```python
from pathlib import Path

# Get package root
package_root = Path(__file__).parent.parent.parent.parent.parent

# Access data
features_dir = package_root / "data/ad_prediction/features/regions"
models_dir = package_root / "data/ad_prediction/models/models"
```

## Data Provenance

### Features
- **Extracted**: 2026-01-XX
- **Model**: Hybrid Saliency V4
- **Checkpoint**: `synthesis_v4_20260119_113843_topk512/checkpoints/best_model.pth`
- **Dataset**: ADNI AD/MCI (584 samples)

### Models
- **Trained**: 2026-01-XX
- **Dataset**: Healthy controls (1710 samples)
  - ABIDE1: 427 samples
  - ADNI: 146 samples
  - IXI: 449 samples
  - OASIS: 487 samples
  - PPMI: 201 samples
- **Split**: 70% train, 15% val, 15% test
- **Algorithm**: Ridge Regression with bias correction

## Size Information

```
Total size: ~78MB

features/
  - regions/: 73MB (32 × ~2.3MB per region)
  - metadata.csv: 90KB
  - regional_features.npy: 38MB
  
models/
  - models/: 2.5MB (32 × ~80KB per model)
  - bias_correction/: 10KB
  - predictions/: 1MB
```

## Version Control

⚠️ **Note**: These data files are **NOT** tracked by git due to their size.

They are included in the package distribution but excluded from version control via `.gitignore`.

## Updating Data

To update the data files:

1. **Features**: Re-run feature extraction on new data
   ```bash
   python extract_v4_regional_features_adni.py
   ```

2. **Models**: Re-train regional predictors
   ```bash
   python train_v4_regional_predictors.py
   ```

3. **Copy to package**:
   ```bash
   cp -r new_features/* data/ad_prediction/features/
   cp -r new_models/* data/ad_prediction/models/
   ```

## Data Integrity

### Checksums (Optional)

To verify data integrity:

```bash
cd data/ad_prediction
find . -type f -name "*.npy" -o -name "*.pkl" | sort | xargs md5sum > checksums.md5
```

To verify:
```bash
md5sum -c checksums.md5
```

## License

The data files follow the same license as the package. See LICENSE file in package root.

## References

- **ADNI**: http://adni.loni.usc.edu/
- **Hybrid Saliency V4**: See package documentation
- **Regional Analysis**: See experiments/ad_prediction/README.md

## Notes

1. Data is self-contained within package
2. No external dependencies required
3. Paths are relative to package root
4. Compatible with package distribution
5. Can be regenerated from source data if needed
