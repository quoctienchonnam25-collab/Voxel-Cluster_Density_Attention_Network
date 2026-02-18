# Hybrid Saliency V4 Package - Experiments

## Package Structure

```
hybrid_saliency_v4_package/
├── src/
│   └── hybrid_saliency_v4/
│       ├── model/                    # Model architecture
│       │   ├── components/           # Modular components
│       │   └── hybrid_saliency_enhanced_v4.py
│       ├── training/                 # Training scripts
│       │   └── train.py
│       └── experiments/              # 🆕 Experimental applications
│           └── ad_prediction/        # AD brain age gap prediction
│               ├── __init__.py
│               ├── README.md
│               ├── config.json
│               ├── predict_regional_brain_age.py
│               ├── run_experiment.sh
│               └── results/          # Output directory
└── train_saliency_v4.sh             # Main training script
```

## Experiments Module

### Purpose

The `experiments/` module contains practical applications of the trained Hybrid Saliency V4 model for specific research tasks.

### Current Experiments

#### 1. AD Brain Age Gap Prediction (`ad_prediction/`)

**Objective**: Predict brain age gap in Alzheimer's Disease and MCI patients

**Components**:
- `predict_regional_brain_age.py`: Main prediction script (refactored OOP design)
- `run_experiment.sh`: Bash wrapper for easy execution
- `config.json`: Experiment configuration
- `README.md`: Detailed documentation

**Key Features**:
- Regional analysis (32 brain regions)
- VNN-style bias correction
- Ensemble predictions
- Comprehensive statistics

**Usage**:
```bash
cd src/hybrid_saliency_v4/experiments/ad_prediction
bash run_experiment.sh
```

## Design Principles

### 1. Modularity
- Each experiment is self-contained in its own directory
- Reusable components are factored into classes
- Clear separation between data, models, and experiments

### 2. Reproducibility
- Configuration files for all parameters
- Detailed documentation
- Version-controlled paths and dependencies

### 3. Extensibility
- Easy to add new experiments
- Consistent API across experiments
- Shared utilities can be added to parent module

## Adding New Experiments

To add a new experiment:

1. Create a new directory under `experiments/`
2. Add `__init__.py` with module description
3. Create main script with argument parsing
4. Add `run_experiment.sh` wrapper
5. Write `README.md` documentation
6. Add `config.json` for parameters

Example structure:
```
experiments/
└── your_experiment/
    ├── __init__.py
    ├── README.md
    ├── config.json
    ├── main_script.py
    ├── run_experiment.sh
    └── results/
```

## Data Dependencies

### External Data (Not in Package)

The experiments rely on external data located in:
```
/media/devin/WORK/devin/tien/src/brain_age_prediction/
├── regional_features_v4_adni_ad_mci/    # Extracted features
├── regional_predictors_v4_hc1710/       # Trained models
└── FINAL_PREDICTIONS_MASTER/            # Reference results
```

### Why External?

- Large data files (models, features) are not version-controlled
- Shared across multiple scripts and experiments
- Easier to update independently

## Integration with Main Package

The experiments module integrates seamlessly with the main package:

1. **Model Loading**: Uses the same model architecture from `model/`
2. **Feature Extraction**: Can use training utilities from `training/`
3. **PYTHONPATH**: Scripts set up proper Python path to import package modules

## Future Experiments

Potential experiments to add:

1. **MCI Progression Prediction**: Predict conversion from MCI to AD
2. **Regional Vulnerability Analysis**: Identify most affected brain regions
3. **Multi-Site Validation**: Test generalization across datasets
4. **Longitudinal Analysis**: Track brain age changes over time
5. **Biomarker Correlation**: Correlate brain age gap with clinical markers

## Notes

- All experiments should follow the same structure and conventions
- Document all dependencies and data requirements
- Include example outputs and expected results
- Provide clear error messages and validation

## Contact

For questions about experiments or to contribute new experiments, please refer to the main package documentation.
