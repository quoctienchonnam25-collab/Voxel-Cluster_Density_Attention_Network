# Directory Structure Update

## Changes Made

Reorganized the results directory to be within the experiment module for better organization.

### Before
```
vcda_net_package/
├── src/vcda_net/
│   └── experiments/ad_prediction/
│       ├── predict_regional_brain_age.py
│       ├── run_experiment.sh
│       └── ...
└── experiments/                        ← OLD LOCATION
    └── ad_prediction/
        └── results/
```

### After
```
vcda_net_package/
└── src/vcda_net/
    └── experiments/ad_prediction/
        ├── predict_regional_brain_age.py
        ├── run_experiment.sh
        ├── visualize_results.py
        ├── config.json
        └── results/                    ← NEW LOCATION
            ├── predictions_adni_ad_mci.csv
            ├── predictions_adni_ad_mci_statistics.json
            ├── compare_results.py
            └── VALIDATION_REPORT.md
```

## Benefits

1. **Better Organization**: Results are now co-located with experiment code
2. **Cleaner Structure**: No top-level `experiments/` directory
3. **Self-Contained**: Everything related to AD prediction is in one place
4. **Easier Navigation**: All experiment files in single directory

## Updated Paths

### run_experiment.sh
```bash
# Before
OUTPUT_DIR="${PACKAGE_ROOT}/experiments/ad_prediction/results"

# After
OUTPUT_DIR="${SCRIPT_DIR}/results"
```

### config.json
```json
{
  "output": {
    "output_dir": "results",  // Relative to experiment directory
    ...
  }
}
```

## File Locations

### Results Directory
**Full Path**: 
```
/media/devin/WORK/devin/tien/src/brain_age_prediction/vcda_net_package/src/vcda_net/experiments/ad_prediction/results/
```

**Relative to Package Root**:
```
src/vcda_net/experiments/ad_prediction/results/
```

**Relative to Experiment Directory**:
```
results/
```

## Usage

No changes needed! The script automatically uses the new location:

```bash
cd src/vcda_net/experiments/ad_prediction
bash run_experiment.sh
# Results saved to ./results/
```

## Files in Results Directory

- `predictions_adni_ad_mci.csv` - Full predictions (1.6MB)
- `predictions_adni_ad_mci_statistics.json` - Summary statistics
- `compare_results.py` - Validation script
- `VALIDATION_REPORT.md` - Validation documentation

---

**Updated**: 2026-02-17  
**Status**: ✅ Complete
