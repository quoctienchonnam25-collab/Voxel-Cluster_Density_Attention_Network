# Quick Reference - AD Brain Age Gap Prediction

## 🚀 Quick Start

```bash
cd /media/devin/WORK/devin/tien/src/brain_age_prediction/hybrid_saliency_v4_package/src/hybrid_saliency_v4/experiments/ad_prediction
bash run_experiment.sh
```

## 📁 File Structure

```
ad_prediction/
├── predict_regional_brain_age.py    # Main script
├── run_experiment.sh                # Quick launcher
├── visualize_results.py             # Analysis & plots
├── config.json                      # Configuration
├── README.md                        # Full documentation
└── results/                         # Output (auto-created)
```

## 🔧 Common Commands

### Run Prediction
```bash
bash run_experiment.sh
```

### Custom Paths
```bash
python -m hybrid_saliency_v4.experiments.ad_prediction.predict_regional_brain_age \
    --features_dir /path/to/features \
    --metadata_file /path/to/metadata.csv \
    --models_dir /path/to/models \
    --bias_params_file /path/to/params.json \
    --output_file /path/to/output.csv
```

### Visualize Results
```bash
python visualize_results.py \
    --predictions_file results/predictions_adni_ad_mci.csv \
    --output_dir results/visualizations
```

## 📊 Output Files

| File | Description |
|------|-------------|
| `predictions_adni_ad_mci.csv` | Full predictions |
| `predictions_adni_ad_mci_statistics.json` | Summary stats |
| `brain_age_gap_analysis.png` | Distribution plots |
| `regional_analysis.png` | Regional heatmaps |
| `cohort_statistics.csv` | Per-cohort stats |
| `between_group_comparisons.csv` | Statistical tests |

## 🔍 Key Columns in Output

- `ensemble_corrected_mean`: Final brain age prediction
- `ensemble_corrected_delta`: Brain age gap (predicted - true)
- `region_XX_corrected`: Per-region predictions
- `true_age`: Chronological age

## 📈 Interpretation

- **Positive delta**: Brain appears older (accelerated aging)
- **Negative delta**: Brain appears younger
- **Expected**: AD > MCI > HC

## ⚙️ Configuration

Edit `config.json` to change:
- Data paths
- Model parameters
- Output settings

## 🐛 Troubleshooting

### Import Error
```bash
export PYTHONPATH="/path/to/hybrid_saliency_v4_package/src:$PYTHONPATH"
```

### Missing Data
Check paths in `config.json` or script arguments

### Permission Denied
```bash
chmod +x run_experiment.sh
```

## 📚 Documentation

- **Full README**: `README.md`
- **Integration Guide**: `INTEGRATION_COMPLETE.md`
- **Package Docs**: `../../README.md`

## 🔗 Related Scripts

- Original: `/media/devin/WORK/devin/tien/src/brain_age_prediction/predict_all_regions_with_bias_correction_adni.py`
- Feature extraction: `../../training/`
- Model training: `../../../train_saliency_v4.sh`

## 💡 Tips

1. Always check paths in config before running
2. Results are saved to `results/` directory
3. Use visualization script for analysis
4. Check statistics JSON for quick summary
5. Refer to README for detailed documentation
