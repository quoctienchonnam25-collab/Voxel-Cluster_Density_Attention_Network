# Regional Brain Age Predictions - V4 Features

**Generated:** 2026-02-03  
**Dataset:** Healthy Brain 1710 (ABIDE, ADNI, IXI, OASIS, PPMI)  
**Features:** 32 regions × 512 dimensions (extracted from Hybrid GradCAM V4)  
**Models:** 32 Ridge regression predictors (α=1.0)

---

## 📁 Files Overview

### **Prediction Files:**
- `predictions_train.csv` - Training set predictions (1,196 samples)
- `predictions_valid.csv` - Validation set predictions (257 samples)
- `predictions_test.csv` - Test set predictions (257 samples)
- `predictions_all.csv` - All samples combined (1,710 samples)
- `predictions_summary.json` - Performance summary

### **CSV Structure:**

Each CSV file contains the following columns:

#### **Metadata Columns:**
- `idx` - Original sample index
- `mri_path` - Path to MRI file
- `age` - Original age from metadata
- `sex` - Sex (M/F)
- `dataset` - Source dataset (ABIDE, ADNI, IXI, OASIS, PPMI)
- `subject_id` - Subject identifier
- `split` - Data split (train/valid/test)
- `true_age` - True chronological age

#### **Regional Predictions (32 columns):**
- `region_00_pred` through `region_31_pred` - Brain age prediction from each region

#### **Ensemble Predictions:**
- `average_pred` - Average of all 32 regional predictions
- `median_pred` - Median of all 32 regional predictions
- `std_pred` - Standard deviation across 32 regional predictions

#### **Error Metrics:**
- `average_error` - Error of average prediction (predicted - true)
- `average_abs_error` - Absolute error of average prediction
- `median_error` - Error of median prediction
- `median_abs_error` - Absolute error of median prediction

---

## 📊 Performance Summary

### **Test Set Performance (n=257):**

| Metric | Average Ensemble | Median Ensemble |
|--------|------------------|-----------------|
| **MAE** | **7.00 ± 5.04 years** | **6.18 ± 4.96 years** |
| **RMSE** | **8.62 years** | **7.92 years** |
| **Bias** | **+0.55 years** | **+0.23 years** |

### **Comparison with Individual Regions:**

| Metric | Individual Regions (avg) | Ensemble (average) | Improvement |
|--------|--------------------------|-------------------|-------------|
| **MAE** | 9.81 ± 3.90 years | 7.00 ± 5.04 years | **-28.6%** ✅ |

**Key Finding:** Ensemble of regional predictions significantly outperforms individual regions!

### **Regional Variability:**
- Mean std across regions: **9.78 years**
- Range: 5.44 - 16.62 years
- Indicates substantial variation in predictions across different brain regions

---

## 🎯 Usage Examples

### **1. Load predictions:**
```python
import pandas as pd

# Load test set predictions
df = pd.read_csv('predictions_test.csv')

# View sample
print(df[['true_age', 'average_pred', 'median_pred', 'std_pred']].head())
```

### **2. Analyze regional patterns:**
```python
# Get all regional predictions
region_cols = [f'region_{i:02d}_pred' for i in range(32)]
regional_preds = df[region_cols]

# Find which regions contribute most to prediction
correlations = regional_preds.corrwith(df['true_age'])
print("Top 5 most predictive regions:")
print(correlations.nlargest(5))
```

### **3. Identify outliers:**
```python
# Samples with high regional variability
high_var = df[df['std_pred'] > 15]
print(f"Samples with high regional disagreement: {len(high_var)}")

# Samples with large prediction errors
large_error = df[df['average_abs_error'] > 15]
print(f"Samples with large errors: {len(large_error)}")
```

### **4. Compare ensemble methods:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['true_age'], df['average_pred'], alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel('True Age')
plt.ylabel('Average Prediction')
plt.title('Average Ensemble')

plt.subplot(1, 2, 2)
plt.scatter(df['true_age'], df['median_pred'], alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel('True Age')
plt.ylabel('Median Prediction')
plt.title('Median Ensemble')

plt.tight_layout()
plt.savefig('ensemble_comparison.png')
```

---

## 🔬 Regional Analysis

### **Per-Region Performance:**

See `../regional_predictors_results.csv` for detailed per-region metrics:
- MAE, RMSE, R² for train/valid/test
- Pearson correlation with true age
- Statistical significance

### **Best Performing Regions (by test MAE):**
Load `regional_predictors_results.csv` and sort by `mae_test` ascending.

### **Regional Contribution Analysis:**
```python
# Calculate how much each region contributes to ensemble
region_cols = [f'region_{i:02d}_pred' for i in range(32)]
regional_preds = df[region_cols]

# Correlation with ensemble
ensemble_corr = regional_preds.corrwith(df['average_pred'])
print("Regions most aligned with ensemble:")
print(ensemble_corr.nlargest(10))
```

---

## 📈 Key Insights

### **1. Ensemble Benefits:**
- ✅ **28.6% improvement** over individual regions
- ✅ **Median ensemble** slightly better than average (6.18 vs 7.00 MAE)
- ✅ Robust to outlier regions

### **2. Regional Variability:**
- High std_pred (>15 years) indicates regional disagreement
- May suggest:
  - Challenging cases
  - Pathological changes
  - Data quality issues

### **3. Error Patterns:**
- Small positive bias (+0.23 to +0.55 years)
- Some large errors (worst: -44 years)
- Generally consistent across age range

---

## 🚀 Next Steps

### **1. Apply to Disease Cohorts:**
```bash
# Extract features for ADNI/OASIS disease data
# Apply 32 regional models
# Calculate regional delta-ages
```

### **2. Regional Delta-Age Analysis:**
- Which regions show highest acceleration in AD/MCI?
- Regional patterns vs global patterns
- Correlation with clinical measures

### **3. Compare with Global V4:**
- Global prediction: ~4.5 MAE
- Regional ensemble: ~7.0 MAE
- Weighted ensemble: Global + Regional

### **4. Interpretability:**
- Visualize regional contributions
- Map predictions to brain anatomy
- Clinical interpretation of regional patterns

---

## 📚 References

- **Feature Extraction:** `extract_v4_regional_features.py`
- **Model Training:** `train_v4_regional_predictors.py`
- **Prediction Generation:** `generate_regional_predictions.py`
- **Model Files:** `../models/region_XX_model.pkl`
- **Training Results:** `../regional_predictors_results.csv`

---

**Last Updated:** 2026-02-03 18:10 JST
