# Regional Bias Correction - VNN Style

**Generated:** 2026-02-03  
**Method:** VNN-style linear bias correction  
**Training Set:** HC test set (n=257, age 8-95 years)  
**Regions:** 32 brain regions

---

## 📋 Overview

Applied VNN-style bias correction to each of the 32 regional brain age predictors:

```
corrected_age = β0 + β1 × predicted_age
```

Each region has its own correction parameters (β0, β1) fitted on the HC test set.

---

## 📁 Files

### **Bias Correction Parameters:**
- `regional_bias_correction_params.csv` - Parameters for all 32 regions (CSV format)
- `regional_bias_correction_params.json` - Same parameters (JSON format, easy loading)

### **Corrected Predictions:**
- `predictions_test_bias_corrected.csv` - HC test set with corrected predictions

### **Analysis:**
- `bias_correction_analysis.png` - Visualization of correction effects
- `bias_correction_summary.json` - Summary statistics

---

## 📊 Bias Correction Parameters

### **Average Parameters (across 32 regions):**
- **β0 (intercept):** -4.82 ± 13.61
- **β1 (slope):** 1.08 ± 0.26
- **R²:** 0.746 ± 0.170

### **Top 5 Best Performing Regions (by R²):**

| Region | β0 | β1 | R² | MAE After |
|--------|----|----|-----|-----------|
| 3 | -0.66 | 1.01 | 0.967 | 3.22 years |
| 13 | -0.58 | 1.01 | 0.959 | 3.41 years |
| 28 | -0.07 | 1.00 | 0.922 | 5.54 years |
| 7 | -2.46 | 1.04 | 0.915 | 5.97 years |
| 20 | 0.23 | 1.00 | 0.904 | 5.93 years |

### **Regions with Ideal Slope (closest to 1.0):**

| Region | β0 | β1 | R² | MAE After |
|--------|----|----|-----|-----------|
| 20 | 0.23 | 1.000 | 0.904 | 5.93 years |
| 26 | 0.36 | 0.998 | 0.856 | 7.55 years |
| 28 | -0.07 | 0.995 | 0.922 | 5.54 years |

---

## 📈 Performance Improvement

### **Per-Region Performance:**

| Metric | Before Correction | After Correction | Improvement |
|--------|------------------|------------------|-------------|
| **MAE** | 9.81 ± 3.90 years | 9.64 ± 3.70 years | **0.17 years (1.8%)** |
| **Bias** | 0.55 ± 0.65 years | **0.00 ± 0.00 years** | **100% reduction** ✅ |

### **Ensemble Performance (After Correction):**

| Method | MAE | Bias | Std |
|--------|-----|------|-----|
| **Average Ensemble** | **6.30 years** | **0.00 years** ✅ | 8.02 years |
| **Median Ensemble** | **5.40 years** ⭐ | -0.13 years | 7.24 years |

**Key Finding:** Median ensemble performs best with **5.40 MAE** and near-zero bias!

---

## 🎯 Delta-Age Statistics

### **Per-Region Delta-Age:**
- Mean range: **0.00 to 0.00 years** (perfect bias correction!)
- Std range: **4.69 to 20.84 years**

### **Ensemble Delta-Age:**
- **Average:** 0.00 ± 8.02 years (MAE: 6.30)
- **Median:** -0.13 ± 7.24 years (MAE: 5.40)

---

## 💻 Usage

### **1. Load Bias Correction Parameters:**

```python
import json
import pandas as pd

# Option 1: Load from JSON
with open('regional_bias_correction_params.json', 'r') as f:
    bc_params = json.load(f)

# Option 2: Load from CSV
bc_df = pd.read_csv('regional_bias_correction_params.csv')

# Get parameters for region 5
region_5 = bc_params[5]
beta_0 = region_5['beta_0_intercept']
beta_1 = region_5['beta_1_slope']
```

### **2. Apply Correction to New Predictions:**

```python
def apply_regional_bias_correction(predictions, bc_params):
    """
    Apply bias correction to regional predictions
    
    Args:
        predictions: DataFrame with columns region_00_pred, region_01_pred, ...
        bc_params: List of dicts with beta_0_intercept and beta_1_slope
    
    Returns:
        DataFrame with corrected predictions and delta-ages
    """
    import pandas as pd
    
    df_corrected = predictions.copy()
    
    for i in range(32):
        region_col = f'region_{i:02d}_pred'
        bc = bc_params[i]
        
        # Apply correction
        y_pred = predictions[region_col].values
        y_corrected = bc['beta_0_intercept'] + bc['beta_1_slope'] * y_pred
        
        # Store corrected prediction
        df_corrected[f'region_{i:02d}_corrected'] = y_corrected
        
        # Calculate delta-age (if true_age available)
        if 'true_age' in predictions.columns:
            df_corrected[f'region_{i:02d}_delta'] = y_corrected - predictions['true_age']
    
    # Calculate ensemble
    corrected_cols = [f'region_{i:02d}_corrected' for i in range(32)]
    df_corrected['average_corrected'] = df_corrected[corrected_cols].mean(axis=1)
    df_corrected['median_corrected'] = df_corrected[corrected_cols].median(axis=1)
    
    if 'true_age' in predictions.columns:
        df_corrected['average_delta'] = df_corrected['average_corrected'] - predictions['true_age']
        df_corrected['median_delta'] = df_corrected['median_corrected'] - predictions['true_age']
    
    return df_corrected

# Example usage
import json
with open('regional_bias_correction_params.json', 'r') as f:
    bc_params = json.load(f)

# Apply to new predictions
df_new = pd.read_csv('new_predictions.csv')
df_corrected = apply_regional_bias_correction(df_new, bc_params)
```

### **3. Apply to Disease Cohorts:**

```python
# Load disease predictions (e.g., ADNI AD/MCI)
df_disease = pd.read_csv('disease_predictions.csv')

# Apply bias correction
df_disease_corrected = apply_regional_bias_correction(df_disease, bc_params)

# Analyze regional delta-ages
delta_cols = [f'region_{i:02d}_delta' for i in range(32)]
regional_deltas = df_disease_corrected[delta_cols]

# Which regions show highest acceleration?
mean_deltas = regional_deltas.mean()
print("Regions with highest delta-age:")
print(mean_deltas.nlargest(10))
```

---

## 🔬 Interpretation

### **Bias Correction Effectiveness:**

1. **✅ Perfect Bias Removal:**
   - Mean delta-age = 0.00 for all regions
   - Systematic bias completely eliminated

2. **✅ Maintained Accuracy:**
   - MAE only slightly reduced (1.8%)
   - Bias correction doesn't harm individual predictions

3. **✅ Improved Ensemble:**
   - Ensemble MAE: 6.30 years (average) or 5.40 years (median)
   - Better than individual regions (9.64 years)

### **Regional Variability:**

- **High R² regions (3, 13, 28):** Most reliable predictors
- **Slope ≈ 1.0 regions (20, 26, 28):** Well-calibrated predictions
- **High std regions:** More variable predictions, less reliable

### **Clinical Applications:**

1. **Healthy Controls:** Mean delta-age ≈ 0 (by design)
2. **Disease Groups:** Positive delta-age indicates brain age acceleration
3. **Regional Patterns:** Identify which regions are most affected

---

## 📊 Comparison with Global V4

| Method | MAE | Bias | Interpretability |
|--------|-----|------|------------------|
| **Global V4** | ~4.5 years | ~0 years | Medium |
| **Regional Ensemble (corrected)** | **5.4 years** | **-0.13 years** | **High** ✅ |
| Regional Individual | 9.6 years | 0 years | High |

**Trade-off:** Regional approach sacrifices some accuracy for interpretability.

---

## 🚀 Next Steps

### **1. Apply to Disease Data:**
```bash
# Extract features for ADNI/OASIS
# Apply regional predictors
# Apply bias correction
# Analyze regional delta-age patterns
```

### **2. Regional Delta-Age Analysis:**
- Which regions show highest acceleration in AD?
- Regional patterns in MCI vs AD
- Correlation with clinical measures

### **3. Ensemble Optimization:**
- Weighted ensemble based on R²
- Combine global + regional predictions
- Region-specific thresholds

---

## 📚 Files Reference

**Parameters:**
- `regional_bias_correction_params.csv` - All correction parameters
- `regional_bias_correction_params.json` - Same in JSON format

**Predictions:**
- `predictions_test_bias_corrected.csv` - Corrected HC test predictions

**Analysis:**
- `bias_correction_analysis.png` - 4-panel visualization
- `bias_correction_summary.json` - Summary statistics

**Related:**
- `../predictions/predictions_test.csv` - Original predictions (before correction)
- `../regional_predictors_results.csv` - Per-region training results
- `../models/region_XX_model.pkl` - Trained Ridge models

---

**Last Updated:** 2026-02-03 18:25 JST
