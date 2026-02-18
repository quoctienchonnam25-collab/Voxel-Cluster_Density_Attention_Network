#!/usr/bin/env python3
"""Quick comparison of reference vs new predictions."""

import pandas as pd
import numpy as np

# Load files
print("Loading files...")
ref_df = pd.read_csv("/media/devin/WORK/devin/tien/src/brain_age_prediction/FINAL_PREDICTIONS_MASTER/predictions_all_regions_adni.csv")
new_df = pd.read_csv("/media/devin/WORK/devin/tien/src/brain_age_prediction/vcda_net_package/experiments/ad_prediction/results/predictions_adni_ad_mci.csv")

print(f"\n{'='*80}")
print("COMPARISON: Reference vs New Implementation")
print(f"{'='*80}")

print(f"\nSamples: {len(ref_df)} vs {len(new_df)}")

# Compare key column
ref_pred = ref_df['ensemble_corrected_mean'].values
new_pred = new_df['ensemble_corrected_mean'].values

diff = np.abs(ref_pred - new_pred)
print(f"\nEnsemble Corrected Mean:")
print(f"  Max diff:  {diff.max():.6f} years")
print(f"  Mean diff: {diff.mean():.6f} years")
print(f"  Correlation: {np.corrcoef(ref_pred, new_pred)[0,1]:.6f}")

if diff.max() < 0.001:
    print(f"\n✓ IDENTICAL!")
elif diff.max() < 0.01:
    print(f"\n✓ VERY CLOSE!")
else:
    print(f"\n⚠ DIFFERENT")

# Stats
print(f"\nReference stats:")
print(f"  Mean delta: {ref_df['ensemble_corrected_delta'].mean():.3f}")
print(f"  Std delta:  {ref_df['ensemble_corrected_delta'].std():.3f}")

print(f"\nNew stats:")
print(f"  Mean delta: {new_df['ensemble_corrected_delta'].mean():.3f}")
print(f"  Std delta:  {new_df['ensemble_corrected_delta'].std():.3f}")
