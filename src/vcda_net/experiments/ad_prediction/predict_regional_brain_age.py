#!/usr/bin/env python3
"""
Regional Brain Age Prediction with Bias Correction for AD/MCI Patients

This script applies 32 regional brain age predictors to ADNI AD/MCI dataset
and performs VNN-style bias correction to calculate brain age gap.

Pipeline:
1. Load regional features extracted from disease cohort
2. Load trained regional predictors (32 models)
3. Generate predictions for each region
4. Apply bias correction using parameters from healthy controls
5. Calculate ensemble predictions
6. Save results with detailed statistics

Author: Anonymous
Date: 2026-02-17
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import Dict, Tuple, List
import argparse


class RegionalBrainAgePredictor:
    """Regional brain age predictor with bias correction."""
    
    def __init__(
        self,
        models_dir: Path,
        bias_params_file: Path,
        num_regions: int = 32
    ):
        """
        Initialize the predictor.
        
        Args:
            models_dir: Directory containing trained regional models
            bias_params_file: JSON file with bias correction parameters
            num_regions: Number of brain regions (default: 32)
        """
        self.models_dir = Path(models_dir)
        self.bias_params_file = Path(bias_params_file)
        self.num_regions = num_regions
        self.models = []
        self.bias_params = []
        
    def load_models(self):
        """Load all regional models."""
        print(f"Loading {self.num_regions} regional models...")
        for region_idx in range(self.num_regions):
            model_file = self.models_dir / f"region_{region_idx:02d}_model.pkl"
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            self.models.append(model)
        print(f"  ✓ Loaded {len(self.models)} models")
        
    def load_bias_params(self):
        """Load bias correction parameters."""
        print(f"Loading bias correction parameters...")
        with open(self.bias_params_file, 'r') as f:
            self.bias_params = json.load(f)
        print(f"  ✓ Loaded parameters for {len(self.bias_params)} regions")
        
    def predict_region(
        self,
        features: np.ndarray,
        region_idx: int,
        apply_correction: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict brain age for a specific region.
        
        Args:
            features: Regional features [n_samples, feature_dim]
            region_idx: Region index (0-31)
            apply_correction: Whether to apply bias correction
            
        Returns:
            raw_predictions: Raw predictions
            corrected_predictions: Bias-corrected predictions
        """
        # Raw prediction
        raw_pred = self.models[region_idx].predict(features)
        
        if apply_correction:
            # Apply bias correction
            params = self.bias_params[region_idx]
            beta_0 = params['beta_0_intercept']
            beta_1 = params['beta_1_slope']
            corrected_pred = beta_0 + beta_1 * raw_pred
        else:
            corrected_pred = raw_pred
            
        return raw_pred, corrected_pred
    
    def predict_all_regions(
        self,
        features_dir: Path,
        n_samples: int,
        apply_correction: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict brain age for all regions.
        
        Args:
            features_dir: Directory containing regional features
            n_samples: Number of samples
            apply_correction: Whether to apply bias correction
            
        Returns:
            all_raw_predictions: [n_samples, num_regions]
            all_corrected_predictions: [n_samples, num_regions]
        """
        all_raw = np.zeros((n_samples, self.num_regions))
        all_corrected = np.zeros((n_samples, self.num_regions))
        
        for region_idx in tqdm(range(self.num_regions), desc="Processing regions"):
            # Load features for this region
            features_file = features_dir / f"region_{region_idx:02d}_features.npy"
            features = np.load(features_file)
            
            # Predict
            raw_pred, corr_pred = self.predict_region(
                features, region_idx, apply_correction
            )
            
            all_raw[:, region_idx] = raw_pred
            all_corrected[:, region_idx] = corr_pred
            
        return all_raw, all_corrected
    
    def calculate_ensemble(
        self,
        predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate ensemble predictions.
        
        Args:
            predictions: [n_samples, num_regions]
            
        Returns:
            Dictionary with ensemble statistics
        """
        return {
            'mean': predictions.mean(axis=1),
            'median': np.median(predictions, axis=1),
            'std': predictions.std(axis=1)
        }


def create_results_dataframe(
    metadata_df: pd.DataFrame,
    true_ages: np.ndarray,
    all_raw_predictions: np.ndarray,
    all_corrected_predictions: np.ndarray,
    num_regions: int
) -> pd.DataFrame:
    """
    Create comprehensive results dataframe.
    
    Args:
        metadata_df: Original metadata
        true_ages: True ages
        all_raw_predictions: Raw predictions [n_samples, num_regions]
        all_corrected_predictions: Corrected predictions [n_samples, num_regions]
        num_regions: Number of regions
        
    Returns:
        Results dataframe with all predictions and statistics
    """
    results_df = metadata_df.copy()
    results_df['true_age'] = true_ages
    
    # Add per-region predictions
    for region_idx in range(num_regions):
        results_df[f'region_{region_idx:02d}_raw'] = all_raw_predictions[:, region_idx]
        results_df[f'region_{region_idx:02d}_corrected'] = all_corrected_predictions[:, region_idx]
        results_df[f'region_{region_idx:02d}_delta_raw'] = all_raw_predictions[:, region_idx] - true_ages
        results_df[f'region_{region_idx:02d}_delta_corrected'] = all_corrected_predictions[:, region_idx] - true_ages
    
    # Calculate ensemble predictions (mean, median, std)
    raw_ensemble = {
        'mean': all_raw_predictions.mean(axis=1),
        'median': np.median(all_raw_predictions, axis=1),
        'std': all_raw_predictions.std(axis=1)
    }
    corr_ensemble = {
        'mean': all_corrected_predictions.mean(axis=1),
        'median': np.median(all_corrected_predictions, axis=1),
        'std': all_corrected_predictions.std(axis=1)
    }
    
    results_df['ensemble_raw_mean'] = raw_ensemble['mean']
    results_df['ensemble_raw_median'] = raw_ensemble['median']
    results_df['ensemble_raw_std'] = raw_ensemble['std']
    
    results_df['ensemble_corrected_mean'] = corr_ensemble['mean']
    results_df['ensemble_corrected_median'] = corr_ensemble['median']
    results_df['ensemble_corrected_std'] = corr_ensemble['std']

    
    # Calculate errors and deltas
    results_df['ensemble_raw_error'] = raw_ensemble['mean'] - true_ages
    results_df['ensemble_raw_abs_error'] = np.abs(results_df['ensemble_raw_error'])
    results_df['ensemble_corrected_error'] = corr_ensemble['mean'] - true_ages
    results_df['ensemble_corrected_abs_error'] = np.abs(results_df['ensemble_corrected_error'])
    
    results_df['ensemble_raw_delta'] = results_df['ensemble_raw_error']
    results_df['ensemble_corrected_delta'] = results_df['ensemble_corrected_error']
    
    return results_df


def calculate_statistics(
    results_df: pd.DataFrame,
    true_ages: np.ndarray,
    num_regions: int
) -> Dict:
    """Calculate comprehensive statistics."""
    
    stats = {
        'ensemble': {},
        'per_region': []
    }
    
    # Ensemble statistics
    ensemble_raw_mae = results_df['ensemble_raw_abs_error'].mean()
    ensemble_raw_bias = results_df['ensemble_raw_error'].mean()
    ensemble_raw_rmse = np.sqrt((results_df['ensemble_raw_error']**2).mean())
    ensemble_raw_r, _ = pearsonr(true_ages, results_df['ensemble_raw_mean'])
    
    ensemble_corr_mae = results_df['ensemble_corrected_abs_error'].mean()
    ensemble_corr_bias = results_df['ensemble_corrected_error'].mean()
    ensemble_corr_rmse = np.sqrt((results_df['ensemble_corrected_error']**2).mean())
    ensemble_corr_r, _ = pearsonr(true_ages, results_df['ensemble_corrected_mean'])
    
    stats['ensemble'] = {
        'raw': {
            'mae': float(ensemble_raw_mae),
            'bias': float(ensemble_raw_bias),
            'rmse': float(ensemble_raw_rmse),
            'r': float(ensemble_raw_r),
            'r2': float(ensemble_raw_r**2)
        },
        'corrected': {
            'mae': float(ensemble_corr_mae),
            'bias': float(ensemble_corr_bias),
            'rmse': float(ensemble_corr_rmse),
            'r': float(ensemble_corr_r),
            'r2': float(ensemble_corr_r**2)
        }
    }
    
    # Per-region statistics
    for region_idx in range(num_regions):
        raw_errors = results_df[f'region_{region_idx:02d}_delta_raw'].values
        corr_errors = results_df[f'region_{region_idx:02d}_delta_corrected'].values
        
        stats['per_region'].append({
            'region': region_idx,
            'mae_raw': float(np.abs(raw_errors).mean()),
            'mae_corrected': float(np.abs(corr_errors).mean()),
            'bias_raw': float(raw_errors.mean()),
            'bias_corrected': float(corr_errors.mean())
        })
    
    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Regional brain age prediction for AD/MCI patients'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        required=True,
        help='Directory containing regional features'
    )
    parser.add_argument(
        '--metadata_file',
        type=str,
        required=True,
        help='Metadata CSV file'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        required=True,
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--bias_params_file',
        type=str,
        required=True,
        help='Bias correction parameters JSON file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--num_regions',
        type=int,
        default=32,
        help='Number of brain regions'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("REGIONAL BRAIN AGE PREDICTION - AD/MCI COHORT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Features: {args.features_dir}")
    print(f"  Metadata: {args.metadata_file}")
    print(f"  Models: {args.models_dir}")
    print(f"  Bias params: {args.bias_params_file}")
    print(f"  Output: {args.output_file}")
    print(f"  Regions: {args.num_regions}")
    
    # Load metadata
    print(f"\n{'='*80}")
    print("STEP 1: LOADING DATA")
    print(f"{'='*80}")
    
    metadata_df = pd.read_csv(args.metadata_file)
    n_samples = len(metadata_df)
    true_ages = metadata_df['age'].values
    
    print(f"  ✓ Loaded {n_samples} samples")
    print(f"  Age range: {true_ages.min():.0f} - {true_ages.max():.0f} years")
    
    # Initialize predictor
    print(f"\n{'='*80}")
    print("STEP 2: INITIALIZING PREDICTOR")
    print(f"{'='*80}")
    
    predictor = RegionalBrainAgePredictor(
        models_dir=args.models_dir,
        bias_params_file=args.bias_params_file,
        num_regions=args.num_regions
    )
    
    predictor.load_models()
    predictor.load_bias_params()
    
    # Generate predictions
    print(f"\n{'='*80}")
    print("STEP 3: GENERATING PREDICTIONS")
    print(f"{'='*80}")
    
    all_raw, all_corrected = predictor.predict_all_regions(
        features_dir=Path(args.features_dir),
        n_samples=n_samples,
        apply_correction=True
    )
    
    print(f"  ✓ Generated predictions for all {args.num_regions} regions")
    
    # Create results dataframe
    print(f"\n{'='*80}")
    print("STEP 4: CREATING RESULTS")
    print(f"{'='*80}")
    
    results_df = create_results_dataframe(
        metadata_df=metadata_df,
        true_ages=true_ages,
        all_raw_predictions=all_raw,
        all_corrected_predictions=all_corrected,
        num_regions=args.num_regions
    )
    
    # Calculate statistics
    stats = calculate_statistics(
        results_df=results_df,
        true_ages=true_ages,
        num_regions=args.num_regions
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("ENSEMBLE STATISTICS")
    print(f"{'='*80}")
    print(f"\nRaw Predictions:")
    print(f"  MAE:  {stats['ensemble']['raw']['mae']:.3f} years")
    print(f"  RMSE: {stats['ensemble']['raw']['rmse']:.3f} years")
    print(f"  Bias: {stats['ensemble']['raw']['bias']:.3f} years")
    print(f"  R:    {stats['ensemble']['raw']['r']:.4f}")
    
    print(f"\nCorrected Predictions:")
    print(f"  MAE:  {stats['ensemble']['corrected']['mae']:.3f} years")
    print(f"  RMSE: {stats['ensemble']['corrected']['rmse']:.3f} years")
    print(f"  Bias: {stats['ensemble']['corrected']['bias']:.3f} years")
    print(f"  R:    {stats['ensemble']['corrected']['r']:.4f}")
    
    # Save results
    print(f"\n{'='*80}")
    print("STEP 5: SAVING RESULTS")
    print(f"{'='*80}")
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved predictions: {output_path}")
    
    # Save statistics
    stats_file = output_path.parent / f"{output_path.stem}_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved statistics: {stats_file}")
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
