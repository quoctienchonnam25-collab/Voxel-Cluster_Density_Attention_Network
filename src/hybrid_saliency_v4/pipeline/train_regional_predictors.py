#!/usr/bin/env python3
"""
Train Regional Brain Age Predictors

Trains 32 regional predictors using Ridge regression on extracted features.
Includes train/valid/test split and bias correction parameter calculation.

Author: Anonymous
Date: 2026-02-17
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


class RegionalPredictorTrainer:
    """Train regional brain age predictors with bias correction."""
    
    def __init__(
        self,
        features_dir: str,
        output_dir: str,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        alpha: float = 1.0,
        random_seed: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            features_dir: Directory containing extracted features
            output_dir: Directory to save trained models
            train_ratio: Training set ratio
            valid_ratio: Validation set ratio
            test_ratio: Test set ratio
            alpha: Ridge regression regularization parameter
            random_seed: Random seed for reproducibility
        """
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.alpha = alpha
        self.random_seed = random_seed
        
        self.all_features = None
        self.ages = None
        self.metadata_df = None
        self.num_regions = None
        
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        
        self.models = []
        self.results = []
        
    def load_data(self):
        """Load extracted features and metadata."""
        print(f"\nLoading data from {self.features_dir}...")
        
        # Load features
        features_file = self.features_dir / "regional_features.npy"
        self.all_features = np.load(features_file)
        
        n_samples, self.num_regions, embedding_dim = self.all_features.shape
        print(f"  Features shape: {self.all_features.shape}")
        
        # Load metadata
        metadata_file = self.features_dir / "metadata.csv"
        self.metadata_df = pd.read_csv(metadata_file)
        self.ages = self.metadata_df['age'].values
        
        print(f"  Samples: {n_samples}")
        print(f"  Regions: {self.num_regions}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Age range: {self.ages.min():.0f} - {self.ages.max():.0f} years")
        
    def split_data(self):
        """Split data into train/valid/test sets."""
        print(f"\nSplitting data...")
        
        n_samples = len(self.ages)
        indices = np.arange(n_samples)
        
        # First split: train+valid vs test
        train_valid_idx, self.test_idx = train_test_split(
            indices,
            test_size=self.test_ratio,
            random_state=self.random_seed
        )
        
        # Second split: train vs valid
        valid_ratio_adjusted = self.valid_ratio / (self.train_ratio + self.valid_ratio)
        self.train_idx, self.valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_ratio_adjusted,
            random_state=self.random_seed
        )
        
        print(f"  Train: {len(self.train_idx)} samples ({100*len(self.train_idx)/n_samples:.1f}%)")
        print(f"  Valid: {len(self.valid_idx)} samples ({100*len(self.valid_idx)/n_samples:.1f}%)")
        print(f"  Test:  {len(self.test_idx)} samples ({100*len(self.test_idx)/n_samples:.1f}%)")
        
        # Save split info
        split_info = {
            'train_indices': self.train_idx.tolist(),
            'valid_indices': self.valid_idx.tolist(),
            'test_indices': self.test_idx.tolist(),
            'train_size': len(self.train_idx),
            'valid_size': len(self.valid_idx),
            'test_size': len(self.test_idx),
            'random_seed': self.random_seed
        }
        
        split_file = self.output_dir / "data_split.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"  ✓ Saved split info: {split_file}")
        
    def train_models(self):
        """Train regional predictors."""
        print(f"\nTraining {self.num_regions} regional predictors...")
        
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        ages_train = self.ages[self.train_idx]
        ages_valid = self.ages[self.valid_idx]
        ages_test = self.ages[self.test_idx]
        
        for region_idx in tqdm(range(self.num_regions), desc="Training regions"):
            # Get features for this region
            X_train = self.all_features[self.train_idx, region_idx, :]
            X_valid = self.all_features[self.valid_idx, region_idx, :]
            X_test = self.all_features[self.test_idx, region_idx, :]
            
            # Train model
            model = Ridge(alpha=self.alpha, random_state=self.random_seed)
            model.fit(X_train, ages_train)
            
            # Predictions
            pred_train = model.predict(X_train)
            pred_valid = model.predict(X_valid)
            pred_test = model.predict(X_test)
            
            # Calculate metrics
            result = {
                'region_idx': region_idx,
                'mae_train': float(mean_absolute_error(ages_train, pred_train)),
                'rmse_train': float(np.sqrt(mean_squared_error(ages_train, pred_train))),
                'r2_train': float(r2_score(ages_train, pred_train)),
                'r_train': float(pearsonr(ages_train, pred_train)[0]),
                'mae_valid': float(mean_absolute_error(ages_valid, pred_valid)),
                'rmse_valid': float(np.sqrt(mean_squared_error(ages_valid, pred_valid))),
                'r2_valid': float(r2_score(ages_valid, pred_valid)),
                'r_valid': float(pearsonr(ages_valid, pred_valid)[0]),
                'mae_test': float(mean_absolute_error(ages_test, pred_test)),
                'rmse_test': float(np.sqrt(mean_squared_error(ages_test, pred_test))),
                'r2_test': float(r2_score(ages_test, pred_test)),
                'r_test': float(pearsonr(ages_test, pred_test)[0]),
            }
            
            self.results.append(result)
            self.models.append(model)
            
            # Save model
            model_file = models_dir / f"region_{region_idx:02d}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"\n✓ Trained {self.num_regions} regional predictors")
        
    def calculate_bias_correction(self):
        """Calculate bias correction parameters using training set."""
        print(f"\nCalculating bias correction parameters...")
        
        bias_dir = self.output_dir / "bias_correction"
        bias_dir.mkdir(exist_ok=True)
        
        ages_train = self.ages[self.train_idx]
        bias_params = []
        
        for region_idx in range(self.num_regions):
            X_train = self.all_features[self.train_idx, region_idx, :]
            pred_train = self.models[region_idx].predict(X_train)
            
            # Fit linear correction: true_age = beta_0 + beta_1 * predicted_age
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(pred_train.reshape(-1, 1), ages_train)
            
            beta_0 = float(lr.intercept_)
            beta_1 = float(lr.coef_[0])
            
            bias_params.append({
                'region_idx': region_idx,
                'beta_0_intercept': beta_0,
                'beta_1_slope': beta_1
            })
        
        # Save bias correction parameters
        bias_file = bias_dir / "regional_bias_correction_params.json"
        with open(bias_file, 'w') as f:
            json.dump(bias_params, f, indent=2)
        
        print(f"  ✓ Saved bias correction parameters: {bias_file}")
        
    def save_results(self):
        """Save training results and summary."""
        print(f"\nSaving results...")
        
        # Save results CSV
        results_df = pd.DataFrame(self.results)
        results_file = self.output_dir / "regional_predictors_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"  ✓ Saved results: {results_file}")
        
        # Print aggregate statistics
        print(f"\nAggregate Statistics:")
        print(f"  Train MAE: {results_df['mae_train'].mean():.3f} ± {results_df['mae_train'].std():.3f} years")
        print(f"  Valid MAE: {results_df['mae_valid'].mean():.3f} ± {results_df['mae_valid'].std():.3f} years")
        print(f"  Test MAE:  {results_df['mae_test'].mean():.3f} ± {results_df['mae_test'].std():.3f} years")
        print(f"\n  Train R²:  {results_df['r2_train'].mean():.3f} ± {results_df['r2_train'].std():.3f}")
        print(f"  Valid R²:  {results_df['r2_valid'].mean():.3f} ± {results_df['r2_valid'].std():.3f}")
        print(f"  Test R²:   {results_df['r2_test'].mean():.3f} ± {results_df['r2_test'].std():.3f}")
        
        # Save summary
        summary = {
            'n_regions': self.num_regions,
            'n_train': len(self.train_idx),
            'n_valid': len(self.valid_idx),
            'n_test': len(self.test_idx),
            'model_type': 'Ridge',
            'alpha': self.alpha,
            'aggregate_stats': {
                'train_mae_mean': float(results_df['mae_train'].mean()),
                'valid_mae_mean': float(results_df['mae_valid'].mean()),
                'test_mae_mean': float(results_df['mae_test'].mean()),
                'train_r2_mean': float(results_df['r2_train'].mean()),
                'valid_r2_mean': float(results_df['r2_valid'].mean()),
                'test_r2_mean': float(results_df['r2_test'].mean()),
            }
        }
        
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary: {summary_file}")
        
        return results_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train regional brain age predictors'
    )
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for trained models')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Ridge regression alpha parameter')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TRAIN REGIONAL BRAIN AGE PREDICTORS")
    print("="*80)
    
    # Initialize trainer
    trainer = RegionalPredictorTrainer(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        alpha=args.alpha,
        random_seed=args.random_seed
    )
    
    # Run pipeline
    trainer.load_data()
    trainer.split_data()
    trainer.train_models()
    trainer.calculate_bias_correction()
    results_df = trainer.save_results()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput: {args.output_dir}/")
    print(f"  ├── data_split.json")
    print(f"  ├── regional_predictors_results.csv")
    print(f"  ├── training_summary.json")
    print(f"  ├── models/ (32 .pkl files)")
    print(f"  └── bias_correction/")
    print(f"      └── regional_bias_correction_params.json")
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
