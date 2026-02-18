#!/usr/bin/env python3
"""
Generate Regional Brain Age Predictions

Uses trained regional predictors to generate brain age predictions
with bias correction for new datasets.

This is a wrapper that uses the prediction functionality from
experiments.ad_prediction module.

Author: Anonymous
Date: 2026-02-17
"""

import sys
from pathlib import Path

# Add package to path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

# Import from experiments module
from vcda_net.experiments.ad_prediction.predict_regional_brain_age import (
    RegionalBrainAgePredictor,
    create_results_dataframe,
    calculate_statistics
)

import argparse
import pandas as pd
import numpy as np
import json


class RegionalPredictor:
    """Wrapper for regional brain age prediction."""
    
    def __init__(
        self,
        features_dir: str,
        metadata_file: str,
        models_dir: str,
        bias_params_file: str,
        output_file: str,
        num_regions: int = 32
    ):
        """
        Initialize predictor.
        
        Args:
            features_dir: Directory containing regional features
            metadata_file: CSV file with metadata
            models_dir: Directory containing trained models
            bias_params_file: JSON file with bias correction parameters
            output_file: Output CSV file path
            num_regions: Number of brain regions
        """
        self.predictor = RegionalBrainAgePredictor(
            models_dir=models_dir,
            bias_params_file=bias_params_file,
            num_regions=num_regions
        )
        
        self.features_dir = Path(features_dir)
        self.metadata_file = Path(metadata_file)
        self.output_file = Path(output_file)
        self.num_regions = num_regions
        
    def run(self):
        """Run prediction pipeline."""
        print("="*80)
        print("REGIONAL BRAIN AGE PREDICTION")
        print("="*80)
        
        # Load models and bias parameters
        print("\nLoading models and bias correction parameters...")
        self.predictor.load_models()
        self.predictor.load_bias_params()
        
        # Load metadata
        print("\nLoading metadata...")
        metadata_df = pd.read_csv(self.metadata_file)
        n_samples = len(metadata_df)
        true_ages = metadata_df['age'].values
        
        print(f"  Samples: {n_samples}")
        print(f"  Age range: {true_ages.min():.0f} - {true_ages.max():.0f} years")
        
        # Generate predictions
        print("\nGenerating predictions...")
        all_raw, all_corrected = self.predictor.predict_all_regions(
            features_dir=self.features_dir,
            n_samples=n_samples,
            apply_correction=True
        )
        
        print(f"  ✓ Generated predictions for all {self.num_regions} regions")
        
        # Create results dataframe
        print("\nCreating results dataframe...")
        results_df = create_results_dataframe(
            metadata_df=metadata_df,
            true_ages=true_ages,
            all_raw_predictions=all_raw,
            all_corrected_predictions=all_corrected,
            num_regions=self.num_regions
        )
        
        # Calculate statistics
        stats = calculate_statistics(
            results_df=results_df,
            true_ages=true_ages,
            num_regions=self.num_regions
        )
        
        # Print summary
        print("\nStatistics:")
        print(f"  MAE (corrected):  {stats['ensemble']['corrected']['mae']:.3f} years")
        print(f"  RMSE (corrected): {stats['ensemble']['corrected']['rmse']:.3f} years")
        print(f"  Bias (corrected): {stats['ensemble']['corrected']['bias']:.3f} years")
        print(f"  R (corrected):    {stats['ensemble']['corrected']['r']:.4f}")
        
        # Save results
        print("\nSaving results...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(self.output_file, index=False)
        print(f"  ✓ Saved predictions: {self.output_file}")
        
        # Save statistics
        stats_file = self.output_file.parent / f"{self.output_file.stem}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved statistics: {stats_file}")
        
        print("\n" + "="*80)
        print("PREDICTION COMPLETE!")
        print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate regional brain age predictions'
    )
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing regional features')
    parser.add_argument('--metadata_file', type=str, required=True,
                       help='Metadata CSV file')
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--bias_params_file', type=str, required=True,
                       help='Bias correction parameters JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--num_regions', type=int, default=32,
                       help='Number of brain regions')
    
    args = parser.parse_args()
    
    # Initialize and run predictor
    predictor = RegionalPredictor(
        features_dir=args.features_dir,
        metadata_file=args.metadata_file,
        models_dir=args.models_dir,
        bias_params_file=args.bias_params_file,
        output_file=args.output_file,
        num_regions=args.num_regions
    )
    
    predictor.run()


if __name__ == '__main__':
    main()
