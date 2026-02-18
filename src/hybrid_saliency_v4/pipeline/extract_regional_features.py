#!/usr/bin/env python3
"""
Extract Regional Features from Hybrid Saliency V4 Model

This script extracts 32 regional feature vectors from the trained
Hybrid Saliency V4 model for use in regional predictor training.

Features are extracted from the features_concat layer which combines:
- Original feature path (256-dim)
- Saliency map path (256-dim)
= 512-dim per region × 32 regions

Author: Anonymous
Date: 2026-02-17
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import json
import argparse
import sys

# Add package to path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

from hybrid_saliency_v4.model.hybrid_saliency_enhanced_v4 import HybridSaliencyEnhanced


class RegionalFeatureExtractor:
    """Extract regional features from trained Hybrid Saliency V4 model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        unet_checkpoint: str,
        output_dir: str,
        device: str = 'cuda'
    ):
        """
        Initialize feature extractor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            unet_checkpoint: Path to UNet checkpoint
            output_dir: Directory to save extracted features
            device: Device to use ('cuda' or 'cpu')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.unet_checkpoint = Path(unet_checkpoint)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        self.model = None
        self.config = {}
        
    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")
        
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Get configuration
        self.config = checkpoint.get('config', {})
        num_regions = self.config.get('num_regions', 32)
        embedding_dim = self.config.get('embedding_dim', 256)
        top_k = self.config.get('top_k', 128)
        
        print(f"  Model config:")
        print(f"    num_regions: {num_regions}")
        print(f"    embedding_dim: {embedding_dim}")
        print(f"    top_k: {top_k}")
        
        # Initialize model
        self.model = HybridSaliencyEnhanced(
            unet_checkpoint=str(self.unet_checkpoint),
            num_regions=num_regions,
            embedding_dim=embedding_dim,
            top_k=top_k,
            freeze_unet=True  # Freeze UNet during extraction
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        
    def load_dataset(self, data_dir: str, metadata_csv: str):
        """
        Load dataset for feature extraction.
        
        Args:
            data_dir: Directory containing MRI files
            metadata_csv: CSV file with metadata
            
        Returns:
            List of sample dictionaries
        """
        print(f"\nLoading dataset...")
        print(f"  Data dir: {data_dir}")
        print(f"  Metadata: {metadata_csv}")
        
        df = pd.read_csv(metadata_csv)
        df.columns = df.columns.str.upper()
        
        print(f"  Total samples: {len(df)}")
        print(f"  Age range: {df['AGE'].min():.0f} - {df['AGE'].max():.0f} years")
        print(f"  Mean age: {df['AGE'].mean():.1f} ± {df['AGE'].std():.1f} years")
        
        # Prepare samples
        samples = []
        data_path = Path(data_dir)
        
        for idx, row in df.iterrows():
            mri_path = data_path / row['FILENAME']
            if mri_path.exists():
                samples.append({
                    'idx': idx,
                    'mri_path': str(mri_path),
                    'age': float(row['AGE']),
                    'sex': row['SEX'],
                    'dataset': row.get('DATASET', 'unknown'),
                    'subject_id': row.get('SUBJECT_ID', f'subject_{idx}')
                })
        
        print(f"  Valid samples: {len(samples)}")
        return samples
    
    def extract_features(self, samples: list):
        """
        Extract regional features from all samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            all_features: numpy array [n_samples, num_regions, 512]
            all_metadata: list of metadata dictionaries
        """
        num_regions = self.config.get('num_regions', 32)
        
        print(f"\nExtracting regional features...")
        print(f"  This will extract {num_regions} feature vectors per sample")
        print(f"  Each vector has 512 dimensions (256 original + 256 saliency)")
        
        all_features = []
        all_metadata = []
        
        with torch.no_grad():
            for sample in tqdm(samples, desc="Processing"):
                try:
                    # Load and preprocess MRI
                    mri_data = self._load_and_preprocess_mri(sample['mri_path'])
                    mri_tensor = torch.from_numpy(mri_data).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    
                    # Extract features using hook
                    features = self._extract_features_with_hook(mri_tensor)
                    
                    if features is not None:
                        all_features.append(features)
                        all_metadata.append(sample)
                    
                except Exception as e:
                    print(f"\n  ✗ Error processing {sample['mri_path']}: {e}")
                    continue
        
        print(f"\n✓ Extracted features from {len(all_features)} samples")
        
        # Convert to numpy array
        all_features = np.array(all_features)  # [n_samples, num_regions, 512]
        
        return all_features, all_metadata
    
    def _load_and_preprocess_mri(self, mri_path: str) -> np.ndarray:
        """Load and preprocess MRI scan."""
        # Load MRI
        mri_nii = nib.load(mri_path)
        mri_data = mri_nii.get_fdata().astype(np.float32)
        
        # Normalize
        mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min() + 1e-8)
        
        # Resize if needed
        if mri_data.shape != (128, 128, 128):
            import torch.nn.functional as F
            mri_tensor = torch.from_numpy(mri_data).unsqueeze(0).unsqueeze(0)
            mri_tensor = F.interpolate(
                mri_tensor,
                size=(128, 128, 128),
                mode='trilinear',
                align_corners=False
            )
            mri_data = mri_tensor.squeeze().numpy()
        
        return mri_data
    
    def _extract_features_with_hook(self, mri_tensor: torch.Tensor) -> np.ndarray:
        """Extract features using forward hook."""
        features_container = []
        
        def hook_fn(module, input, output):
            # Hook into gated_fusion to get features_concat
            # Input to gated_fusion is features_concat [B, 32, 512]
            if isinstance(input, tuple):
                features_container.append(input[0].detach().cpu())
            else:
                features_container.append(input.detach().cpu())
        
        # Register hook on gated_fusion (which takes features_concat as input)
        hook = self.model.gated_fusion.register_forward_hook(hook_fn)
        
        # Forward pass
        _ = self.model(mri_tensor)
        
        # Remove hook
        hook.remove()
        
        if len(features_container) > 0:
            # features shape: [1, num_regions, 512]
            features = features_container[0].squeeze(0).numpy()  # [num_regions, 512]
            return features
        
        return None
    
    def save_features(self, features: np.ndarray, metadata: list):
        """
        Save extracted features and metadata.
        
        Args:
            features: numpy array [n_samples, num_regions, 512]
            metadata: list of metadata dictionaries
        """
        print(f"\nSaving regional features...")
        print(f"  Feature array shape: {features.shape}")
        
        # Save combined features
        features_file = self.output_dir / "regional_features.npy"
        np.save(features_file, features)
        print(f"  ✓ Saved features: {features_file}")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_file = self.output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        print(f"  ✓ Saved metadata: {metadata_file}")
        
        # Save individual region features
        regions_dir = self.output_dir / "regions"
        regions_dir.mkdir(exist_ok=True)
        
        num_regions = features.shape[1]
        for region_idx in range(num_regions):
            region_features = features[:, region_idx, :]  # [n_samples, 512]
            region_file = regions_dir / f"region_{region_idx:02d}_features.npy"
            np.save(region_file, region_features)
        
        print(f"  ✓ Saved {num_regions} individual region files to: {regions_dir}")
        
        # Save summary
        summary = {
            'n_samples': len(features),
            'num_regions': num_regions,
            'feature_dim': features.shape[2],
            'age_range': [float(metadata_df['age'].min()), float(metadata_df['age'].max())],
            'age_mean': float(metadata_df['age'].mean()),
            'age_std': float(metadata_df['age'].std()),
            'checkpoint': str(self.checkpoint_path),
            'config': self.config
        }
        
        summary_file = self.output_dir / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Saved summary: {summary_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract regional features from Hybrid Saliency V4 model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--unet_checkpoint',
        type=str,
        required=True,
        help='Path to UNet checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing MRI files'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='CSV file with metadata'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for features'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("EXTRACT REGIONAL FEATURES - HYBRID SALIENCY V4")
    print("="*80)
    
    # Initialize extractor
    extractor = RegionalFeatureExtractor(
        checkpoint_path=args.checkpoint,
        unet_checkpoint=args.unet_checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Load model
    extractor.load_model()
    
    # Load dataset
    samples = extractor.load_dataset(args.data_dir, args.metadata)
    
    # Extract features
    features, metadata = extractor.extract_features(samples)
    
    # Save results
    extractor.save_features(features, metadata)
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── regional_features.npy          [{features.shape}]")
    print(f"  ├── metadata.csv                   [{len(metadata)} samples]")
    print(f"  ├── extraction_summary.json")
    print(f"  └── regions/")
    print(f"      ├── region_00_features.npy     [{features.shape[0]}, {features.shape[2]}]")
    print(f"      └── ... (32 files total)")
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
