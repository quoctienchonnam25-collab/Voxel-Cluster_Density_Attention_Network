"""
Training Script for Saliency Map Enhanced Hybrid Model

This script trains the VCDANet model with:
- Dual-path features: Original (256) + Saliency Map (256) = 512-dim
- Per-channel activation magnitude-based density attention maps (NO gradients!)
- Top-K voxel extraction (K=128)
- Gaussian distance matrix encoding
- Unified graph with enriched node features

Usage:
    python train.py \
        --data_dir /path/to/data \
        --metadata_csv /path/to/metadata.csv \
        --unet_checkpoint checkpoints/IXI_3dunet_best_model.pth \
        --top_k 128 \
        --sigma 10.0 \
        --epochs 300 \
        --batch_size 4 \
        --lr 1e-3

Author: Anonymous
Date: 2026-02-15
Version: 4.0.2 (Saliency Map - renamed from GradCAM for accuracy)
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
import nibabel as nib
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from local modules
from vcda_net.model.vcda_net import VCDANet
from vcda_net.training.loss_functions import get_loss_function, HuberLoss


class SynthesisDataset(Dataset):
    """Synthesis Dataset loader for ABIDE1+ADNI+IXI"""
    def __init__(self, data_dir, metadata_csv, split='train', train_ratio=0.7, val_ratio=0.15,
                 stratify=True, random_seed=42):
        """
        Args:
            data_dir: Path to dataset directory (contains ABIDE1_ADNI_IXI_turboprep/)
            metadata_csv: Path to CSV metadata file
            split: 'train', 'val', or 'test'
            train_ratio: Training split ratio (default: 0.7)
            val_ratio: Validation split ratio (default: 0.15, remaining 0.15 for test)
            stratify: Whether to stratify by dataset and age groups (default: True)
            random_seed: Random seed for reproducibility
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        self.data_dir = Path(data_dir)
        np.random.seed(random_seed)
        
        # Read metadata CSV
        df = pd.read_csv(metadata_csv)
        
        # Normalize column names to uppercase for consistency
        df.columns = df.columns.str.upper()
        print(f"\nLoading Synthesis Dataset from: {metadata_csv}")
        print(f"Total samples in metadata: {len(df)}")
        # Count datasets
        dataset_counts = df['DATASET'].value_counts()
        dataset_str = ", ".join([f"{k}={v}" for k, v in dataset_counts.items()])
        print(f"Datasets: {dataset_str}")
        
        # Create full paths to MRI files (data_dir already points to ABIDE1_ADNI_IXI_turboprep)
        df['mri_path'] = df['FILENAME'].apply(lambda x: str(self.data_dir / x))
        
        # Verify files exist
        df['exists'] = df['mri_path'].apply(lambda x: Path(x).exists())
        missing = len(df[~df['exists']])
        if missing > 0:
            print(f"⚠ Warning: {missing} files not found, removing from dataset")
            df = df[df['exists']].copy()
        
        # Create age groups for stratification
        if stratify:
            # Use wider age bins to avoid small groups
            df['age_group'] = pd.cut(df['AGE'], bins=[0, 18, 40, 70, 100],
                                     labels=['youth', 'adult', 'middle_age', 'senior'])
            df['stratify_key'] = df['DATASET'] + '_' + df['age_group'].astype(str)
            
            # Check group sizes
            group_sizes = df['stratify_key'].value_counts()
            min_group_size = group_sizes.min()
            
            if min_group_size < 2:
                print(f"⚠ Warning: Some stratification groups have only {min_group_size} sample(s)")
                print("  Falling back to stratification by DATASET only (not age groups)")
                df['stratify_key'] = df['DATASET']
                stratify_by_age = False
            else:
                stratify_by_age = True
        
        # Split data: train, val, test
        test_ratio = 1 - train_ratio - val_ratio
        
        # Handle floating point errors: if test_ratio is very small or negative, set to 0
        if abs(test_ratio) < 0.01:
            test_ratio = 0.0
            print(f"✓ No test set (using {train_ratio:.0%}/{val_ratio:.0%} train/val split)")
        
        if stratify:
            try:
                if test_ratio == 0.0:
                    # No test set - direct train/val split
                    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                    train_df, val_df = train_test_split(
                        df, test_size=val_ratio_adjusted, random_state=random_seed,
                        stratify=df['stratify_key']
                    )
                    test_df = pd.DataFrame()  # Empty test set
                else:
                    # Stratified split into train+val and test first
                    train_val_df, test_df = train_test_split(
                        df, test_size=test_ratio, random_state=random_seed,
                        stratify=df['stratify_key']
                    )
                    
                    # Then split train+val into train and val
                    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=val_ratio_adjusted, random_state=random_seed,
                        stratify=train_val_df['stratify_key']
                    )
                
                print(f"✓ Stratified split successful (by {'dataset + age' if stratify_by_age else 'dataset only'})")
                
            except ValueError as e:
                print(f"⚠ Warning: Stratified split failed: {str(e)}")
                print("  Falling back to random split...")
                
                # Fallback to simple random split
                if test_ratio == 0.0:
                    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                    train_df, val_df = train_test_split(
                        df, test_size=val_ratio_adjusted, random_state=random_seed
                    )
                    test_df = pd.DataFrame()
                else:
                    train_val_df, test_df = train_test_split(
                        df, test_size=test_ratio, random_state=random_seed
                    )
                    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=val_ratio_adjusted, random_state=random_seed
                    )
                print("✓ Random split completed")
        else:
            # Simple random split
            if test_ratio == 0.0:
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                train_df, val_df = train_test_split(
                    df, test_size=val_ratio_adjusted, random_state=random_seed
                )
                test_df = pd.DataFrame()
            else:
                train_val_df, test_df = train_test_split(
                    df, test_size=test_ratio, random_state=random_seed
                )
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                train_df, val_df = train_test_split(
                    train_val_df, test_size=val_ratio_adjusted, random_state=random_seed
                )
            print("✓ Random split completed")
        
        # Select split
        if split == 'train':
            self.df = train_df.reset_index(drop=True)
        elif split == 'val':
            self.df = val_df.reset_index(drop=True)
        else:  # test
            self.df = test_df.reset_index(drop=True)
        
        # Prepare samples
        self.samples = []
        
        # Determine subject_id column name (support both formats)
        subj_col = 'SUBJECT_ID' if 'SUBJECT_ID' in self.df.columns else 'SUB_ID'
        
        for _, row in self.df.iterrows():
            self.samples.append({
                'mri_path': row['mri_path'],
                'age': float(row['AGE']),
                'dataset': row['DATASET'],
                'subject_id': row[subj_col],
                'sex': row['SEX']
            })
        
        # Print split statistics
        print(f"\n{split.upper()} SET Statistics:")
        print(f"  Total samples: {len(self.samples)}")
        
        # Only print detailed stats if dataset is not empty
        if len(self.samples) > 0:
            print(f"  Age range: {self.df['AGE'].min():.1f} - {self.df['AGE'].max():.1f} years")
            print(f"  Mean age: {self.df['AGE'].mean():.1f} ± {self.df['AGE'].std():.1f} years")
            print(f"  Dataset distribution:")
            for dataset in ['ABIDE1', 'ADNI', 'IXI']:
                count = len(self.df[self.df['DATASET'] == dataset])
                if count > 0:
                    age_mean = self.df[self.df['DATASET'] == dataset]['AGE'].mean()
                    print(f"    {dataset}: {count} samples (mean age: {age_mean:.1f})")
            print(f"  Sex distribution: M={len(self.df[self.df['SEX']=='M'])}, "
                  f"F={len(self.df[self.df['SEX']=='F'])}")
        else:
            print(f"  (Empty dataset - no test set for 80/20 split)")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load MRI
        mri_nii = nib.load(sample['mri_path'])
        mri_data = mri_nii.get_fdata().astype(np.float32)
        
        # Normalize
        mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min() + 1e-8)
        
        # Resize if needed
        if mri_data.shape != (128, 128, 128):
            import torch.nn.functional as F
            mri_tensor = torch.from_numpy(mri_data).unsqueeze(0).unsqueeze(0)
            mri_tensor = F.interpolate(mri_tensor, size=(128, 128, 128),
                                      mode='trilinear', align_corners=False)
            mri_data = mri_tensor.squeeze().numpy()
        
        mri_tensor = torch.from_numpy(mri_data).unsqueeze(0).float()  # [1, 128, 128, 128]
        age_tensor = torch.tensor([sample['age']], dtype=torch.float32)
        
        return mri_tensor, age_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions, targets = [], []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for mri, age in pbar:
        mri, age = mri.to(device), age.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred = model(mri)
        loss = criterion(pred, age)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions.extend(pred.detach().cpu().numpy().flatten())
        targets.extend(age.detach().cpu().numpy().flatten())
        
        # Update progress
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r, _ = pearsonr(predictions, targets)
    
    # R² (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return avg_loss, mae, rmse, r, r2


def validate(model, dataloader, criterion, device, epoch, split='Val'):
    """Validate model"""
    model.eval()
    total_loss = 0
    model.eval()
    total_loss = 0
    predictions, targets, gates = [], [], []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [{split}]')
        for mri, age in pbar:
            mri, age = mri.to(device), age.to(device)
            
            # Model in eval mode returns (prediction, gate_mean)
            output = model(mri)
            if isinstance(output, tuple):
                pred, gate = output
                gates.extend(gate.cpu().numpy().flatten())
            else:
                pred = output
                
            loss = criterion(pred, age)
            
            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(age.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    avg_gate = np.mean(gates) if gates else 0.0
    
    # Handle empty dataloader (e.g., no test set in 80/20 split)
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, predictions, targets, avg_gate
    
    avg_loss = total_loss / len(dataloader)
    mae = np.mean(np.abs(predictions - targets)) if len(predictions) > 0 else 0.0
    rmse = np.sqrt(np.mean((predictions - targets) ** 2)) if len(predictions) > 0 else 0.0
    r, _ = pearsonr(predictions, targets) if len(predictions) > 1 else (0.0, 0.0)
    
    # R² (coefficient of determination)
    if len(predictions) > 1:
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        r2 = 0.0
    
    return avg_loss, mae, rmse, r, r2, predictions, targets, avg_gate


def plot_training_curves(history, save_path):
    """
    Plot training and validation loss/MAE curves
    
    Args:
        history: dict with train_loss, val_loss, train_mae, val_mae
        save_path: path to save figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (L1)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE (years)', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Train Saliency Map Enhanced Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to synthesis_dataset directory')
    parser.add_argument('--metadata_csv', type=str, required=True,
                       help='Path to ABIDE1_ADNI_IXI_turboprep_metadata.csv')
    parser.add_argument('--unet_checkpoint', type=str, required=True,
                       help='Path to pretrained UNet checkpoint')

    
    # Model
    parser.add_argument('--num_regions', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--resnet_depth', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'])
    
    # UNet Fine-tuning
    parser.add_argument('--freeze_unet', action='store_true', default=False,
                       help='Freeze UNet weights (use as fixed feature extractor)')
    parser.add_argument('--unfreeze_unet', dest='freeze_unet', action='store_false',
                       help='Allow UNet fine-tuning (default)')
    parser.add_argument('--unet_lr', type=float, default=None,
                       help='Learning rate for UNet (if None, use main lr)')
    parser.add_argument('--unfreeze_unet_after_epoch', type=int, default=0,
                       help='Unfreeze UNet after this epoch (0=unfreeze from start, -1=keep frozen)')
    
    # Saliency Map
    parser.add_argument('--top_k', type=int, default=128,
                       help='Number of top voxels from density attention map')
    parser.add_argument('--sigma', type=float, default=10.0,
                       help='Gaussian bandwidth for distance matrix')
    parser.add_argument('--matrix_resize', type=int, default=64,
                       help='Resize matrix to this size')
    
    # GNN
    parser.add_argument('--edge_num', type=int, default=31)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--use_edge_attention', action='store_true',
                       help='Use edge attention mechanism')
    
    # Transformer
    parser.add_argument('--transformer_d_model', type=int, default=256)
    parser.add_argument('--transformer_nhead', type=int, default=8)
    parser.add_argument('--transformer_num_layers', type=int, default=3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Loss Function
    parser.add_argument('--loss_type', type=str, default='huber',
                       choices=['l1', 'mse', 'huber', 'smooth_l1', 'logcosh'],
                       help='Loss function type (default: huber)')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss (default: 1.0)')
    parser.add_argument('--smooth_l1_beta', type=float, default=1.0,
                       help='Beta parameter for Smooth L1 loss (default: 1.0)')
    
    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15, remaining 0.15 for test)')
    parser.add_argument('--stratify', action='store_true', default=True,
                       help='Stratify split by dataset and age groups')
    parser.add_argument('--no_stratify', dest='stratify', action='store_false',
                       help='Disable stratified splitting')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for data splitting')
    
    # Early stopping
    parser.add_argument('--early_stop', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience (epochs without improvement)')
    
    # Other
    parser.add_argument('--output_dir', type=str, default='vcda_runs',
                       help='Output directory')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f'vcda_net_{timestamp}'
    
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Logging
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    print("="*70)
    print("Training Saliency Map Enhanced Hybrid Model (V4)")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Top-K voxels: {args.top_k}")
    print(f"Gaussian sigma: {args.sigma}")
    print(f"Matrix resize: {args.matrix_resize}")
    print("⚠️  Note: Using activation magnitude-based density attention maps (NOT GradCAM)")
    print("="*70)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Datasets with stratified split (train/val/test: 70/15/15)
    print("\nLoading Synthesis Datasets...")
    print(f"Data directory: {args.data_dir}")
    print(f"Metadata CSV: {args.metadata_csv}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={1-args.train_ratio-args.val_ratio}")
    print(f"Stratified splitting: {'Enabled' if args.stratify else 'Disabled'}")
    print(f"Random seed: {args.random_seed}")
    print("="*70)
    
    train_dataset = SynthesisDataset(
        args.data_dir, 
        args.metadata_csv,
        split='train',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stratify=args.stratify,
        random_seed=args.random_seed
    )
    
    val_dataset = SynthesisDataset(
        args.data_dir,
        args.metadata_csv,
        split='val',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stratify=args.stratify,
        random_seed=args.random_seed
    )
    
    test_dataset = SynthesisDataset(
        args.data_dir,
        args.metadata_csv,
        split='test',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stratify=args.stratify,
        random_seed=args.random_seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, 
                              drop_last=True)  # Drop last incomplete batch to avoid BatchNorm error
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    print("\nInitializing model...")
    
    # Determine initial freeze state
    initial_freeze = args.freeze_unet or (args.unfreeze_unet_after_epoch > 0)
    
    model = VCDANet(
        unet_checkpoint=args.unet_checkpoint,
        num_regions=args.num_regions,
        embedding_dim=args.embedding_dim,
        resnet_depth=args.resnet_depth,
        top_k=args.top_k,
        sigma=args.sigma,
        matrix_resize=args.matrix_resize,
        edge_num=args.edge_num,
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        use_edge_attention=args.use_edge_attention,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        dropout=args.dropout,
        freeze_unet=initial_freeze  # Pass freeze parameter
    ).to(device)
    
    # Apply gradual unfreezing if needed
    if args.unfreeze_unet_after_epoch > 0:
        print(f"\n⏰ Gradual Unfreezing: UNet will be unfrozen after epoch {args.unfreeze_unet_after_epoch}")
    elif args.unfreeze_unet_after_epoch == -1:
        print(f"\n🔒 UNet will remain frozen throughout training")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unet_params = sum(p.numel() for p in model.unet_extractor.parameters())
    unet_trainable = sum(p.numel() for p in model.unet_extractor.parameters() if p.requires_grad)
    
    print(f"\nParameter Summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  UNet parameters:      {unet_params:,} ({'trainable' if unet_trainable > 0 else 'frozen'})")
    print(f"  Other parameters:     {total_params - unet_params:,} (trainable)")
    
    # Optimizer with differential learning rates
    if args.unet_lr is not None and unet_trainable > 0:
        # Separate UNet and other parameters
        unet_params_list = list(model.unet_extractor.parameters())
        other_params_list = [p for n, p in model.named_parameters() 
                            if 'unet_extractor' not in n]
        
        param_groups = [
            {'params': unet_params_list, 'lr': args.unet_lr, 'name': 'unet'},
            {'params': other_params_list, 'lr': args.lr, 'name': 'other'}
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"\n✓ Using differential learning rates:")
        print(f"    UNet LR:  {args.unet_lr:.2e}")
        print(f"    Other LR: {args.lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        print(f"\n✓ Using single learning rate: {args.lr:.2e}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    # Loss function
    print(f"\n{'='*70}")
    print("Loss Function Configuration")
    print(f"{'='*70}")
    
    if args.loss_type == 'huber':
        criterion = get_loss_function('huber', delta=args.huber_delta)
        print(f"Loss: Huber Loss (delta={args.huber_delta})")
        print(f"  - Robust to outliers from multi-dataset")
        print(f"  - Faster convergence than L1")
    elif args.loss_type == 'smooth_l1':
        criterion = get_loss_function('smooth_l1', beta=args.smooth_l1_beta)
        print(f"Loss: Smooth L1 Loss (beta={args.smooth_l1_beta})")
        print(f"  - Similar to Huber with different parameterization")
    elif args.loss_type == 'l1':
        criterion = get_loss_function('l1')
        print(f"Loss: L1 Loss (MAE)")
        print(f"  - Simple and robust baseline")
    elif args.loss_type == 'mse':
        criterion = get_loss_function('mse')
        print(f"Loss: MSE Loss (L2)")
        print(f"  - Sensitive to outliers, use with caution")
    elif args.loss_type == 'logcosh':
        criterion = get_loss_function('logcosh')
        print(f"Loss: LogCosh Loss")
        print(f"  - Smooth and robust, no hyperparameters")
    
    print(f"{'='*70}\n")
    
    # Training loop
    print("\nStarting training...")
    training_start_time = datetime.now()
    training_start_timestamp = training_start_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training started at: {training_start_timestamp}")
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    history = {
        'train_loss': [], 'train_mae': [], 'train_rmse': [], 'train_r': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r': [], 'val_r2': [], 'val_gate': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Gradual unfreezing logic
        if args.unfreeze_unet_after_epoch > 0 and epoch == args.unfreeze_unet_after_epoch + 1:
            print(f"\n{'='*70}")
            print(f"🔓 UNFREEZING UNet at epoch {epoch}")
            print(f"{'='*70}")
            
            # Unfreeze UNet parameters
            for param in model.unet_extractor.parameters():
                param.requires_grad = True
            
            # Update optimizer with differential learning rate if specified
            if args.unet_lr is not None:
                unet_params_list = list(model.unet_extractor.parameters())
                other_params_list = [p for n, p in model.named_parameters() 
                                    if 'unet_extractor' not in n]
                
                param_groups = [
                    {'params': unet_params_list, 'lr': args.unet_lr, 'name': 'unet'},
                    {'params': other_params_list, 'lr': args.lr, 'name': 'other'}
                ]
                
                # Create new optimizer
                optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=20
                )
                
                print(f"✓ Updated optimizer with differential learning rates:")
                print(f"    UNet LR:  {args.unet_lr:.2e}")
                print(f"    Other LR: {args.lr:.2e}")
            else:
                # Just update optimizer to include new parameters
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                              weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=20
                )
                print(f"✓ Updated optimizer (single LR: {args.lr:.2e})")
            
            # Print parameter status
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            unet_trainable = sum(p.numel() for p in model.unet_extractor.parameters() if p.requires_grad)
            print(f"✓ Trainable parameters: {trainable_params:,} (UNet: {unet_trainable:,})")
            print(f"{'='*70}\n")
        
        # Train
        train_loss, train_mae, train_rmse, train_r, train_r2 = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_mae, val_rmse, val_r, val_r2, val_preds, val_targets, val_gate = validate(
            model, val_loader, criterion, device, epoch, 'Val'
        )
        
        # Update scheduler
        scheduler.step(val_mae)
        
        # Log (convert to Python float for JSON serialization)
        history['train_loss'].append(float(train_loss))
        history['train_mae'].append(float(train_mae))
        history['train_rmse'].append(float(train_rmse))
        history['train_r'].append(float(train_r))
        history['train_r2'].append(float(train_r2))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_rmse'].append(float(val_rmse))
        history['val_r'].append(float(val_r))
        history['val_r2'].append(float(val_r2))
        history['val_gate'].append(float(val_gate))
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/train', train_rmse, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('Correlation/train', train_r, epoch)
        writer.add_scalar('Correlation/val', val_r, epoch)
        writer.add_scalar('R2/train', train_r2, epoch)
        writer.add_scalar('R2/val', val_r2, epoch)
        writer.add_scalar('Gate/val_mean', val_gate, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R: {train_r:.3f}, R²: {train_r2:.3f}")
        print(f"  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, R: {val_r:.3f}, R²: {val_r2:.3f}")
        print(f"  Gate  - Mean: {val_gate:.3f} (>0.5: Transformer, <0.5: Bottleneck)")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            
            # Calculate training duration so far
            current_time = datetime.now()
            elapsed_time = current_time - training_start_time
            elapsed_hours = elapsed_time.total_seconds() / 3600
            
            torch.save({
                # Model state
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                # Performance metrics
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r': val_r,
                'val_r2': val_r2,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r': train_r,
                'train_r2': train_r2,
                
                # Training metadata
                'training_start_time': training_start_timestamp,
                'checkpoint_save_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'training_duration_hours': elapsed_hours,
                'epochs_trained': epoch,
                
                # Model configuration
                'config': {
                    'num_regions': args.num_regions,
                    'embedding_dim': args.embedding_dim,
                    'resnet_depth': args.resnet_depth,
                    'top_k': args.top_k,
                    'sigma': args.sigma,
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'unet_lr': args.unet_lr,
                    'loss_type': args.loss_type,
                    'dropout': args.dropout
                },
                
                # Dataset info
                'dataset': {
                    'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'test_samples': len(test_dataset),
                    'data_dir': str(args.data_dir),
                    'random_seed': args.random_seed
                }
            }, checkpoint_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! MAE: {val_mae:.2f} (trained {elapsed_hours:.2f}h)")
        else:
            epochs_no_improve += 1
            if args.early_stop:
                print(f"  No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping check
        if args.early_stop and epochs_no_improve >= args.patience:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered!")
            print(f"No improvement for {epochs_no_improve} epochs (patience={args.patience})")
            print(f"Best validation MAE: {best_val_mae:.2f}")
            print(f"{'='*70}")
            break
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Final evaluation on best model
    print("\n" + "="*70)
    print("Final Evaluation on Best Model...")
    print("="*70)
    
    checkpoint = torch.load(checkpoint_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_loss, final_val_mae, final_val_rmse, final_val_r, final_val_r2, val_preds, val_targets, val_gate = validate(
        model, val_loader, criterion, device, 0, 'Final Val'
    )
    
    # Only evaluate on test set if it exists
    if len(test_loader) > 0:
        final_test_loss, final_test_mae, final_test_rmse, final_test_r, final_test_r2, test_preds, test_targets, test_gate = validate(
            model, test_loader, criterion, device, 0, 'Final Test'
        )
    else:
        print("Skipping test evaluation (no test set - 80/20 split)")
        final_test_loss = final_test_mae = final_test_rmse = final_test_r = final_test_r2 = 0.0
        test_preds = test_targets = np.array([])
        test_gate = 0.0
    
    print(f"\nFinal Results:")
    print(f"  Validation:")
    print(f"    MAE:  {final_val_mae:.2f} years")
    print(f"    RMSE: {final_val_rmse:.2f} years")
    print(f"    R (Pearson correlation): {final_val_r:.3f}")
    print(f"    R² (coefficient of determination): {final_val_r2:.3f}")
    
    if len(test_loader) > 0:
        print(f"  Test:")
        print(f"    MAE:  {final_test_mae:.2f} years")
        print(f"    RMSE: {final_test_rmse:.2f} years")
        print(f"    R (Pearson correlation): {final_test_r:.3f}")
        print(f"    R² (coefficient of determination): {final_test_r2:.3f}")
    else:
        print(f"  Test: N/A (no test set)")
    print(f"  Best epoch: {checkpoint['epoch']}")
    
    # Save final results
    results = {
        'final_val_mae': float(final_val_mae),
        'final_val_rmse': float(final_val_rmse),
        'final_val_correlation': float(final_val_r),
        'final_val_r2': float(final_val_r2),
        'final_test_mae': float(final_test_mae),
        'final_test_rmse': float(final_test_rmse),
        'final_test_correlation': float(final_test_r),
        'final_test_r2': float(final_test_r2),
        'best_val_mae': float(best_val_mae),
        'best_epoch': int(checkpoint['epoch']),
        'data_split': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': 1 - args.train_ratio - args.val_ratio,
            'stratified': args.stratify,
            'random_seed': args.random_seed
        },
        'dataset_info': {
            'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.savez(output_dir / 'predictions.npz',
             val_predictions=val_preds,
             val_targets=val_targets,
             test_predictions=test_preds,
             test_targets=test_targets)
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(history, output_dir / 'training_curves.png')
    print(f"  ✓ Saved: training_curves.png")
    
    writer.close()
    
    print(f"\n✓ Training complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
