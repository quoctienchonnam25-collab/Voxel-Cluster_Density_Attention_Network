#!/usr/bin/env python
"""
Quick Test Training Script for VCDA-Net Package
Tests 1 epoch to verify package functionality
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split

# Import from package
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from vcda_net.model.vcda_net import VCDANet


class QuickDataset(Dataset):
    """Quick dataset for testing"""
    def __init__(self, data_dir, metadata_csv, split='train', train_ratio=0.8):
        self.data_dir = Path(data_dir)
        
        # Read metadata
        df = pd.read_csv(metadata_csv)
        df.columns = df.columns.str.upper()
        
        # Create full paths
        df['mri_path'] = df['FILENAME'].apply(lambda x: str(self.data_dir / x))
        
        # Verify files exist
        df['exists'] = df['mri_path'].apply(lambda x: Path(x).exists())
        df = df[df['exists']].copy()
        
        print(f"Total samples: {len(df)}")
        
        # Simple split
        train_df, val_df = train_test_split(df, test_size=1-train_ratio, random_state=42)
        
        self.df = train_df if split == 'train' else val_df
        
        print(f"{split.upper()} set: {len(self.df)} samples")
        print(f"Age range: {self.df['AGE'].min():.1f} - {self.df['AGE'].max():.1f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load MRI
        mri_nii = nib.load(row['mri_path'])
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
        
        mri_tensor = torch.from_numpy(mri_data).unsqueeze(0).float()
        age_tensor = torch.tensor([row['AGE']], dtype=torch.float32)
        
        return mri_tensor, age_tensor


def test_training():
    """Test 1 epoch of training"""
    
    print("="*70)
    print("VCDA-NET - QUICK TEST TRAINING")
    print("="*70)
    
    # Paths
    data_dir = Path('/media/devin/WORK/devin/tien/synthesis_data/healthy_brain_1710/ABIDE_ADNI_IXI_OASIS_PPMI_Turboprep_balanced_1710')
    metadata_csv = Path('/media/devin/WORK/devin/tien/synthesis_data/healthy_brain_1710/ABIDE_ADNI_IXI_OASIS_PPMI_Turboprep_balanced_1710_metadata.csv')
    unet_checkpoint = Path('/media/devin/WORK/devin/tien/src/brain_age_prediction/vcda_net_package/src/vcda_net/checkpoints/IXI_3dunet_best_model.pth')
    
    # Check paths
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    if not metadata_csv.exists():
        print(f"❌ Metadata CSV not found: {metadata_csv}")
        return
    if not unet_checkpoint.exists():
        print(f"❌ UNet checkpoint not found: {unet_checkpoint}")
        return
    
    print(f"✓ Data directory: {data_dir}")
    print(f"✓ Metadata CSV: {metadata_csv}")
    print(f"✓ UNet checkpoint: {unet_checkpoint}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Datasets
    print("Loading datasets...")
    train_dataset = QuickDataset(data_dir, metadata_csv, split='train', train_ratio=0.8)
    val_dataset = QuickDataset(data_dir, metadata_csv, split='val', train_ratio=0.8)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                             num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # Model
    print("Creating model...")
    model = VCDANet(
        unet_checkpoint=str(unet_checkpoint),
        num_regions=32,
        embedding_dim=256,
        resnet_depth='resnet18',
        top_k=128,
        sigma=10.0,
        matrix_resize=64,
        edge_num=31,
        hidden_channels=64,
        num_gnn_layers=3,
        use_edge_attention=True,
        transformer_d_model=256,
        transformer_nhead=8,
        transformer_num_layers=3,
        dropout=0.3,
        freeze_unet=True  # Freeze UNet for faster testing
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()
    
    # Test 1 epoch
    print("="*70)
    print("TRAINING - EPOCH 1")
    print("="*70)
    
    model.train()
    total_loss = 0
    predictions, targets = [], []
    
    pbar = tqdm(train_loader, desc='Training')
    for i, (mri, age) in enumerate(pbar):
        mri, age = mri.to(device), age.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred = model(mri)
        loss = criterion(pred, age)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions.extend(pred.detach().cpu().numpy().flatten())
        targets.extend(age.detach().cpu().numpy().flatten())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Test only first 10 batches for speed
        if i >= 9:
            print(f"\n✓ Tested {i+1} batches (stopping early for quick test)")
            break
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    avg_loss = total_loss / min(len(train_loader), 10)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    print(f"\nTraining Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  MAE: {mae:.2f} years")
    print(f"  RMSE: {rmse:.2f} years")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    model.eval()
    val_loss = 0
    val_predictions, val_targets, val_gates = [], [], []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for i, (mri, age) in enumerate(pbar):
            mri, age = mri.to(device), age.to(device)
            
            # Model in eval mode returns (prediction, gate_mean)
            output = model(mri)
            if isinstance(output, tuple):
                pred, gate = output
                val_gates.extend(gate.cpu().numpy().flatten())
            else:
                pred = output
            
            loss = criterion(pred, age)
            
            val_loss += loss.item()
            val_predictions.extend(pred.cpu().numpy().flatten())
            val_targets.extend(age.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Test only first 5 batches
            if i >= 4:
                print(f"\n✓ Tested {i+1} batches (stopping early for quick test)")
                break
    
    # Calculate metrics
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    
    val_avg_loss = val_loss / min(len(val_loader), 5)
    val_mae = np.mean(np.abs(val_predictions - val_targets))
    val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
    val_gate_mean = np.mean(val_gates) if val_gates else 0.0
    
    print(f"\nValidation Results:")
    print(f"  Loss: {val_avg_loss:.4f}")
    print(f"  MAE: {val_mae:.2f} years")
    print(f"  RMSE: {val_rmse:.2f} years")
    print(f"  Gate Mean: {val_gate_mean:.3f} (>0.5: Transformer, <0.5: Bottleneck)")
    
    print("\n" + "="*70)
    print("✅ TEST TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nPackage is working correctly!")
    print("You can now run full training with:")
    print("  python -m vcda_net.training.train \\")
    print("    --data_dir <path> \\")
    print("    --metadata_csv <path> \\")
    print("    --unet_checkpoint <path> \\")
    print("    --epochs 300")


if __name__ == '__main__':
    test_training()
