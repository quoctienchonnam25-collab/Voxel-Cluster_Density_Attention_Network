"""
Loss Functions for Brain Age Prediction
Implements various loss functions optimized for regression tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss variant)
    
    Combines MSE for small errors and MAE for large errors.
    More robust to outliers than MSE, faster convergence than MAE.
    
    Args:
        delta: Threshold for switching between MSE and MAE (default: 1.0)
               - Smaller delta → more like L1 (robust to outliers)
               - Larger delta → more like L2 (faster convergence)
    
    Recommended for brain age prediction with multi-dataset (ABIDE1, ADNI, IXI)
    """
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] predictions
            target: [B, 1] ground truth ages
        Returns:
            loss: scalar
        """
        error = torch.abs(pred - target)
        
        # Huber loss formula
        # if |error| <= delta: 0.5 * error^2
        # else: delta * (|error| - 0.5 * delta)
        is_small_error = error <= self.delta
        
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (error - 0.5 * self.delta)
        
        loss = torch.where(is_small_error, squared_loss, linear_loss)
        
        return loss.mean()


class AdaptiveHuberLoss(nn.Module):
    """
    Adaptive Huber Loss with learnable delta
    
    Delta is learned during training to automatically adapt
    to the data distribution.
    """
    def __init__(self, init_delta=1.0):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(init_delta))
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        delta = torch.abs(self.delta)  # Ensure positive
        
        is_small_error = error <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (error - 0.5 * delta)
        
        loss = torch.where(is_small_error, squared_loss, linear_loss)
        
        return loss.mean()


class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber Loss for handling age imbalance
    
    Applies different weights to different age ranges to handle
    data imbalance (e.g., more young people in ABIDE1, more elderly in ADNI)
    
    Args:
        delta: Huber delta parameter
        age_bins: List of age bin edges (default: [0, 18, 40, 70, 100])
        weights: Weights for each age bin (default: uniform)
    """
    def __init__(self, delta=1.0, age_bins=None, weights=None):
        super().__init__()
        self.delta = delta
        
        if age_bins is None:
            self.age_bins = [0, 18, 40, 70, 100]
        else:
            self.age_bins = age_bins
        
        if weights is None:
            # Default: uniform weights
            self.weights = torch.ones(len(self.age_bins) - 1)
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, pred, target):
        error = torch.abs(pred - target)
        
        # Compute Huber loss
        is_small_error = error <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (error - 0.5 * self.delta)
        loss_per_sample = torch.where(is_small_error, squared_loss, linear_loss)
        
        # Compute weights based on target age
        weights = torch.ones_like(target)
        for i in range(len(self.age_bins) - 1):
            mask = (target >= self.age_bins[i]) & (target < self.age_bins[i + 1])
            weights[mask] = self.weights[i].to(target.device)
        
        # Apply weights
        weighted_loss = loss_per_sample * weights
        
        return weighted_loss.mean()


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss
    
    log(cosh(pred - target))
    
    Advantages:
    - Smooth everywhere (twice differentiable)
    - Approximately equal to (x^2)/2 for small x
    - Approximately equal to |x| - log(2) for large x
    - Less sensitive to outliers than MSE
    
    Good alternative to Huber loss without hyperparameter tuning.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        error = pred - target
        loss = torch.log(torch.cosh(error))
        return loss.mean()


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss)
    
    Useful for uncertainty estimation in age prediction.
    Can predict confidence intervals.
    
    Args:
        quantile: Target quantile (default: 0.5 for median)
    """
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, pred, target):
        error = target - pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return loss.mean()


def get_loss_function(loss_type='huber', **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_type: 'l1', 'mse', 'huber', 'adaptive_huber', 
                   'weighted_huber', 'logcosh', 'quantile'
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        loss_fn: Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'l1' or loss_type == 'mae':
        return nn.L1Loss()
    
    elif loss_type == 'mse' or loss_type == 'l2':
        return nn.MSELoss()
    
    elif loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        return HuberLoss(delta=delta)
    
    elif loss_type == 'adaptive_huber':
        init_delta = kwargs.get('init_delta', 1.0)
        return AdaptiveHuberLoss(init_delta=init_delta)
    
    elif loss_type == 'weighted_huber':
        delta = kwargs.get('delta', 1.0)
        age_bins = kwargs.get('age_bins', None)
        weights = kwargs.get('weights', None)
        return WeightedHuberLoss(delta=delta, age_bins=age_bins, weights=weights)
    
    elif loss_type == 'logcosh':
        return LogCoshLoss()
    
    elif loss_type == 'quantile':
        quantile = kwargs.get('quantile', 0.5)
        return QuantileLoss(quantile=quantile)
    
    elif loss_type == 'smooth_l1':
        # PyTorch's built-in Smooth L1 (similar to Huber)
        beta = kwargs.get('beta', 1.0)
        return nn.SmoothL1Loss(beta=beta)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Testing
if __name__ == '__main__':
    print("="*70)
    print("Loss Functions for Brain Age Prediction")
    print("="*70)
    
    # Test data
    pred = torch.tensor([[25.5], [45.2], [70.1], [85.8]])
    target = torch.tensor([[25.0], [50.0], [70.0], [60.0]])  # Last one is outlier
    
    print("\nPredictions:", pred.squeeze().tolist())
    print("Targets:    ", target.squeeze().tolist())
    print("Errors:     ", (pred - target).squeeze().abs().tolist())
    
    # Test different losses
    losses = {
        'L1 (MAE)': nn.L1Loss(),
        'MSE (L2)': nn.MSELoss(),
        'Huber (δ=1.0)': HuberLoss(delta=1.0),
        'Huber (δ=5.0)': HuberLoss(delta=5.0),
        'Smooth L1': nn.SmoothL1Loss(),
        'LogCosh': LogCoshLoss(),
    }
    
    print("\n" + "="*70)
    print("Loss Comparison:")
    print("="*70)
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target).item()
        print(f"{name:20s}: {loss_value:.4f}")
    
    print("\n" + "="*70)
    print("Recommendation for Brain Age Prediction:")
    print("="*70)
    print("✓ Primary choice: Huber Loss (delta=1.0)")
    print("  - Robust to outliers from mixed datasets")
    print("  - Faster convergence than L1")
    print("  - Balances precision and robustness")
    print("\n✓ Alternative: LogCosh Loss")
    print("  - No hyperparameter tuning needed")
    print("  - Smooth and robust")
    print("\n✓ Baseline: L1 Loss")
    print("  - Simple and reliable")
    print("  - Good for comparison")
    print("="*70)
