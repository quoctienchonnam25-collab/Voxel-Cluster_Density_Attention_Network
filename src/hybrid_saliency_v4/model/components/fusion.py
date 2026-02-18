"""
Fusion Components

Modules for fusing multiple feature streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """3D Residual Block for Bottleneck Stream"""
    
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out, inplace=True)


class BottleneckStream(nn.Module):
    """Process UNet bottleneck with ResNet blocks - V3 addition"""
    
    def __init__(self, bottleneck_channels=128, hidden_channels=256, 
                 num_blocks=3, use_batchnorm=True):
        super().__init__()
        
        # 1x1x1 Conv to expand channels: 128 → 256
        self.conv_expand = nn.Conv3d(bottleneck_channels, hidden_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm3d(hidden_channels) if use_batchnorm else nn.Identity()
        
        # ResNet blocks (3 blocks of 3×3×3 convs)
        self.res_blocks = nn.ModuleList([
            ResBlock3D(hidden_channels, hidden_channels, stride=1, use_batchnorm=use_batchnorm)
            for _ in range(num_blocks)
        ])
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        print(f"✓ BottleneckStream: {bottleneck_channels}→{hidden_channels}, {num_blocks} ResBlocks")
    
    def forward(self, x):
        """
        Args:
            x: UNet bottleneck [B, 128, H/8, W/8, D/8]
        Returns:
            embedding: [B, 256]
        """
        # Expand channels: 128 → 256
        x = self.conv_expand(x)
        x = self.bn_expand(x)
        x = F.relu(x, inplace=True)
        
        # ResNet blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Global pooling: [B, 256, H, W, D] → [B, 256, 1, 1, 1]
        x = self.global_pool(x)
        
        # Flatten: [B, 256, 1, 1, 1] → [B, 256]
        embedding = x.view(x.size(0), -1)
        
        return embedding


class GatedFusion(nn.Module):
    """
    Gated Fusion for Adaptive Dual-Stream Combination (V4)
    
    Learns element-wise gates to adaptively blend transformer and bottleneck features.
    Gate values in [0, 1] control the contribution of each stream.
    
    Args:
        input_dim: Dimension of each input stream (default: 256)
        output_dim: Dimension of fused output (default: 512)
        dropout: Dropout rate (default: 0.3)
    """
    def __init__(self, input_dim=256, output_dim=512, dropout=0.3):
        super().__init__()
        
        # Transform stream 1 (transformer) to output dimension
        self.transform1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transform stream 2 (bottleneck) to output dimension
        self.transform2 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gating network: learns adaptive blending weights
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Sigmoid()  # Gate values in [0, 1]
        )
        
        print(f"✓ GatedFusion: {input_dim}+{input_dim} → {output_dim} (adaptive)")
    
    def forward(self, stream1, stream2):
        """
        Adaptively fuse two streams using learned gates
        
        Args:
            stream1: [B, 256] - Transformer embedding
            stream2: [B, 256] - Bottleneck embedding
        
        Returns:
            fused: [B, 512] - Adaptively fused features
            gate: [B, 512] - Gate values (for analysis/visualization)
                            - gate close to 1: prefers stream1 (transformer)
                            - gate close to 0: prefers stream2 (bottleneck)
        """
        # Transform both streams to output dimension
        h1 = self.transform1(stream1)  # [B, 512]
        h2 = self.transform2(stream2)  # [B, 512]
        
        # Learn adaptive gate from both streams
        combined_input = torch.cat([stream1, stream2], dim=1)  # [B, 512]
        gate = self.gate_network(combined_input)  # [B, 512], values in [0, 1]
        
        # Gated fusion: adaptive element-wise blending
        # gate=1 → use h1 (transformer)
        # gate=0 → use h2 (bottleneck)
        # gate=0.5 → equal blend
        fused = gate * h1 + (1 - gate) * h2  # [B, 512]
        
        return fused, gate
