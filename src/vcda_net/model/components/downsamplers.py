"""
Downsampling Components

Various downsampling modules for feature map processing
"""

import torch.nn as nn


class StridedConvDownsampler(nn.Module):
    """Downsample feature maps using strided convolutions"""
    
    def __init__(self, in_channels=32, out_channels=32, num_stages=1):
        super().__init__()
        
        layers = []
        for _ in range(num_stages):
            layers.append(nn.Conv3d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1
            ))
            layers.append(nn.InstanceNorm3d(out_channels))
            layers.append(nn.GELU())
        
        self.downsample = nn.Sequential(*layers)
        print(f"StridedConvDownsampler: {in_channels}→{out_channels}, stages={num_stages}")
    
    def forward(self, x):
        return self.downsample(x)
