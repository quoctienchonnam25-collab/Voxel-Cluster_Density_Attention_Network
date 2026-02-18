"""
Transformer Components

Transformer-based aggregation modules
"""

import torch
import torch.nn as nn


class TransformerAggregation(nn.Module):
    """Transformer to aggregate node features"""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        
        # internal storage for attention
        self.last_attn_weights = None
        
        print(f"TransformerAggregation initialized:")
        print(f"  - Input: {input_dim}, Model: {d_model}")
        print(f"  - Heads: {nhead}, Layers: {num_layers}")
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B, N, input_dim]
            return_attention: bool
        Returns:
            out: [B, d_model]
            (optional) attn: [B, N, N] if return_attention=True
        """
        x = self.input_proj(x)  # [B, N, d_model]
        x = x + self.pos_encoding
        
        # Transform
        x = self.transformer(x)  # [B, N, d_model]
        
        # Global average pooling
        out = x.mean(dim=1)  # [B, d_model]
        
        if return_attention:
            # Return dummy attention for now (full implementation would need hooks)
            return out, None
        
        return out
