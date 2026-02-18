"""
GNN Components

Graph Neural Network components for brain region connectivity
"""

import torch.nn as nn


class MultiEdgeAttention(nn.Module):
    """Attention over 3D edge features"""
    
    def __init__(self, edge_dim=3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.Tanh(),
            nn.Linear(edge_dim, edge_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, edge_attr):
        weights = self.attention(edge_attr)
        return edge_attr * weights
