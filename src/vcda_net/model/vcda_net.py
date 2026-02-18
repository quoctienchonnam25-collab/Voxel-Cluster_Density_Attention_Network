"""
Voxel Cluster Density Attention Network (VCDA-Net) - Gated Fusion Architecture

V4 improvements over V3:
1. V3 features: Dual-stream (Transformer + Bottleneck) with skip connections
2. NEW: Gated Fusion - Attention-based adaptive fusion of streams
3. Learns optimal combination weights for each sample
4. More flexible than simple concatenation (V3)

Architecture:
    Stream 1 (Transformer): UNet features → Dual paths → GNN → Transformer → [256]
    Stream 2 (Bottleneck): UNet bottleneck → 1×1×1 Conv → ResNet blocks → Pool → [256]
    Fusion: Gated Attention → [512] (adaptive learned weights)
    Prediction: MLP (512→256→128→1) → Age

Key Innovation (V4):
    - Replaces concat([s1, s2]) with gate*transform(s1) + (1-gate)*transform(s2)
    - Gate learned from both streams
    - Adaptive fusion per sample

Author: Anonymous
Date: 2026-02-16 (Refactored)
Version: 4.0.2 (Gated Fusion + Modular Components)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv

# Import all components from components module
from .components import (
    # Utility functions
    extract_top_k_coordinates,
    compute_gaussian_distance_matrix,
    # Feature extraction
    UNetFeatureExtractor,
    # Downsampling
    StridedConvDownsampler,
    # Encoders
    ResNetEncoder,
    MatrixResNetEncoder,
    # GNN
    MultiEdgeAttention,
    # Transformers
    TransformerAggregation,
    # Fusion
    BottleneckStream,
    GatedFusion,
)


class VCDANet(nn.Module):
    """
    Hybrid model with Saliency Map enhanced features
    
    For each of 32 feature maps:
    1. Original path: FeatureMap → ResNet → 256-dim
    2. SaliencyMap path: FeatureMap → SaliencyMap → TopK coords → Gaussian matrix → ResNet → 256-dim
    3. Concatenate: 256 + 256 = 512-dim per region
    4. Build graph with 512-dim nodes
    """
    
    def __init__(
        self,
        # UNet
        unet_checkpoint: str,
        num_regions: int = 32,
        freeze_unet: bool = False,  # NEW: control UNet freezing
        
        # Embeddings
        embedding_dim: int = 256,
        resnet_depth: str = 'resnet18',
        
        # Saliency Map
        top_k: int = 128,
        sigma: float = 10.0,
        matrix_resize: int = 64,  # Resize KxK to 64x64 for ResNet
        
        # GNN
        edge_num: int = 31,
        hidden_channels: int = 64,
        num_gnn_layers: int = 3,
        use_edge_attention: bool = True,
        
        # Transformer
        transformer_d_model: int = 256,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 3,
        
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_regions = num_regions
        self.top_k = top_k
        self.sigma = sigma
        self.matrix_resize = matrix_resize
        self.edge_num = edge_num
        self.use_edge_attention = use_edge_attention
        
        # UNet feature extractor (controlled by freeze_unet parameter)
        self.unet_extractor = UNetFeatureExtractor(unet_checkpoint, freeze=freeze_unet)
        
        # Downsampler - restore for faster training
        self.downsampler = StridedConvDownsampler(32, 32, num_stages=1)
        
        # Original path: ResNet for feature maps
        self.resnet_features = ResNetEncoder(
            embedding_dim=embedding_dim,
            resnet_depth=resnet_depth,
            dropout=dropout
        )
        
        # SaliencyMap path: ResNet for Gaussian matrices
        self.resnet_matrices = MatrixResNetEncoder(
            resnet_depth=resnet_depth,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        
        # Project concatenated features (512) to GNN space
        self.feature_proj = nn.Linear(embedding_dim * 2, hidden_channels)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                ), edge_dim=3
            ))
        
        # Edge attention
        if use_edge_attention:
            self.edge_attention = MultiEdgeAttention(edge_dim=3)
        
        # Transformer
        self.transformer = TransformerAggregation(
            input_dim=hidden_channels,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dropout=dropout
        )
        
        # Bottleneck Stream (V3 addition)
        self.bottleneck_stream = BottleneckStream(
            bottleneck_channels=128,
            hidden_channels=256,
            num_blocks=3,
            use_batchnorm=True
        )
        
        # Gated Fusion (V4 innovation - replaces simple concatenation)
        self.fusion = GatedFusion(
            input_dim=256,      # Each stream outputs 256
            output_dim=512,     # Fused output is 512
            dropout=dropout
        )
        print("✓ V4: Using Gated Fusion (attention-based adaptive fusion)")
        
        # Prediction head (V4: same as V3, input is 512 from fusion)
        self.head = nn.Sequential(
            # Input: 512 (256 from transformer + 256 from bottleneck)
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        print(f"\n{'='*70}")
        print(f"VCDANet Architecture:")
        print(f"{'='*70}")
        print(f"Feature dims: {embedding_dim} + {embedding_dim} = {embedding_dim*2}")
        print(f"Top-K voxels: {top_k}")
        print(f"Gaussian sigma: {sigma}")
        print(f"Matrix resize: {matrix_resize}x{matrix_resize}")
        print(f"GNN: {num_gnn_layers} layers, hidden={hidden_channels}")
        print(f"Edge Attention: {'Yes' if use_edge_attention else 'No'}")
        print(f"{'='*70}\n")
    
    def compute_weighted_feature_matrices(self, feature_maps):
        """
        Compute importance-weighted feature matrices for all channels
        
        Uses feature magnitude as importance weights (NOT Saliency Map gradients).
        This simplified approach avoids backward passes during training.
        
        Args:
            feature_maps: [B, 32, H, W, D]
        
        Returns:
            matrices: [B, 32, matrix_resize, matrix_resize] - Gaussian distance matrices
        """
        B, C, H, W, D = feature_maps.shape
        device = feature_maps.device
        
        # Magnitude-based weighting: Use feature map magnitudes as importance
        # Compute per-channel importance using global average pooling
        channel_importance = feature_maps.abs().mean(dim=(2, 3, 4), keepdim=True)  # [B, 32, 1, 1, 1]
        
        # Weight feature maps by importance
        weighted_maps = feature_maps * channel_importance  # [B, 32, H, W, D]
        
        # Vectorized normalization (much faster than nested loops!)
        wmap_min = weighted_maps.amin(dim=(2, 3, 4), keepdim=True)  # [B, 32, 1, 1, 1]
        wmap_max = weighted_maps.amax(dim=(2, 3, 4), keepdim=True)  # [B, 32, 1, 1, 1]
        weighted_maps_normalized = (weighted_maps - wmap_min) / (wmap_max - wmap_min + 1e-8)
        
        # Extract top-K and compute matrices
        # Use no_grad here since Gaussian matrices don't need gradients
        # (they're just spatial encodings, not learned parameters)
        with torch.no_grad():
            all_matrices = []
            
            for b in range(B):
                batch_matrices = []
                
                for c in range(C):
                    wmap_3d = weighted_maps_normalized[b, c].cpu().numpy()  # [H, W, D]
                    
                    # Extract top-K coordinates
                    coords, _ = extract_top_k_coordinates(wmap_3d, k=self.top_k)
                    
                    # Compute Gaussian matrix
                    matrix = compute_gaussian_distance_matrix(coords, sigma=self.sigma)  # [K, K]
                    
                    # Resize to fixed size
                    matrix_tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).float()  # [1, 1, K, K]
                    matrix_resized = F.interpolate(
                        matrix_tensor,
                        size=(self.matrix_resize, self.matrix_resize),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # [1, resize, resize]
                    
                    batch_matrices.append(matrix_resized)
                
                batch_matrices = torch.stack(batch_matrices, dim=0)  # [32, 1, resize, resize]
                all_matrices.append(batch_matrices)
            
            all_matrices = torch.stack(all_matrices, dim=0)  # [B, 32, 1, resize, resize]
            all_matrices = all_matrices.squeeze(2)  # [B, 32, resize, resize]
        
        return all_matrices.to(device)
    
    def build_unified_graph(self, node_features, k=31):
        """
        Build k-NN graph with 3D edge features
        
        Args:
            node_features: [N, hidden_dim]
            k: number of nearest neighbors
        
        Returns:
            edge_index: [2, E]
            edge_attr: [E, 3] - distance, cosine similarity, dot product
        """
        N = node_features.size(0)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(node_features, node_features, p=2)
        
        # k-NN edges
        _, indices = torch.topk(dist_matrix, k=k+1, largest=False, dim=1)
        indices = indices[:, 1:]  # Remove self-loops
        
        # Build edge_index
        src = torch.arange(N, device=node_features.device).repeat_interleave(k)
        dst = indices.flatten()
        edge_index = torch.stack([src, dst], dim=0)
        
        # Compute edge attributes (VECTORIZED - much faster than loop!)
        src_features = node_features[edge_index[0]]  # [E, hidden_dim]
        dst_features = node_features[edge_index[1]]  # [E, hidden_dim]
        
        # 1. Distance (Euclidean) - vectorized
        dist = torch.norm(src_features - dst_features, dim=1, keepdim=True)  # [E, 1]
        
        # 2. Cosine similarity (direction) - vectorized
        cos = F.cosine_similarity(src_features, dst_features, dim=1, eps=1e-8).unsqueeze(1)  # [E, 1]
        
        # 3. Dot product (correlation strength = direction + magnitude) - vectorized
        dot = (src_features * dst_features).sum(dim=1, keepdim=True)  # [E, 1]
        
        # Stack edge attributes
        edge_attr = torch.cat([dist, cos, dot], dim=1)  # [E, 3]
        
        return edge_index, edge_attr
    
    def forward(self, mri, return_graph=False):
        """
        Args:
            mri: [B, 1, 128, 128, 128]
            return_graph: bool - If True, return graph structure for PGExplainer
        
        Returns:
            age: [B, 1]
            gate_mean: [B, 1] (during inference)
            graph_info: dict (if return_graph=True) - Contains:
                - edge_index: Graph connectivity [2, E]
                - edge_attr: Edge features [E, 3]
                - node_features: Node features after GNN [B, 32, hidden]
                - batch: Batch assignment [N]
                - prediction: Age prediction [B, 1]
        """
        B = mri.size(0)
        device = mri.device
        
        # Step 1: Extract feature maps AND bottleneck (V3)
        feature_maps, bottleneck = self.unet_extractor(mri)
        # feature_maps: [B, 32, 64, 64, 64]
        # bottleneck: [B, 128, 32, 32, 32]
        
        feature_maps_down = self.downsampler(feature_maps)  # [B, 32, 32, 32, 32]
        
        # Step 2: Original path - ResNet encoding
        features_original = self.resnet_features(feature_maps_down)  # [B, 32, 256]
        
        # Step 3: Weighted feature path - Compute matrices and encode
        gaussian_matrices = self.compute_weighted_feature_matrices(feature_maps)  # [B, 32, resize, resize]
        
        # Reshape for batch processing through ResNet
        B_mat, C, H, W = gaussian_matrices.shape
        matrices_flat = gaussian_matrices.view(B_mat * C, 1, H, W)  # [B*32, 1, resize, resize]
        
        features_matrices = self.resnet_matrices(matrices_flat)  # [B*32, 256]
        features_matrices = features_matrices.view(B_mat, C, -1)  # [B, 32, 256]
        
        # Step 4: Concatenate features
        features_concat = torch.cat([features_original, features_matrices], dim=2)  # [B, 32, 512]
        
        # Step 5: Project to GNN space
        features_gnn = self.feature_proj(features_concat)  # [B, 32, hidden]
        
        # Step 6: Build graphs
        data_list = []
        for i in range(B):
            edge_index, edge_attr = self.build_unified_graph(features_gnn[i], k=self.edge_num)
            
            if self.use_edge_attention:
                edge_attr = self.edge_attention(edge_attr)
            
            data = Data(
                x=features_gnn[i],
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            data_list.append(data)
        
        batch_data = Batch.from_data_list(data_list).to(device)
        
        # Step 7: GNN processing
        x = batch_data.x
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, batch_data.edge_index, batch_data.edge_attr)
            x = F.relu(x)
        
        # Reshape
        x_per_graph = []
        for i in range(B):
            mask = (batch_data.batch == i)
            x_per_graph.append(x[mask])
        x_batched = torch.stack(x_per_graph, dim=0)  # [B, 32, hidden]
        
        # Step 8a: Transformer aggregation (Stream 1)
        if return_graph:
             transformer_embedding, transformer_attn = self.transformer(x_batched, return_attention=True)  # [B, 256], [B, N, N]
        else:
             transformer_embedding = self.transformer(x_batched)  # [B, 256]
        
        # Step 8b: Bottleneck stream (Stream 2)
        bottleneck_embedding = self.bottleneck_stream(bottleneck)  # [B, 256]
        
        # Step 9: Fusion - Gated attention fusion (V4)
        combined_embedding, gate_values = self.fusion(
            transformer_embedding,  # Stream 1: [B, 256]
            bottleneck_embedding    # Stream 2: [B, 256]
        )  # Returns: [B, 512], [B, 512]
        
        # Step 10: Prediction
        age = self.head(combined_embedding)  # [B, 1]
        
        # Optional: Return graph structure for PGExplainer
        if return_graph:
            graph_info = {
                'edge_index': batch_data.edge_index,
                'edge_attr': batch_data.edge_attr,
                'node_features': x_batched,  # After GNN processing
                'batch': batch_data.batch,
                'prediction': age,
                'gate_values': gate_values,  # Include gate values for analysis
            }
            
            # Only add transformer_attn if it was successfully captured
            if transformer_attn is not None:
                graph_info['transformer_attn'] = transformer_attn
            
            
            if self.training:
                return age, graph_info
            else:
                gate_mean = gate_values.mean(dim=1, keepdim=True)
                return age, gate_mean, graph_info
        
        # Normal return (backward compatible)
        # Return gate values for analysis during inference
        # gate close to 1 → prefers transformer (local structure)
        # gate close to 0 → prefers bottleneck (global features)
        if self.training:
            return age
        else:
            # During inference/validation, return gate values for monitoring
            gate_mean = gate_values.mean(dim=1, keepdim=True)  # [B, 1] - average gate per sample
            return age, gate_mean


if __name__ == '__main__':
    # Test
    print("Testing VCDANet...")
    
    model = VCDANet(
        unet_checkpoint='checkpoints/3dunet_checkpoints/best_model.pth',
        num_regions=32,
        embedding_dim=256,
        top_k=128,
        sigma=10.0,
        matrix_resize=64
    )
    
    # Dummy input
    mri = torch.randn(2, 1, 128, 128, 128)
    
    print("\nForward pass...")
    with torch.no_grad():
        output = model(mri)
    
    print(f"Output shape: {output.shape}")
    print("Test passed!")
