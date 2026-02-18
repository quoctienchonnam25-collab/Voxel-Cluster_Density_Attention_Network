"""
Encoder Components

ResNet-based encoders for feature extraction
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MatrixResNetEncoder(nn.Module):
    """
    Encode KxK matrix to 256-dim vector using ResNet
    Treats matrix as single-channel image
    """
    def __init__(self, resnet_depth='resnet18', embedding_dim=256, dropout=0.3):
        super().__init__()
        
        # Load pretrained ResNet
        if resnet_depth == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif resnet_depth == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif resnet_depth == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown resnet_depth: {resnet_depth}")
        
        # Modify first conv to accept 1 channel (grayscale matrix)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        # Copy other layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Custom FC layer
        if resnet_depth in ['resnet18', 'resnet34']:
            in_features = 512
        else:  # resnet50
            in_features = 2048
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, K, K] - batch of matrices
        
        Returns:
            emb: [B, embedding_dim]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResNetEncoder(nn.Module):
    """Encode each region's 3D volume to embedding"""
    
    def __init__(self, embedding_dim=256, resnet_depth='resnet18', dropout=0.3):
        super().__init__()
        
        # 3D to 2D projection (max pooling along depth)
        self.pool3d_to_2d = nn.AdaptiveMaxPool3d((32, 32, 1))
        
        # Load pretrained ResNet
        if resnet_depth == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif resnet_depth == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        
        # Modify first conv
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        if resnet_depth in ['resnet18', 'resnet34']:
            in_features = 512
        else:
            in_features = 2048
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 32, 32, 32, 32] - 32 regions
        Returns:
            emb: [B, 32, embedding_dim]
        """
        B, N = x.shape[0], x.shape[1]
        embeddings = []
        
        for i in range(N):
            region = x[:, i:i+1, :, :, :]  # [B, 1, 32, 32, 32]
            
            # 3D -> 2D
            region_2d = self.pool3d_to_2d(region).squeeze(-1)  # [B, 1, 32, 32]
            
            # ResNet
            out = self.conv1(region_2d)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)  # [B, embedding_dim]
            
            embeddings.append(out)
        
        embeddings = torch.stack(embeddings, dim=1)  # [B, 32, embedding_dim]
        return embeddings
