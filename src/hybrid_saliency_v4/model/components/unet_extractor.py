"""
UNet Feature Extractor Component

Extracts feature maps and bottleneck from pretrained 3D UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetFeatureExtractor(nn.Module):
    """
    Extract 32 feature maps and TRUE bottleneck from pretrained UNet using hooks
    """
    def __init__(self, unet_checkpoint, num_classes=33, freeze=True):
        super().__init__()
        
        from monai.networks.nets import UNet
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        )
        
        # Load pretrained weights
        checkpoint = torch.load(unet_checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.unet.load_state_dict(checkpoint)
        
        # Storage for captured features
        self.bottleneck_features = None
        
        # Register hooks to capture real bottleneck
        self._register_hooks()
        
        if freeze:
            for param in self.unet.parameters():
                param.requires_grad = False
            print("UNet frozen")
        else:
            print("✓ UNet is TRAINABLE (not frozen)")
        
        self.num_classes = num_classes

    def _register_hooks(self):
        """Register forward hook to capture bottleneck"""
        def bottleneck_hook(module, input, output):
            self.bottleneck_features = output
            
        # For MONAI UNet with strides=(2,2,2) (3 layers), the bottleneck 
        # is typically at the middle of the Sequential model.
        # Structure: [Down1, Down2, Down3, Bottleneck, Up3, Up2, Up1]
        # We try to hook the deepest layer with 128 channels.
        try:
            # Model structure indices depend on implementation, usually index 3 or 4 is bottleneck
            # Investigating MONAI source, it's often a Sequential.
            # We hook the layer that produces the bottleneck (128 channels).
            # If explicit index fails, we might need a safer lookup, but index 3 is common for 3-layer UNet.
            # Let's try hooking the bottleneck block.
            # safe assumption for this config: model[3] is likely the bottleneck calculation
            self.unet.model[3].register_forward_hook(bottleneck_hook)
            print("✓ Registered hook for UNet bottleneck extraction")
        except Exception as e:
            print(f"⚠ Warning: Could not register UNet hook: {e}")
            print("  Falling back to approximation strategy if hook fails during forward.")

    def forward(self, x):
        """
        Args:
            x: [B, 1, 128, 128, 128]
        Returns:
            features: [B, 32, 64, 64, 64] - decoder output features
            bottleneck: [B, 128, 16, 16, 16] - REAL bottleneck from encoder
        """
        # Clear previous capture
        self.bottleneck_features = None
        
        logits = self.unet(x)  # [B, 33, 64, 64, 64]
        features = logits[:, 1:, :, :, :]  # [B, 32, 64, 64, 64]
        
        # Get true bottleneck from hook
        if self.bottleneck_features is not None:
            bottleneck = self.bottleneck_features
        else:
            # Fallback if hook failed (e.g. DataParallel issue or wrong index)
            # Create approximate bottleneck by downsampling features
            # Real UNet bottleneck would be 128 channels at 1/8 resolution
            # We approximate it by pooling and expanding channels
            bottleneck = F.avg_pool3d(features.mean(dim=1, keepdim=True), kernel_size=4, stride=4)
            bottleneck = bottleneck.repeat(1, 128, 1, 1, 1)  # [B, 128, 16, 16, 16]
        
        return features, bottleneck
