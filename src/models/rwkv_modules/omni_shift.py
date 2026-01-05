"""
OmniShift: Reparameterizable 5x5 depth-wise convolution
Adapted from RestoreRWKV and LALIC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniShift(nn.Module):
    """
    Reparameterized 5x5 depth-wise convolution for spatial mixing
    During training: uses 1x1 + 3x3 + 5x5 + identity with learnable weights
    During inference: merges into a single 5x5 conv for efficiency
    """
    
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        
        # Training-time layers
        self.conv1x1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            groups=dim,
            bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)
        
        # Inference-time layer (reparameterized)
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False
        )
        self.reparam_flag = True
    
    def forward_train(self, x):
        """Forward pass during training"""
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        
        out = (
            self.alpha[0] * x +
            self.alpha[1] * out1x1 +
            self.alpha[2] * out3x3 +
            self.alpha[3] * out5x5
        )
        return out
    
    def reparam_5x5(self):
        """
        Merge all convolutions into a single 5x5 depth-wise conv
        Called once when switching from training to inference
        """
        # Pad smaller kernels to 5x5
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))
        
        # Combine with learned alpha weights
        combined_weight = (
            self.alpha[0] * identity_weight +
            self.alpha[1] * padded_weight_1x1 +
            self.alpha[2] * padded_weight_3x3 +
            self.alpha[3] * self.conv5x5.weight
        )
        
        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)
    
    def forward(self, x):
        """Adaptive forward pass"""
        if self.training:
            self.reparam_flag = True
            return self.forward_train(x)
        elif self.training is False and self.reparam_flag is True:
            self.reparam_5x5()
            self.reparam_flag = False
            return self.conv5x5_reparam(x)
        elif self.training is False and self.reparam_flag is False:
            return self.conv5x5_reparam(x)
        else:
            return self.forward_train(x)
