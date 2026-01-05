"""
Channel Mix module for HPCM
FFN with spatial shift and gating mechanism
"""

import torch
import torch.nn as nn
from einops import rearrange

from .omni_shift import OmniShift


class ChannelMix_HPCM(nn.Module):
    """
    Channel-wise FFN with spatial shift
    Gated feed-forward network for feature transformation
    """
    
    def __init__(self, dim, hidden_rate=4):
        super().__init__()
        self.n_embd = dim
        hidden_dim = int(hidden_rate * dim)
        
        # Spatial shifting
        self.omni_shift = OmniShift(dim=dim)
        
        # FFN projections
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x, resolution):
        """
        Args:
            x: (B, C, H, W) feature tensor
            resolution: (H, W) tuple
        
        Returns:
            (B, C, H, W) output tensor
        """
        H, W = resolution
        
        # Apply spatial shift
        x = self.omni_shift(x)
        
        # Convert to sequence format
        x = rearrange(x, "b c h w -> b (h w) c")
        
        # Gated FFN: squared ReLU activation
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        # Gate with sigmoid receptance
        x = torch.sigmoid(self.receptance(x)) * kv
        
        # Convert back to spatial format
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        
        return x
