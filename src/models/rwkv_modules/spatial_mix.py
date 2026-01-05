"""
Spatial Mix module with Bi-directional RWKV for HPCM
Linear attention over spatial dimensions
"""

import torch
import torch.nn as nn
from einops import rearrange

from .omni_shift import OmniShift
from .biwkv4 import RUN_BiWKV4_HPCM


class SpatialMix_HPCM(nn.Module):
    """
    Spatial mixing with Bi-directional linear attention
    Captures long-range spatial dependencies with O(N*HW) complexity
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        attn_dim = dim
        
        # Spatial shifting with reparameterizable conv
        self.omni_shift = OmniShift(dim=dim)
        
        # Key, Value, Receptance projections
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, dim, bias=False)
        
        # Time-varying parameters for RWKV
        self.decay = nn.Parameter(torch.randn((self.dim,)))
        self.boost = nn.Parameter(torch.randn((self.dim,)))
    
    def jit_func(self, x, resolution):
        """Compute K, V, R with spatial shift"""
        H, W = resolution
        
        # Apply spatial shift
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        
        # Compute projections
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        
        return sr, k, v
    
    def forward(self, x, resolution):
        """
        Args:
            x: (B, C, H, W) feature tensor
            resolution: (H, W) tuple
        
        Returns:
            (B, C, H, W) output tensor
        """
        B, C, H, W = x.size()
        T = H * W
        
        # Convert to sequence format
        x = rearrange(x, "b c h w -> b (h w) c")
        
        # Get K, V, R
        sr, k, v = self.jit_func(x, resolution)
        
        # Bi-directional WKV4 (linear attention)
        x = RUN_BiWKV4_HPCM(self.decay / T, self.boost / T, k, v)
        
        # Gate with receptance and project
        x = sr * x
        x = self.output(x)
        
        # Convert back to spatial format
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        
        return x
