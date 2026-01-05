"""
RWKV Fusion Network for Context Integration
Replaces simple conv1x1 in context_net with RWKV-based processing
"""

import torch
import torch.nn as nn
from einops import rearrange

from .spatial_mix import SpatialMix_HPCM
from .channel_mix import ChannelMix_HPCM


class RWKVFusionBlock(nn.Module):
    """
    Single RWKV block for context fusion
    Similar to RWKVContextCell but without input projection (single stream)
    """
    
    def __init__(self, dim, hidden_rate=4, use_checkpoint=False):
        """
        Args:
            dim: input/output dimension
            hidden_rate: expansion ratio for channel mixing
            use_checkpoint: whether to use gradient checkpointing
        """
        super().__init__()
        self.dim = dim
        self.hidden_rate = hidden_rate
        self.use_checkpoint = use_checkpoint
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # RWKV blocks
        self.spatial_mix = SpatialMix_HPCM(dim)
        self.channel_mix = ChannelMix_HPCM(dim, hidden_rate=hidden_rate)
        
        # Learnable residual scaling factors
        self.gamma1 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(dim), requires_grad=True)
    
    def _forward_impl(self, x, resolution):
        """Core forward implementation"""
        H, W = resolution
        
        # Spatial Mix with residual connection
        x_spatial = self.spatial_mix(x, resolution)
        x_spatial_norm = rearrange(x_spatial, "b c h w -> b h w c")
        x_spatial_norm = self.ln1(x_spatial_norm)
        x_norm = rearrange(x, "b c h w -> b h w c")
        x = x_norm + self.gamma1.view(1, 1, 1, -1) * (x_spatial_norm - x_norm)
        x = rearrange(x, "b h w c -> b c h w")
        
        # Channel Mix with residual connection
        x_channel = self.channel_mix(x, resolution)
        x_channel_norm = rearrange(x_channel, "b c h w -> b h w c")
        x_channel_norm = self.ln2(x_channel_norm)
        x_norm = rearrange(x, "b c h w -> b h w c")
        x = x_norm + self.gamma2.view(1, 1, 1, -1) * (x_channel_norm - x_norm)
        x = rearrange(x, "b h w c -> b c h w")
        
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input tensor (B, C, H, W)
        
        Returns:
            output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        resolution = (H, W)
        
        # RWKV processing with optional gradient checkpointing
        if self.use_checkpoint and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, resolution, use_reentrant=False
            )
        else:
            x = self._forward_impl(x, resolution)
        
        return x


class RWKVFusionNet(nn.Module):
    """
    RWKV-based context fusion network
    Replaces conv1x1(2*M, 2*M) in context_net
    """
    
    def __init__(self, dim, num_blocks=1, hidden_rate=4, use_checkpoint=False):
        """
        Args:
            dim: input/output dimension (2*M for context fusion)
            num_blocks: number of RWKV blocks (default: 1 for minimal change)
            hidden_rate: expansion ratio for channel mixing
            use_checkpoint: whether to use gradient checkpointing
        """
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        
        # RWKV blocks for context processing
        self.blocks = nn.ModuleList([
            RWKVFusionBlock(dim, hidden_rate=hidden_rate, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks)
        ])
        
        # Optional output projection (for compatibility with original conv1x1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input context tensor (B, C, H, W)
        
        Returns:
            fused context tensor (B, C, H, W)
        """
        # Sequential RWKV blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.out_proj(x)
        
        return x
