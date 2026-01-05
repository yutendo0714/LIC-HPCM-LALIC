"""
RWKV Context Cell: Replacement for CrossAttentionCell in HPCM
Linear attention-based context aggregation with O(N*T) complexity
"""

import torch
import torch.nn as nn
from einops import rearrange

from .spatial_mix import SpatialMix_HPCM
from .channel_mix import ChannelMix_HPCM


class RWKVContextCell(nn.Module):
    """
    RWKV-based context cell for HPCM progressive coding
    
    Replaces O(N^2) CrossAttentionCell with O(N*T) linear attention
    Aggregates information from current context and previous hidden state
    
    Architecture:
        - Input projection: concatenate x_t and h_prev
        - RWKV Block: LayerNorm -> SpatialMix -> LayerNorm -> ChannelMix
        - Output projection: project to output dimensions
    """
    
    def __init__(self, input_dim, hidden_rate=4, use_checkpoint=False):
        """
        Args:
            input_dim: dimension of input features (e.g., 640 for M=320)
            hidden_rate: expansion ratio for FFN
            use_checkpoint: whether to use gradient checkpointing
        """
        super().__init__()
        self.input_dim = input_dim
        self.use_checkpoint = use_checkpoint
        
        # Input projection: handle concatenation of x_t and h_prev
        self.input_proj = nn.Conv2d(input_dim * 2, input_dim, 1)
        
        # Layer normalization (applied in channel-last format)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        
        # RWKV components
        self.spatial_mix = SpatialMix_HPCM(input_dim)
        self.channel_mix = ChannelMix_HPCM(input_dim, hidden_rate)
        
        # Learnable scaling parameters (like LALIC's gamma)
        self.gamma1 = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        
        # Output projection
        self.output_proj = nn.Conv2d(input_dim, input_dim, 1)
    
    def _forward_impl(self, x):
        """Core forward implementation"""
        B, C, H, W = x.shape
        resolution = (H, W)
        
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
    
    def forward(self, x_t, h_prev):
        """
        Forward pass
        
        Args:
            x_t: current context tensor (B, C, H, W)
            h_prev: previous hidden state (B, C, H, W)
        
        Returns:
            h_t: updated hidden state (B, C, H, W)
        """
        # Concatenate current context and previous state
        x_combined = torch.cat([x_t, h_prev], dim=1)
        x = self.input_proj(x_combined)
        
        # RWKV processing with optional gradient checkpointing
        if self.use_checkpoint and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        else:
            x = self._forward_impl(x)
        
        # Output projection
        h_t = self.output_proj(x)
        
        return h_t
