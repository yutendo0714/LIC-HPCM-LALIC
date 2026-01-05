"""
RWKV-enhanced Spatial Prior for HPCM
Replaces DWConvRB-based y_spatial_prior with RWKV processing
"""

import torch
import torch.nn as nn
from einops import rearrange

from .spatial_mix import SpatialMix_HPCM
from .channel_mix import ChannelMix_HPCM
from src.layers import conv1x1


class RWKVSpatialPriorBlock(nn.Module):
    """
    Single RWKV block for spatial prior processing
    Similar to RWKVFusionBlock but optimized for spatial prior tasks
    """
    
    def __init__(self, dim, hidden_rate=4, use_checkpoint=False):
        """
        Args:
            dim: input/output dimension (3*M for spatial prior)
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
        x_norm = rearrange(x, "b c h w -> b h w c")
        x_spatial_norm = rearrange(x_spatial, "b c h w -> b h w c")
        x_norm = self.ln1(x_norm)
        x = x + self.gamma1.view(1, 1, 1, -1) * (x_spatial_norm - x_norm)
        x = rearrange(x, "b h w c -> b c h w")
        
        # Channel Mix with residual connection
        x_channel = self.channel_mix(x, resolution)
        x_norm = rearrange(x, "b c h w -> b h w c")
        x_channel_norm = rearrange(x_channel, "b c h w -> b h w c")
        x_norm = self.ln2(x_norm)
        x = x + self.gamma2.view(1, 1, 1, -1) * (x_channel_norm - x_norm)
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


class RWKVSpatialPrior_S1_S2(nn.Module):
    """
    RWKV-enhanced spatial prior for s1 and s2 scales
    Replaces y_spatial_prior_s1_s2 with RWKV processing
    """
    
    def __init__(self, M, num_rwkv_blocks=2, hidden_rate=4, use_checkpoint=False):
        """
        Args:
            M: base channel dimension
            num_rwkv_blocks: number of RWKV blocks (default: 2 to match DWConvRB×2)
            hidden_rate: expansion ratio for channel mixing
            use_checkpoint: whether to use gradient checkpointing
        """
        super().__init__()
        self.M = M
        self.num_rwkv_blocks = num_rwkv_blocks
        
        # Branch 1: RWKV feature extraction
        self.branch_1 = nn.Sequential(*[
            RWKVSpatialPriorBlock(M*3, hidden_rate=hidden_rate, use_checkpoint=use_checkpoint)
            for _ in range(num_rwkv_blocks)
        ])
        
        # Branch 2: Output processing (keep similar to baseline for compatibility)
        self.branch_2 = nn.Sequential(
            RWKVSpatialPriorBlock(M*3, hidden_rate=hidden_rate, use_checkpoint=use_checkpoint),
            conv1x1(3*M, 2*M),
        )
    
    def forward(self, x, quant_step):
        """
        Forward pass
        
        Args:
            x: input tensor (B, 3*M, H, W)
            quant_step: quantization step scaling factor
        
        Returns:
            output tensor (B, 2*M, H, W) - scales and means
        """
        # Branch 1: RWKV feature extraction with quant_step modulation
        x = self.branch_1(x) * quant_step
        
        # Branch 2: Output projection
        x = self.branch_2(x)
        
        return x


class RWKVSpatialPrior_S3(nn.Module):
    """
    RWKV-enhanced spatial prior for s3 scale (full resolution)
    Replaces y_spatial_prior_s3 with RWKV processing
    """
    
    def __init__(self, M, num_rwkv_blocks=3, hidden_rate=4, use_checkpoint=False):
        """
        Args:
            M: base channel dimension
            num_rwkv_blocks: number of RWKV blocks (default: 3 to match DWConvRB×3)
            hidden_rate: expansion ratio for channel mixing
            use_checkpoint: whether to use gradient checkpointing
        """
        super().__init__()
        self.M = M
        self.num_rwkv_blocks = num_rwkv_blocks
        
        # Branch 1: RWKV feature extraction (3 blocks for higher capacity at full resolution)
        self.branch_1 = nn.Sequential(*[
            RWKVSpatialPriorBlock(M*3, hidden_rate=hidden_rate, use_checkpoint=use_checkpoint)
            for _ in range(num_rwkv_blocks)
        ])
        
        # Branch 2: Output processing (2 RWKV blocks + projection)
        branch_2_blocks = [
            RWKVSpatialPriorBlock(M*3, hidden_rate=hidden_rate, use_checkpoint=use_checkpoint)
            for _ in range(2)
        ]
        branch_2_blocks.append(conv1x1(3*M, 2*M))
        self.branch_2 = nn.Sequential(*branch_2_blocks)
    
    def forward(self, x, quant_step):
        """
        Forward pass
        
        Args:
            x: input tensor (B, 3*M, H, W)
            quant_step: quantization step scaling factor
        
        Returns:
            output tensor (B, 2*M, H, W) - scales and means
        """
        # Branch 1: RWKV feature extraction with quant_step modulation
        x = self.branch_1(x) * quant_step
        
        # Branch 2: Output projection
        x = self.branch_2(x)
        
        return x
