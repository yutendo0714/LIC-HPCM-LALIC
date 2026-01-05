"""
RWKV modules for HPCM integration
Bi-directional RWKV (linear attention) components for learned image compression
"""

from .biwkv4 import ensure_biwkv4_loaded, RUN_BiWKV4_HPCM
from .omni_shift import OmniShift
from .spatial_mix import SpatialMix_HPCM
from .channel_mix import ChannelMix_HPCM
from .rwkv_context_cell import RWKVContextCell
from .rwkv_fusion_net import RWKVFusionNet, RWKVFusionBlock
from .rwkv_spatial_prior import (
    RWKVSpatialPrior_S1_S2,
    RWKVSpatialPrior_S3,
    RWKVSpatialPriorBlock
)

__all__ = [
    'ensure_biwkv4_loaded',
    'RUN_BiWKV4_HPCM',
    'OmniShift',
    'SpatialMix_HPCM',
    'ChannelMix_HPCM',
    'RWKVContextCell',
    'RWKVFusionNet',
    'RWKVFusionBlock',
    'RWKVSpatialPrior_S1_S2',
    'RWKVSpatialPrior_S3',
    'RWKVSpatialPriorBlock',
]
