"""
HPCM variants with RWKV integration
Progressive integration phases from baseline to fully RWKV-enhanced
"""

from .hpcm_phase1 import HPCM_Phase1
from .hpcm_phase2 import HPCM_Phase2
from .hpcm_phase3 import HPCM_Phase3
from .hpcm_phase4 import HPCM_Phase4

__all__ = [
    'HPCM_Phase1',
    'HPCM_Phase2',
    'HPCM_Phase3',
    'HPCM_Phase4',
]
