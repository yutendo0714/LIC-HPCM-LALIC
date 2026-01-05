"""
Bi-directional WKV4 CUDA kernel wrapper for HPCM
Adapted from LALIC (RwkvCompress)
"""

import os
import torch
from torch.utils.cpp_extension import load


_biwkv4_loaded = False
_biwkv4_cuda = None


def load_biwkv4():
    """Load the Bi-directional WKV version 4 CUDA kernel"""
    global _biwkv4_cuda
    
    # Get the path to CUDA sources in RwkvCompress
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))
    rwkv_cuda_dir = os.path.join(project_root, "RwkvCompress", "models", "cuda")
    
    if not os.path.exists(rwkv_cuda_dir):
        raise RuntimeError(f"CUDA sources not found at {rwkv_cuda_dir}")
    
    _biwkv4_cuda = load(
        name="biwkv4_hpcm",
        sources=[
            os.path.join(rwkv_cuda_dir, "biwkv4_op_new.cpp"),
            os.path.join(rwkv_cuda_dir, "biwkv4_cuda_new.cu"),
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--expt-relaxed-constexpr",
        ],
    )
    return _biwkv4_cuda


def ensure_biwkv4_loaded():
    """Ensure CUDA kernel is loaded (load once, use many times)"""
    global _biwkv4_loaded, _biwkv4_cuda
    if not _biwkv4_loaded:
        _biwkv4_cuda = load_biwkv4()
        _biwkv4_loaded = True


class BiWKV4_HPCM(torch.autograd.Function):
    """
    Bi-directional WKV4 operation with autograd support
    O(N*T) linear attention mechanism
    """
    
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        
        ctx.save_for_backward(w, u, k, v)
        
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        
        y = torch.ops.biwkv4_hpcm.forward(w, u, k, v)
        
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        
        return y
    
    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        
        gw, gu, gk, gv = torch.ops.biwkv4_hpcm.backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
        )
        
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_BiWKV4_HPCM(w, u, k, v):
    """
    Run Bi-directional WKV4 operation
    
    Args:
        w: decay weights (C,) or (B, C)
        u: boost weights (C,) or (B, C)  
        k: key tensor (B, T, C)
        v: value tensor (B, T, C)
    
    Returns:
        output tensor (B, T, C)
    """
    return BiWKV4_HPCM.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())
