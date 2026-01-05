#!/usr/bin/env python3
"""
Test script for HPCM Phase 1 (RWKV integration at s3)
Tests forward pass, compression, and benchmarking
"""

import argparse
import time
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.hpcm_variants import HPCM_Phase1


def test_forward_pass(resolution=256, batch_size=1, device='cuda'):
    """Test basic forward pass"""
    print(f"\n{'='*60}")
    print(f"Testing Forward Pass - Resolution: {resolution}x{resolution}")
    print(f"{'='*60}\n")
    
    # Create model
    model = HPCM_Phase1(M=320, N=256).to(device)
    model.eval()
    
    # Random input
    x = torch.randn(batch_size, 3, resolution, resolution).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, training=False)
    
    # Timed run
    print("Running inference...")
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        output = model(x, training=False)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Results
    x_hat = output['x_hat']
    y_likelihood = output['likelihoods']['y']
    z_likelihood = output['likelihoods']['z']
    
    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_hat.shape}")
    print(f"   Y likelihood shape: {y_likelihood.shape}")
    print(f"   Z likelihood shape: {z_likelihood.shape}")
    print(f"   Time: {elapsed*1000:.2f} ms")
    print(f"   Throughput: {batch_size/elapsed:.2f} images/sec")
    
    # Compute MSE
    mse = torch.mean((x - x_hat) ** 2).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    return True


def test_rwkv_modules(device='cuda'):
    """Test individual RWKV modules"""
    print(f"\n{'='*60}")
    print(f"Testing RWKV Modules")
    print(f"{'='*60}\n")
    
    from src.models.rwkv_modules import (
        OmniShift, SpatialMix_HPCM, ChannelMix_HPCM, RWKVContextCell
    )
    
    B, C, H, W = 2, 640, 32, 32
    
    # Test OmniShift
    print("1. Testing OmniShift...")
    omni = OmniShift(C).to(device)
    x = torch.randn(B, C, H, W).to(device)
    y = omni(x)
    assert y.shape == x.shape, f"OmniShift output shape mismatch: {y.shape} vs {x.shape}"
    print(f"   ✅ OmniShift: {x.shape} -> {y.shape}")
    
    # Test SpatialMix
    print("2. Testing SpatialMix_HPCM...")
    spatial = SpatialMix_HPCM(C).to(device)
    y = spatial(x, (H, W))
    assert y.shape == x.shape, f"SpatialMix output shape mismatch: {y.shape} vs {x.shape}"
    print(f"   ✅ SpatialMix: {x.shape} -> {y.shape}")
    
    # Test ChannelMix
    print("3. Testing ChannelMix_HPCM...")
    channel = ChannelMix_HPCM(C, hidden_rate=4).to(device)
    y = channel(x, (H, W))
    assert y.shape == x.shape, f"ChannelMix output shape mismatch: {y.shape} vs {x.shape}"
    print(f"   ✅ ChannelMix: {x.shape} -> {y.shape}")
    
    # Test RWKVContextCell
    print("4. Testing RWKVContextCell...")
    cell = RWKVContextCell(C, hidden_rate=4).to(device)
    x_t = torch.randn(B, C, H, W).to(device)
    h_prev = torch.randn(B, C, H, W).to(device)
    h_t = cell(x_t, h_prev)
    assert h_t.shape == x_t.shape, f"RWKVContextCell output shape mismatch: {h_t.shape} vs {x_t.shape}"
    print(f"   ✅ RWKVContextCell: ({x_t.shape}, {h_prev.shape}) -> {h_t.shape}")
    
    print(f"\n✅ All RWKV modules working correctly!\n")
    return True


def test_model_comparison(resolution=256, device='cuda'):
    """Compare baseline HPCM with Phase 1"""
    print(f"\n{'='*60}")
    print(f"Comparing Models")
    print(f"{'='*60}\n")
    
    try:
        from src.models.HPCM_Base import HPCM as HPCM_Baseline
        has_baseline = True
    except:
        print("⚠️  Baseline HPCM not available for comparison")
        has_baseline = False
    
    # Phase 1 model
    phase1 = HPCM_Phase1(M=320, N=256).to(device)
    phase1.eval()
    
    # Count parameters
    phase1_params = sum(p.numel() for p in phase1.parameters())
    print(f"Phase 1 Parameters: {phase1_params:,}")
    
    if has_baseline:
        baseline = HPCM_Baseline(M=320, N=256).to(device)
        baseline.eval()
        baseline_params = sum(p.numel() for p in baseline.parameters())
        print(f"Baseline Parameters: {baseline_params:,}")
        print(f"Difference: {phase1_params - baseline_params:+,} ({(phase1_params/baseline_params - 1)*100:+.2f}%)")
    
    # Test forward pass speed
    x = torch.randn(1, 3, resolution, resolution).to(device)
    
    print(f"\nSpeed comparison on {resolution}x{resolution} image:")
    
    # Phase 1
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = phase1(x, training=False)
        torch.cuda.synchronize()
        phase1_time = (time.time() - start) / 10
    
    print(f"  Phase 1: {phase1_time*1000:.2f} ms/image")
    
    if has_baseline:
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = baseline(x, training=False)
            torch.cuda.synchronize()
            baseline_time = (time.time() - start) / 10
        
        print(f"  Baseline: {baseline_time*1000:.2f} ms/image")
        print(f"  Speedup: {baseline_time/phase1_time:.2f}x ({(1-phase1_time/baseline_time)*100:.1f}% faster)")
    
    return True


def test_with_image(image_path, device='cuda'):
    """Test with real image"""
    print(f"\n{'='*60}")
    print(f"Testing with Real Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0).to(device)
    
    # Model
    model = HPCM_Phase1(M=320, N=256).to(device)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(x, training=False)
    
    x_hat = output['x_hat']
    
    # Compute metrics
    mse = torch.mean((x - x_hat) ** 2).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"✅ Reconstruction successful!")
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")
    
    # Compute bits per pixel (approximate)
    y_bits = -torch.log2(output['likelihoods']['y']).sum()
    z_bits = -torch.log2(output['likelihoods']['z']).sum()
    total_bits = y_bits + z_bits
    num_pixels = x.shape[2] * x.shape[3]
    bpp = total_bits.item() / num_pixels
    
    print(f"   BPP: {bpp:.4f}")
    print(f"   Total bits: {total_bits.item():.0f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test HPCM Phase 1')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'forward', 'modules', 'compare', 'image'],
                       help='Test mode')
    parser.add_argument('--resolution', type=int, default=256,
                       help='Image resolution for testing')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"\n{'#'*60}")
    print(f"  HPCM Phase 1 Test Suite")
    print(f"  Device: {args.device.upper()}")
    print(f"{'#'*60}")
    
    success = True
    
    try:
        if args.mode in ['all', 'modules']:
            success &= test_rwkv_modules(args.device)
        
        if args.mode in ['all', 'forward']:
            success &= test_forward_pass(args.resolution, device=args.device)
        
        if args.mode in ['all', 'compare']:
            success &= test_model_comparison(args.resolution, device=args.device)
        
        if args.mode == 'image':
            if args.image is None:
                print("❌ Please provide --image path for image mode")
                success = False
            else:
                success &= test_with_image(args.image, device=args.device)
        
        if success:
            print(f"\n{'='*60}")
            print(f"  ✅ All tests passed!")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"  ❌ Some tests failed")
            print(f"{'='*60}\n")
            
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
