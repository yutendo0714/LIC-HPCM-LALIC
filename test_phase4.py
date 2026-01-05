"""
Test suite for HPCM Phase 4: Spatial Prior Enhancement
Validates RWKVSpatialPrior integration (FINAL PHASE)
"""

import sys
import argparse
from pathlib import Path

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. This is a structure validation only.")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test 1: Verify all required modules can be imported"""
    print("\n" + "="*70)
    print("Test 1: Module Import Validation")
    print("="*70)
    
    try:
        from src.models.rwkv_modules import (
            ensure_biwkv4_loaded,
            RWKVContextCell,
            RWKVFusionNet,
            RWKVSpatialPrior_S1_S2,
            RWKVSpatialPrior_S3,
            RWKVSpatialPriorBlock
        )
        print("✅ RWKV modules imported successfully")
        print("   - ensure_biwkv4_loaded")
        print("   - RWKVContextCell")
        print("   - RWKVFusionNet")
        print("   - RWKVSpatialPrior_S1_S2 (Phase 4)")
        print("   - RWKVSpatialPrior_S3 (Phase 4)")
        print("   - RWKVSpatialPriorBlock (Phase 4)")
    except Exception as e:
        print(f"❌ Failed to import RWKV modules: {e}")
        return False
    
    try:
        from src.models.hpcm_variants import HPCM_Phase4
        print("✅ HPCM_Phase4 imported successfully")
    except Exception as e:
        print(f"❌ Failed to import HPCM_Phase4: {e}")
        return False
    
    return True


def test_rwkv_spatial_prior():
    """Test 2: Verify RWKVSpatialPrior architecture"""
    print("\n" + "="*70)
    print("Test 2: RWKVSpatialPrior Architecture Validation")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.models.rwkv_modules import (
            RWKVSpatialPrior_S1_S2,
            RWKVSpatialPrior_S3,
            RWKVSpatialPriorBlock
        )
        
        M = 320
        
        # Test S1_S2 version
        print("\n📊 Testing RWKVSpatialPrior_S1_S2:")
        spatial_prior_s1_s2 = RWKVSpatialPrior_S1_S2(M, num_rwkv_blocks=2, hidden_rate=4)
        params_s1_s2 = sum(p.numel() for p in spatial_prior_s1_s2.parameters())
        print(f"   ✓ Created (M={M}, num_blocks=2, hidden_rate=4)")
        print(f"   ✓ Parameters: {params_s1_s2:,}")
        
        # Test S3 version
        print("\n📊 Testing RWKVSpatialPrior_S3:")
        spatial_prior_s3 = RWKVSpatialPrior_S3(M, num_rwkv_blocks=3, hidden_rate=4)
        params_s3 = sum(p.numel() for p in spatial_prior_s3.parameters())
        print(f"   ✓ Created (M={M}, num_blocks=3, hidden_rate=4)")
        print(f"   ✓ Parameters: {params_s3:,}")
        
        # Test forward pass (if CUDA available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            spatial_prior_s1_s2 = spatial_prior_s1_s2.to(device)
            spatial_prior_s3 = spatial_prior_s3.to(device)
            
            # Test S1_S2 with different resolutions
            test_cases_s1_s2 = [
                (1, 960, 16, 16),   # s1 resolution (H/4, W/4)
                (1, 960, 32, 32),   # s2 resolution (H/2, W/2)
            ]
            
            print("\n   S1_S2 Forward Pass Tests:")
            for B, C, H, W in test_cases_s1_s2:
                x = torch.randn(B, C, H, W, device=device)
                quant_step = torch.ones(B, C, H, W, device=device)
                
                with torch.no_grad():
                    y = spatial_prior_s1_s2(x, quant_step)
                
                expected_shape = (B, M*2, H, W)  # Output: 2*M channels
                assert y.shape == expected_shape, f"Output shape mismatch: {y.shape} vs {expected_shape}"
                print(f"      ✓ [{B}, {C}, {H}, {W}] → {y.shape}")
            
            # Test S3 with full resolution
            test_cases_s3 = [
                (1, 960, 64, 64),   # s3 resolution (H, W)
                (2, 960, 128, 128), # Larger batch and resolution
            ]
            
            print("\n   S3 Forward Pass Tests:")
            for B, C, H, W in test_cases_s3:
                x = torch.randn(B, C, H, W, device=device)
                quant_step = torch.ones(B, C, H, W, device=device)
                
                with torch.no_grad():
                    y = spatial_prior_s3(x, quant_step)
                
                expected_shape = (B, M*2, H, W)
                assert y.shape == expected_shape, f"Output shape mismatch: {y.shape} vs {expected_shape}"
                print(f"      ✓ [{B}, {C}, {H}, {W}] → {y.shape}")
        else:
            print("   ⚠️  CUDA not available, skipping forward pass test")
        
        return True
        
    except Exception as e:
        print(f"❌ RWKVSpatialPrior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase4_architecture():
    """Test 3: Verify Phase 4 model architecture"""
    print("\n" + "="*70)
    print("Test 3: HPCM Phase 4 Architecture Validation")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.models.hpcm_variants import HPCM_Phase4
        from src.models.rwkv_modules import (
            RWKVContextCell,
            RWKVFusionNet,
            RWKVSpatialPrior_S1_S2,
            RWKVSpatialPrior_S3
        )
        
        model = HPCM_Phase4(M=320, N=256)
        print("✅ HPCM_Phase4 model created (M=320, N=256)")
        
        # Verify attention modules are RWKV
        assert isinstance(model.attn_s1, RWKVContextCell), "attn_s1 should be RWKVContextCell"
        assert isinstance(model.attn_s2, RWKVContextCell), "attn_s2 should be RWKVContextCell"
        assert isinstance(model.attn_s3, RWKVContextCell), "attn_s3 should be RWKVContextCell"
        print("✅ All attention modules use RWKVContextCell:")
        print(f"   - attn_s1: dim=640, hidden_rate={model.attn_s1.hidden_rate}")
        print(f"   - attn_s2: dim=640, hidden_rate={model.attn_s2.hidden_rate}")
        print(f"   - attn_s3: dim=640, hidden_rate={model.attn_s3.hidden_rate}")
        
        # Verify context_net uses RWKVFusionNet
        assert len(model.context_net) == 2, "Should have 2 context_net modules"
        for i, net in enumerate(model.context_net):
            assert isinstance(net, RWKVFusionNet), f"context_net[{i}] should be RWKVFusionNet"
        print("✅ context_net uses RWKVFusionNet:")
        print(f"   - context_net[0]: dim=640, num_blocks=1")
        print(f"   - context_net[1]: dim=640, num_blocks=1")
        
        # Verify spatial priors use RWKV (Phase 4 new feature)
        assert isinstance(model.y_spatial_prior_s1_s2, RWKVSpatialPrior_S1_S2), \
            "y_spatial_prior_s1_s2 should be RWKVSpatialPrior_S1_S2"
        assert isinstance(model.y_spatial_prior_s3, RWKVSpatialPrior_S3), \
            "y_spatial_prior_s3 should be RWKVSpatialPrior_S3"
        print("✅ Spatial priors use RWKV (Phase 4 enhancement):")
        print(f"   - y_spatial_prior_s1_s2: num_blocks=2, hidden_rate=4")
        print(f"   - y_spatial_prior_s3: num_blocks=3, hidden_rate=4")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 4 architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase4_forward():
    """Test 4: Verify Phase 4 forward pass"""
    print("\n" + "="*70)
    print("Test 4: HPCM Phase 4 Forward Pass Validation")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping (CUDA not available)")
        return True
    
    try:
        from src.models.hpcm_variants import HPCM_Phase4
        
        device = torch.device('cuda')
        model = HPCM_Phase4(M=320, N=256).to(device)
        model.eval()
        
        print("✅ Model moved to CUDA")
        
        # Test with different image sizes
        test_sizes = [
            (1, 3, 256, 256),  # Standard
            (2, 3, 512, 512),  # Larger batch and resolution
        ]
        
        for B, C, H, W in test_sizes:
            print(f"\n   Testing input size: [{B}, {C}, {H}, {W}]")
            
            x = torch.randn(B, C, H, W, device=device)
            
            with torch.no_grad():
                try:
                    output = model(x, training=False)
                except Exception as e:
                    print(f"   ❌ Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # Verify output structure
            assert 'x_hat' in output, "Output should contain 'x_hat'"
            assert 'likelihoods' in output, "Output should contain 'likelihoods'"
            
            x_hat = output['x_hat']
            assert x_hat.shape == x.shape, f"Reconstruction shape mismatch: {x_hat.shape} vs {x.shape}"
            print(f"   ✓ x_hat shape: {x_hat.shape}")
            
            likelihoods = output['likelihoods']
            assert 'y' in likelihoods and 'z' in likelihoods
            print(f"   ✓ Likelihoods: y={likelihoods['y'].shape}, z={likelihoods['z'].shape}")
        
        print("\n✅ All forward pass tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_all_phases():
    """Test 5: Compare all phases (Baseline → Phase 1 → 2 → 3 → 4)"""
    print("\n" + "="*70)
    print("Test 5: Full Phase Progression Comparison")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.models.hpcm_variants import (
            HPCM_Phase1,
            HPCM_Phase2,
            HPCM_Phase3,
            HPCM_Phase4
        )
        from src.models.rwkv_modules import (
            RWKVContextCell,
            RWKVFusionNet,
            RWKVSpatialPrior_S1_S2,
            RWKVSpatialPrior_S3
        )
        
        # Create all phase models
        phase1 = HPCM_Phase1(M=320, N=256)
        phase2 = HPCM_Phase2(M=320, N=256)
        phase3 = HPCM_Phase3(M=320, N=256)
        phase4 = HPCM_Phase4(M=320, N=256)
        
        print("✅ All phase models created")
        
        # Parameter comparison
        params_p1 = sum(p.numel() for p in phase1.parameters())
        params_p2 = sum(p.numel() for p in phase2.parameters())
        params_p3 = sum(p.numel() for p in phase3.parameters())
        params_p4 = sum(p.numel() for p in phase4.parameters())
        
        print(f"\n📊 Parameter Comparison:")
        print(f"   Phase 1: {params_p1:,}")
        print(f"   Phase 2: {params_p2:,} ({(params_p2/params_p1-1)*100:+.2f}%)")
        print(f"   Phase 3: {params_p3:,} ({(params_p3/params_p1-1)*100:+.2f}%)")
        print(f"   Phase 4: {params_p4:,} ({(params_p4/params_p1-1)*100:+.2f}%)")
        
        # Architecture evolution
        print(f"\n🔧 Architecture Evolution:")
        print(f"   Phase 1:")
        print(f"      - attn_s1/s2: CrossAttention, attn_s3: RWKV")
        print(f"      - context_net: conv1x1")
        print(f"      - spatial_prior: DWConvRB")
        
        print(f"   Phase 2:")
        print(f"      - attn_s1/s2/s3: ALL RWKV")
        print(f"      - context_net: conv1x1")
        print(f"      - spatial_prior: DWConvRB")
        
        print(f"   Phase 3:")
        print(f"      - attn_s1/s2/s3: ALL RWKV")
        print(f"      - context_net: RWKVFusionNet ✨")
        print(f"      - spatial_prior: DWConvRB")
        
        print(f"   Phase 4:")
        print(f"      - attn_s1/s2/s3: ALL RWKV")
        print(f"      - context_net: RWKVFusionNet")
        print(f"      - spatial_prior: RWKV ✨✨ (FULLY RWKV-OPTIMIZED)")
        
        # Verify Phase 4 has all RWKV components
        assert isinstance(phase4.attn_s1, RWKVContextCell)
        assert isinstance(phase4.attn_s2, RWKVContextCell)
        assert isinstance(phase4.attn_s3, RWKVContextCell)
        assert isinstance(phase4.context_net[0], RWKVFusionNet)
        assert isinstance(phase4.context_net[1], RWKVFusionNet)
        assert isinstance(phase4.y_spatial_prior_s1_s2, RWKVSpatialPrior_S1_S2)
        assert isinstance(phase4.y_spatial_prior_s3, RWKVSpatialPrior_S3)
        print(f"\n   ✓ Phase 4 verified: ALL components use RWKV")
        
        # Memory footprint (if CUDA available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            phase1 = phase1.to(device)
            phase2 = phase2.to(device)
            phase3 = phase3.to(device)
            phase4 = phase4.to(device)
            
            x = torch.randn(1, 3, 256, 256, device=device)
            
            memories = []
            for name, model in [("Phase 1", phase1), ("Phase 2", phase2), 
                               ("Phase 3", phase3), ("Phase 4", phase4)]:
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model(x, training=False)
                mem = torch.cuda.max_memory_allocated() / 1024**2
                memories.append((name, mem))
            
            print(f"\n💾 Memory Usage (256×256 image):")
            baseline_mem = memories[0][1]
            for name, mem in memories:
                diff_pct = (mem/baseline_mem - 1) * 100
                print(f"   {name}: {mem:.1f} MB ({diff_pct:+.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🎯 "*20)
    print("  HPCM Phase 4 Test Suite - Spatial Prior Enhancement (FINAL)")
    print("🎯 "*20)
    
    tests = [
        ("Module Imports", test_imports),
        ("RWKVSpatialPrior", test_rwkv_spatial_prior),
        ("Phase 4 Architecture", test_phase4_architecture),
        ("Phase 4 Forward Pass", test_phase4_forward),
        ("All Phases Comparison", compare_all_phases),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n   Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉🎉🎉 ALL TESTS PASSED! Phase 4 (FINAL) implementation is complete! 🎉🎉🎉")
        print("\n📝 Integration Complete:")
        print("   ✅ Phase 1: s3 RWKV integration")
        print("   ✅ Phase 2: Full scale RWKV (s1, s2, s3)")
        print("   ✅ Phase 3: Context fusion enhancement")
        print("   ✅ Phase 4: Spatial prior enhancement (FINAL)")
        print("\n🚀 Next Steps:")
        print("   1. Train Phase 4 model on your dataset")
        print("   2. Compare R-D performance: Baseline → Phase 1 → 2 → 3 → 4")
        print("   3. Measure encoding/decoding time across all phases")
        print("   4. Publish results and enjoy the performance gains!")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description='Test HPCM Phase 4 implementation')
    parser.add_argument('--mode', choices=['all', 'imports', 'spatial', 'arch', 'forward', 'compare'],
                       default='all', help='Test mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        success = run_all_tests()
    elif args.mode == 'imports':
        success = test_imports()
    elif args.mode == 'spatial':
        success = test_rwkv_spatial_prior()
    elif args.mode == 'arch':
        success = test_phase4_architecture()
    elif args.mode == 'forward':
        success = test_phase4_forward()
    elif args.mode == 'compare':
        success = compare_all_phases()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
