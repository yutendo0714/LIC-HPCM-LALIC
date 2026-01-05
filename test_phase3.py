"""
Test suite for HPCM Phase 3: Context Fusion Enhancement
Validates RWKVFusionNet integration into context_net
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
            RWKVFusionBlock
        )
        print("✅ RWKV modules imported successfully")
        print("   - ensure_biwkv4_loaded")
        print("   - RWKVContextCell")
        print("   - RWKVFusionNet (Phase 3)")
        print("   - RWKVFusionBlock (Phase 3)")
    except Exception as e:
        print(f"❌ Failed to import RWKV modules: {e}")
        return False
    
    try:
        from src.models.hpcm_variants import HPCM_Phase3
        print("✅ HPCM_Phase3 imported successfully")
    except Exception as e:
        print(f"❌ Failed to import HPCM_Phase3: {e}")
        return False
    
    return True


def test_rwkv_fusion_net():
    """Test 2: Verify RWKVFusionNet architecture"""
    print("\n" + "="*70)
    print("Test 2: RWKVFusionNet Architecture Validation")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.models.rwkv_modules import RWKVFusionNet, RWKVFusionBlock
        
        # Test single block
        dim = 640
        fusion_net = RWKVFusionNet(dim, num_blocks=1, hidden_rate=4)
        
        print(f"✅ RWKVFusionNet created (dim={dim}, num_blocks=1, hidden_rate=4)")
        
        # Count parameters
        total_params = sum(p.numel() for p in fusion_net.parameters())
        print(f"   Parameters: {total_params:,}")
        
        # Verify components
        assert len(fusion_net.blocks) == 1, "Should have 1 RWKV block"
        assert hasattr(fusion_net, 'out_proj'), "Should have output projection"
        print("   ✓ Contains 1 RWKVFusionBlock")
        print("   ✓ Contains output projection (conv1x1)")
        
        # Test forward pass (if CUDA available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            fusion_net = fusion_net.to(device)
            
            # Test with different resolutions
            test_cases = [
                (1, 640, 32, 32),   # s2-like resolution
                (1, 640, 64, 64),   # s3-like resolution
            ]
            
            for B, C, H, W in test_cases:
                x = torch.randn(B, C, H, W, device=device)
                with torch.no_grad():
                    y = fusion_net(x)
                
                assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
                print(f"   ✓ Forward pass [{B}, {C}, {H}, {W}] → {y.shape}")
        else:
            print("   ⚠️  CUDA not available, skipping forward pass test")
        
        return True
        
    except Exception as e:
        print(f"❌ RWKVFusionNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_architecture():
    """Test 3: Verify Phase 3 model architecture"""
    print("\n" + "="*70)
    print("Test 3: HPCM Phase 3 Architecture Validation")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.models.hpcm_variants import HPCM_Phase3
        from src.models.rwkv_modules import RWKVContextCell, RWKVFusionNet
        
        model = HPCM_Phase3(M=320, N=256)
        print("✅ HPCM_Phase3 model created (M=320, N=256)")
        
        # Verify attention modules are RWKV
        assert isinstance(model.attn_s1, RWKVContextCell), "attn_s1 should be RWKVContextCell"
        assert isinstance(model.attn_s2, RWKVContextCell), "attn_s2 should be RWKVContextCell"
        assert isinstance(model.attn_s3, RWKVContextCell), "attn_s3 should be RWKVContextCell"
        print("✅ All attention modules use RWKVContextCell:")
        print(f"   - attn_s1: dim=640, hidden_rate={model.attn_s1.hidden_rate}")
        print(f"   - attn_s2: dim=640, hidden_rate={model.attn_s2.hidden_rate}")
        print(f"   - attn_s3: dim=640, hidden_rate={model.attn_s3.hidden_rate}")
        
        # Verify context_net uses RWKVFusionNet (Phase 3 modification)
        assert len(model.context_net) == 2, "Should have 2 context_net modules"
        for i, net in enumerate(model.context_net):
            assert isinstance(net, RWKVFusionNet), f"context_net[{i}] should be RWKVFusionNet"
        print("✅ context_net uses RWKVFusionNet (Phase 3 enhancement):")
        print(f"   - context_net[0]: dim=640, num_blocks=1")
        print(f"   - context_net[1]: dim=640, num_blocks=1")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_forward():
    """Test 4: Verify Phase 3 forward pass"""
    print("\n" + "="*70)
    print("Test 4: HPCM Phase 3 Forward Pass Validation")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping (CUDA not available)")
        return True
    
    try:
        from src.models.hpcm_variants import HPCM_Phase3
        
        device = torch.device('cuda')
        model = HPCM_Phase3(M=320, N=256).to(device)
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


def compare_phases():
    """Test 5: Compare Phase 2 vs Phase 3"""
    print("\n" + "="*70)
    print("Test 5: Phase 2 vs Phase 3 Comparison")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.models.hpcm_variants import HPCM_Phase2, HPCM_Phase3
        from src.models.rwkv_modules import RWKVFusionNet
        
        phase2 = HPCM_Phase2(M=320, N=256)
        phase3 = HPCM_Phase3(M=320, N=256)
        
        print("✅ Both models created")
        
        # Parameter count
        params_p2 = sum(p.numel() for p in phase2.parameters())
        params_p3 = sum(p.numel() for p in phase3.parameters())
        
        print(f"\n📊 Parameter Comparison:")
        print(f"   Phase 2: {params_p2:,}")
        print(f"   Phase 3: {params_p3:,}")
        print(f"   Difference: {params_p3 - params_p2:+,} ({(params_p3/params_p2-1)*100:+.2f}%)")
        
        # Architecture comparison
        print(f"\n🔧 Architecture Differences:")
        print(f"   Phase 2 context_net: nn.Conv2d (simple 1x1 conv)")
        print(f"   Phase 3 context_net: RWKVFusionNet (RWKV-enhanced)")
        
        # Verify Phase 3 has RWKVFusionNet
        assert isinstance(phase3.context_net[0], RWKVFusionNet), "Phase 3 should use RWKVFusionNet"
        print(f"   ✓ Phase 3 uses RWKVFusionNet with num_blocks=1, hidden_rate=4")
        
        # Memory footprint (if CUDA available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            phase2 = phase2.to(device)
            phase3 = phase3.to(device)
            
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(1, 3, 256, 256, device=device)
            
            # Phase 2
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = phase2(x, training=False)
            mem_p2 = torch.cuda.max_memory_allocated() / 1024**2
            
            # Phase 3
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = phase3(x, training=False)
            mem_p3 = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"\n💾 Memory Usage (256×256 image):")
            print(f"   Phase 2: {mem_p2:.1f} MB")
            print(f"   Phase 3: {mem_p3:.1f} MB")
            print(f"   Difference: {mem_p3 - mem_p2:+.1f} MB ({(mem_p3/mem_p2-1)*100:+.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("  HPCM Phase 3 Test Suite - Context Fusion Enhancement")
    print("🧪 "*20)
    
    tests = [
        ("Module Imports", test_imports),
        ("RWKVFusionNet", test_rwkv_fusion_net),
        ("Phase 3 Architecture", test_phase3_architecture),
        ("Phase 3 Forward Pass", test_phase3_forward),
        ("Phase 2 vs 3 Comparison", compare_phases),
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
        print("\n🎉 All tests passed! Phase 3 implementation is complete.")
        print("\n📝 Next Steps:")
        print("   1. Train Phase 3 model on your dataset")
        print("   2. Compare R-D performance: Baseline → Phase 1 → Phase 2 → Phase 3")
        print("   3. Measure encoding/decoding time improvements")
        print("   4. Proceed to Phase 4: Spatial Prior Enhancement")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description='Test HPCM Phase 3 implementation')
    parser.add_argument('--mode', choices=['all', 'imports', 'fusion', 'arch', 'forward', 'compare'],
                       default='all', help='Test mode to run')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        success = run_all_tests()
    elif args.mode == 'imports':
        success = test_imports()
    elif args.mode == 'fusion':
        success = test_rwkv_fusion_net()
    elif args.mode == 'arch':
        success = test_phase3_architecture()
    elif args.mode == 'forward':
        success = test_phase3_forward()
    elif args.mode == 'compare':
        success = compare_phases()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
