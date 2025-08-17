#!/usr/bin/env python3
"""
Quick test script to verify MoP models work correctly.

Run this after setting up your repository:
python test_models.py

Author: Eran Ben Artzy
"""

import torch
import sys
import os

# Add the current directory to path so we can import mop
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mop_models():
    """Test that MoP models can be imported and run."""
    
    print("🧪 Testing MoP Models")
    print("=" * 30)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from mop import ViT_MoP, ViT_Baseline
        from mop.models import ViewsLinear, Kernels3, FuseExcInh
        print("✅ All imports successful!")
        
        # Test model creation
        print("\n🏗️  Testing model creation...")
        baseline = ViT_Baseline(dim=256, depth=6, heads=4, n_classes=10)
        mop_model = ViT_MoP(dim=256, depth=6, heads=4, n_classes=10, n_views=5, n_kernels=3)
        print("✅ Models created successfully!")
        
        # Count parameters
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        baseline_params = count_params(baseline)
        mop_params = count_params(mop_model)
        
        print(f"\n📊 Parameter counts:")
        print(f"   Baseline: {baseline_params:,} ({baseline_params/1e6:.2f}M)")
        print(f"   MoP:      {mop_params:,} ({mop_params/1e6:.2f}M)")
        print(f"   Difference: {mop_params - baseline_params:+,}")
        
        # Test forward pass
        print("\n🚀 Testing forward pass...")
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)  # CIFAR-10 format
        
        # Baseline forward pass
        baseline.eval()
        with torch.no_grad():
            baseline_out = baseline(x)
        print(f"✅ Baseline output: {baseline_out.shape}")
        
        # MoP forward pass
        mop_model.eval()
        with torch.no_grad():
            mop_out = mop_model(x)
        print(f"✅ MoP output: {mop_out.shape}")
        
        # Test MoP gate extraction
        print("\n🔍 Testing gate extraction...")
        with torch.no_grad():
            gates, views, kernels = mop_model.get_gate_maps(x)
        print(f"✅ Gates: {gates.shape}")
        print(f"✅ Views: {views.shape}")
        print(f"✅ Kernels: {kernels.shape}")
        
        # Sanity checks
        assert baseline_out.shape == (batch_size, 10), f"Expected {(batch_size, 10)}, got {baseline_out.shape}"
        assert mop_out.shape == (batch_size, 10), f"Expected {(batch_size, 10)}, got {mop_out.shape}"
        assert gates.shape[0] == batch_size, f"Gates batch size mismatch"
        assert views.shape == (batch_size, 5, 8, 8), f"Views shape mismatch"
        assert kernels.shape == (batch_size, 3, 8, 8), f"Kernels shape mismatch"
        
        print("\n🎉 All tests passed!")
        print("✅ Your MoP implementation is working correctly!")
        print(f"\n📋 Summary:")
        print(f"   • Models can be imported and created")
        print(f"   • Forward passes work correctly")
        print(f"   • Gate extraction works")
        print(f"   • Parameter counts: Baseline {baseline_params/1e6:.1f}M, MoP {mop_params/1e6:.1f}M")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Make sure you:")
        print("   1. Created all the model files (components.py, vit_mop.py, vit_baseline.py)")
        print("   2. Updated the __init__.py files")
        print("   3. Are running from the repository root directory")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mop_models()
    if success:
        print(f"\n🚀 Ready to commit and push to GitHub!")
    else:
        print(f"\n🔧 Fix the issues above, then try again.")
    
    sys.exit(0 if success else 1)