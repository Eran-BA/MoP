#!/usr/bin/env python3
"""
Test script for GPT-MoP implementation and three-way comparison

This script demonstrates:
1. Building Baseline, Quartet, and MoP GPT models
2. Parameter matching analysis
3. Forward pass testing
4. MoP gate map extraction

Author: Eran Ben Artzy
"""

import os
import sys

import torch

# Add the mop directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mop"))

from mop.models import (ComparisonConfig, GPT_MoP, create_comparison_framework,
                        create_gpt_baseline, create_gpt_mop,
                        create_gpt_quartet)
from mop.models.quartet_attn_patch import TransformerConfig


def test_individual_models():
    """Test individual model creation and functionality"""
    print("🧪 Testing Individual Model Creation")
    print("=" * 50)

    # Configuration
    config = TransformerConfig(
        n_layer=4, n_head=4, n_embd=128, block_size=64, dropout=0.1, bias=False
    )

    vocab_size = 1000

    try:
        # Test Baseline
        print("📝 Testing GPT Baseline...")
        baseline = create_gpt_baseline(vocab_size, config)
        print(f"   ✅ Created: {baseline.__class__.__name__}")
        print(
            f"   ✅ Parameters: {sum(p.numel() for p in baseline.parameters() if p.requires_grad):,}"
        )

        # Test Quartet
        print("\n📝 Testing GPT Quartet...")
        quartet = create_gpt_quartet(vocab_size, config)
        print(f"   ✅ Created: {quartet.__class__.__name__}")
        print(
            f"   ✅ Parameters: {sum(p.numel() for p in quartet.parameters() if p.requires_grad):,}"
        )

        # Test MoP
        print("\n📝 Testing GPT MoP...")
        mop = create_gpt_mop(vocab_size, config, n_views=3, n_kernels=2)
        print(f"   ✅ Created: {mop.__class__.__name__}")
        print(
            f"   ✅ Parameters: {sum(p.numel() for p in mop.parameters() if p.requires_grad):,}"
        )

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass for all models"""
    print("\n🚀 Testing Forward Pass")
    print("=" * 50)

    config = TransformerConfig(
        n_layer=2, n_head=2, n_embd=64, block_size=32, dropout=0.1, bias=False
    )

    vocab_size = 100
    batch_size = 2
    seq_len = 16

    try:
        # Create test input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Test Baseline
        print("📝 Testing Baseline forward pass...")
        baseline = create_gpt_baseline(vocab_size, config)
        baseline.eval()
        with torch.no_grad():
            logits, loss = baseline(x, targets=y)
        print(f"   ✅ Logits: {logits.shape}, Loss: {loss.item():.4f}")

        # Test Quartet
        print("\n📝 Testing Quartet forward pass...")
        quartet = create_gpt_quartet(vocab_size, config)
        quartet.eval()
        with torch.no_grad():
            logits, loss = quartet(x, targets=y)
        print(f"   ✅ Logits: {logits.shape}, Loss: {loss.item():.4f}")

        # Test MoP
        print("\n📝 Testing MoP forward pass...")
        mop = create_gpt_mop(vocab_size, config, n_views=2, n_kernels=1)
        mop.eval()
        with torch.no_grad():
            logits, loss = mop(x, targets=y)
        print(f"   ✅ Logits: {logits.shape}, Loss: {loss.item():.4f}")

        # Test MoP gate extraction
        print("\n🔍 Testing MoP gate extraction...")
        gates, views, kernels = mop.get_gate_maps(x)
        print(f"   ✅ Gates: {gates.shape}")
        print(f"   ✅ Views: {views.shape}")
        print(f"   ✅ Kernels: {kernels.shape}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_comparison_framework():
    """Test the three-way comparison framework"""
    print("\n⚖️  Testing Comparison Framework")
    print("=" * 50)

    try:
        # Create configuration
        config = ComparisonConfig(
            n_layer=3, n_head=4, n_embd=96, block_size=64, n_views=3, n_kernels=2
        )

        # Create framework
        framework = create_comparison_framework(config)

        # Build models
        models = framework.build_models(vocab_size=500)

        # Print comparison summary
        framework.print_comparison_summary()

        # Test forward pass
        print("\n🧪 Testing Framework Forward Pass:")
        print("-" * 40)
        results = framework.test_forward_pass(batch_size=2, seq_len=32, vocab_size=500)

        for name, result in results.items():
            if "error" not in result:
                print(
                    f"{name:>10}: ✅ Logits {result['logits_shape']}, Loss: {result['loss_value']:.4f}"
                )
                if "mop_maps" in result:
                    maps = result["mop_maps"]
                    print(
                        f"{'':>10}  MoP Maps: Gates {maps['gates_shape']}, "
                        f"Views {maps['views_shape']}, Kernels {maps['kernels_shape']}"
                    )
            else:
                print(f"{name:>10}: ❌ Error: {result['error']}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧠 GPT-MoP Implementation Test Suite")
    print("=" * 60)
    print("Testing: Baseline vs Quartet vs MoP GPT models")
    print("=" * 60)

    tests = [
        ("Individual Model Creation", test_individual_models),
        ("Forward Pass Testing", test_forward_pass),
        ("Comparison Framework", test_comparison_framework),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔧 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:>25}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! GPT-MoP implementation is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Use the comparison framework for fair evaluations")
        print("   2. Train models on your language modeling tasks")
        print("   3. Analyze MoP gate patterns for interpretability")
        print("   4. Extend to other architectures (Whisper, multimodal, etc.)")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the errors above.")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
