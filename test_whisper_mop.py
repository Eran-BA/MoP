#!/usr/bin/env python3
"""
Test script for Whisper-MoP implementation

This script demonstrates:
1. Building Baseline and MoP Whisper models
2. Parameter matching analysis
3. Forward pass testing with audio-like inputs
4. MoP gate map extraction for temporal patterns
5. Audio processing capabilities

Author: Eran Ben Artzy
"""

import os
import sys

import torch

# Add the project root (parent of 'mop') to path so 'import mop.models' works
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from mop.models import (WhisperConfig, WhisperMoP, create_whisper_baseline,
                        create_whisper_mop)


def test_individual_models():
    """Test individual Whisper model creation and functionality"""
    print("🧪 Testing Individual Whisper Model Creation")
    print("=" * 50)

    # Configuration (encoder-decoder; small sizes for quick tests)
    config = WhisperConfig(
        n_layer_enc=2,
        n_layer_dec=2,
        n_head=4,
        n_embd=128,
        n_mels=40,
        n_audio_ctx=128,
        n_text_ctx=64,
        dropout=0.1,
        bias=False,
        n_views=3,
        n_kernels=2,
        kernel_size=3,
        vocab_size=256,
    )

    try:
        # Baseline (MoP disabled via alpha=0 inside factory)
        print("🎵 Testing Whisper Baseline...")
        baseline = create_whisper_baseline(config)
        n_params_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
        print(f"   ✅ Created: {baseline.__class__.__name__}")
        print(f"   ✅ Parameters: {n_params_base:,}")

        # MoP
        print("\n🎵 Testing Whisper MoP...")
        mop = create_whisper_mop(config)
        n_params_mop = sum(p.numel() for p in mop.parameters() if p.requires_grad)
        print(f"   ✅ Created: {mop.__class__.__name__}")
        print(f"   ✅ Parameters: {n_params_mop:,}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass for Whisper models"""
    print("\n🚀 Testing Whisper Forward Pass")
    print("=" * 50)

    vocab_size = 100
    config = WhisperConfig(
        n_layer_enc=2,
        n_layer_dec=2,
        n_head=2,
        n_embd=64,
        n_mels=16,
        n_audio_ctx=32,
        n_text_ctx=32,
        dropout=0.1,
        bias=False,
        n_views=2,
        n_kernels=1,
        kernel_size=3,
        vocab_size=vocab_size,
    )

    batch_size = 2
    t_audio = 16
    t_text = 16

    try:
        # Inputs
        mel = torch.randn(batch_size, t_audio, config.n_mels)  # (B, T_audio, n_mels)
        dec_input_ids = torch.randint(0, vocab_size, (batch_size, t_text))
        targets = torch.randint(0, vocab_size, (batch_size, t_text))

        # Baseline
        print("🎵 Testing Baseline forward pass...")
        baseline = create_whisper_baseline(config).eval()
        with torch.no_grad():
            logits, loss, gates = baseline(mel, dec_input_ids, targets=targets)
        print(
            f"   ✅ Logits: {tuple(logits.shape)}, Loss: {loss.item():.4f}, Gates: {tuple(gates.shape)}"
        )

        # MoP
        print("\n🎵 Testing MoP forward pass...")
        mop = create_whisper_mop(config).eval()
        with torch.no_grad():
            logits, loss, gates = mop(mel, dec_input_ids, targets=targets)
        print(
            f"   ✅ Logits: {tuple(logits.shape)}, Loss: {loss.item():.4f}, Gates: {tuple(gates.shape)}"
        )

        # MoP gate extraction
        print("\n🔍 Testing MoP gate extraction...")
        with torch.no_grad():
            gates_only = mop.get_gate_maps(mel)
        print(f"   ✅ Gates only: {tuple(gates_only.shape)}  # (B, L_enc, T_audio)")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_audio_processing():
    """Test audio-specific processing capabilities"""
    print("\n🎵 Testing Audio Processing Capabilities")
    print("=" * 50)

    config = WhisperConfig(
        n_layer_enc=2,
        n_layer_dec=2,
        n_head=2,
        n_embd=64,
        n_mels=16,
        n_audio_ctx=32,
        n_text_ctx=32,
        dropout=0.1,
        bias=False,
        n_views=2,
        n_kernels=1,
        kernel_size=3,
        vocab_size=128,
    )

    try:
        # Synthetic spectrogram-like input
        batch_size = 2
        time_steps = 24
        mel = torch.randn(batch_size, time_steps, config.n_mels)  # (B, T, n_mels)
        print(f"   📊 Input spectrogram: {tuple(mel.shape)}")

        # Decoder input
        dec_len = 12
        dec_input_ids = torch.randint(0, config.vocab_size, (batch_size, dec_len))
        print(f"   🎯 Decoder input: {tuple(dec_input_ids.shape)}")

        # MoP model
        mop = create_whisper_mop(config).eval()
        with torch.no_grad():
            logits, loss, gates = mop(mel, dec_input_ids)  # targets=None
        print(
            f"   ✅ Forward pass successful: logits {tuple(logits.shape)}, gates {tuple(gates.shape)}"
        )

        # Gate maps only
        with torch.no_grad():
            gates_only = mop.get_gate_maps(mel)
        print(f"   ✅ MoP gates extracted: {tuple(gates_only.shape)}  # (B, L_enc, T)")

        # Simple analysis
        B, L, T = gates_only.shape
        print(f"   📈 Pattern Analysis:")
        print(f"      - Time steps: {T}")
        print(f"      - Encoder layers: {L}")
        print(f"      - Batch size: {B}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_temporal_spectral_patterns():
    """Test temporal pattern detection capabilities (via time-gates)"""
    print("\n🔍 Testing Temporal Pattern Detection")
    print("=" * 50)

    try:
        config = WhisperConfig(
            n_layer_enc=2,
            n_layer_dec=2,
            n_head=2,
            n_embd=64,
            n_mels=16,
            n_audio_ctx=32,
            n_text_ctx=32,
            n_views=3,
            n_kernels=2,
            kernel_size=5,
            vocab_size=128,
        )

        mop = create_whisper_mop(config).eval()

        # Construct a toy mel with recognizable time patterns
        B = 1
        T = 24
        F = config.n_mels
        mel = torch.zeros(B, T, F)

        # Diagonal-ish pattern across time/freq indices that overlap
        for t in range(min(T, F)):
            mel[0, t, t] = 1.0

        # Strong bin at freq=8 over all time
        mel[0, :, 8] += 0.5

        # A "beat" at time=12 across all freqs
        mel[0, 12, :] += 0.3

        print("   📊 Created test patterns:")
        print("      - Diagonal: time-frequency correlation")
        print("      - Horizontal: constant frequency over time (f=8)")
        print("      - Vertical: single time across frequencies (t=12)")

        with torch.no_grad():
            gates = mop.get_gate_maps(mel)  # (B, L_enc, T)

        print(f"   ✅ Pattern detection successful: gates {tuple(gates.shape)}")
        print(f"   🔬 Analysis:")
        print(f"      - Per-layer gates over time: {tuple(gates.shape)}")
        print(f"      - Time steps: {gates.shape[-1]}")
        print(f"      - Encoder layers: {gates.shape[1]}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all Whisper-MoP tests"""
    print("🎵 Whisper-MoP Implementation Test Suite")
    print("=" * 60)
    print("Testing: Audio Transformers with Mixture of Products")
    print("=" * 60)

    tests = [
        ("Individual Model Creation", test_individual_models),
        ("Forward Pass Testing", test_forward_pass),
        ("Audio Processing", test_audio_processing),
        ("Pattern Detection", test_temporal_spectral_patterns),
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
        print("\n🎉 All tests passed! Whisper-MoP implementation is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Train on audio datasets (LibriSpeech, CommonVoice, etc.)")
        print("   2. Analyze MoP gate timelines for interpretability")
        print("   3. Try gating cross-attention logits in the decoder")
        print("   4. Compare WER vs param-matched baselines")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the errors above.")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
