#!/usr/bin/env python3
"""
Simple test to debug Whisper-MoP dimension issues
"""

import os
import sys

import torch

# Add the mop directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mop"))

from mop.models import WhisperConfig, create_whisper_mop


def test_dimensions():
    """Test the dimension handling in Whisper-MoP"""
    print("🔍 Testing Whisper-MoP Dimensions")
    print("=" * 40)

    # Use a simple configuration
    config = WhisperConfig(
        n_layer_enc=1,
        n_layer_dec=1,
        n_head=2,
        n_embd=64,
        n_mels=16,
        n_audio_ctx=32,
        n_text_ctx=16,
        dropout=0.0,
        bias=False,
        n_views=2,
        n_kernels=1,
        kernel_size=3,
    )

    try:
        # Create model
        model = create_whisper_mop(config)
        print(
            f"✅ Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"
        )

        # Test input
        batch_size = 2
        time_steps = 8
        freq_bins = 16

        # Create mel spectrogram input: (B, T_audio, n_mels)
        mel = torch.randn(batch_size, time_steps, freq_bins)
        print(f"📥 Mel spectrogram input: {mel.shape}")

        # Create decoder input: (B, T_text)
        dec_input_ids = torch.randint(0, 100, (batch_size, 6))
        print(f"📥 Decoder input: {dec_input_ids.shape}")

        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, loss, gates = model(mel, dec_input_ids)
            print(f"✅ Forward pass successful: {logits.shape}")
            print(f"✅ Gates shape: {gates.shape}")

            # Test gate extraction
            gates_only = model.get_gate_maps(mel)
            print(f"✅ Gate extraction successful: {gates_only.shape}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dimensions()
    if success:
        print("\n🎉 Dimension test passed!")
    else:
        print("\n⚠️  Dimension test failed!")
