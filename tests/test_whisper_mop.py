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
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mop.models import (WhisperConfig, WhisperMoP, create_whisper_baseline,
                        create_whisper_mop)


def test_individual_models():
    """Test individual Whisper model creation and functionality"""
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

    # Baseline
    baseline = create_whisper_baseline(config)
    n_params_base = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    assert n_params_base > 0

    # MoP
    mop = create_whisper_mop(config)
    n_params_mop = sum(p.numel() for p in mop.parameters() if p.requires_grad)
    assert n_params_mop > 0


def test_forward_pass():
    """Test forward pass for Whisper models"""
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

    # Inputs
    mel = torch.randn(batch_size, t_audio, config.n_mels)
    dec_input_ids = torch.randint(0, vocab_size, (batch_size, t_text))
    targets = torch.randint(0, vocab_size, (batch_size, t_text))

    # Baseline
    baseline = create_whisper_baseline(config).eval()
    with torch.no_grad():
        logits, loss, gates = baseline(mel, dec_input_ids, targets=targets)
    assert logits.shape[0] == batch_size and loss is not None and gates is not None

    # MoP
    mop = create_whisper_mop(config).eval()
    with torch.no_grad():
        logits, loss, gates = mop(mel, dec_input_ids, targets=targets)
    assert logits.shape[0] == batch_size and loss is not None and gates is not None


def test_audio_processing():
    """Test audio-specific processing capabilities"""
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

    # Synthetic spectrogram-like input
    batch_size = 2
    time_steps = 24
    mel = torch.randn(batch_size, time_steps, config.n_mels)

    # Decoder input
    dec_len = 12
    dec_input_ids = torch.randint(0, config.vocab_size, (batch_size, dec_len))

    # MoP model
    mop = create_whisper_mop(config).eval()
    with torch.no_grad():
        logits, loss, gates = mop(mel, dec_input_ids)
    assert logits.ndim == 3 and gates is not None


def test_temporal_spectral_patterns():
    """Test temporal pattern detection capabilities (via time-gates)"""
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

    # Construct a toy mel with recognizable time/freq patterns
    B, T, F = 1, 24, config.n_mels
    mel = torch.zeros(B, T, F)
    for t in range(min(T, F)):
        mel[0, t, t] = 1.0
    mel[0, :, 8] += 0.5
    mel[0, 12, :] += 0.3

    with torch.no_grad():
        gates = mop.get_gate_maps(mel)
    assert gates.shape[0] == B and gates.shape[-1] == T


