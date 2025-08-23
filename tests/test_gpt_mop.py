#!/usr/bin/env python3
"""
Test script for GPT-MoP implementation and three-way comparison

Author: Eran Ben Artzy
"""

import os
import sys
import torch

# Add the project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mop.models import (ComparisonConfig, create_comparison_framework,
                        create_gpt_baseline, create_gpt_mop, create_gpt_quartet)
from mop.models.quartet_attn_patch import TransformerConfig


def test_individual_models():
    config = TransformerConfig(
        n_layer=2, n_head=2, n_embd=64, block_size=32, dropout=0.1, bias=False
    )
    vocab_size = 200

    baseline = create_gpt_baseline(vocab_size, config)
    quartet = create_gpt_quartet(vocab_size, config)
    mop = create_gpt_mop(vocab_size, config, n_views=2, n_kernels=1)

    assert sum(p.numel() for p in baseline.parameters() if p.requires_grad) > 0
    assert sum(p.numel() for p in quartet.parameters() if p.requires_grad) > 0
    assert sum(p.numel() for p in mop.parameters() if p.requires_grad) > 0


def test_forward_pass():
    config = TransformerConfig(
        n_layer=2, n_head=2, n_embd=64, block_size=32, dropout=0.1, bias=False
    )
    vocab_size = 100
    x = torch.randint(0, vocab_size, (2, 16))
    y = torch.randint(0, vocab_size, (2, 16))

    baseline = create_gpt_baseline(vocab_size, config).eval()
    with torch.no_grad():
        logits, loss = baseline(x, targets=y)
    assert logits.shape[:2] == (2, 16)

    quartet = create_gpt_quartet(vocab_size, config).eval()
    with torch.no_grad():
        logits, loss = quartet(x, targets=y)
    assert logits.shape[:2] == (2, 16)

    mop = create_gpt_mop(vocab_size, config, n_views=2, n_kernels=1).eval()
    with torch.no_grad():
        logits, loss = mop(x, targets=y)
    assert logits.shape[:2] == (2, 16)


def test_comparison_framework():
    cfg = ComparisonConfig(n_layer=2, n_head=2, n_embd=64, block_size=32)
    fw = create_comparison_framework(cfg)
    models = fw.build_models(vocab_size=300)
    assert set(models.keys()) == {"baseline", "quartet", "mop"}


