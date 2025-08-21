"""
GPT-MoP: Mixture of Products for Language Modeling

Extends the Quartet attention mechanism with spatial boolean logic for sequential data.
Author: Eran Ben Artzy
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quartet_attn_patch import (MLP, Block, CausalSelfAttention,
                                 TinyTransformerLM, TransformerConfig)


class ViewsLinear1D(nn.Module):
    """Multi-view projection for 1D sequences (tokens)"""

    def __init__(self, dim, n_views=5):
        super().__init__()
        self.n_views = n_views
        self.proj = nn.Linear(dim, n_views, bias=False)

    def forward(self, tok):
        # tok: (B, T, D) -> (B, V, T)
        B, T, D = tok.shape
        views = self.proj(tok)  # (B, T, V)
        views = views.transpose(1, 2)  # (B, V, T)
        return views


class Kernels1D(nn.Module):
    """1D convolutional kernels for sequential pattern detection"""

    def __init__(self, in_ch, n_kernels=3, kernel_size=3):
        super().__init__()
        self.n_kernels = n_kernels
        self.conv = nn.Conv1d(
            in_ch, n_kernels, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x):
        # x: (B, V, T) -> (B, K, T)
        return self.conv(x)


class FuseExcInh1D(nn.Module):
    """Excitatory/inhibitory gating for 1D sequences"""

    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, 2, kernel_size=1, bias=False
        )  # 2 channels: pos, neg
        self.alpha = nn.Parameter(torch.ones(2))  # Learnable weights

    def forward(self, x):
        # x: (B, V+K, T) -> gates: (B, 2, T)
        gates = self.conv(x)  # (B, 2, T)
        g_pos, g_neg = gates.chunk(2, dim=1)  # Split into positive/negative

        # Apply learnable weights
        a_pos, a_neg = self.alpha[0], self.alpha[1]

        return g_pos, g_neg, a_pos, a_neg


class MoPBlock(nn.Module):
    """Transformer block with MoP mechanism applied after attention"""

    def __init__(self, config: TransformerConfig, n_views=5, n_kernels=3):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        # MoP components
        self.views = ViewsLinear1D(config.n_embd, n_views=n_views)
        self.kernels = Kernels1D(in_ch=n_views, n_kernels=n_kernels)
        self.fuse = FuseExcInh1D(in_ch=n_views + n_kernels)

        self.n_views = n_views
        self.n_kernels = n_kernels

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Standard transformer block
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)

        # Apply MoP mechanism
        x = self.apply_mop(x)

        # MLP
        x = x + self.mlp(self.ln2(x))
        return x

    def apply_mop(self, x):
        """Apply Mixture of Products mechanism to token representations"""
        B, T, D = x.shape

        # Multi-view projection
        V = self.views(x)  # (B, V, T)

        # Learnable kernels
        K = self.kernels(V)  # (B, K, T)

        # Combine views and kernels
        maps = torch.cat([V, K], dim=1)  # (B, V+K, T)

        # Excitatory/inhibitory gating
        G_pos, G_neg, a_pos, a_neg = self.fuse(maps)
        gate = 1 + a_pos * G_pos - a_neg * G_neg  # (B, 1, T)

        # Apply gate to tokens
        gate_flat = gate.transpose(1, 2)  # (B, T, 1)
        x = x * gate_flat  # (B, T, D)

        return x

    def get_gate_maps(self, x):
        """Extract gate maps for visualization/analysis"""
        B, T, D = x.shape

        V = self.views(x)
        K = self.kernels(V)
        maps = torch.cat([V, K], dim=1)

        G_pos, G_neg, a_pos, a_neg = self.fuse(maps)
        gate = 1 + a_pos * G_pos - a_neg * G_neg

        return gate, V, K


class GPT_MoP(nn.Module):
    """GPT-style language model with Mixture of Products mechanism"""

    def __init__(
        self, vocab_size: int, config: TransformerConfig, n_views=5, n_kernels=3
    ):
        super().__init__()
        self.config = config
        self.n_views = n_views
        self.n_kernels = n_kernels

        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.use_abs_pos_emb
            else None
        )
        self.drop = nn.Dropout(config.dropout)

        # MoP-enhanced transformer blocks
        self.blocks = nn.ModuleList(
            [
                MoPBlock(config, n_views=n_views, n_kernels=n_kernels)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence length > block size"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok = self.wte(idx)
        if self.wpe is not None:
            x = self.drop(tok + self.wpe(pos))
        else:
            x = self.drop(tok)

        # Pass through MoP-enhanced blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_gate_maps(self, x):
        """Extract gate maps from all layers for analysis"""
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        tok = self.wte(x)
        if self.wpe is not None:
            x = self.drop(tok + self.wpe(pos))
        else:
            x = self.drop(tok)

        all_gates = []
        all_views = []
        all_kernels = []

        for block in self.blocks:
            # Get intermediate representations before MoP
            x_ln = block.ln1(x)
            x_attn = block.attn(x_ln)
            x_res = x + x_attn

            # Extract MoP maps
            gate, views, kernels = block.get_gate_maps(x_res)
            all_gates.append(gate)
            all_views.append(views)
            all_kernels.append(kernels)

            # Continue through the block
            x = x_res
            x = block.apply_mop(x)
            x = x + block.mlp(block.ln2(x))

        # Stack across layers
        gates = torch.stack(all_gates, dim=1)  # (B, L, 1, T)
        views = torch.stack(all_views, dim=1)  # (B, L, V, T)
        kernels = torch.stack(all_kernels, dim=1)  # (B, L, K, T)

        return gates, views, kernels


# Factory functions for easy model creation
def create_gpt_mop(vocab_size: int, config: TransformerConfig, n_views=5, n_kernels=3):
    """Create a GPT-MoP model with the specified configuration"""
    return GPT_MoP(
        vocab_size=vocab_size, config=config, n_views=n_views, n_kernels=n_kernels
    )


def create_gpt_baseline(vocab_size: int, config: TransformerConfig):
    """Create a baseline GPT model (no Quartet, no MoP)"""
    base_config = TransformerConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        block_size=config.block_size,
        bias=config.bias,
        use_quartet=False,
    )
    return TinyTransformerLM(vocab_size=vocab_size, config=base_config)


def create_gpt_quartet(vocab_size: int, config: TransformerConfig):
    """Create a GPT-Quartet model (Quartet attention, no MoP)"""
    quartet_config = TransformerConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        block_size=config.block_size,
        bias=config.bias,
        use_quartet=True,
    )
    return TinyTransformerLM(vocab_size=vocab_size, config=quartet_config)
