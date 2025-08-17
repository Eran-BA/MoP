"""
Quartet Attention: Dual-path attention mechanism

Augments standard scaled dot-product attention with a second path (Q₂·K₂ᵀ)
and learns to mix them via a learnable gate σ(m).

Author: Eran Ben Artzy
"""

from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    block_size: int = 512
    bias: bool = False
    # Quartet extras
    use_quartet: bool = True
    quartet_scale: float = 1.0
    quartet_gate_init: float = -5.0  # sigmoid(-5) ~ 0.0067
    score_norm_eps: float = 1e-5
    use_abs_pos_emb: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        if config.use_quartet:
            self.q2_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.k2_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.mixture = nn.Parameter(torch.tensor([config.quartet_gate_init], dtype=torch.float32))
            self.quartet_scale = nn.Parameter(torch.tensor([config.quartet_scale], dtype=torch.float32))
        else:
            self.q2_proj = None
            self.k2_proj = None
            self.register_parameter("mixture", None)
            self.register_parameter("quartet_scale", None)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, need_weights: bool = False):
        B, T, C = x.size()
        H, Dh = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        if self.config.use_quartet:
            q2 = self.q2_proj(x).view(B, T, H, Dh).transpose(1, 2)
            k2 = self.k2_proj(x).view(B, T, H, Dh).transpose(1, 2)
            q2k2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

            def _norm(scores):
                mu = scores.mean(dim=-1, keepdim=True)
                sigma = scores.std(dim=-1, keepdim=True)
                return (scores - mu) / (sigma + self.config.score_norm_eps)

            qk_norm = _norm(qk)
            q2k2_norm = _norm(q2k2)

            m = torch.sigmoid(self.mixture)  # scalar in [0,1]
            scores = (1.0 - m) * qk_norm + m * (qk_norm * q2k2_norm) * self.quartet_scale
        else:
            mu = qk.mean(dim=-1, keepdim=True)
            sigma = qk.std(dim=-1, keepdim=True)
            scores = (qk - mu) / (sigma + 1e-5)

        causal = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(causal == 0, float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.o_proj(y))

        if need_weights:
            return y, attn
        return y


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if config.use_abs_pos_emb else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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

    def forward(self, idx: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence length > block size"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok = self.wte(idx)
        if self.wpe is not None:
            x = self.drop(tok + self.wpe(pos))
        else:
            x = self.drop(tok)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
