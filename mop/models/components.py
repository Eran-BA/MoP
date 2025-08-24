"""
Core MoP Components

Author: Eran Ben Artzy
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) implementation."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, in_ch=3, dim=256, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch, bias=False)

    def forward(self, x):
        x = self.proj(x)  # (B,D,H/P,W/P)
        B, D, Gh, Gw = x.shape
        return x.flatten(2).transpose(1, 2), (Gh, Gw)


class MSA(nn.Module):
    """Multi-Head Self-Attention."""

    def __init__(self, dim, heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.dk))
        attn = self.attn_drop(attn.softmax(dim=-1))

        y = attn @ v
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(y))


class MLP(nn.Module):
    """MLP block."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid, bias=False)
        self.fc2 = nn.Linear(hid, dim, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class MoEMLP(nn.Module):
    """Mixture-of-Experts feed-forward MLP (token-level), simple top-1 routing.

    Notes:
    - Implements E experts, each a standard 2-layer MLP with GELU.
    - Gating is token-wise over the hidden dimension of the input.
    - For simplicity and stability, we compute all experts and combine with a
      one-hot gate (top-1). This preserves correctness and simplicity, while not
      attempting compute savings.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, num_experts: int = 4):
        super().__init__()
        assert num_experts >= 2, "MoE requires at least 2 experts"
        hidden = int(dim * mlp_ratio)
        self.num_experts = int(num_experts)
        # Per-expert MLPs
        self.fc1 = nn.ModuleList([nn.Linear(dim, hidden, bias=False) for _ in range(self.num_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(hidden, dim, bias=False) for _ in range(self.num_experts)])
        self.act = nn.GELU(approximate="tanh")
        # Gating over experts
        self.gate = nn.Linear(dim, self.num_experts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        logits = self.gate(x_flat)  # [B*N, E]
        # Top-1 routing (one-hot)
        top_idx = logits.argmax(dim=-1)  # [B*N]
        one_hot = torch.zeros_like(logits).scatter_(1, top_idx.unsqueeze(-1), 1.0)
        y_sum = torch.zeros_like(x_flat)
        # Compute all experts and mix with one-hot gates
        for e in range(self.num_experts):
            y_e = self.fc2[e](self.act(self.fc1[e](x_flat)))  # [B*N, D]
            w_e = one_hot[:, e].unsqueeze(-1)  # [B*N,1]
            y_sum = y_sum + w_e * y_e
        return y_sum.reshape(B, N, D)


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self, dim, heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MSA(dim, heads, attn_drop, drop)
        self.dp1 = DropPath(drop_path)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp1(self.attn(self.ln1(x)))
        x = x + self.dp2(self.mlp(self.ln2(x)))
        return x


class BlockMoE(nn.Module):
    """Transformer block variant with MoE MLP instead of dense MLP."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        num_experts: int = 4,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MSA(dim, heads, attn_drop, drop)
        self.dp1 = DropPath(drop_path)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MoEMLP(dim, mlp_ratio, num_experts)
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp1(self.attn(self.ln1(x)))
        x = x + self.dp2(self.mlp(self.ln2(x)))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder."""

    def __init__(
        self,
        dim=256,
        depth=6,
        heads=4,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
        patch=4,
        num_tokens=64,
    ):
        super().__init__()
        self.patch = PatchEmbed(dim=dim, patch=patch)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))

        dps = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, drop, 0.0, dps[i]) for i in range(depth)]
        )
        self.ln_f = nn.LayerNorm(dim)

        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        tok, (Gh, Gw) = self.patch(x)
        tok = tok + self.pos

        for blk in self.blocks:
            tok = blk(tok)

        tok = self.ln_f(tok)
        return tok, (Gh, Gw)


class ViTEncoderMoE(nn.Module):
    """Vision Transformer Encoder with MoE MLP blocks."""

    def __init__(
        self,
        dim=256,
        depth=6,
        heads=4,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.1,
        patch=4,
        num_tokens=64,
        num_experts: int = 4,
    ):
        super().__init__()
        self.patch = PatchEmbed(dim=dim, patch=patch)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))

        dps = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                BlockMoE(
                    dim,
                    heads,
                    mlp_ratio,
                    drop,
                    0.0,
                    dps[i],
                    num_experts=num_experts,
                )
                for i in range(depth)
            ]
        )
        self.ln_f = nn.LayerNorm(dim)

        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        tok, (Gh, Gw) = self.patch(x)
        tok = tok + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln_f(tok)
        return tok, (Gh, Gw)


class ViewsLinear(nn.Module):
    """Multi-view projection layer."""

    def __init__(self, dim, n_views=5):
        super().__init__()
        self.proj = nn.Linear(dim, n_views, bias=False)
        self.n_views = n_views

    def forward(self, tok, grid):
        B, N, D = tok.shape
        Gh, Gw = grid
        V = self.proj(tok)  # (B,N,V)
        return V.transpose(1, 2).reshape(B, self.n_views, Gh, Gw)


class Kernels3(nn.Module):
    """Learnable 3x3 spatial kernels."""

    def __init__(self, in_ch, n_kernels=3):
        super().__init__()
        self.k = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, n_kernels, kernel_size=1, bias=False),
        )

    def forward(self, maps):
        return self.k(maps)


class FuseExcInh(nn.Module):
    """Excitatory/Inhibitory fusion layer."""

    def __init__(self, in_ch):
        super().__init__()
        hid = max(8, in_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hid, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hid, 2, kernel_size=1, bias=True),
        )
        self.alpha_pos = nn.Parameter(torch.tensor(0.8))
        self.alpha_neg = nn.Parameter(torch.tensor(0.8))

    def forward(self, x):
        G = self.fuse(x)  # (B,2,H,W)
        G_pos, G_neg = torch.sigmoid(G[:, :1]), torch.sigmoid(G[:, 1:])
        a_pos, a_neg = F.softplus(self.alpha_pos), F.softplus(self.alpha_neg)
        return G_pos, G_neg, a_pos, a_neg
