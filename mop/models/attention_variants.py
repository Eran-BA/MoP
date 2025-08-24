"""
Unified multi-head attention variants (A/B/C/D/E):

- A: Baseline MSA (standard scaled dot-product attention)
- B: MoP-compatible (uses baseline attention here; MoP gating is applied outside attention)
- C: Cross-View Mixer attention (2-view binding + transpose cues + optional per-key prior)
- D: Multi-Hop dual-path attention with value-aware transport
- E: Edgewise-gated attention with per-edge gates from a small conv head; supports shared-QKV and multi-views

These classes are architecture-agnostic and can be used inside ViT-like blocks.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMSA(nn.Module):
    def __init__(
        self, dim: int, heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        y = attn @ v
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(y))


class CrossViewMixerMSA(nn.Module):
    """MSA with cross-view binding, 2x2 mixing, transpose cues, and optional per-key prior sharpening."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_transpose_cues: bool = True,
        t1: float = 0.0,
        t2: float = 0.0,
        enable_per_key_prior: bool = False,
        prior_weight: float = 0.5,
        anchor_mode: str = "argmax_row_sum",
        fixed_k_star: int = 0,
    ):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.qkv1 = nn.Linear(dim, dim * 3, bias=False)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        # 2x2 mixing matrix M initialized to identity
        self.mix = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        self.use_transpose_cues = bool(use_transpose_cues)
        self.t1 = float(t1)
        self.t2 = float(t2)

        self.enable_per_key_prior = bool(enable_per_key_prior)
        self.prior_weight = float(prior_weight)
        self.anchor_mode = str(anchor_mode)
        self.fixed_k_star = int(fixed_k_star)

    def _compute_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        scale = 1.0 / math.sqrt(self.dk)
        qkv1 = self.qkv1(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        qkv2 = self.qkv2(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q2, k2, _v2 = qkv2[0], qkv2[1], qkv2[2]
        S1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        S2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        S12 = torch.matmul(q1, k2.transpose(-2, -1)) * scale
        S21 = torch.matmul(q2, k1.transpose(-2, -1)) * scale
        m11, m12 = self.mix[0, 0], self.mix[0, 1]
        m21, m22 = self.mix[1, 0], self.mix[1, 1]
        S = m11 * S1 + m12 * S12 + m21 * S21 + m22 * S2
        if self.use_transpose_cues:
            if self.t1 != 0.0:
                S = S + self.t1 * S1.transpose(-2, -1)
            if self.t2 != 0.0:
                S = S + self.t2 * S2.transpose(-2, -1)
        return S, S1, S2, v1

    def _apply_mask(
        self, S: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return S
        return S.masked_fill((mask == 0), float("-inf"))

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        S, S1, S2, v1 = self._compute_logits(x)
        S_masked = self._apply_mask(S, attn_mask)
        A_mix = F.softmax(S_masked, dim=-1)
        if self.enable_per_key_prior and self.prior_weight > 0.0:
            S1m = self._apply_mask(S1, attn_mask)
            S2m = self._apply_mask(S2, attn_mask)
            A1 = F.softmax(S1m, dim=-1)
            A2 = F.softmax(S2m, dim=-1)
            if self.anchor_mode == "fixed":
                B, H, N, _ = A1.shape
                k_star = torch.full(
                    (B, H),
                    max(0, min(N - 1, self.fixed_k_star)),
                    device=x.device,
                    dtype=torch.long,
                )
            elif self.anchor_mode == "argmax_row_sum":
                row_sum = A2.sum(dim=-1)
                k_star = row_sum.argmax(dim=-1)  # [B,H]
            else:
                B, H, _N, _ = A1.shape
                k_star = torch.zeros((B, H), device=x.device, dtype=torch.long)
            B, H, N, _ = A2.shape
            k_idx = k_star.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, N)
            A2_anchor = torch.gather(A2, 2, k_idx)  # [B,H,1,N]
            A_sharp = A1 * A2_anchor.expand(-1, -1, N, -1)
            A_sharp = A_sharp / (A_sharp.sum(dim=-1, keepdim=True) + 1e-9)
            A = (1.0 - self.prior_weight) * A_mix + self.prior_weight * A_sharp
        else:
            A = A_mix
        A = self.attn_drop(A)
        y = torch.matmul(A, v1)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.proj_drop(self.proj(y))


def _lse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(torch.stack([a, b], dim=0), dim=0)


class MultiHopMSA(nn.Module):
    """Dual-path logits with gated multi-hop composition; value transport follows the chain."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_not: float = 0.5,
        gates: Optional[Dict[str, float]] = None,
        hops: int = 3,
    ):
        super().__init__()
        assert dim % heads == 0
        assert hops >= 2
        self.h = heads
        self.dk = dim // heads
        self.hops = int(hops)
        self.qkv1 = nn.Linear(dim, dim * 3, bias=False)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.beta_not = float(beta_not)
        self.gates = gates or dict(and_=1.0, or_=0.0, not_=0.0, chain=0.0, base=1.0)
        self.chain_value_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        scale = 1.0 / math.sqrt(self.dk)
        qkv1 = self.qkv1(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv2(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        S1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        S2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        if attn_mask is not None:
            m = attn_mask == 0
            S1 = S1.masked_fill(m, float("-inf"))
            S2 = S2.masked_fill(m, float("-inf"))
        A1 = F.softmax(S1, dim=-1)
        A2 = F.softmax(S2, dim=-1)
        # Mix logits with gates
        Smix = S1.clone()
        Smix = Smix + self.gates.get("and_", 1.0) * (S2)
        Smix = Smix + self.gates.get("or_", 0.0) * (_lse(S1, S2) - S1)
        Smix = Smix - self.gates.get("not_", 0.0) * (self.beta_not * S2)
        # Multi-hop composition forward chain C_fwd = A1 @ A2 @ ...
        C_fwd = A1 @ A2
        for _ in range(max(0, self.hops - 2)):
            C_fwd = C_fwd @ A2
        eps = 1e-6
        Smix = Smix + self.gates.get("chain", 0.0) * torch.log(C_fwd + eps)
        if attn_mask is not None:
            Smix = Smix.masked_fill(attn_mask == 0, float("-inf"))
        A = F.softmax(Smix, dim=-1)
        A = self.attn_drop(A)
        # Value transport along chain
        transport = v2
        for _ in range(max(0, self.hops - 1)):
            transport = A2 @ transport
        y_chain = A1 @ transport
        w = torch.sigmoid(self.chain_value_logit)
        y = (A @ v1) + w * y_chain
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(y))


class EdgewiseGateHead(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int = 16,
        use_k3: bool = False,
        gate_mode: str = "dense",
        gate_rank: int = 4,
        gate_init: str = "neutral",
    ):
        super().__init__()
        self.use_k3 = bool(use_k3)
        self.gate_mode = str(gate_mode)
        self.gate_rank = int(gate_rank)
        self.gate_init = str(gate_init)

        if self.gate_mode == "dense":
            self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True)
            self.act = nn.GELU(approximate="tanh")
            if self.use_k3:
                self.mid3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True)
            self.conv2 = nn.Conv2d(hidden, 4, kernel_size=1, bias=True)
            # Bias preset
            nn.init.constant_(self.conv2.bias, -5.0)
            with torch.no_grad():
                if self.gate_init == "and":
                    self.conv2.bias[0] = 2.0  # g_and high
                elif self.gate_init == "or":
                    self.conv2.bias[1] = 2.0  # g_or high
                elif self.gate_init == "not":
                    self.conv2.bias[2] = 2.0  # g_not high
                elif self.gate_init == "nor":
                    # favor inhibition; keep OR low (already -5)
                    self.conv2.bias[2] = 2.0
                elif self.gate_init == "xor":
                    # favor OR while suppressing AND (AND already low)
                    self.conv2.bias[1] = 2.0
                elif self.gate_init == "chain":
                    self.conv2.bias[3] = 2.0  # g_chain high
        else:
            # Low-rank gate: produce row/col factors per gate using Conv1d over channels
            # Inputs to this head should be provided as feat [B*H, C, N, N]
            # We compute row factors a and col factors b from row/col pooled features
            self.row_proj = nn.Conv1d(in_ch, 4 * self.gate_rank, kernel_size=1, bias=True)
            self.col_proj = nn.Conv1d(in_ch, 4 * self.gate_rank, kernel_size=1, bias=True)
            # Initialize biases to favor presets
            with torch.no_grad():
                # Start close to zero gates
                nn.init.constant_(self.row_proj.bias, 0.0)
                nn.init.constant_(self.col_proj.bias, 0.0)
                # choose channel index: 0=and,1=or,2=not,3=chain
                idx_map = {"and": 0, "or": 1, "not": 2, "chain": 3}
                if self.gate_init in idx_map:
                    idx = idx_map[self.gate_init]
                    c = float(max(0.0, (2.0 / max(1, self.gate_rank)) ** 0.5))
                    s, e = idx * self.gate_rank, (idx + 1) * self.gate_rank
                    self.row_proj.bias[s:e] = c
                    self.col_proj.bias[s:e] = c
                elif self.gate_init in ("nor", "xor"):
                    # For NOR/XOR, bias towards NOT or OR respectively
                    if self.gate_init == "nor":
                        idx = 2  # not
                    else:
                        idx = 1  # or
                    c = float(max(0.0, (2.0 / max(1, self.gate_rank)) ** 0.5))
                    s, e = idx * self.gate_rank, (idx + 1) * self.gate_rank
                    self.row_proj.bias[s:e] = c
                    self.col_proj.bias[s:e] = c
                elif self.gate_init == "mix5":
                    # Initialize a mixture of five presets: and, or, not, nor, xor
                    c = float(max(0.0, (2.0 / max(1, self.gate_rank)) ** 0.5))
                    for idx in [0, 1, 2]:  # and(0), or(1), not(2)
                        s, e = idx * self.gate_rank, (idx + 1) * self.gate_rank
                        self.row_proj.bias[s:e] = c
                        self.col_proj.bias[s:e] = c
                    # nor ~ not(2) already covered; xor ~ or(1) already covered

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "dense":
            x = self.conv1(feat)
            x = self.act(x)
            if self.use_k3:
                x = self.mid3(self.act(x))
            x = self.conv2(x)
            return torch.sigmoid(x)
        # Low-rank: build gates from row/col factors
        # feat: [B*H, C, N, N]
        BtH, C, N, _ = feat.shape
        # row/col pooled features
        row_feat = feat.mean(dim=3)  # [B*H, C, N]
        col_feat = feat.mean(dim=2)  # [B*H, C, N]
        a = self.row_proj(row_feat)  # [B*H, 4*r, N]
        b = self.col_proj(col_feat)  # [B*H, 4*r, N]
        a = a.view(BtH, 4, self.gate_rank, N)
        b = b.view(BtH, 4, self.gate_rank, N)
        # Outer-product sum over rank: G[b,c,i,j] = sum_k a[b,c,k,i]*b[b,c,k,j]
        G = torch.einsum("bcrn,bcrm->bcnm", a, b)
        return torch.sigmoid(G)


class EdgewiseMSA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_not: float = 0.5,
        use_k3: bool = False,
        n_views: int = 2,
        share_qkv: bool = False,
        gate_mode: str = "dense",
        gate_rank: int = 4,
        gate_init: str = "neutral",
        use_lens_bank: bool = False,
        lens_kernel_size: int = 3,
        lens_dilations: Optional[Tuple[int, ...]] = None,
        # Q/K lens bank (preferred for lensing):
        use_lens_bank_qk: bool = False,
        lens_qk_kernel_size: int = 3,
        lens_qk_dilations: Optional[Tuple[int, ...]] = None,
        lens_qk_causal: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.beta_not = beta_not
        self.n_views = max(2, int(n_views))
        self.share_qkv = bool(share_qkv)
        self.use_lens_bank = bool(use_lens_bank)
        self.lens_kernel_size = int(lens_kernel_size)
        self.lens_dilations = tuple(lens_dilations) if lens_dilations is not None else (1, 2)
        # Q/K lens bank
        self.use_lens_bank_qk = bool(use_lens_bank_qk)
        self.lens_qk_kernel_size = int(lens_qk_kernel_size)
        self.lens_qk_dilations = (
            tuple(lens_qk_dilations) if lens_qk_dilations is not None else (1, 2)
        )
        self.lens_qk_causal = bool(lens_qk_causal)
        if self.share_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_scale = nn.Parameter(torch.ones(self.n_views, self.h, 1, self.dk))
            self.k_scale = nn.Parameter(torch.ones(self.n_views, self.h, 1, self.dk))
            self.v_scale = nn.Parameter(torch.ones(self.n_views, self.h, 1, self.dk))
        else:
            self.qkv_list = nn.ModuleList(
                [nn.Linear(dim, dim * 3, bias=False) for _ in range(self.n_views)]
            )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        in_ch = 2 * self.n_views + 2
        if self.use_lens_bank_qk and not self.share_qkv:
            raise ValueError("use_lens_bank_qk=True requires share_qkv=True for now")
        if self.use_lens_bank_qk:
            # Depthwise over sequence length for Q and K per head/channel
            # Conv1d over [B*H, dk, N]
            pad_same = [d * (self.lens_qk_kernel_size - 1) // 2 for d in self.lens_qk_dilations]
            self.q_lens = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.dk,
                        out_channels=self.dk,
                        kernel_size=self.lens_qk_kernel_size,
                        padding=0 if self.lens_qk_causal else pad_same[i],
                        dilation=d,
                        groups=self.dk,
                        bias=False,
                    )
                    for i, d in enumerate(self.lens_qk_dilations)
                ]
            )
            self.k_lens = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=self.dk,
                        out_channels=self.dk,
                        kernel_size=self.lens_qk_kernel_size,
                        padding=0 if self.lens_qk_causal else pad_same[i],
                        dilation=d,
                        groups=self.dk,
                        bias=False,
                    )
                    for i, d in enumerate(self.lens_qk_dilations)
                ]
            )
            # When using Q/K lens bank, number of S views follows #dilations
            self._lens_qk_num = len(self.lens_qk_dilations)
        # Optional depthwise lens bank over S_i channels with multi-scale dilations
        if self.use_lens_bank:
            self.lens_bank = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.n_views,
                        out_channels=self.n_views,
                        kernel_size=self.lens_kernel_size,
                        padding=d,
                        dilation=d,
                        groups=self.n_views,
                        bias=False,
                    )
                    for d in self.lens_dilations
                ]
            )
            in_ch = in_ch + self.n_views * len(self.lens_dilations)
        self.edge_head = EdgewiseGateHead(
            in_ch=in_ch,
            hidden=16,
            use_k3=use_k3,
            gate_mode=gate_mode,
            gate_rank=gate_rank,
            gate_init=gate_init,
        )
        self.chain_value_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        qs, ks, vs = [], [], []
        if self.share_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
            q_base, k_base, v_base = qkv[0], qkv[1], qkv[2]
            for i in range(self.n_views):
                qs.append(q_base * self.q_scale[i])
                ks.append(k_base * self.k_scale[i])
                vs.append(v_base * self.v_scale[i])
        else:
            for lin in self.qkv_list:
                qkv = lin(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
                qs.append(qkv[0])
                ks.append(qkv[1])
                vs.append(qkv[2])
        scale = 1.0 / math.sqrt(self.dk)
        if self.use_lens_bank_qk:
            # Build lensed Q/K for each dilation
            q_base = qs[0]
            k_base = ks[0]
            Bq, Hq, Nq, Dq = q_base.shape
            q_flat = q_base.reshape(Bq * Hq, Dq, Nq)
            k_flat = k_base.reshape(Bq * Hq, Dq, Nq)
            q_l_list = []
            k_l_list = []
            for i, (qconv, kconv) in enumerate(zip(self.q_lens, self.k_lens)):
                if self.lens_qk_causal:
                    left = (self.lens_qk_kernel_size - 1) * self.lens_qk_dilations[i]
                    q_in = torch.nn.functional.pad(q_flat, (left, 0))
                    k_in = torch.nn.functional.pad(k_flat, (left, 0))
                else:
                    q_in = q_flat
                    k_in = k_flat
                q_l = qconv(q_in)
                k_l = kconv(k_in)
                q_l = q_l.view(Bq, Hq, Dq, Nq).transpose(2, 3)  # -> [B,H,N,D]
                k_l = k_l.view(Bq, Hq, Dq, Nq).transpose(2, 3)
                q_l_list.append(q_l)
                k_l_list.append(k_l)
            S_list = [
                torch.matmul(q_l_list[i], k_l_list[i].transpose(-2, -1)) * scale
                for i in range(self._lens_qk_num)
            ]
        else:
            S_list = [
                torch.matmul(qs[i], ks[i].transpose(-2, -1)) * scale
                for i in range(self.n_views)
            ]
        if attn_mask is not None:
            m = attn_mask == 0
            S_list = [S.masked_fill(m, float("-inf")) for S in S_list]
        A_list = [F.softmax(S, dim=-1) for S in S_list]
        C_fwd = A_list[0]
        for i in range(1, self.n_views):
            C_fwd = torch.matmul(C_fwd, A_list[i])
        C_bwd = A_list[-1]
        for i in range(self.n_views - 2, -1, -1):
            C_bwd = torch.matmul(C_bwd, A_list[i])
        eps = 1e-6
        BtH = B * self.h
        S_imgs = [S.view(BtH, N, N) for S in S_list]
        ST_imgs = [img.transpose(1, 2) for img in S_imgs]
        Cr_img = torch.log(C_fwd + eps).view(BtH, N, N)
        Cl_img = torch.log(C_bwd + eps).view(BtH, N, N)
        feat_list = S_imgs + ST_imgs + [Cr_img, Cl_img]
        if self.use_lens_bank:
            # Stack S_i as channels [B*H, V, N, N] and apply depthwise conv per dilation
            S_stack = torch.stack(S_imgs, dim=1)  # [B*H, V, N, N]
            lens_feats = []
            for conv in self.lens_bank:
                lens_feats.append(conv(S_stack))  # [B*H, V, N, N]
            if lens_feats:
                lens_cat = torch.cat(lens_feats, dim=1)  # [B*H, V*L, N, N]
                # Split back into list of [B*H, N, N] per channel
                lens_list = [lens_cat[:, i] for i in range(lens_cat.shape[1])]
                feat_list = feat_list + lens_list
        feat = torch.stack(feat_list, dim=1)
        gates = self.edge_head(feat)
        g_and, g_or, g_not, g_chain = gates[:, 0], gates[:, 1], gates[:, 2], gates[:, 3]
        S1_img = S_imgs[0]
        S_sum = S1_img
        for i in range(1, self.n_views):
            S_sum = S_sum + S_imgs[i]
        lse_all = torch.logsumexp(torch.stack(S_imgs, dim=1), dim=1)
        S_mean_others = (S_sum - S1_img) / max(1, self.n_views - 1)
        Smix = S1_img
        Smix = Smix + g_and * (S_sum - S1_img)
        Smix = Smix + g_or * (lse_all - S1_img)
        Smix = Smix - g_not * (self.beta_not * S_mean_others)
        Smix = Smix + g_chain * Cr_img
        Smix = Smix.view(B, self.h, N, N)
        if attn_mask is not None:
            Smix = Smix.masked_fill(attn_mask == 0, float("-inf"))
        A = F.softmax(Smix, dim=-1)
        A = self.attn_drop(A)
        v1 = vs[0]
        y_base = torch.matmul(A, v1)
        transport = vs[-1]
        for i in range(self.n_views - 1, 0, -1):
            transport = torch.matmul(A_list[i], transport)
        y_chain = torch.matmul(A_list[0], transport)
        w = torch.sigmoid(self.chain_value_logit)
        y = y_base + w * y_chain
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(y))


class UnifiedMSA(nn.Module):
    """Switchable attention wrapper for modes A/B/C/D/E.

    - mode 'A': Baseline MSA
    - mode 'B': Baseline MSA (MoP applies gating outside attention)
    - mode 'C': CrossViewMixerMSA
    - mode 'D': MultiHopMSA
    - mode 'E': EdgewiseMSA
    """

    def __init__(self, mode: str, dim: int, heads: int = 4, **kwargs):
        super().__init__()
        mode = str(mode).upper()
        self.mode = mode
        if mode in ("A", "B"):
            self.impl = BaselineMSA(
                dim, heads, kwargs.get("attn_drop", 0.0), kwargs.get("proj_drop", 0.0)
            )
        elif mode == "C":
            self.impl = CrossViewMixerMSA(
                dim,
                heads,
                kwargs.get("attn_drop", 0.0),
                kwargs.get("proj_drop", 0.0),
                use_transpose_cues=kwargs.get("use_transpose_cues", True),
                t1=kwargs.get("t1", 0.0),
                t2=kwargs.get("t2", 0.0),
                enable_per_key_prior=kwargs.get("enable_per_key_prior", False),
                prior_weight=kwargs.get("prior_weight", 0.5),
                anchor_mode=kwargs.get("anchor_mode", "argmax_row_sum"),
                fixed_k_star=kwargs.get("fixed_k_star", 0),
            )
        elif mode == "D":
            self.impl = MultiHopMSA(
                dim,
                heads,
                kwargs.get("attn_drop", 0.0),
                kwargs.get("proj_drop", 0.0),
                beta_not=kwargs.get("beta_not", 0.5),
                gates=kwargs.get("gates", None),
                hops=kwargs.get("hops", 3),
            )
        elif mode == "E":
            self.impl = EdgewiseMSA(
                dim,
                heads,
                kwargs.get("attn_drop", 0.0),
                kwargs.get("proj_drop", 0.0),
                beta_not=kwargs.get("beta_not", 0.5),
                use_k3=kwargs.get("use_k3", False),
                n_views=kwargs.get("n_views", 2),
                share_qkv=kwargs.get("share_qkv", False),
                gate_mode=kwargs.get("gate_mode", "dense"),
                gate_rank=kwargs.get("gate_rank", 4),
                gate_init=kwargs.get("gate_init", "neutral"),
            )
        else:
            raise ValueError(f"Unknown attention mode: {mode}")

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.impl(x, attn_mask)
