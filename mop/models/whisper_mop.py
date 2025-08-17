"""
Whisper-MoP: Mixture of Products for Audio Transformers (Encoder-Decoder)
- Encoder: non-causal SA over audio frames; MoP gate applied after SA, before MLP
- Decoder: causal SA over text; cross-attention to encoder memory
Author: Eran Ben Artzy
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- Config ---------------------------

@dataclass
class WhisperConfig:
    # Audio
    n_mels: int = 80
    n_audio_ctx: int = 1500   # max audio frames (time steps)

    # Text
    vocab_size: int = 51865
    n_text_ctx: int = 448     # max text tokens

    # Transformer dims
    n_embd: int = 1024
    n_head: int = 16
    n_layer_enc: int = 12
    n_layer_dec: int = 12
    dropout: float = 0.0
    bias: bool = False
    use_abs_pos_emb: bool = True

    # MoP (used in encoder only)
    n_views: int = 5
    n_kernels: int = 3
    kernel_size: int = 5


# --------------------------- MoP (2D over mel) ---------------------------

class ViewsConv2D(nn.Module):
    """1x1 conv to generate V 'views' from a single-channel mel map (B,1,T,F)."""
    def __init__(self, n_views: int):
        super().__init__()
        self.conv = nn.Conv2d(1, n_views, kernel_size=1, bias=False)

    def forward(self, mel2d: torch.Tensor) -> torch.Tensor:
        # mel2d: (B, 1, T, F) -> (B, V, T, F)
        return self.conv(mel2d)


class Kernels2D(nn.Module):
    """Depthwise-ish conv over (T,F) to produce K pattern maps from V views."""
    def __init__(self, in_ch: int, n_kernels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, n_kernels, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, T, F) -> (B, K, T, F)
        return self.conv(x)


class FuseExcInh2D(nn.Module):
    """Produce excitatory & inhibitory fields from concatenated [views|kernels]."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2, kernel_size=1, bias=False)
        # learnable global scalars (alpha_pos, alpha_neg)
        self.alpha = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, V+K, T, F)
        gates = self.conv(x)                       # (B, 2, T, F)
        g_pos, g_neg = gates.chunk(2, dim=1)       # (B,1,T,F) each
        a_pos, a_neg = self.alpha[0], self.alpha[1]
        return g_pos, g_neg, a_pos, a_neg


class MoP2D(nn.Module):
    """
    Full MoP block over mel spectrograms.
    Returns a per-time-step scalar gate g(t) used to modulate the encoder stream.
    """
    def __init__(self, n_views: int, n_kernels: int, kernel_size: int):
        super().__init__()
        self.views = ViewsConv2D(n_views)
        self.kernels = Kernels2D(n_views, n_kernels, kernel_size)
        self.fuse = FuseExcInh2D(n_views + n_kernels)

    def forward(self, mel2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mel2d: (B, 1, T, F)
        returns:
           gate_t: (B, T, 1)  # scalar per time step
           V: (B, V, T, F)
           K: (B, K, T, F)
        """
        V = self.views(mel2d)              # (B,V,T,F)
        K = self.kernels(V)                # (B,K,T,F)
        maps = torch.cat([V, K], dim=1)    # (B,V+K,T,F)

        g_pos, g_neg, a_pos, a_neg = self.fuse(maps)  # (B,1,T,F)
        # average across frequency to get a per-time factor
        g_pos_t = g_pos.mean(dim=3, keepdim=False)    # (B,1,T)
        g_neg_t = g_neg.mean(dim=3, keepdim=False)    # (B,1,T)

        gate_t = 1 + a_pos * g_pos_t - a_neg * g_neg_t   # (B,1,T)
        gate_t = gate_t.transpose(1, 2)                  # -> (B,T,1)
        return gate_t, V, K


# --------------------------- Attention blocks ---------------------------

def _sa_mask(T_q: int, T_k: int, device) -> torch.Tensor:
    """Causal mask for decoder self-attention: (1,1,T_q,T_k)."""
    return torch.tril(torch.ones(T_q, T_k, device=device, dtype=torch.bool)).view(1, 1, T_q, T_k)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float, bias: bool, causal: bool):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        B, T, D = x.shape
        H, Dh = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)        # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)        # (B,H,T,Dh)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)        # (B,H,T,Dh)

        att = (q @ k.transpose(-2, -1)) * self.scale                # (B,H,T,T)

        if self.causal:
            mask = _sa_mask(T, T, x.device)                         # (1,1,T,T)
            att = att.masked_fill(~mask, float("-inf"))

        if attn_bias is not None:
            att = att + attn_bias  # broadcast if provided

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                                  # (B,H,T,Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)             # (B,T,D)
        return self.resid_drop(self.o_proj(y))


class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, n_head: int, dropout: float, bias: bool):
        super().__init__()
        assert dim_q % n_head == 0
        self.n_head = n_head
        self.head_dim = dim_q // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim_q, dim_q, bias=bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=bias)
        self.o_proj = nn.Linear(dim_q, dim_q, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x_q: (B, Tq, Dq), x_kv: (B, Tk, Dk)
        B, Tq, Dq = x_q.shape
        Tk = x_kv.shape[1]
        H, Dh = self.n_head, self.head_dim

        q = self.q_proj(x_q).view(B, Tq, H, Dh).transpose(1, 2)     # (B,H,Tq,Dh)
        k = self.k_proj(x_kv).view(B, Tk, H, Dh).transpose(1, 2)    # (B,H,Tk,Dh)
        v = self.v_proj(x_kv).view(B, Tk, H, Dh).transpose(1, 2)    # (B,H,Tk,Dh)

        att = (q @ k.transpose(-2, -1)) * self.scale                # (B,H,Tq,Tk)
        if attn_mask is not None:
            att = att + attn_mask

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                                 # (B,H,Tq,Dh)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Dq)          # (B,Tq,Dq)
        return self.resid_drop(self.o_proj(y))


class MLP(nn.Module):
    def __init__(self, dim: int, dropout: float, bias: bool):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim, bias=bias)
        self.proj = nn.Linear(4 * dim, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        return self.drop(x)


# --------------------------- Encoder/Decoder blocks ---------------------------

class EncoderBlock(nn.Module):
    """Non-causal SA + MoP gate (from mel2d) + MLP."""
    def __init__(self, cfg: WhisperConfig):
        super().__init__()
        D = cfg.n_embd
        self.ln1 = nn.LayerNorm(D)
        self.attn = MultiheadSelfAttention(D, cfg.n_head, cfg.dropout, cfg.bias, causal=False)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = MLP(D, cfg.dropout, cfg.bias)

        self.mop = MoP2D(cfg.n_views, cfg.n_kernels, cfg.kernel_size)

    def forward(self, x: torch.Tensor, mel2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,D), mel2d: (B,1,T,F)
        x = x + self.attn(self.ln1(x))                 # encoder SA
        gate_t, V, K = self.mop(mel2d)                 # (B,T,1)
        x = x * gate_t                                 # modulate stream per time step
        x = x + self.mlp(self.ln2(x))
        return x, gate_t.squeeze(-1)                   # also return gate per time for analysis


class DecoderBlock(nn.Module):
    """Causal SA + cross-attn to encoder + MLP."""
    def __init__(self, cfg: WhisperConfig):
        super().__init__()
        D = cfg.n_embd
        self.ln1 = nn.LayerNorm(D)
        self.self_attn = MultiheadSelfAttention(D, cfg.n_head, cfg.dropout, cfg.bias, causal=True)

        self.ln2 = nn.LayerNorm(D)
        self.cross_attn = MultiheadCrossAttention(D, D, cfg.n_head, cfg.dropout, cfg.bias)

        self.ln3 = nn.LayerNorm(D)
        self.mlp = MLP(D, cfg.dropout, cfg.bias)

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), enc)
        x = x + self.mlp(self.ln3(x))
        return x


# --------------------------- Whisper-MoP Model ---------------------------

class WhisperMoP(nn.Module):
    """
    Encoder-decoder with MoP gating in the encoder.
    forward(mel, dec_input_ids, targets=None) -> (logits, loss)
    """
    def __init__(self, cfg: WhisperConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.n_embd

        # Encoder input projection: mel -> embedding
        self.audio_proj = nn.Linear(cfg.n_mels, D, bias=cfg.bias)
        self.audio_pos = nn.Embedding(cfg.n_audio_ctx, D) if cfg.use_abs_pos_emb else None

        # Decoder embeddings
        self.wte = nn.Embedding(cfg.vocab_size, D)
        self.text_pos = nn.Embedding(cfg.n_text_ctx, D) if cfg.use_abs_pos_emb else None

        self.drop = nn.Dropout(cfg.dropout)

        # Stacks
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layer_enc)])
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layer_dec)])

        # Final LN + LM head (tied)
        self.enc_ln_f = nn.LayerNorm(D)
        self.dec_ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    # ----- init -----
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ----- forward -----
    @torch.no_grad()
    def _pos(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)  # (1,T)

    def encode(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mel: (B, T_audio, n_mels)
        returns: enc_out (B, T_audio, D), gate_times (B, L_enc, T_audio)
        """
        B, T_a, F = mel.shape
        assert F == self.cfg.n_mels, "mel dim mismatch"

        x = self.audio_proj(mel)                                 # (B,T_a,D)
        if self.audio_pos is not None:
            pos = self._pos(T_a, mel.device)
            x = x + self.audio_pos(pos)                          # (B,T_a,D)
        x = self.drop(x)

        mel2d = mel.transpose(1, 2).unsqueeze(1)                 # (B,1,F,T) -> want (B,1,T,F)
        mel2d = mel2d.transpose(2, 3).contiguous()               # (B,1,T,F)

        gate_layers = []
        for blk in self.encoder:
            x, gate_t = blk(x, mel2d)                            # gate_t: (B,T)
            gate_layers.append(gate_t)
        x = self.enc_ln_f(x)

        gates = torch.stack(gate_layers, dim=1)                  # (B, L_enc, T)
        return x, gates

    def decode(self, enc_out: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """
        enc_out: (B, T_audio, D)
        dec_input_ids: (B, T_text)
        returns: logits (B, T_text, vocab)
        """
        B, T_t = dec_input_ids.shape
        x = self.wte(dec_input_ids)                              # (B,T_t,D)
        if self.text_pos is not None:
            pos = self._pos(T_t, dec_input_ids.device)
            x = x + self.text_pos(pos)
        x = self.drop(x)

        for blk in self.decoder:
            x = blk(x, enc_out)

        x = self.dec_ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(
        self,
        mel: torch.Tensor,             # (B, T_audio, n_mels)
        dec_input_ids: torch.Tensor,   # (B, T_text)
        targets: Optional[torch.Tensor] = None,  # (B, T_text)
    ):
        enc_out, gates = self.encode(mel)
        logits = self.decode(enc_out, dec_input_ids)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, gates  # gates: (B, L_enc, T_audio)

    # ----- analysis -----
    @torch.no_grad()
    def get_gate_maps(self, mel: torch.Tensor):
        """
        Run only the encoder and return per-layer time gates.
        """
        _, gates = self.encode(mel)
        return gates  # (B, L_enc, T_audio)


# --------------------------- Factories ---------------------------

def create_whisper_mop(cfg: WhisperConfig) -> WhisperMoP:
    return WhisperMoP(cfg)


def create_whisper_baseline(cfg: WhisperConfig) -> WhisperMoP:
    # Same architecture with MoP weights zeroed: set alpha -> 0 so gate == 1
    model = WhisperMoP(cfg)
    with torch.no_grad():
        for blk in model.encoder:
            blk.mop.fuse.alpha.zero_()  # gate = 1 + 0*pos - 0*neg => identity
    return model
