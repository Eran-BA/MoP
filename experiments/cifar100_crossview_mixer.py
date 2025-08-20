#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Ensure project root on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from mop.models import DropPath, PatchEmbed
except Exception:
    PatchEmbed = None
    DropPath = None


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossViewMixerMSA(nn.Module):
    """MSA with cross-view binding, 2x2 mixing, transpose cues, and per-key prior sharpening."""

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
        B, N, D = x.shape
        scale = 1.0 / math.sqrt(self.dk)

        qkv1 = self.qkv1(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.qkv2(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q2, k2, _v2 = qkv2[0], qkv2[1], qkv2[2]

        S1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # [B,H,N,N]
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
                row_sum = A2.sum(dim=-1)  # [B,H,N]
                k_star = row_sum.argmax(dim=-1)  # [B,H]
            else:
                B, H, N, _ = A1.shape
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


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class BlockXView(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_transpose_cues: bool = True,
        t1: float = 0.0,
        t2: float = 0.0,
        enable_per_key_prior: bool = False,
        prior_weight: float = 0.5,
        anchor_mode: str = "argmax_row_sum",
        fixed_k_star: int = 0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CrossViewMixerMSA(
            dim,
            heads,
            attn_drop,
            drop,
            use_transpose_cues=use_transpose_cues,
            t1=t1,
            t2=t2,
            enable_per_key_prior=enable_per_key_prior,
            prior_weight=prior_weight,
            anchor_mode=anchor_mode,
            fixed_k_star=fixed_k_star,
        )
        self.dp1 = DropPath(drop_path) if DropPath is not None else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.dp2 = DropPath(drop_path) if DropPath is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.ln1(x)))
        x = x + self.dp2(self.mlp(self.ln2(x)))
        return x


class ViTCrossView(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        n_classes: int = 100,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        patch: int = 4,
        num_tokens: int = 64,
        use_transpose_cues: bool = True,
        t1: float = 0.0,
        t2: float = 0.0,
        enable_per_key_prior: bool = False,
        prior_weight: float = 0.5,
        anchor_mode: str = "argmax_row_sum",
        fixed_k_star: int = 0,
    ):
        super().__init__()
        if PatchEmbed is None:
            self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)
            self._use_builtin_patch = True
        else:
            self.patch = PatchEmbed(in_ch=3, dim=dim, patch=patch)
            self._use_builtin_patch = False
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))

        dps = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                BlockXView(
                    dim,
                    heads,
                    mlp_ratio,
                    drop,
                    0.0,
                    dps[i],
                    use_transpose_cues=use_transpose_cues,
                    t1=t1,
                    t2=t2,
                    enable_per_key_prior=enable_per_key_prior,
                    prior_weight=prior_weight,
                    anchor_mode=anchor_mode,
                    fixed_k_star=fixed_k_star,
                )
                for i in range(depth)
            ]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes, bias=False)

        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_use_builtin_patch", False):
            x = self.patch(x)
            B, D, Gh, Gw = x.shape
            tok = x.flatten(2).transpose(1, 2)
        else:
            tok, (Gh, Gw) = self.patch(x)
        tok = tok + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln_f(tok)
        cls = tok.mean(dim=1)
        return self.head(cls)


def get_loaders(
    batch: int = 256, tiny: bool = False, workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    tfm_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train = datasets.CIFAR100("./data", train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR100("./data", train=False, download=True, transform=tfm_test)
    if tiny:
        train = torch.utils.data.Subset(train, range(5000))
        test = torch.utils.data.Subset(test, range(1000))
    train_loader = DataLoader(
        train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=False
    )
    test_loader = DataLoader(
        test, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="CIFAR-100 cross-view mixer with per-key prior sharpening and transpose cues"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--tiny", action="store_true")
    # Model capacity
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop_path", type=float, default=0.1)
    # Mixer knobs
    ap.add_argument("--use_transpose_cues", action="store_true")
    ap.add_argument("--t1", type=float, default=0.0)
    ap.add_argument("--t2", type=float, default=0.0)
    # Per-key prior
    ap.add_argument("--enable_prior", action="store_true")
    ap.add_argument("--prior_weight", type=float, default=0.5)
    ap.add_argument(
        "--anchor_mode",
        type=str,
        choices=["argmax_row_sum", "fixed", "none"],
        default="argmax_row_sum",
    )
    ap.add_argument("--k_star", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/cifar100_crossview_mixer")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m: nn.Module):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    accs = []
    for s in args.seeds:
        print(f"\n🔬 Running seed {s}")
        set_seed(s)

        model = ViTCrossView(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            n_classes=100,
            drop_path=args.drop_path,
            use_transpose_cues=args.use_transpose_cues,
            t1=args.t1,
            t2=args.t2,
            enable_per_key_prior=args.enable_prior,
            prior_weight=args.prior_weight,
            anchor_mode=args.anchor_mode,
            fixed_k_star=args.k_star,
        ).to(device)

        print(
            f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )
        opt, sch = make_opt(model)
        steps = 0
        model.train()
        it = iter(train_loader)

        while steps < args.steps:
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(train_loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            sch.step()

            steps += 1
            if steps % max(args.eval_every, 1) == 0 or steps == 1:
                acc = evaluate(model, val_loader, device)
                print(f"step {steps:4d} | loss={loss.item():.3f} | acc={acc:.3f}")

        final_acc = evaluate(model, val_loader, device)
        accs.append(final_acc)
        print(f"seed {s}: acc={final_acc:.4f}")

    csv_path = os.path.join(args.out, "cifar100_crossview_mixer.csv")
    with open(csv_path, "w") as f:
        f.write("seed,acc\n")
        for i, s in enumerate(args.seeds):
            f.write(f"{s},{accs[i]:.4f}\n")

    print(f"\n📊 Final: {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
