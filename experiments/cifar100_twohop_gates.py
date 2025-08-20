#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
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


def lse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(torch.stack([a, b], dim=0), dim=0)


def dual_path_mix(
    S1: torch.Tensor,
    S2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    beta_not: float = 0.5,
    gates: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    if gates is None:
        gates = dict(and_=1.0, or_=0.0, not_=0.0, chain=0.0, base=1.0)

    Smix = gates["base"] * (S1 + 0.0)
    Smix = Smix + gates["and_"] * ((S1 + S2) - S1)
    Smix = Smix + gates["or_"] * (lse(S1, S2) - S1)
    Smix = Smix - gates["not_"] * (beta_not * S2)

    if mask is not None:
        m = mask == 0
        A1 = F.softmax(S1.masked_fill(m, float("-inf")), dim=-1)
        A2 = F.softmax(S2.masked_fill(m, float("-inf")), dim=-1)
    else:
        A1 = F.softmax(S1, dim=-1)
        A2 = F.softmax(S2, dim=-1)
    C_right = torch.matmul(A1, A2)
    eps = 1e-6
    Smix = Smix + gates["chain"] * torch.log(C_right + eps)

    if mask is not None:
        Smix = Smix.masked_fill((mask == 0), float("-inf"))
    return F.softmax(Smix, dim=-1)


class DualPathMSA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_not: float = 0.5,
        gates: Optional[Dict[str, float]] = None,
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
        self.beta_not = beta_not
        self.gates = gates or dict(and_=1.0, or_=0.0, not_=0.0, chain=0.0, base=1.0)
        self.chain_value_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv1(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv[0], qkv[1], qkv[2]
        qkv_b = self.qkv2(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv_b[0], qkv_b[1], qkv_b[2]

        scale = 1.0 / math.sqrt(self.dk)
        S1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        S2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        A = dual_path_mix(
            S1, S2, mask=attn_mask, beta_not=self.beta_not, gates=self.gates
        )
        A = self.attn_drop(A)

        y_base = torch.matmul(A, v1)

        if attn_mask is not None:
            m = attn_mask == 0
            A1 = F.softmax(S1.masked_fill(m, float("-inf")), dim=-1)
            A2 = F.softmax(S2.masked_fill(m, float("-inf")), dim=-1)
        else:
            A1 = F.softmax(S1, dim=-1)
            A2 = F.softmax(S2, dim=-1)
        y_chain = torch.matmul(A1, torch.matmul(A2, v2))

        w = torch.sigmoid(self.chain_value_logit)
        y = y_base + w * y_chain
        y = y.transpose(1, 2).reshape(B, N, D)
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


class BlockGated(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        beta_not: float = 0.5,
        gates: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = DualPathMSA(
            dim, heads, attn_drop, drop, beta_not=beta_not, gates=gates
        )
        self.dp1 = DropPath(drop_path) if DropPath is not None else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.dp2 = DropPath(drop_path) if DropPath is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.ln1(x)))
        x = x + self.dp2(self.mlp(self.ln2(x)))
        return x


class ViTGated(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        heads: int = 4,
        n_classes: int = 100,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        patch: int = 4,
        num_tokens: int = 64,
        beta_not: float = 0.5,
        gates: Optional[Dict[str, float]] = None,
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
                BlockGated(
                    dim,
                    heads,
                    mlp_ratio,
                    drop,
                    0.0,
                    dps[i],
                    beta_not=beta_not,
                    gates=gates,
                )
                for i in range(depth)
            ]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes, bias=False)

        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_builtin_patch:
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
        description="CIFAR-100 two-hop gated attention with value transport"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--beta_not", type=float, default=0.5)
    ap.add_argument("--gate_base", type=float, default=1.0)
    ap.add_argument("--gate_and", type=float, default=1.0)
    ap.add_argument("--gate_or", type=float, default=0.0)
    ap.add_argument("--gate_not", type=float, default=0.0)
    ap.add_argument("--gate_chain", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="results/cifar100_twohop_gates")
    args = ap.parse_args()

    gates = dict(
        base=args.gate_base,
        and_=args.gate_and,
        or_=args.gate_or,
        not_=args.gate_not,
        chain=args.gate_chain,
    )

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    print(f"Gates: {gates} | beta_not={args.beta_not}")

    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m: nn.Module):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    accs = []
    for s in args.seeds:
        print(f"\n🔬 Running seed {s}")
        set_seed(s)

        model = ViTGated(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            n_classes=100,
            drop_path=args.drop_path,
            beta_not=args.beta_not,
            gates=gates,
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

    csv_path = os.path.join(args.out, "cifar100_twohop_gates.csv")
    with open(csv_path, "w") as f:
        f.write("seed,acc\n")
        for i, s in enumerate(args.seeds):
            f.write(f"{s},{accs[i]:.4f}\n")

    print(f"\n📊 Final: {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
