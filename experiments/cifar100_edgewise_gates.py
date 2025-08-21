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


class EdgewiseGateHead(nn.Module):
    def __init__(self, in_ch: int = 6, hidden: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(hidden, 4, kernel_size=1, bias=True)
        nn.init.constant_(self.conv2.bias, -5.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.conv1(feat)
        x = self.act(x)
        x = self.conv2(x)
        return torch.sigmoid(x)


class EdgewiseMSA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_not: float = 0.5,
    ):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.beta_not = beta_not
        self.qkv1 = nn.Linear(dim, dim * 3, bias=False)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.edge_head = EdgewiseGateHead(in_ch=6, hidden=16)
        self.chain_value_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv1 = self.qkv1(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        qkv2 = self.qkv2(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        scale = 1.0 / math.sqrt(self.dk)
        S1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        S2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        if attn_mask is not None:
            m = attn_mask == 0
            S1 = S1.masked_fill(m, float("-inf"))
            S2 = S2.masked_fill(m, float("-inf"))
        A1 = F.softmax(S1, dim=-1)
        A2 = F.softmax(S2, dim=-1)
        C_right = torch.matmul(A1, A2)
        C_left = torch.matmul(A2, A1)
        eps = 1e-6
        BtH = B * self.h
        S1_img = S1.view(BtH, N, N)
        S2_img = S2.view(BtH, N, N)
        S1T_img = S1_img.transpose(1, 2)
        S2T_img = S2_img.transpose(1, 2)
        Cr_img = torch.log(C_right + eps).view(BtH, N, N)
        Cl_img = torch.log(C_left + eps).view(BtH, N, N)
        feat = torch.stack([S1_img, S2_img, S1T_img, S2T_img, Cr_img, Cl_img], dim=1)
        gates = self.edge_head(feat)
        g_and, g_or, g_not, g_chain = gates[:, 0], gates[:, 1], gates[:, 2], gates[:, 3]
        Smix = S1_img
        Smix = Smix + g_and * (S1_img + S2_img - S1_img)
        Smix = Smix + g_or * (lse(S1_img, S2_img) - S1_img)
        Smix = Smix - g_not * (self.beta_not * S2_img)
        Smix = Smix + g_chain * Cr_img
        Smix = Smix.view(B, self.h, N, N)
        if attn_mask is not None:
            Smix = Smix.masked_fill(attn_mask == 0, float("-inf"))
        A = F.softmax(Smix, dim=-1)
        A = self.attn_drop(A)
        y_base = torch.matmul(A, v1)
        transport = torch.matmul(A2, v2)
        y_chain = torch.matmul(A1, transport)
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


class BlockEdgewise(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        beta_not: float = 0.5,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = EdgewiseMSA(dim, heads, attn_drop, drop, beta_not=beta_not)
        self.dp1 = DropPath(drop_path) if DropPath is not None else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.dp2 = DropPath(drop_path) if DropPath is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.ln1(x)))
        x = x + self.dp2(self.mlp(self.ln2(x)))
        return x


class ViTEdgewise(nn.Module):
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
                BlockEdgewise(
                    dim,
                    heads,
                    mlp_ratio,
                    drop,
                    0.0,
                    dps[i],
                    beta_not=beta_not,
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
        description="CIFAR-100 Edgewise-gated dual-path attention"
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
    ap.add_argument("--out", type=str, default="results/cifar100_edgewise_gates")
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
        print(f"\n🔬 Seed {s}")
        set_seed(s)
        model = ViTEdgewise(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            n_classes=100,
            drop_path=args.drop_path,
            beta_not=args.beta_not,
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

    csv_path = os.path.join(args.out, "cifar100_edgewise_gates.csv")
    with open(csv_path, "w") as f:
        f.write("seed,acc\n")
        for i, s in enumerate(args.seeds):
            f.write(f"{s},{accs[i]:.4f}\n")
    print(f"\n📊 Final: {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
