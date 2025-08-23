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

# Project path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from mop.models import DropPath, PatchEmbed
except Exception:
    PatchEmbed = None
    DropPath = None


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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
            nn.init.constant_(self.conv2.bias, -5.0)
            with torch.no_grad():
                if self.gate_init == "and":
                    self.conv2.bias[0] = 2.0
                elif self.gate_init == "or":
                    self.conv2.bias[1] = 2.0
                elif self.gate_init == "not":
                    self.conv2.bias[2] = 2.0
                elif self.gate_init == "nor":
                    self.conv2.bias[2] = 2.0
                elif self.gate_init == "xor":
                    self.conv2.bias[1] = 2.0
                elif self.gate_init == "chain":
                    self.conv2.bias[3] = 2.0
        else:
            self.row_proj = nn.Conv1d(in_ch, 4 * self.gate_rank, kernel_size=1, bias=True)
            self.col_proj = nn.Conv1d(in_ch, 4 * self.gate_rank, kernel_size=1, bias=True)
            with torch.no_grad():
                nn.init.constant_(self.row_proj.bias, 0.0)
                nn.init.constant_(self.col_proj.bias, 0.0)
                idx_map = {"and": 0, "or": 1, "not": 2, "chain": 3}
                if self.gate_init in idx_map:
                    idx = idx_map[self.gate_init]
                    c = float(max(0.0, (2.0 / max(1, self.gate_rank)) ** 0.5))
                    s, e = idx * self.gate_rank, (idx + 1) * self.gate_rank
                    self.row_proj.bias[s:e] = c
                    self.col_proj.bias[s:e] = c
                elif self.gate_init in ("nor", "xor"):
                    idx = 2 if self.gate_init == "nor" else 1
                    c = float(max(0.0, (2.0 / max(1, self.gate_rank)) ** 0.5))
                    s, e = idx * self.gate_rank, (idx + 1) * self.gate_rank
                    self.row_proj.bias[s:e] = c
                    self.col_proj.bias[s:e] = c

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B*H, C, T, T]
        if self.gate_mode == "dense":
            x = self.conv1(feat)
            x = self.act(x)
            if self.use_k3:
                x = self.mid3(self.act(x))
            x = self.conv2(x)
            return torch.sigmoid(x)
        BtH, C, N, _ = feat.shape
        row_feat = feat.mean(dim=3)
        col_feat = feat.mean(dim=2)
        a = self.row_proj(row_feat)
        b = self.col_proj(col_feat)
        a = a.view(BtH, 4, self.gate_rank, N)
        b = b.view(BtH, 4, self.gate_rank, N)
        G = torch.einsum("bcrn,bcrm->bcnm", a, b)
        return torch.sigmoid(G)


class EdgewiseMSA(nn.Module):
    """Multi-view attention with edgewise boolean gates per (i,j).

    For n_views=M, build features: [S1..SM, S1^T..SM^T, log(C_fwd), log(C_bwd)], where
    C_fwd = A1 @ A2 @ ... @ AM and C_bwd = AM @ ... @ A1. Predict per-edge gates and mix logits.
    Includes value-aware chain transport with a learnable global weight.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        beta_not: float = 0.5,
        use_k3: bool = False,
        n_views: int = 2,
        gate_mode: str = "dense",
        gate_rank: int = 4,
        gate_init: str = "neutral",
    ):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.beta_not = beta_not
        self.n_views = max(2, int(n_views))

        self.qkv_list = nn.ModuleList(
            [nn.Linear(dim, dim * 3, bias=False) for _ in range(self.n_views)]
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        in_ch = 2 * self.n_views + 2
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
        qs: list[torch.Tensor] = []
        ks: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for lin in self.qkv_list:
            qkv = lin(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
            qs.append(qkv[0])
            ks.append(qkv[1])
            vs.append(qkv[2])

        scale = 1.0 / math.sqrt(self.dk)
        S_list = [
            torch.matmul(qs[i], ks[i].transpose(-2, -1)) * scale
            for i in range(self.n_views)
        ]

        if attn_mask is not None:
            m = attn_mask == 0
            S_list = [S.masked_fill(m, float("-inf")) for S in S_list]

        A_list = [F.softmax(S, dim=-1) for S in S_list]
        # Forward and backward chains
        C_fwd = A_list[0]
        for i in range(1, self.n_views):
            C_fwd = torch.matmul(C_fwd, A_list[i])
        C_bwd = A_list[-1]
        for i in range(self.n_views - 2, -1, -1):
            C_bwd = torch.matmul(C_bwd, A_list[i])
        eps = 1e-6

        # Build feature stack per head as image: [B*H, C, N, N]
        BtH = B * self.h
        S_imgs = [S.view(BtH, N, N) for S in S_list]
        ST_imgs = [img.transpose(1, 2) for img in S_imgs]
        Cr_img = torch.log(C_fwd + eps).view(BtH, N, N)
        Cl_img = torch.log(C_bwd + eps).view(BtH, N, N)
        feat = torch.stack(S_imgs + ST_imgs + [Cr_img, Cl_img], dim=1)
        gates = self.edge_head(feat)  # [B*H, 4, N, N]
        g_and, g_or, g_not, g_chain = gates[:, 0], gates[:, 1], gates[:, 2], gates[:, 3]

        # Mix logits per edge
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

        # Restore shape and mask, then softmax
        Smix = Smix.view(B, self.h, N, N)
        if attn_mask is not None:
            Smix = Smix.masked_fill(attn_mask == 0, float("-inf"))
        A = F.softmax(Smix, dim=-1)
        A = self.attn_drop(A)

        # Value paths
        v1 = vs[0]
        y_base = torch.matmul(A, v1)
        # Value transport along forward chain using last view's values
        transport = vs[-1]
        for i in range(self.n_views - 1, 0, -1):
            transport = torch.matmul(A_list[i], transport)
        y_chain = torch.matmul(A_list[0], transport)
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
        use_k3: bool = False,
        gate_mode: str = "dense",
        gate_rank: int = 4,
        gate_init: str = "neutral",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = EdgewiseMSA(
            dim,
            heads,
            attn_drop,
            drop,
            beta_not=beta_not,
            use_k3=use_k3,
            gate_mode=gate_mode,
            gate_rank=gate_rank,
            gate_init=gate_init,
        )
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
        depth: int = 6,
        heads: int = 4,
        n_classes: int = 10,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        patch: int = 4,
        num_tokens: int = 64,
        beta_not: float = 0.5,
        use_k3: bool = False,
        gate_mode: str = "dense",
        gate_rank: int = 4,
        gate_init: str = "neutral",
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
                    use_k3=use_k3,
                    gate_mode=gate_mode,
                    gate_rank=gate_rank,
                    gate_init=gate_init,
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
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm_train)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test)
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
        description="CIFAR-10 Edgewise-gated dual-path attention"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--beta_not", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results/cifar10_edgewise_gates")
    ap.add_argument("--ew_gate_mode", type=str, default="dense", choices=["dense", "lowrank"])
    ap.add_argument("--ew_gate_rank", type=int, default=4)
    ap.add_argument(
        "--ew_gate_init",
        type=str,
        default="neutral",
        choices=["neutral", "and", "or", "not", "nor", "xor", "chain"],
    )
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
            n_classes=10,
            drop_path=args.drop_path,
            beta_not=args.beta_not,
            gate_mode=args.ew_gate_mode,
            gate_rank=args.ew_gate_rank,
            gate_init=args.ew_gate_init,
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

    csv_path = os.path.join(args.out, "cifar10_edgewise_gates.csv")
    with open(csv_path, "w") as f:
        f.write("seed,acc\n")
        for i, s in enumerate(args.seeds):
            f.write(f"{s},{accs[i]:.4f}\n")
    print(f"\n📊 Final: {float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
