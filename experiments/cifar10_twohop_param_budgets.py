#!/usr/bin/env python3
import argparse
import os
import random
import sys
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root and experiments dir to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)


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


def try_import_twohop():
    try:
        from cifar10_twohop_gates import ViTGated  # type: ignore

        print("✅ Imported ViTGated (two-hop, CIFAR-10) from experiments")
        return ViTGated
    except Exception as e:
        raise ImportError(f"Could not import ViTGated from cifar10_twohop_gates: {e}")


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_params(
    ctor,
    n_classes: int,
    dim: int,
    depth: int,
    heads: int,
    extra_kwargs: Optional[Dict] = None,
) -> int:
    kwargs = dict(dim=dim, depth=depth, heads=heads, n_classes=n_classes)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    model = ctor(**kwargs)
    return count_parameters(model)


def find_config_for_target(
    ctor,
    n_classes: int,
    target_params: int,
    dims: Iterable[int] = (128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768),
    depths: Iterable[int] = (4, 6, 8, 10, 12),
    heads_list: Iterable[int] = (4, 6, 8),
    extra_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, int], int]:
    best_diff = None
    best_cfg = None
    best_params = None
    for heads in heads_list:
        for dim in dims:
            if dim % heads != 0:
                continue
            for depth in depths:
                try:
                    p = estimate_params(
                        ctor, n_classes, dim, depth, heads, extra_kwargs
                    )
                except Exception:
                    continue
                diff = abs(int(target_params) - p)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_cfg = {"dim": dim, "depth": depth, "heads": heads}
                    best_params = p
    if best_cfg is None:
        raise RuntimeError("Could not find a configuration close to target params.")
    return best_cfg, int(best_params)


def main():
    ap = argparse.ArgumentParser(
        description="Param-match TwoHop ViT on CIFAR-10 at specified budgets"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--targets", type=int, nargs="+", default=[5_000_000, 50_000_000])
    # two-hop gates
    ap.add_argument("--gate_base", type=float, default=1.0)
    ap.add_argument("--gate_and", type=float, default=1.0)
    ap.add_argument("--gate_or", type=float, default=0.0)
    ap.add_argument("--gate_not", type=float, default=0.0)
    ap.add_argument("--gate_chain", type=float, default=0.0)
    ap.add_argument("--beta_not", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results/cifar10_twohop_param_budgets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    ViTGated = try_import_twohop()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    gates = dict(
        base=args.gate_base,
        and_=args.gate_and,
        or_=args.gate_or,
        not_=args.gate_not,
        chain=args.gate_chain,
    )

    def make_opt(m: nn.Module):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    for target in args.targets:
        print(f"\n🎯 Target parameters: {int(target):,}")
        cfg, p = find_config_for_target(
            ViTGated,
            n_classes=10,
            target_params=int(target),
            extra_kwargs={"gates": gates, "beta_not": args.beta_not},
        )
        print(f"TwoHop cfg: {cfg} | params={p:,}")

        accs: list[float] = []
        for s in args.seeds:
            print(f"\n🔬 Seed {s}")
            set_seed(s)
            model = ViTGated(
                n_classes=10, **cfg, gates=gates, beta_not=args.beta_not
            ).to(device)
            print(f"Params (instantiated): {count_parameters(model):,}")
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
                loss = nn.functional.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
                sch.step()

                steps += 1
                if steps % max(args.eval_every, 1) == 0 or steps == 1:
                    acc = evaluate(model, val_loader, device)
                    print(f"step {steps:4d} | loss={loss.item():.3f} | acc={acc:.3f}")

            acc = evaluate(model, val_loader, device)
            accs.append(acc)
            print(f"seed {s}: acc={acc:.4f}")

        csv_path = os.path.join(args.out, f"cifar10_twohop_target_{int(target)}.csv")
        with open(csv_path, "w") as f:
            f.write("seed,acc,dim,depth,heads,params\n")
            for i, s in enumerate(args.seeds):
                f.write(
                    f"{s},{accs[i]:.4f},{cfg['dim']},{cfg['depth']},{cfg['heads']},{p}\n"
                )
        print(
            f"📊 Target {int(target):,} | mean acc={float(np.mean(accs)):.4f} ± {float(np.std(accs)):.4f}"
        )
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
