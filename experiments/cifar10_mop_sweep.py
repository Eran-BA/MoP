#!/usr/bin/env python3
import argparse
import itertools
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Fix import path - add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def try_import_models():
    """Import MoP models with robust error handling"""
    import_attempts = [
        ("mop", lambda: __import__("mop")),
        ("mop.models", lambda: __import__("mop.models", fromlist=[""])),
    ]

    for desc, import_func in import_attempts:
        try:
            module = import_func()
            if hasattr(module, "ViT_MoP"):
                print(f"✅ Successfully imported from {desc}")
                return module.ViT_MoP
            elif hasattr(module, "models") and hasattr(module.models, "ViT_MoP"):
                print(f"✅ Successfully imported from {desc}.models")
                return module.models.ViT_MoP
        except ImportError as e:
            print(f"❌ Failed to import from {desc}: {e}")
            continue

    raise ImportError("Could not import ViT_MoP. Please check installation.")


def get_loaders(batch=256, tiny=False, workers=2):
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
        # fast smoke: ~5k train / 1k test
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
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument(
        "--tiny", action="store_true", help="use small subset for a quick smoke run"
    )
    ap.add_argument("--views", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--kernels", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--out", type=str, default="results/cifar10_mop_sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    print(f"Project root: {PROJECT_ROOT}")

    ViT_MoP = try_import_models()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    results = []
    for seed in args.seeds:
        for n_views, n_kernels in itertools.product(args.views, args.kernels):
            cfg_name = f"views{n_views}_kernels{n_kernels}_seed{seed}"
            print(f"\n🔬 Running {cfg_name}")
            set_seed(seed)

            model = ViT_MoP(
                dim=256,
                depth=6,
                heads=4,
                n_classes=10,
                n_views=n_views,
                n_kernels=n_kernels,
            ).to(device)

            params = count_parameters(model)
            print(f"Model params: {params:,}")

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

            final_acc = evaluate(model, val_loader, device)
            results.append((seed, n_views, n_kernels, final_acc))
            print(f"done {cfg_name}: acc={final_acc:.4f}")

    # save CSV
    csv_path = os.path.join(args.out, "cifar10_mop_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("seed,views,kernels,acc\n")
        for seed, v, k, acc in results:
            f.write(f"{seed},{v},{k},{acc:.4f}\n")

    # simple aggregation
    print("\n📊 Aggregate by (views,kernels):")

    def group_key(item):
        _, v, k, _ = item
        return (v, k)

    grouped = {}
    for item in results:
        key = group_key(item)
        grouped.setdefault(key, []).append(item[3])
    for (v, k), accs in sorted(grouped.items()):
        mean_acc = float(np.mean(accs)) if len(accs) else float("nan")
        std_acc = float(np.std(accs)) if len(accs) else float("nan")
        print(
            f"views={v:2d} kernels={k:2d} | acc={mean_acc:.4f} ± {std_acc:.4f} (n={len(accs)})"
        )

    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
