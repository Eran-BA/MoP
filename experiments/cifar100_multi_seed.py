#!/usr/bin/env python3
import argparse
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

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


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
            if hasattr(module, "ViT_Baseline") and hasattr(module, "ViT_MoP"):
                print(f"✅ Successfully imported from {desc}")
                return module.ViT_Baseline, module.ViT_MoP
            elif hasattr(module, "models"):
                models = module.models
                if hasattr(models, "ViT_Baseline") and hasattr(models, "ViT_MoP"):
                    print(f"✅ Successfully imported from {desc}.models")
                    return models.ViT_Baseline, models.ViT_MoP
        except ImportError as e:
            print(f"❌ Failed to import from {desc}: {e}")
            continue

    # If all imports fail, provide detailed debugging
    print(f"\n🔍 Import debugging:")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python path includes project root: {PROJECT_ROOT in sys.path}")

    mop_path = os.path.join(PROJECT_ROOT, "mop")
    print(f"MoP directory exists: {os.path.exists(mop_path)}")

    if os.path.exists(mop_path):
        print(f"MoP directory contents: {os.listdir(mop_path)}")
        init_file = os.path.join(mop_path, "__init__.py")
        print(f"MoP __init__.py exists: {os.path.exists(init_file)}")

        models_path = os.path.join(mop_path, "models")
        if os.path.exists(models_path):
            print(f"Models directory contents: {os.listdir(models_path)}")
        else:
            print("❌ No models subdirectory found")

    raise ImportError(
        "Could not import ViT_Baseline and ViT_MoP. Please check installation."
    )


def get_loaders(batch=256, tiny=False, workers=2):
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
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument(
        "--tiny", action="store_true", help="use small subset for a quick smoke run"
    )
    ap.add_argument("--out", type=str, default="results/cifar100")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")  # helps on MPS
    print(f"Device: {device}")
    print(f"Project root: {PROJECT_ROOT}")

    ViT_Baseline, ViT_MoP = try_import_models()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    accs_base = []
    accs_mop = []
    for s in args.seeds:
        print(f"\n🔬 Running experiment with seed {s}")
        set_seed(s)
        # small-ish configs for quick runs
        base = ViT_Baseline(dim=256, depth=8, heads=4, n_classes=100).to(device)
        mop = ViT_MoP(
            dim=256, depth=8, heads=4, n_classes=100, n_views=5, n_kernels=3
        ).to(device)

        # Show parameter counts
        base_params = count_parameters(base)
        mop_params = count_parameters(mop)
        print(f"Baseline params: {base_params:,}")
        print(f"MoP params: {mop_params:,}")
        print(f"Param ratio: {mop_params/base_params:.3f}")

        opt_b, sch_b = make_opt(base)
        opt_m, sch_m = make_opt(mop)

        steps = 0
        base.train()
        mop.train()
        it = iter(train_loader)

        while steps < args.steps:
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(train_loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)

            # baseline
            opt_b.zero_grad(set_to_none=True)
            loss_b = nn.functional.cross_entropy(base(xb), yb)
            loss_b.backward()
            opt_b.step()
            sch_b.step()

            # mop
            opt_m.zero_grad(set_to_none=True)
            loss_m = nn.functional.cross_entropy(mop(xb), yb)
            loss_m.backward()
            opt_m.step()
            sch_m.step()

            steps += 1
            if steps % max(args.eval_every, 1) == 0 or steps == 1:
                a_b = evaluate(base, val_loader, device)
                a_m = evaluate(mop, val_loader, device)
                print(
                    f"step {steps:4d} | loss_b={loss_b.item():.3f} loss_m={loss_m.item():.3f} | "
                    f"acc_b={a_b:.3f} acc_m={a_m:.3f} | diff={a_m-a_b:+.3f}"
                )

        # final eval per seed
        a_b = evaluate(base, val_loader, device)
        a_m = evaluate(mop, val_loader, device)
        accs_base.append(a_b)
        accs_mop.append(a_m)
        print(f"seed {s}: baseline={a_b:.4f}  mop={a_m:.4f}  diff={a_m-a_b:+.4f}")

    # save CSV
    csv_path = os.path.join(args.out, "cifar100_acc.csv")
    with open(csv_path, "w") as f:
        f.write("seed,baseline,mop,diff\n")
        for s, (b, m) in enumerate(zip(accs_base, accs_mop)):
            f.write(f"{args.seeds[s]},{b:.4f},{m:.4f},{m-b:.4f}\n")

    print(f"\n📊 Final Results (across {len(args.seeds)} seeds):")
    mean_base = np.mean(accs_base)
    mean_mop = np.mean(accs_mop)
    std_base = np.std(accs_base)
    std_mop = np.std(accs_mop)
    mean_diff = mean_mop - mean_base

    print(f"Baseline: {mean_base:.4f} ± {std_base:.4f}")
    print(f"MoP:      {mean_mop:.4f} ± {std_mop:.4f}")
    print(f"Diff:     {mean_diff:+.4f}")
    print(f"Results saved to: {csv_path}")

    if mean_diff > 0:
        print("🎉 MoP shows improvement!")
    else:
        print("🤔 MoP shows no improvement")


if __name__ == "__main__":
    main()
