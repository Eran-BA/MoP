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

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


def try_import_models():
    """Import MoP models with robust error handling."""
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
            if hasattr(module, "models"):
                models = module.models
                if hasattr(models, "ViT_Baseline") and hasattr(models, "ViT_MoP"):
                    print(f"✅ Successfully imported from {desc}.models")
                    return models.ViT_Baseline, models.ViT_MoP
        except Exception as e:
            print(f"❌ Failed to import from {desc}: {e}")
            continue

    raise ImportError(
        "Could not import ViT_Baseline and ViT_MoP. Please check installation."
    )


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
    """Grid-search small space to match parameter target as closely as possible."""
    best_diff = None
    best_cfg = None
    best_params = None
    for heads in heads_list:
        for dim in dims:
            # Require divisibility typical for multi-head attention
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


def find_config_for_target_under_budget(
    ctor,
    n_classes: int,
    target_params: int,
    max_params: int,
    dims: Iterable[int] = (128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768),
    depths: Iterable[int] = (4, 6, 8, 10, 12),
    heads_list: Iterable[int] = (4, 6, 8),
    extra_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, int], int]:
    """Find config closest to target but not exceeding max_params."""
    best_diff = None
    best_cfg = None
    best_params = None
    best_under = None
    best_under_params = None
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
                if p <= max_params:
                    diff = abs(int(target_params) - p)
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_cfg = {"dim": dim, "depth": depth, "heads": heads}
                        best_params = p
                    if best_under_params is None or p > best_under_params:
                        best_under_params = p
                        best_under = {"dim": dim, "depth": depth, "heads": heads}
    # Prefer closest-to-target under budget; otherwise take the largest under budget
    if best_cfg is not None:
        return best_cfg, int(best_params)
    if best_under is not None:
        return best_under, int(best_under_params)
    raise RuntimeError("Could not find a configuration under the specified budget.")


def find_config_for_target_within_ratio_under_budget(
    ctor,
    n_classes: int,
    target_params: int,
    max_params: int,
    max_ratio_diff: float = 0.01,
    dims: Iterable[int] = (128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768),
    depths: Iterable[int] = (4, 6, 8, 10, 12),
    heads_list: Iterable[int] = (4, 6, 8),
    extra_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, int], int, bool]:
    """Find config under budget and within relative diff to max_params when possible.

    Returns (cfg, params, within_ratio_flag).
    """
    best_within = None
    best_within_params = None
    best_within_target_diff = None
    best_under = None
    best_under_params = None
    best_under_target_diff = None
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
                if p > max_params:
                    continue
                rel_gap = abs(max_params - p) / max(1, max_params)
                target_diff = abs(int(target_params) - p)
                if rel_gap <= max_ratio_diff:
                    if best_within is None or target_diff < best_within_target_diff:
                        best_within = {"dim": dim, "depth": depth, "heads": heads}
                        best_within_params = p
                        best_within_target_diff = target_diff
                # Track the closest-to-target under budget regardless of ratio for fallback
                if best_under is None or target_diff < best_under_target_diff:
                    best_under = {"dim": dim, "depth": depth, "heads": heads}
                    best_under_params = p
                    best_under_target_diff = target_diff
    if best_within is not None:
        return best_within, int(best_within_params), True
    if best_under is not None:
        return best_under, int(best_under_params), False
    raise RuntimeError("Could not find a configuration under the specified budget.")


def find_mop_config_match_baseline(
    ctor,
    n_classes: int,
    target_params: int,
    baseline_cfg: Dict[str, int],
    baseline_params: int,
    max_ratio_diff: float = 0.01,
    dims_choices: Iterable[int] = (
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        448,
        512,
        640,
        768,
    ),
    depths_choices: Iterable[int] = (4, 6, 8, 10, 12),
    heads_choices: Iterable[int] = (4, 6, 8),
    extra_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, int], int, bool]:
    """Prefer exact same (dim,depth,heads) as baseline; otherwise choose closest under-budget with cfg <= baseline.

    Returns (cfg, params, within_ratio_flag).
    """
    base_dim = baseline_cfg["dim"]
    base_depth = baseline_cfg["depth"]
    base_heads = baseline_cfg["heads"]

    # Build candidate lists restricted to <= baseline
    dims = [d for d in dims_choices if d <= base_dim]
    if base_dim not in dims:
        dims.append(base_dim)
    depths = [d for d in depths_choices if d <= base_depth]
    if base_depth not in depths:
        depths.append(base_depth)
    heads_list = [h for h in heads_choices if h <= base_heads]
    if base_heads not in heads_list:
        heads_list.append(base_heads)

    # Try exact baseline config first
    try:
        p_same = estimate_params(
            ctor, n_classes, base_dim, base_depth, base_heads, extra_kwargs
        )
        if p_same <= baseline_params:
            rel_gap = abs(baseline_params - p_same) / max(1, baseline_params)
            return (
                {"dim": base_dim, "depth": base_depth, "heads": base_heads},
                int(p_same),
                (rel_gap <= max_ratio_diff),
            )
    except Exception:
        pass

    # Otherwise search under budget with cfg <= baseline
    best_within = None
    best_within_params = None
    best_within_target_diff = None
    best_under = None
    best_under_params = None
    best_under_target_diff = None
    for heads in sorted(set(heads_list)):
        for dim in sorted(set(dims)):
            if dim % heads != 0:
                continue
            for depth in sorted(set(depths)):
                try:
                    p = estimate_params(
                        ctor, n_classes, dim, depth, heads, extra_kwargs
                    )
                except Exception:
                    continue
                if p > baseline_params:
                    continue
                rel_gap = abs(baseline_params - p) / max(1, baseline_params)
                target_diff = abs(int(target_params) - p)
                if rel_gap <= max_ratio_diff:
                    if best_within is None or target_diff < best_within_target_diff:
                        best_within = {"dim": dim, "depth": depth, "heads": heads}
                        best_within_params = p
                        best_within_target_diff = target_diff
                if best_under is None or target_diff < best_under_target_diff:
                    best_under = {"dim": dim, "depth": depth, "heads": heads}
                    best_under_params = p
                    best_under_target_diff = target_diff

    if best_within is not None:
        return best_within, int(best_within_params), True
    if best_under is not None:
        return best_under, int(best_under_params), False
    raise RuntimeError("Could not find an MoP configuration under baseline budget.")


def main():
    ap = argparse.ArgumentParser(
        description="A/B test Baseline vs MoP at fixed parameter budgets on CIFAR-10"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="random seeds")
    ap.add_argument("--steps", type=int, default=1000, help="training steps per run")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument(
        "--tiny", action="store_true", help="use small subset for a quick smoke run"
    )
    ap.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[5_000_000, 50_000_000],
        help="parameter targets",
    )
    ap.add_argument("--mop_views", type=int, default=5)
    ap.add_argument("--mop_kernels", type=int, default=3)
    ap.add_argument("--out", type=str, default="results/cifar10_ab_param_budgets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    print(f"Project root: {PROJECT_ROOT}")

    ViT_Baseline, ViT_MoP = try_import_models()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m: nn.Module):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    # Results aggregated across targets
    for target in args.targets:
        print(f"\n🎯 Target parameters: {int(target):,}")

        base_cfg, base_p = find_config_for_target(
            ViT_Baseline,
            n_classes=10,
            target_params=int(target),
        )
        mop_cfg, mop_p, within_1pct = find_mop_config_match_baseline(
            ViT_MoP,
            n_classes=10,
            target_params=int(target),
            baseline_cfg=base_cfg,
            baseline_params=base_p,
            max_ratio_diff=0.01,
            extra_kwargs={"n_views": args.mop_views, "n_kernels": args.mop_kernels},
        )

        print(f"Baseline config: {base_cfg} | params={base_p:,}")
        if within_1pct:
            print(
                f"MoP config     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={mop_p:,} (≤ base, within 1%)"
            )
        else:
            gap = (base_p - mop_p) / max(1, base_p)
            print(
                f"MoP config     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={mop_p:,} (≤ base, gap {gap:.2%} > 1% target)"
            )

        accs_base: list[float] = []
        accs_mop: list[float] = []

        for s in args.seeds:
            print(f"\n🔬 Running A/B with seed {s}")
            set_seed(s)

            base = ViT_Baseline(n_classes=10, **base_cfg).to(device)
            mop = ViT_MoP(
                n_classes=10,
                **mop_cfg,
                n_views=args.mop_views,
                n_kernels=args.mop_kernels,
            ).to(device)

            print(f"Baseline params (instantiated): {count_parameters(base):,}")
            print(f"MoP params (instantiated)     : {count_parameters(mop):,}")

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

        # save CSV per target
        csv_path = os.path.join(args.out, f"cifar10_ab_target_{int(target)}.csv")
        with open(csv_path, "w") as f:
            f.write(
                "seed,baseline_acc,mop_acc,diff,baseline_params,mop_params,baseline_cfg,mop_cfg\n"
            )
            for i, s in enumerate(args.seeds):
                f.write(
                    f'{s},{accs_base[i]:.4f},{accs_mop[i]:.4f},{accs_mop[i]-accs_base[i]:.4f},{base_p},{mop_p},"{base_cfg}","{mop_cfg}"\n'
                )

        print(
            f"\n📊 Final Results for target {int(target):,} (across {len(args.seeds)} seeds):"
        )
        mean_base = float(np.mean(accs_base))
        mean_mop = float(np.mean(accs_mop))
        std_base = float(np.std(accs_base))
        std_mop = float(np.std(accs_mop))
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
