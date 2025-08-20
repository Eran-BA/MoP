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


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def set_seed(seed: int) -> None:
    import random as _random

    _random.seed(seed)
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
    """Import Baseline/MoP and the CIFAR-100 Cross-View Mixer model."""
    import_attempts = [
        ("mop", lambda: __import__("mop")),
        ("mop.models", lambda: __import__("mop.models", fromlist=[""])),
    ]
    ViT_Baseline = None
    ViT_MoP = None
    for desc, import_func in import_attempts:
        try:
            module = import_func()
            src = None
            if hasattr(module, "ViT_Baseline") and hasattr(module, "ViT_MoP"):
                ViT_Baseline, ViT_MoP, src = module.ViT_Baseline, module.ViT_MoP, desc
            elif (
                hasattr(module, "models")
                and hasattr(module.models, "ViT_Baseline")
                and hasattr(module.models, "ViT_MoP")
            ):
                ViT_Baseline, ViT_MoP, src = (
                    module.models.ViT_Baseline,
                    module.models.ViT_MoP,
                    f"{desc}.models",
                )
            if src is not None:
                print(f"✅ Imported Baseline/MoP from {src}")
                break
        except Exception as e:
            print(f"❌ Failed to import Baseline/MoP from {desc}: {e}")
    if ViT_Baseline is None or ViT_MoP is None:
        raise ImportError("Could not import ViT_Baseline and ViT_MoP")

    try:
        from cifar100_crossview_mixer import ViTCrossView  # type: ignore

        print("✅ Imported ViTCrossView (CIFAR-100) from experiments")
    except Exception as e:
        raise ImportError(f"Could not import ViTCrossView (CIFAR-100): {e}")

    return ViT_Baseline, ViT_MoP, ViTCrossView


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
    depths: Iterable[int] = (6, 8, 10, 12),
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


def find_config_for_target_under_budget(
    ctor,
    n_classes: int,
    target_params: int,
    max_params: int,
    dims: Iterable[int] = (128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768),
    depths: Iterable[int] = (6, 8, 10, 12),
    heads_list: Iterable[int] = (4, 6, 8),
    extra_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, int], int]:
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
    if best_cfg is not None:
        return best_cfg, int(best_params)
    if best_under is not None:
        return best_under, int(best_under_params)
    raise RuntimeError("Could not find a configuration under the specified budget.")


def main():
    ap = argparse.ArgumentParser(
        description="A/B/C test Baseline vs MoP vs CrossView at fixed parameter budgets on CIFAR-100"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--targets", type=int, nargs="+", default=[5_000_000, 50_000_000])
    # MoP hyperparams
    ap.add_argument("--mop_views", type=int, default=5)
    ap.add_argument("--mop_kernels", type=int, default=3)
    # CrossView knobs
    ap.add_argument("--xview_transpose", action="store_true")
    ap.add_argument("--xview_t1", type=float, default=0.0)
    ap.add_argument("--xview_t2", type=float, default=0.0)
    ap.add_argument("--xview_enable_prior", action="store_true")
    ap.add_argument("--xview_prior_weight", type=float, default=0.5)
    ap.add_argument(
        "--xview_anchor_mode",
        type=str,
        choices=["argmax_row_sum", "fixed", "none"],
        default="argmax_row_sum",
    )
    ap.add_argument("--xview_k_star", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/cifar100_ab3_param_budgets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    ViT_Baseline, ViT_MoP, ViTCrossView = try_import_models()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m: nn.Module):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    for target in args.targets:
        print(f"\n🎯 Target parameters: {int(target):,}")

        base_cfg, base_p = find_config_for_target(
            ViT_Baseline, n_classes=100, target_params=int(target)
        )
        mop_cfg, mop_p = find_config_for_target_under_budget(
            ViT_MoP,
            n_classes=100,
            target_params=int(target),
            max_params=base_p,
            extra_kwargs={"n_views": args.mop_views, "n_kernels": args.mop_kernels},
        )
        xview_extra = dict(
            use_transpose_cues=args.xview_transpose,
            t1=args.xview_t1,
            t2=args.xview_t2,
            enable_per_key_prior=args.xview_enable_prior,
            prior_weight=args.xview_prior_weight,
            anchor_mode=args.xview_anchor_mode,
            fixed_k_star=args.xview_k_star,
        )
        xview_cfg, xview_p = find_config_for_target(
            ViTCrossView,
            n_classes=100,
            target_params=int(target),
            extra_kwargs=xview_extra,
        )

        print(f"Baseline cfg: {base_cfg} | params={base_p:,}")
        print(
            f"MoP cfg     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={mop_p:,} (≤ base)"
        )
        print(f"XView cfg   : {xview_cfg} + {xview_extra} | params={xview_p:,}")

        accs_base: list[float] = []
        accs_mop: list[float] = []
        accs_xv: list[float] = []

        for s in args.seeds:
            print(f"\n🔬 Seed {s}")
            set_seed(s)

            base = ViT_Baseline(n_classes=100, **base_cfg).to(device)
            mop = ViT_MoP(
                n_classes=100,
                **mop_cfg,
                n_views=args.mop_views,
                n_kernels=args.mop_kernels,
            ).to(device)
            xv = ViTCrossView(n_classes=100, **xview_cfg, **xview_extra).to(device)

            print(
                f"Params | base: {count_parameters(base):,} | mop: {count_parameters(mop):,} | xview: {count_parameters(xv):,}"
            )

            opt_b, sch_b = make_opt(base)
            opt_m, sch_m = make_opt(mop)
            opt_x, sch_x = make_opt(xv)

            steps = 0
            base.train()
            mop.train()
            xv.train()
            it = iter(train_loader)

            while steps < args.steps:
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    xb, yb = next(it)
                xb, yb = xb.to(device), yb.to(device)

                opt_b.zero_grad(set_to_none=True)
                loss_b = nn.functional.cross_entropy(base(xb), yb)
                loss_b.backward()
                opt_b.step()
                sch_b.step()

                opt_m.zero_grad(set_to_none=True)
                loss_m = nn.functional.cross_entropy(mop(xb), yb)
                loss_m.backward()
                opt_m.step()
                sch_m.step()

                opt_x.zero_grad(set_to_none=True)
                loss_x = nn.functional.cross_entropy(xv(xb), yb)
                loss_x.backward()
                opt_x.step()
                sch_x.step()

                steps += 1
                if steps % max(args.eval_every, 1) == 0 or steps == 1:
                    a_b = evaluate(base, val_loader, device)
                    a_m = evaluate(mop, val_loader, device)
                    a_x = evaluate(xv, val_loader, device)
                    print(
                        f"step {steps:4d} | Lb={loss_b.item():.3f} Lm={loss_m.item():.3f} Lx={loss_x.item():.3f} | Ab={a_b:.3f} Am={a_m:.3f} Ax={a_x:.3f}"
                    )

            a_b = evaluate(base, val_loader, device)
            a_m = evaluate(mop, val_loader, device)
            a_x = evaluate(xv, val_loader, device)
            accs_base.append(a_b)
            accs_mop.append(a_m)
            accs_xv.append(a_x)
            print(f"seed {s}: base={a_b:.4f}  mop={a_m:.4f}  xview={a_x:.4f}")

        csv_path = os.path.join(args.out, f"cifar100_ab3_target_{int(target)}.csv")
        with open(csv_path, "w") as f:
            f.write(
                "seed,acc_base,acc_mop,acc_xview,params_base,params_mop,params_xview,base_cfg,mop_cfg,xview_cfg,xview_extra\n"
            )
            for i, s in enumerate(args.seeds):
                f.write(
                    f'{s},{accs_base[i]:.4f},{accs_mop[i]:.4f},{accs_xv[i]:.4f},{base_p},{mop_p},{xview_p},"{base_cfg}","{mop_cfg}","{xview_cfg}","{xview_extra}"\n'
                )

        print(
            f"\n📊 Target {int(target):,}: base={float(np.mean(accs_base)):.4f}±{float(np.std(accs_base)):.4f} | mop={float(np.mean(accs_mop)):.4f}±{float(np.std(accs_mop)):.4f} | xview={float(np.mean(accs_xv)):.4f}±{float(np.std(accs_xv)):.4f}"
        )
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
