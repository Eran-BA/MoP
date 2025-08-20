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
    """Import Baseline/MoP, Cross-View Mixer, and Multi-Hop models for CIFAR-100."""
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

    try:
        from cifar100_multihop_gates import ViTMultiHop  # type: ignore

        print("✅ Imported ViTMultiHop (CIFAR-100) from experiments")
    except Exception as e:
        raise ImportError(f"Could not import ViTMultiHop (CIFAR-100): {e}")

    return ViT_Baseline, ViT_MoP, ViTCrossView, ViTMultiHop


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
    depths_choices: Iterable[int] = (6, 8, 10, 12),
    heads_choices: Iterable[int] = (4, 6, 8),
    extra_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, int], int, bool]:
    base_dim = baseline_cfg["dim"]
    base_depth = baseline_cfg["depth"]
    base_heads = baseline_cfg["heads"]

    dims = [d for d in dims_choices if d <= base_dim]
    if base_dim not in dims:
        dims.append(base_dim)
    depths = [d for d in depths_choices if d <= base_depth]
    if base_depth not in depths:
        depths.append(base_depth)
    heads_list = [h for h in heads_choices if h <= base_heads]
    if base_heads not in heads_list:
        heads_list.append(base_heads)

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
        description="A/B/C/D test on CIFAR-100: Baseline vs MoP vs CrossView vs MultiHop at fixed parameter budgets"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--lr_large", type=float, default=1e-3, help="LR when target >= large threshold"
    )
    ap.add_argument("--large_threshold", type=int, default=50_000_000)
    ap.add_argument(
        "--warmup_frac",
        type=float,
        default=0.1,
        help="fraction of steps for linear warmup",
    )
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
    # MultiHop knobs
    ap.add_argument("--mh_hops", type=int, default=3)
    ap.add_argument("--mh_beta_not", type=float, default=0.5)
    ap.add_argument("--mh_gate_base", type=float, default=1.0)
    ap.add_argument("--mh_gate_and", type=float, default=1.0)
    ap.add_argument("--mh_gate_or", type=float, default=0.0)
    ap.add_argument("--mh_gate_not", type=float, default=0.0)
    ap.add_argument("--mh_gate_chain", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="results/cifar100_ab4_param_budgets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    ViT_Baseline, ViT_MoP, ViTCrossView, ViTMultiHop = try_import_models()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m: nn.Module, lr_value: float):
        opt = optim.AdamW(m.parameters(), lr=lr_value, weight_decay=args.weight_decay)
        warmup_steps = int(max(args.steps, 1) * max(args.warmup_frac, 0.0))
        if warmup_steps > 0:
            sched1 = optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, total_iters=warmup_steps
            )
            sched2 = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(args.steps - warmup_steps, 1)
            )
            sch = optim.lr_scheduler.SequentialLR(
                opt, schedulers=[sched1, sched2], milestones=[warmup_steps]
            )
        else:
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    for target in args.targets:
        print(f"\n🎯 Target parameters: {int(target):,}")
        lr_current = (
            args.lr if int(target) < int(args.large_threshold) else args.lr_large
        )
        print(f"Using learning rate: {lr_current} (warmup_frac={args.warmup_frac})")

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
        xview_cfg, xview_p, _ = find_mop_config_match_baseline(
            ViTCrossView,
            n_classes=100,
            target_params=int(target),
            baseline_cfg=base_cfg,
            baseline_params=base_p,
            max_ratio_diff=0.01,
            extra_kwargs=xview_extra,
        )
        mh_gates = dict(
            base=args.mh_gate_base,
            and_=args.mh_gate_and,
            or_=args.mh_gate_or,
            not_=args.mh_gate_not,
            chain=args.mh_gate_chain,
        )
        mh_extra = dict(gates=mh_gates, beta_not=args.mh_beta_not, hops=args.mh_hops)
        mh_cfg, mh_p, _ = find_mop_config_match_baseline(
            ViTMultiHop,
            n_classes=100,
            target_params=int(target),
            baseline_cfg=base_cfg,
            baseline_params=base_p,
            max_ratio_diff=0.01,
            extra_kwargs=mh_extra,
        )

        print(f"Baseline cfg: {base_cfg} | params={base_p:,}")
        # Prefer MoP ≤ Baseline and within 1% if possible, with cfg ≤ baseline
        try:
            cfg1, p1, within = find_mop_config_match_baseline(
                ViT_MoP,
                n_classes=100,
                target_params=int(target),
                baseline_cfg=base_cfg,
                baseline_params=base_p,
                max_ratio_diff=0.01,
                extra_kwargs={"n_views": args.mop_views, "n_kernels": args.mop_kernels},
            )
            mop_cfg, mop_p = cfg1, p1
            if within:
                print(
                    f"MoP cfg     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={mop_p:,} (≤ base, within 1% & cfg ≤ baseline)"
                )
            else:
                gap = (base_p - mop_p) / max(1, base_p)
                print(
                    f"MoP cfg     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={mop_p:,} (≤ base, gap {gap:.2%}, cfg ≤ baseline)"
                )
        except Exception:
            print(
                f"MoP cfg     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={mop_p:,} (≤ base)"
            )
        print(f"XView cfg   : {xview_cfg} + {xview_extra} | params={xview_p:,}")
        print(f"MultiHop cfg: {mh_cfg} + {mh_extra} | params={mh_p:,}")

        accs_base: list[float] = []
        accs_mop: list[float] = []
        accs_xv: list[float] = []
        accs_mh: list[float] = []

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
            mh = ViTMultiHop(n_classes=100, **mh_cfg, **mh_extra).to(device)

            print(
                f"Params | base: {count_parameters(base):,} | mop: {count_parameters(mop):,} | xview: {count_parameters(xv):,} | multihop: {count_parameters(mh):,}"
            )

            opt_b, sch_b = make_opt(base, lr_current)
            opt_m, sch_m = make_opt(mop, lr_current)
            opt_x, sch_x = make_opt(xv, lr_current)
            opt_h, sch_h = make_opt(mh, lr_current)

            steps = 0
            base.train()
            mop.train()
            xv.train()
            mh.train()
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

                # cross-view
                opt_x.zero_grad(set_to_none=True)
                loss_x = nn.functional.cross_entropy(xv(xb), yb)
                loss_x.backward()
                opt_x.step()
                sch_x.step()

                # multihop
                opt_h.zero_grad(set_to_none=True)
                loss_h = nn.functional.cross_entropy(mh(xb), yb)
                loss_h.backward()
                opt_h.step()
                sch_h.step()

                steps += 1
                if steps % max(args.eval_every, 1) == 0 or steps == 1:
                    a_b = evaluate(base, val_loader, device)
                    a_m = evaluate(mop, val_loader, device)
                    a_x = evaluate(xv, val_loader, device)
                    a_h = evaluate(mh, val_loader, device)
                    print(
                        f"step {steps:4d} | Lb={loss_b.item():.3f} Lm={loss_m.item():.3f} Lx={loss_x.item():.3f} Lh={loss_h.item():.3f} | "
                        f"Ab={a_b:.3f} Am={a_m:.3f} Ax={a_x:.3f} Ah={a_h:.3f}"
                    )

            # final eval per seed
            a_b = evaluate(base, val_loader, device)
            a_m = evaluate(mop, val_loader, device)
            a_x = evaluate(xv, val_loader, device)
            a_h = evaluate(mh, val_loader, device)
            accs_base.append(a_b)
            accs_mop.append(a_m)
            accs_xv.append(a_x)
            accs_mh.append(a_h)
            print(
                f"seed {s}: base={a_b:.4f}  mop={a_m:.4f}  xview={a_x:.4f}  multihop={a_h:.4f}"
            )

        # save CSV per target
        csv_path = os.path.join(args.out, f"cifar100_ab4_target_{int(target)}.csv")
        with open(csv_path, "w") as f:
            f.write(
                "seed,acc_base,acc_mop,acc_xview,acc_multihop,params_base,params_mop,params_xview,params_multihop,base_cfg,mop_cfg,xview_cfg,xview_extra,multihop_cfg,multihop_extra\n"
            )
            for i, s in enumerate(args.seeds):
                f.write(
                    f'{s},{accs_base[i]:.4f},{accs_mop[i]:.4f},{accs_xv[i]:.4f},{accs_mh[i]:.4f},{base_p},{mop_p},{xview_p},{mh_p},"{base_cfg}","{mop_cfg}","{xview_cfg}","{xview_extra}","{mh_cfg}","{mh_extra}"\n'
                )

        print(
            f"\n📊 Target {int(target):,}: base={float(np.mean(accs_base)):.4f}±{float(np.std(accs_base)):.4f} | "
            f"mop={float(np.mean(accs_mop)):.4f}±{float(np.std(accs_mop)):.4f} | "
            f"xview={float(np.mean(accs_xv)):.4f}±{float(np.std(accs_xv)):.4f} | "
            f"multihop={float(np.mean(accs_mh)):.4f}±{float(np.std(accs_mh)):.4f}"
        )
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
