#!/usr/bin/env python3
import argparse
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

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
    """Import Baseline, MoP, Cross-View, Multi-Hop, and Edgewise models (CIFAR-100)."""
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

    try:
        from cifar100_edgewise_gates import ViTEdgewise  # type: ignore

        print("✅ Imported ViTEdgewise (CIFAR-100) from experiments")
    except Exception as e:
        raise ImportError(f"Could not import ViTEdgewise (CIFAR-100): {e}")

    return ViT_Baseline, ViT_MoP, ViTCrossView, ViTMultiHop, ViTEdgewise


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


def find_model_config_match_baseline(
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
    raise RuntimeError("Could not find configuration under baseline budget.")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "A/B/C/D/E on CIFAR-100 at fixed parameter budgets: "
            "A=Baseline, B=MoP, C=CrossView, D=MultiHop, E=Edgewise"
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_large", type=float, default=1e-3)
    ap.add_argument("--large_threshold", type=int, default=50_000_000)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--targets", type=int, nargs="+", default=[5_000_000, 50_000_000])
    # Model selection
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["A", "B", "C", "D", "E"],
        default=["A", "B", "C", "D", "E"],
        help="Which models to run: A=Baseline, B=MoP, C=CrossView, D=MultiHop, E=Edgewise",
    )
    # MoP
    ap.add_argument("--mop_views", type=int, default=5)
    ap.add_argument("--mop_kernels", type=int, default=3)
    # CrossView
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
    # MultiHop
    ap.add_argument("--mh_hops", type=int, default=3)
    ap.add_argument("--mh_beta_not", type=float, default=0.5)
    ap.add_argument("--mh_gate_chain", type=float, default=1.0)
    # Edgewise
    ap.add_argument("--ew_beta_not", type=float, default=0.5)
    ap.add_argument(
        "--ew_use_k3", action="store_true", help="use 3x3 conv stage in edgewise head"
    )
    ap.add_argument(
        "--ew_views", type=int, default=5, help="number of views in Edgewise model"
    )
    ap.add_argument("--out", type=str, default="results/cifar100_ab5_param_budgets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    ViT_Baseline, ViT_MoP, ViTCrossView, ViTMultiHop, ViTEdgewise = try_import_models()
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
                opt, [sched1, sched2], milestones=[warmup_steps]
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

        # Prepare per-model configs if selected
        cfgs: Dict[str, Tuple[Dict[str, int], int]] = {}

        if "B" in args.models:
            cfgs["B"] = find_model_config_match_baseline(
                ViT_MoP,
                n_classes=100,
                target_params=int(target),
                baseline_cfg=base_cfg,
                baseline_params=base_p,
                max_ratio_diff=0.01,
                extra_kwargs={"n_views": args.mop_views, "n_kernels": args.mop_kernels},
            )[:2]
        if "C" in args.models:
            xview_extra = dict(
                use_transpose_cues=args.xview_transpose,
                t1=args.xview_t1,
                t2=args.xview_t2,
                enable_per_key_prior=args.xview_enable_prior,
                prior_weight=args.xview_prior_weight,
                anchor_mode=args.xview_anchor_mode,
                fixed_k_star=args.xview_k_star,
            )
            cfgs["C"] = find_model_config_match_baseline(
                ViTCrossView,
                n_classes=100,
                target_params=int(target),
                baseline_cfg=base_cfg,
                baseline_params=base_p,
                max_ratio_diff=0.01,
                extra_kwargs=xview_extra,
            )[:2]
        if "D" in args.models:
            mh_extra = dict(
                gates=dict(
                    base=1.0, and_=1.0, or_=0.0, not_=0.0, chain=args.mh_gate_chain
                ),
                beta_not=args.mh_beta_not,
                hops=args.mh_hops,
            )
            cfgs["D"] = find_model_config_match_baseline(
                ViTMultiHop,
                n_classes=100,
                target_params=int(target),
                baseline_cfg=base_cfg,
                baseline_params=base_p,
                max_ratio_diff=0.01,
                extra_kwargs=mh_extra,
            )[:2]
        if "E" in args.models:
            cfgs["E"] = find_model_config_match_baseline(
                ViTEdgewise,
                n_classes=100,
                target_params=int(target),
                baseline_cfg=base_cfg,
                baseline_params=base_p,
                max_ratio_diff=0.01,
                extra_kwargs={
                    "beta_not": args.ew_beta_not,
                    "use_k3": args.ew_use_k3,
                    "n_views": args.ew_views,
                },
            )[:2]

        print(f"Baseline cfg: {base_cfg} | params={base_p:,}")
        if "B" in cfgs:
            print(
                f"MoP cfg     : {cfgs['B'][0]} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}}} | params={cfgs['B'][1]:,}"
            )
        if "C" in cfgs:
            print(f"XView cfg   : {cfgs['C'][0]} | params={cfgs['C'][1]:,}")
        if "D" in cfgs:
            print(f"MultiHop cfg: {cfgs['D'][0]} | params={cfgs['D'][1]:,}")
        if "E" in cfgs:
            print(f"Edgewise cfg: {cfgs['E'][0]} | params={cfgs['E'][1]:,}")

        # Accumulator
        accs: Dict[str, List[float]] = {
            k: [] for k in ["A", "B", "C", "D", "E"] if k in (["A"] + args.models)
        }

        for s in args.seeds:
            print(f"\n🔬 Seed {s}")
            set_seed(s)

            models: Dict[str, nn.Module] = {}
            # A
            models["A"] = ViT_Baseline(n_classes=100, **base_cfg).to(device)
            # B
            if "B" in args.models:
                models["B"] = ViT_MoP(
                    n_classes=100,
                    **cfgs["B"][0],
                    n_views=args.mop_views,
                    n_kernels=args.mop_kernels,
                ).to(device)
            # C
            if "C" in args.models:
                xview_extra = dict(
                    use_transpose_cues=args.xview_transpose,
                    t1=args.xview_t1,
                    t2=args.xview_t2,
                    enable_per_key_prior=args.xview_enable_prior,
                    prior_weight=args.xview_prior_weight,
                    anchor_mode=args.xview_anchor_mode,
                    fixed_k_star=args.xview_k_star,
                )
                models["C"] = ViTCrossView(
                    n_classes=100, **cfgs["C"][0], **xview_extra
                ).to(device)
            # D
            if "D" in args.models:
                mh_extra = dict(
                    gates=dict(
                        base=1.0, and_=1.0, or_=0.0, not_=0.0, chain=args.mh_gate_chain
                    ),
                    beta_not=args.mh_beta_not,
                    hops=args.mh_hops,
                )
                models["D"] = ViTMultiHop(n_classes=100, **cfgs["D"][0], **mh_extra).to(
                    device
                )
            # E
            if "E" in args.models:
                models["E"] = ViTEdgewise(
                    n_classes=100,
                    **cfgs["E"][0],
                    beta_not=args.ew_beta_not,
                    use_k3=args.ew_use_k3,
                    n_views=args.ew_views,
                ).to(device)

            # Params line
            params_line = f"Params | A(base): {count_parameters(models['A']):,}"
            for key in ["B", "C", "D", "E"]:
                if key in models:
                    params_line += f" | {key}: {count_parameters(models[key]):,}"
            print(params_line)

            # Opts
            opts: Dict[str, Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]] = (
                {}
            )
            for key, m in models.items():
                opts[key] = make_opt(m, lr_current)

            steps = 0
            for m in models.values():
                m.train()
            it = iter(train_loader)

            while steps < args.steps:
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    xb, yb = next(it)
                xb, yb = xb.to(device), yb.to(device)

                # forward/backward each selected model
                losses: Dict[str, torch.Tensor] = {}
                for key, m in models.items():
                    opt, _ = opts[key]
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(m(xb), yb)
                    loss.backward()
                    opt.step()
                    losses[key] = loss
                for key, (_opt, sch) in opts.items():
                    sch.step()

                steps += 1
                if steps % max(args.eval_every, 1) == 0 or steps == 1:
                    acc_report = []
                    for key, m in models.items():
                        acc = evaluate(m, val_loader, device)
                        acc_report.append((key, acc))
                    loss_str = " ".join(
                        [f"L{key}={losses[key].item():.3f}" for key in losses]
                    )
                    acc_str = " ".join([f"A{key}={acc:.3f}" for key, acc in acc_report])
                    print(f"step {steps:4d} | {loss_str} | {acc_str}")

            # final eval per seed
            for key, m in models.items():
                a = evaluate(m, val_loader, device)
                accs[key].append(a)
            print("seed", s, " " + " ".join([f"{k}={accs[k][-1]:.4f}" for k in accs]))

        # Save CSV per target
        csv_path = os.path.join(args.out, f"cifar100_ab5_target_{int(target)}.csv")
        enabled = ["A"] + args.models
        headers = ["seed"] + [f"acc_{k}" for k in enabled]
        with open(csv_path, "w") as f:
            f.write(",".join(headers) + "\n")
            for i, s in enumerate(args.seeds):
                row = [str(s)]
                for k in enabled:
                    row.append(f"{accs[k][i]:.4f}")
                f.write(",".join(row) + "\n")
        print(
            "\n"
            + " ".join(
                [
                    f"{k}={float(np.mean(v)):.4f}±{float(np.std(v)):.4f}"
                    for k, v in accs.items()
                ]
            )
        )
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
