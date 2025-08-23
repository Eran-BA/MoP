#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make project and experiments importable
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
    """Import Baseline, MoP and experiment variants for CIFAR-100."""
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

    from cifar100_crossview_mixer import ViTCrossView  # type: ignore
    from cifar100_edgewise_gates import ViTEdgewise  # type: ignore
    from cifar100_multihop_gates import ViTMultiHop  # type: ignore

    print("✅ Imported ViTCrossView, ViTMultiHop, ViTEdgewise (CIFAR-100)")

    return ViT_Baseline, ViT_MoP, ViTCrossView, ViTMultiHop, ViTEdgewise


def get_loaders(
    batch: int = 256,
    tiny: bool = False,
    workers: int = 2,
    val_frac: float = 0.1,
    val_seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    # Two copies of training set with different transforms so validation uses eval tfms
    train_full_aug = datasets.CIFAR100(
        "./data", train=True, download=True, transform=tfm_train
    )
    train_full_eval = datasets.CIFAR100(
        "./data", train=True, download=False, transform=tfm_test
    )
    test = datasets.CIFAR100("./data", train=False, download=True, transform=tfm_test)

    # Deterministic split
    num_train = len(train_full_aug)
    n_val = int(max(1, min(num_train - 1, round(float(val_frac) * num_train))))
    rng = np.random.RandomState(int(val_seed))
    idx = rng.permutation(num_train)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train = torch.utils.data.Subset(train_full_aug, train_idx)
    val = torch.utils.data.Subset(train_full_eval, val_idx)

    if tiny:
        train = torch.utils.data.Subset(train, range(min(5000, len(train_idx))))
        val = torch.utils.data.Subset(val, range(min(1000, len(val_idx))))
        test = torch.utils.data.Subset(test, range(1000))

    train_loader = DataLoader(
        train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=False
    )
    val_loader = DataLoader(
        val, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False
    )
    test_loader = DataLoader(
        test, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False
    )
    return train_loader, val_loader, test_loader


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


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Tournament A/B/C/D/E on CIFAR-100 with 5 seeds. Includes plan-only mode for 1B+."
        )
    )
    ap.add_argument("--targets", type=int, nargs="+", default=[5_000_000])
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["A", "B", "C", "D", "E"],
        default=["A", "B", "C", "D", "E"],
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_large", type=float, default=1e-3)
    ap.add_argument("--large_threshold", type=int, default=50_000_000)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_seed", type=int, default=0)
    # CrossView flags
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
    # MultiHop flags
    ap.add_argument("--mh_hops", type=int, default=3)
    ap.add_argument("--mh_beta_not", type=float, default=0.5)
    ap.add_argument("--mh_gate_chain", type=float, default=1.0)
    # MoP flags
    ap.add_argument("--mop_views", type=int, default=5)
    ap.add_argument("--mop_kernels", type=int, default=3)
    # Edgewise flags
    ap.add_argument("--ew_beta_not", type=float, default=0.5)
    ap.add_argument("--ew_views", type=int, default=5)
    ap.add_argument("--ew_mlp_ratio", type=float, default=4.0)
    ap.add_argument("--ew_use_k3", action="store_true")
    ap.add_argument("--ew_share_qkv", action="store_true")
    ap.add_argument("--ew_gate_mode", type=str, default="dense", choices=["dense", "lowrank"])
    ap.add_argument("--ew_gate_rank", type=int, default=4)
    ap.add_argument(
        "--ew_gate_init",
        type=str,
        default="neutral",
        choices=["neutral", "and", "or", "not", "nor", "xor", "chain"],
    )
    # Per-model LR for E
    ap.add_argument("--lr_e", type=float, default=None)
    ap.add_argument("--lr_mult_e", type=float, default=1.0)
    # Planning mode for very large budgets
    ap.add_argument(
        "--plan_only",
        action="store_true",
        help="Do not train; print per-model planned configs and exit.",
    )
    ap.add_argument("--out", type=str, default="results/ab5_tournament")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    ViT_Baseline, ViT_MoP, ViTCrossView, ViTMultiHop, ViTEdgewise = try_import_models()
    train_loader, val_loader, test_loader = get_loaders(
        args.batch,
        tiny=args.tiny,
        workers=2,
        val_frac=float(args.val_frac),
        val_seed=int(args.val_seed),
    )

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

    # Lightweight heuristic planner for 1B+ (no instantiation). Purely indicative.
    def heuristic_plan_1b(target_params: int) -> Dict[str, Dict[str, int]]:
        plans: Dict[str, Dict[str, int]] = {}
        if target_params >= 1_000_000_000:
            # Suggestion inspired by large ViT configs (no guarantee of exact param match)
            plans["A"] = {"dim": 1280, "depth": 32, "heads": 16}
            plans["B"] = {"dim": 1280, "depth": 32, "heads": 16}
            plans["C"] = {"dim": 1280, "depth": 32, "heads": 16}
            plans["D"] = {"dim": 1152, "depth": 36, "heads": 16}
            plans["E"] = {"dim": 1024, "depth": 40, "heads": 16}
        else:
            plans["A"] = {"dim": 768, "depth": 12, "heads": 12}
            plans["B"] = {"dim": 768, "depth": 12, "heads": 12}
            plans["C"] = {"dim": 768, "depth": 12, "heads": 12}
            plans["D"] = {"dim": 640, "depth": 16, "heads": 10}
            plans["E"] = {"dim": 640, "depth": 16, "heads": 10}
        return plans

    # Use param-matching helpers from CIFAR ab5 runner so tournament hits the requested budget
    try:
        from cifar100_ab5_param_budgets import (
            find_config_for_target, find_model_config_match_baseline)
    except Exception as e:
        raise ImportError(
            f"Param-matching helpers not available: {e}. Ensure 'experiments/cifar100_ab5_param_budgets.py' exists."
        )

    for target in args.targets:
        print(f"\n🎯 Tournament target parameters: {int(target):,}")
        lr_current = (
            args.lr if int(target) < int(args.large_threshold) else args.lr_large
        )
        print(f"Using base LR: {lr_current} (warmup_frac={args.warmup_frac})")

        if args.plan_only:
            plan = heuristic_plan_1b(int(target))
            print("Planned per-model configs (heuristic, no instantiation):")
            for k in args.models:
                print(f"  {k}: {plan.get(k, {})}")
            # Write a small plan file
            with open(
                os.path.join(args.out, f"tournament_plan_{int(target)}.txt"), "w"
            ) as f:
                for k in args.models:
                    f.write(f"{k}: {plan.get(k, {})}\n")
            continue

        # Compute param-matched baseline and per-model configs for this target
        base_cfg, base_p = find_config_for_target(
            ViT_Baseline, n_classes=100, target_params=int(target)
        )

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
        if "E" in args.models:
            # Ladder similar to ab5 to help E fit under baseline
            ew_cfg = None
            ew_p = None
            try_views = list(range(int(args.ew_views), 1, -1))
            mlp_order = [args.ew_mlp_ratio, 4.0, 3.0, 2.0, 1.5, 1.0]
            seen = set()
            mlp_try: List[float] = []
            for r in mlp_order:
                if r > 0 and r not in seen:
                    mlp_try.append(r)
                    seen.add(r)
            use_k3_try = (
                [bool(args.ew_use_k3), False] if args.ew_use_k3 else [False, True]
            )
            done = False
            for v in try_views:
                if done:
                    break
                for r in mlp_try:
                    if done:
                        break
                    for use_k3_flag in use_k3_try:
                        try:
                            xkwargs = {
                                "beta_not": args.ew_beta_not,
                                "use_k3": bool(use_k3_flag),
                                "n_views": int(v),
                                "share_qkv": args.ew_share_qkv,
                                "mlp_ratio": float(r),
                            }
                            cfg_e, p_e, _within = find_model_config_match_baseline(
                                ViTEdgewise,
                                n_classes=100,
                                target_params=int(target),
                                baseline_cfg=base_cfg,
                                baseline_params=base_p,
                                max_ratio_diff=0.01,
                                extra_kwargs=xkwargs,
                            )
                            ew_cfg, ew_p = cfg_e, p_e
                            ew_cfg["_ew_views"] = int(v)
                            ew_cfg["_ew_mlp_ratio"] = float(r)
                            ew_cfg["_ew_use_k3"] = bool(use_k3_flag)
                            done = True
                            break
                        except Exception:
                            continue
            if ew_cfg is None:
                raise RuntimeError(
                    "Edgewise (E) could not fit under baseline in tournament. Try reducing --ew_views."
                )
            cfgs["E"] = (ew_cfg, ew_p)

        # Models dict per seed
        accs: Dict[str, List[float]] = {k: [] for k in ["A"] + args.models}

        for s in args.seeds:
            print(f"\n🔬 Seed {s}")
            set_seed(s)

            models: Dict[str, nn.Module] = {}
            # A (param-matched)
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
                models["C"] = ViTCrossView(
                    n_classes=100,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    use_transpose_cues=args.xview_transpose,
                    t1=args.xview_t1,
                    t2=args.xview_t2,
                    enable_per_key_prior=args.xview_enable_prior,
                    prior_weight=args.xview_prior_weight,
                    anchor_mode=args.xview_anchor_mode,
                    fixed_k_star=args.xview_k_star,
                ).to(device)
            # D
            if "D" in args.models:
                models["D"] = ViTMultiHop(
                    n_classes=100,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    gates=dict(
                        base=1.0, and_=1.0, or_=0.0, not_=0.0, chain=args.mh_gate_chain
                    ),
                    beta_not=args.mh_beta_not,
                    hops=args.mh_hops,
                ).to(device)
            # E (Edgewise) param-matched with extras
            if "E" in args.models:
                cfg_e = cfgs["E"][0]
                chosen_ew_views = cfg_e.get("_ew_views", args.ew_views)
                chosen_ew_mlp = cfg_e.get("_ew_mlp_ratio", args.ew_mlp_ratio)
                chosen_ew_k3 = cfg_e.get("_ew_use_k3", args.ew_use_k3)
                base_kwargs = {k: v for k, v in cfg_e.items() if not k.startswith("_")}
                models["E"] = ViTEdgewise(
                    n_classes=100,
                    **base_kwargs,
                    beta_not=args.ew_beta_not,
                    use_k3=bool(chosen_ew_k3),
                    n_views=int(chosen_ew_views),
                    share_qkv=args.ew_share_qkv,
                    mlp_ratio=float(chosen_ew_mlp),
                    gate_mode=args.ew_gate_mode,
                    gate_rank=int(args.ew_gate_rank),
                    gate_init=str(args.ew_gate_init),
                ).to(device)

            # Report params
            params_line = f"Params | A(base): {count_parameters(models['A']):,}"
            for key in ["B", "C", "D", "E"]:
                if key in models:
                    params_line += f" | {key}: {count_parameters(models[key]):,}"
            print(params_line)

            # Opts with per-model LR for E
            opts: Dict[str, Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]] = (
                {}
            )
            lr_current_seed = lr_current
            for key, m in models.items():
                lr_for_model = lr_current_seed
                if key == "E":
                    if args.lr_e is not None and args.lr_e > 0:
                        lr_for_model = float(args.lr_e)
                    else:
                        lr_for_model = float(lr_current_seed) * float(args.lr_mult_e)
                opts[key] = make_opt(m, lr_for_model)

            # Train loop (shared batches across models)
            for m in models.values():
                m.train()
            it = iter(train_loader)
            steps = 0
            while steps < args.steps:
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    xb, yb = next(it)
                xb, yb = xb.to(device), yb.to(device)

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

            # Final eval per seed (validation)
            for key, m in models.items():
                a = evaluate(m, val_loader, device)
                accs[key].append(a)
            print("seed", s, " " + " ".join([f"{k}={accs[k][-1]:.4f}" for k in accs]))

        # Test-set evaluation (last seed models)
        print("\n📏 Test-set evaluation (last seed models):")
        test_acc_report = []
        for key, m in models.items():
            a_test = evaluate(m, test_loader, device)
            test_acc_report.append((key, a_test))
        print(" ".join([f"T{key}={acc:.4f}" for key, acc in test_acc_report]))

        # Save CSV per target
        csv_path = os.path.join(args.out, f"tournament_target_{int(target)}.csv")
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

        if args.plot:
            # Simple bar of mean val acc per model
            labels = list(accs.keys())
            means = [float(np.mean(accs[k])) for k in labels]
            plt.figure(figsize=(6, 4))
            plt.bar(labels, means)
            plt.ylim(0, 1)
            plt.ylabel("Val Accuracy (mean over seeds)")
            plt.title(f"Tournament CIFAR-100 @ {int(target):,}")
            os.makedirs(args.out, exist_ok=True)
            out_path = os.path.join(args.out, f"tournament_{int(target)}_val_bar.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()


if __name__ == "__main__":
    main()
