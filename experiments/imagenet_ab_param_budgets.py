#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add project root and experiments dir to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


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
    # Import Edgewise from experiments implementation
    try:
        from cifar100_edgewise_gates import ViTEdgewise  # type: ignore

        print("✅ Imported ViTEdgewise from experiments")
    except Exception as e:
        raise ImportError(f"Could not import ViTEdgewise: {e}")
    return ViT_Baseline, ViT_MoP, ViTEdgewise


def get_loaders(
    data_root: str,
    batch: int = 256,
    tiny: bool = False,
    workers: int = 8,
    img_size: int = 224,
    use_randaug: bool = False,
    randaug_n: int = 2,
    randaug_m: int = 9,
    random_erasing: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    train_transforms: List[transforms.Transform] = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaug:
        # torchvision RandAugment: num_ops, magnitude
        train_transforms.append(
            transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m)
        )
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )
    if random_erasing and random_erasing > 0.0:
        train_transforms.append(
            transforms.RandomErasing(p=float(random_erasing), inplace=True)
        )
    train_tf = transforms.Compose(train_transforms)
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )

    # Use ImageFolder with standard structure: train/ and val/
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        raise FileNotFoundError(
            f"ImageNet folders not found under {data_root}. Expecting 'train/' and 'val/' subfolders."
        )

    train = datasets.ImageFolder(train_dir, transform=train_tf)
    val = datasets.ImageFolder(val_dir, transform=val_tf)
    if tiny:
        train = Subset(train, range(50_000))
        val = Subset(val, range(5_000))
    train_loader = DataLoader(
        train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
    dims: Iterable[int] = (192, 224, 256, 320, 384, 448, 512, 640, 768, 1024, 1280),
    depths: Iterable[int] = (8, 10, 12, 16, 24, 32),
    heads_list: Iterable[int] = (3, 4, 6, 8, 12, 16),
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
        192,
        224,
        256,
        320,
        384,
        448,
        512,
        640,
        768,
        1024,
        1280,
    ),
    depths_choices: Iterable[int] = (8, 10, 12, 16, 24, 32),
    heads_choices: Iterable[int] = (3, 4, 6, 8, 12, 16),
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
            "ImageNet A/B at fixed parameter budgets: A=Baseline ViT, B=MoP (param-matched)"
        )
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("IMAGENET_ROOT", "./data/imagenet"),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_large", type=float, default=1e-3)
    ap.add_argument("--large_threshold", type=int, default=100_000_000)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--targets", type=int, nargs="+", default=[50_000_000, 300_000_000])
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["A", "B", "E"],
        default=["A", "B"],
        help="Which models to run: A=Baseline, B=MoP, E=Edgewise",
    )
    # MoP
    ap.add_argument("--mop_views", type=int, default=5)
    ap.add_argument("--mop_kernels", type=int, default=3)
    # ViT image/patch
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--drop_path", type=float, default=0.4)
    # Edgewise
    ap.add_argument("--ew_beta_not", type=float, default=0.5)
    ap.add_argument("--ew_use_k3", action="store_true")
    ap.add_argument("--ew_views", type=int, default=5)
    ap.add_argument("--ew_share_qkv", action="store_true")
    ap.add_argument("--ew_mlp_ratio", type=float, default=4.0)
    ap.add_argument("--ew_gate_mode", type=str, default="dense", choices=["dense", "lowrank"])  # noqa: E501
    ap.add_argument("--ew_gate_rank", type=int, default=4)
    ap.add_argument(
        "--ew_gate_init",
        type=str,
        default="neutral",
        choices=["neutral", "and", "or", "not", "nor", "xor", "chain"],
    )
    # Advanced regularization/augmentation
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--use_randaug", action="store_true")
    ap.add_argument("--randaug_n", type=int, default=2)
    ap.add_argument("--randaug_m", type=int, default=9)
    ap.add_argument("--random_erasing", type=float, default=0.25)
    ap.add_argument("--mixup_alpha", type=float, default=0.8)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument(
        "--mix_prob",
        type=float,
        default=0.5,
        help="probability to use mixup when both mixup and cutmix enabled",
    )
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--out", type=str, default="results/imagenet_ab_param_budgets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    ViT_Baseline, ViT_MoP, ViTEdgewise = try_import_models()
    train_loader, val_loader = get_loaders(
        args.data_root,
        batch=args.batch,
        tiny=args.tiny,
        workers=8,
        img_size=args.img_size,
        use_randaug=args.use_randaug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        random_erasing=args.random_erasing,
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

    extra = {
        "patch": args.patch,
        "img_size": args.img_size,
        "drop_path": args.drop_path,
    }

    # Mixup/CutMix helpers
    def rand_bbox(W: int, H: int, lam: float):
        cut_rat = float(np.sqrt(1.0 - lam))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        # uniform center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        return x1, y1, x2, y2

    def apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
        if alpha <= 0:
            return x, y, y, 1.0, False
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, float(lam), True

    def apply_cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float):
        if alpha <= 0:
            return x, y, y, 1.0, False
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)
        y_a, y_b = y, y[index]
        _, _, H, W = x.shape
        x1, y1, x2, y2 = rand_bbox(W, H, lam)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return x, y_a, y_b, float(lam), True

    for target in args.targets:
        print(f"\n🎯 Target parameters: {int(target):,}")
        lr_current = (
            args.lr if int(target) < int(args.large_threshold) else args.lr_large
        )
        print(f"Using learning rate: {lr_current} (warmup_frac={args.warmup_frac})")

        base_cfg, base_p = find_config_for_target(
            ViT_Baseline, n_classes=1000, target_params=int(target), extra_kwargs=extra
        )
        mop_cfg, mop_p, within = find_model_config_match_baseline(
            ViT_MoP,
            n_classes=1000,
            target_params=int(target),
            baseline_cfg=base_cfg,
            baseline_params=base_p,
            max_ratio_diff=0.01,
            extra_kwargs={
                **extra,
                "n_views": args.mop_views,
                "n_kernels": args.mop_kernels,
            },
        )

        print(f"Baseline cfg: {base_cfg} + {extra} | params={base_p:,}")
        print(
            f"MoP cfg     : {mop_cfg} + {{'n_views': {args.mop_views}, 'n_kernels': {args.mop_kernels}, **extra}} | params={mop_p:,} | within1%={within}"
        )
        # Edgewise plan
        ew_cfg = None
        ew_p = None
        ew_within = False
        if "E" in args.models:
            num_tokens = (int(args.img_size) // int(args.patch)) ** 2
            ew_extra = {
                **extra,
                "beta_not": args.ew_beta_not,
                "use_k3": args.ew_use_k3,
                "n_views": int(args.ew_views),
                "share_qkv": args.ew_share_qkv,
                "mlp_ratio": float(args.ew_mlp_ratio),
                "gate_mode": args.ew_gate_mode,
                "gate_rank": int(args.ew_gate_rank),
                "gate_init": str(args.ew_gate_init),
                "num_tokens": int(num_tokens),
            }
            ew_cfg, ew_p, ew_within = find_model_config_match_baseline(
                ViTEdgewise,
                n_classes=1000,
                target_params=int(target),
                baseline_cfg=base_cfg,
                baseline_params=base_p,
                max_ratio_diff=0.01,
                dims_choices=(192, 224, 256, 320, 384, 448, 512, 640, 768, 1024, 1280),
                depths_choices=(8, 10, 12, 16, 24, 32),
                heads_choices=(4, 6, 8, 12, 16),
                extra_kwargs=ew_extra,
            )
            print(
                f"Edgewise cfg: {ew_cfg} + {{'n_views': {args.ew_views}, 'share_qkv': {args.ew_share_qkv}, 'use_k3': {args.ew_use_k3}, 'mlp_ratio': {args.ew_mlp_ratio}, **extra}} | params={ew_p:,} | within1%={ew_within}"
            )

        # Accumulator
        accs: Dict[str, List[float]] = {k: [] for k in ["A"] + args.models}

        for s in args.seeds:
            print(f"\n🔬 Seed {s}")
            set_seed(s)

            models: Dict[str, nn.Module] = {}
            models["A"] = ViT_Baseline(n_classes=1000, **base_cfg, **extra).to(device)
            if "B" in args.models:
                models["B"] = ViT_MoP(
                    n_classes=1000,
                    **mop_cfg,
                    **extra,
                    n_views=args.mop_views,
                    n_kernels=args.mop_kernels,
                ).to(device)
            if "E" in args.models and ew_cfg is not None:
                num_tokens = (int(args.img_size) // int(args.patch)) ** 2
                models["E"] = ViTEdgewise(
                    n_classes=1000,
                    **ew_cfg,
                    beta_not=args.ew_beta_not,
                    use_k3=bool(args.ew_use_k3),
                    n_views=int(args.ew_views),
                    share_qkv=args.ew_share_qkv,
                    mlp_ratio=float(args.ew_mlp_ratio),
                    patch=int(args.patch),
                    num_tokens=int(num_tokens),
                    drop_path=float(args.drop_path),
                    gate_mode=args.ew_gate_mode,
                    gate_rank=int(args.ew_gate_rank),
                    gate_init=str(args.ew_gate_init),
                ).to(device)

            # Optional EMA models
            models_ema: Optional[Dict[str, nn.Module]] = None
            if args.ema:
                models_ema = {}
                for key, m in models.items():
                    m_ema = type(m)(
                        **{
                            k: v
                            for k, v in m.__dict__.items()
                            if k.startswith("_") is False
                        }
                    )  # placeholder
                    # safer: deep copy state
                    m_ema = type(m)(
                        n_classes=1000, **(base_cfg if key == "A" else mop_cfg), **extra
                    ).to(device)
                    m_ema.load_state_dict(m.state_dict())
                    for p in m_ema.parameters():
                        p.requires_grad = False
                    models_ema[key] = m_ema

            # Params line
            params_line = f"Params | A(base): {count_parameters(models['A']):,}"
            if "B" in models:
                params_line += f" | B: {count_parameters(models['B']):,}"
            if "E" in models:
                params_line += f" | E: {count_parameters(models['E']):,}"
            print(params_line)

            # Opts
            opts: Dict[str, Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]] = (
                {}
            )
            for key, m in models.items():
                opts[key] = make_opt(m, lr_current)

            # Criterion
            ce = nn.CrossEntropyLoss(
                label_smoothing=float(max(args.label_smoothing, 0.0))
            )

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
                xb, yb = xb.to(device, non_blocking=True), yb.to(
                    device, non_blocking=True
                )

                # forward/backward each selected model
                losses: Dict[str, torch.Tensor] = {}
                for key, m in models.items():
                    opt, _ = opts[key]
                    opt.zero_grad(set_to_none=True)
                    x_in = xb
                    use_mix = args.mixup_alpha > 0 and (
                        args.cutmix_alpha <= 0 or np.random.rand() < args.mix_prob
                    )
                    use_cut = args.cutmix_alpha > 0 and (
                        args.mixup_alpha <= 0 or not use_mix
                    )
                    if use_mix:
                        x_m, ya, yb_, lam, used = apply_mixup(
                            x_in.clone(), yb, args.mixup_alpha
                        )
                        out = m(x_m)
                        loss = lam * ce(out, ya) + (1.0 - lam) * ce(out, yb_)
                    elif use_cut:
                        x_c, ya, yb_, lam, used = apply_cutmix(
                            x_in.clone(), yb, args.cutmix_alpha
                        )
                        out = m(x_c)
                        loss = lam * ce(out, ya) + (1.0 - lam) * ce(out, yb_)
                    else:
                        out = m(x_in)
                        loss = ce(out, yb)

                    loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            m.parameters(), float(args.grad_clip)
                        )
                    opt.step()
                    losses[key] = loss
                for key, (_opt, sch) in opts.items():
                    sch.step()

                # EMA update
                if models_ema is not None:
                    d = float(args.ema_decay)
                    for key, m in models.items():
                        m_ema = models_ema[key]
                        with torch.no_grad():
                            for p_ema, p in zip(m_ema.parameters(), m.parameters()):
                                p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

                steps += 1
                if steps % max(args.eval_every, 1) == 0 or steps == 1:
                    acc_report = []
                    for key, m in models.items():
                        eval_model = models_ema[key] if models_ema is not None else m
                        acc = evaluate(eval_model, val_loader, device)
                        acc_report.append((key, acc))
                    loss_str = " ".join(
                        [f"L{key}={losses[key].item():.3f}" for key in losses]
                    )
                    acc_str = " ".join([f"A{key}={acc:.3f}" for key, acc in acc_report])
                    print(f"step {steps:5d} | {loss_str} | {acc_str}")

            # final eval per seed
            for key, m in models.items():
                eval_model = models_ema[key] if models_ema is not None else m
                a = evaluate(eval_model, val_loader, device)
                accs[key].append(a)
            print("seed", s, " " + " ".join([f"{k}={accs[k][-1]:.4f}" for k in accs]))

        # Save CSV per target
        csv_path = os.path.join(args.out, f"imagenet_ab_target_{int(target)}.csv")
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
