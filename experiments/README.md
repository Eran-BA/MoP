## Overview

This directory contains runnable experiment scripts for CIFAR-10/100 using the models in `mop`. Datasets are automatically downloaded to `./data` on first run. Each script selects an available device in order: CUDA, Metal (MPS), then CPU.

## Quick start

- Multi-seed CIFAR-10 baseline vs MoP:
```bash
python experiments/cifar10_multi_seed.py --steps 1000 --seeds 0 1 2
```

- Multi-seed CIFAR-100 baseline vs MoP:
```bash
python experiments/cifar100_multi_seed.py --steps 1500 --seeds 0 1 2
```

- Param-matched A/B (Baseline vs MoP) at ~50M params:
```bash
python experiments/cifar10_ab_param_budgets.py --targets 50000000 --seeds 0 1 --steps 1000
```

- Param-matched A/B/C (Baseline vs MoP vs Cross-View Mixer) at ~50M params:
```bash
python experiments/cifar10_ab3_param_budgets.py --targets 50000000 --seeds 0 1 --steps 1000 \
  --xview_transpose --xview_t1 0.2 --xview_t2 0.2 \
  --xview_enable_prior --xview_prior_weight 0.5 --xview_anchor_mode argmax_row_sum
```

- MoP hyperparameter sweep over views/kernels (CIFAR-10):
```bash
python experiments/cifar10_mop_sweep.py --steps 1000 --seeds 0 1 --views 3 5 7 --kernels 2 3 4
```

- Dual-path gating with two-hop composition (CIFAR-10):
```bash
python experiments/cifar10_twohop_gates.py --steps 1000 --seeds 0 1 --gate_chain 1.0
```

- Cross-view mixer with per-key prior sharpening and transpose cues (CIFAR-10):
```bash
python experiments/cifar10_crossview_mixer.py --steps 1000 --seeds 0 1 \
  --enable_prior --prior_weight 0.5 --use_transpose_cues --t1 0.2 --t2 0.2
```

- Dual-path gating with two-hop composition (CIFAR-100):
```bash
python experiments/cifar100_twohop_gates.py --steps 1500 --seeds 0 1 --gate_chain 1.0
```

- Param-matched Two-hop (CIFAR-10) at ~5M params:
```bash
python experiments/cifar10_twohop_param_budgets.py --targets 5000000 --seeds 0 1 --steps 1000
```

- Param-matched Two-hop (CIFAR-100) at ~5M params:
```bash
python experiments/cifar100_twohop_param_budgets.py --targets 5000000 --seeds 0 1 --steps 1500
```

- Multi-hop gating (value-aware) (CIFAR-10):
```bash
python experiments/cifar10_multihop_gates.py --steps 1000 --seeds 0 1 --hops 3 --gate_chain 1.0
```

- Multi-hop gating (value-aware) (CIFAR-100):
```bash
python experiments/cifar100_multihop_gates.py --steps 1500 --seeds 0 1 --hops 3 --gate_chain 1.0
```

- Edgewise-gated boolean mixer (CIFAR-10):
```bash
python experiments/cifar10_edgewise_gates.py --steps 1000 --seeds 0 1
```

- Edgewise-gated boolean mixer (CIFAR-100):
```bash
python experiments/cifar100_edgewise_gates.py --steps 1500 --seeds 0 1
```

## Scripts

### `cifar10_multi_seed.py`
- Baseline `ViT_Baseline` vs `ViT_MoP` on CIFAR-10 across multiple seeds.
- Key args: `--seeds`, `--steps`, `--batch`, `--tiny`, `--out`.

### `cifar100_multi_seed.py`
- Same as above for CIFAR-100 (100 classes). Slightly deeper defaults.

### `cifar10_mop_sweep.py`
- Sweeps MoP hyperparameters `n_views` and `n_kernels` on CIFAR-10.
- Key args: `--views`, `--kernels`, alongside standard training flags.

### `cifar10_ab_param_budgets.py`
- A/B: Param-matched Baseline vs MoP at specified budgets (e.g., ~5M, ~50M).
- Performs a small grid-search over `(dim, depth, heads)` to match the target count for each model.
- Key args: `--targets`, `--mop_views`, `--mop_kernels`.

### `cifar10_ab3_param_budgets.py`
- A/B/C: Param-matched Baseline vs MoP vs Cross-View Mixer.
- Cross-View Mixer combines: cross-view binding (S12, S21), 2×2 learnable mixer, optional transpose-channel cues, and optional per-key prior sharpening.
- Key args (subset):
  - Matching: `--targets`
  - MoP: `--mop_views`, `--mop_kernels`
  - Cross-View: `--xview_transpose`, `--xview_t1`, `--xview_t2`, `--xview_enable_prior`, `--xview_prior_weight`, `--xview_anchor_mode`, `--xview_k_star`

### `cifar100_ab3_param_budgets.py`
- A/B/C: Param-matched Baseline vs MoP vs Cross-View Mixer on CIFAR-100.
- Same options as the CIFAR-10 script, with CIFAR-100 defaults (e.g., more steps).
- Key args (subset):
  - Matching: `--targets`
  - MoP: `--mop_views`, `--mop_kernels`
  - Cross-View: `--xview_transpose`, `--xview_t1`, `--xview_t2`, `--xview_enable_prior`, `--xview_prior_weight`, `--xview_anchor_mode`, `--xview_k_star`

### `cifar10_twohop_gates.py`
- Implements dual-path attention with logical gates (AND/OR/NOT) and a two-hop composition term via `A1 @ A2`.
- Tunable gate weights: `--gate_base`, `--gate_and`, `--gate_or`, `--gate_not`, `--gate_chain`; `--beta_not`.

### `cifar100_twohop_gates.py`
- CIFAR-100 version of two-hop gating with the same options and value-aware chaining.

### `cifar10_multihop_gates.py`
- Multi-hop generalization of two-hop with value-aware transport. Control hop count via `--hops` (≥2).
- Key args: `--hops`, gate weights `--gate_*`, `--beta_not`.

### `cifar100_multihop_gates.py`
- CIFAR-100 version of multi-hop with identical options.

### `cifar10_edgewise_gates.py`
- Per-edge (i,j) gates predicted by a tiny conv head over channels
  `[S1, S2, S1ᵀ, S2ᵀ, log(C→+ε), log(C←+ε)]`. Produces `g_and, g_or, g_not, g_chain ∈ [0,1]^{T×T}`.
- Includes value-aware chain transport `A1 @ (A2 @ V2)` mixed via a learnable sigmoid weight.
 - Option: `--use_k3` to add a 3×3 conv stage in the gate head.

### `cifar100_edgewise_gates.py`
- CIFAR-100 version of the edgewise-gated mixer.
 - Option: `--use_k3` to add a 3×3 conv stage in the gate head.

### `cifar10_crossview_mixer.py`
- Cross-view mixer attention:
  - Standard scores `S1=Q1K1ᵀ`, `S2=Q2K2ᵀ`, cross-view `S12=Q1K2ᵀ`, `S21=Q2K1ᵀ`.
  - Learnable 2×2 mixing of these paths; optional transpose cues `S1ᵀ`, `S2ᵀ`.
  - Per-key prior sharpening: `A_sharp(i,j) ∝ A1(i,j) · A2(k*, j)` with row normalization.
- Key args: `--use_transpose_cues`, `--t1`, `--t2`, `--enable_prior`, `--prior_weight`, `--anchor_mode`, `--k_star`.

### `cifar100_crossview_mixer.py`
- CIFAR-100 version of Cross-View Mixer with identical options.

### `cifar10_twohop_param_budgets.py`
- Param-match the Two-hop model to specified budgets (e.g., ~5M, ~50M) on CIFAR-10.
- Key args: `--targets`, two-hop gates `--gate_*`, `--beta_not`.

### `cifar100_twohop_param_budgets.py`
- Param-match the Two-hop model to specified budgets (e.g., ~5M, ~50M) on CIFAR-100.
- Key args: `--targets`, two-hop gates `--gate_*`, `--beta_not`.

### `cifar100_ab4_param_budgets.py`
- A/B/C/D: Param-matched Baseline vs MoP vs Cross-View Mixer vs Multi-Hop on CIFAR-100.
- Ensures MoP ≤ Baseline parameters (prefer ≤1% gap) and uses the same-or-smaller `(dim, depth, heads)` for MoP.
- Key args (subset):
  - Matching: `--targets`
  - MoP: `--mop_views`, `--mop_kernels`
  - Cross-View: `--xview_transpose`, `--xview_t1`, `--xview_t2`, `--xview_enable_prior`, `--xview_prior_weight`, `--xview_anchor_mode`, `--xview_k_star`
  - Multi-Hop: `--mh_hops`, `--mh_beta_not`, multi-hop gates `--mh_gate_*`

### `cifar100_ab5_param_budgets.py`
- A/B/C/D/E with selectable models (flags):
  - A: Baseline
  - B: MoP
  - C: Cross-View Mixer
  - D: Multi-Hop
  - E: Edgewise-gated mixer
- Choose via `--models` (e.g., `--models A B E`). Param matching keeps non-baselines at or under Baseline params and cfg.
- Edgewise options:
  - `--ew_views` (default 5): number of views
  - `--ew_use_k3`: enable 3×3 conv stage in the gate head
  - `--ew_share_qkv`: share QKV across views with per-view scalings (reduces params)
  - `--ew_mlp_ratio`: MLP ratio for E (lower values reduce params)
  - `--debug_budget`: print detailed search logs for parameter matching
- Automatic fallback ladder for E during matching: views → mlp_ratio → drop 3×3.
- Example (~5M):
```bash
python experiments/cifar100_ab5_param_budgets.py --targets 5000000 --seeds 0 1 --steps 1500 \
  --models A B C D E \
  --xview_transpose --xview_t1 0.2 --xview_t2 0.2 --xview_enable_prior --xview_prior_weight 0.5 \
  --xview_anchor_mode argmax_row_sum --mh_hops 3 --mh_gate_chain 1.0
```
- Example (~50M, with Edgewise-specific flags; adjust `--batch` for device memory):
```bash
python experiments/cifar100_ab5_param_budgets.py --targets 50000000 --seeds 0 1 --steps 1500 \
  --models A B C D E --batch 64 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv --debug_budget \
  --xview_transpose --xview_t1 0.2 --xview_t2 0.2 --xview_enable_prior --xview_prior_weight 0.5 \
  --xview_anchor_mode argmax_row_sum --mh_hops 3 --mh_gate_chain 1.0
```

### `imagenet_ab_param_budgets.py`
- ImageNet A/B/E: Param-matched Baseline vs MoP vs Edgewise at standard ViT scales. Includes strong augmentation (RandAug, Mixup/CutMix, Random Erasing), label smoothing, DropPath, grad clipping, and EMA.
- Key args (subset):
  - Matching: `--targets` (e.g., 86000000, 307000000, 632000000)
  - Image/patch: `--img_size`, `--patch`
  - Regularization: `--use_randaug`, `--randaug_n`, `--randaug_m`, `--random_erasing`, `--label_smoothing`, `--drop_path`, `--grad_clip`, `--ema`, `--ema_decay`
  - MoP: `--mop_views`, `--mop_kernels`
  - Edgewise: `--ew_views`, `--ew_use_k3`, `--ew_share_qkv`, `--ew_mlp_ratio`
- Examples (adjust `--batch` to your hardware):
```bash
# ViT-B/16 (~86M) and ViT-L/16 (~307M) with A/B/E
python experiments/imagenet_ab_param_budgets.py \
  --data_root /path/to/imagenet \
  --targets 86000000 307000000 \
  --models A B E \
  --img_size 224 --patch 16 \
  --steps 90000 --eval_every 1000 --batch 256 \
  --lr_large 0.001 --warmup_frac 0.1 --weight_decay 0.1 \
  --use_randaug --randaug_n 2 --randaug_m 9 --random_erasing 0.25 \
  --mixup_alpha 0.8 --cutmix_alpha 1.0 --mix_prob 0.5 \
  --drop_path 0.4 --grad_clip 1.0 --ema --ema_decay 0.9999 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv --ew_mlp_ratio 4.0

# ViT-H/14 (~632M) with A/B/E
python experiments/imagenet_ab_param_budgets.py \
  --data_root /path/to/imagenet \
  --targets 632000000 \
  --models A B E \
  --img_size 224 --patch 14 \
  --steps 90000 --eval_every 1000 --batch 256 \
  --lr_large 0.001 --warmup_frac 0.1 --weight_decay 0.1 \
  --use_randaug --randaug_n 2 --randaug_m 9 --random_erasing 0.25 \
  --mixup_alpha 0.8 --cutmix_alpha 1.0 --mix_prob 0.5 \
  --drop_path 0.4 --grad_clip 1.0 --ema --ema_decay 0.9999 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv --ew_mlp_ratio 4.0
```

### `ab5_tournament.py`
- Tournament-style A/B/C/D/E over 5 seeds (CIFAR-100 by default). Also supports plan-only mode for 1B+ sizing without training.
- Example (5M, five seeds):
```bash
python experiments/ab5_tournament.py --targets 5000000 --models A B C D E \
  --steps 1000 --eval_every 100 --batch 64 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv \
  --xview_transpose --xview_t1 0.2 --xview_t2 0.2 \
  --xview_enable_prior --xview_prior_weight 0.5 \
  --xview_anchor_mode argmax_row_sum \
  --mh_hops 3 --mh_gate_chain 1.0
```

### `ab5_paper_benchmark.py`
- Aggregates CSVs from CIFAR and ImageNet experiments to produce publication-ready Markdown and LaTeX tables.
- Example:
```bash
python experiments/ab5_paper_benchmark.py
# writes results/paper_benchmark/ab5_benchmark.md and .tex
```

### Multi-Head Attention variants (developer note)
- All A/B/C/D/E correspond to multi-head attention variants. For direct usage in code, see `mop.models.UnifiedMSA` which supports modes:
  - `A` (baseline), `B` (MoP-compatible baseline), `C` (Cross-View), `D` (Multi-Hop), `E` (Edgewise).
  - Example:
```python
from mop.models import UnifiedMSA
attn = UnifiedMSA(mode="E", dim=256, heads=4, n_views=5, use_k3=True, share_qkv=True)
```

### Edgewise (E) gate options (dense and low-rank)

- `--ew_gate_mode {dense,lowrank}`: gate implementation
- `--ew_gate_rank R`: rank for low-rank gates (default 4)
- `--ew_gate_init {neutral,and,or,not,nor,xor,chain}`: preset-biased initialization
- `--ew_variants <mode:init ...>`: run multiple Edgewise variants in the same run (e.g., `lowrank:neutral lowrank:xor`).

Multi-E in one run (example):
```bash
python experiments/cifar100_ab5_param_budgets.py --targets 5000000 --models A B E \
  --steps 30000 --eval_every 500 --batch 256 \
  --val_frac 0.1 --val_seed 0 \
  --lr 0.003 --warmup_frac 0.1 --weight_decay 0.05 --lr_e 0.0007 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv --ew_mlp_ratio 3.0 \
  --ew_variants lowrank:neutral lowrank:xor --ew_gate_rank 4 \
  --plot --out results/ab5_cifar100_5m/E_dual
```

Note: In multi-E runs each E_* is its own model instance. Early-step metrics can differ across A/B/E variants because of (a) different architectures (MoP vs Edgewise gating level), (b) different initializations (e.g., neutral vs xor/mix5 bias), (c) distinct effective parameter counts within the same budget, and (d) per-model optimization dynamics (e.g., lr_e vs base lr).

Four-model run (A, B, E, E+) with E+ initialized by a 5-preset mixture:
```bash
python experiments/cifar100_ab5_param_budgets.py --targets 5000000 --models A B E \
  --steps 30000 --eval_every 500 --batch 256 \
  --val_frac 0.1 --val_seed 0 \
  --lr 0.003 --warmup_frac 0.1 --weight_decay 0.05 --lr_e 0.0007 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv --ew_mlp_ratio 3.0 \
  --ew_variants lowrank:neutral lowrank:mix5 --ew_gate_rank 4 \
  --plot --out results/ab5_cifar100_5m/A_B_E_Emix5
```

Example (CIFAR-100 @ ~5M with low-rank XOR preset):
```bash
python experiments/cifar100_ab5_param_budgets.py --targets 5000000 --models A B E \
  --steps 30000 --eval_every 500 --batch 256 --val_frac 0.1 --val_seed 0 \
  --ew_views 5 --ew_use_k3 --ew_share_qkv \
  --ew_gate_mode lowrank --ew_gate_rank 4 --ew_gate_init xor \
  --plot --out results/ab5_cifar100_5m
```

## Results

- Each script writes CSV summaries under `results/` subdirectories (e.g., `results/cifar10_ab_param_budgets/`).
- CSVs include per-seed accuracies and, for matching scripts, the exact model configs and parameter counts.

## Tips

- If you hit out-of-memory (OOM) at higher parameter budgets, reduce `--batch` (e.g., 64 or 32).
- Use `--tiny` for a fast smoke test (~5k train / 1k test) while validating configurations.
- Training precision: scripts set `torch.set_float32_matmul_precision("high")` to speed up MPS; adjust as needed.


