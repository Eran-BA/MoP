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

## Results

- Each script writes CSV summaries under `results/` subdirectories (e.g., `results/cifar10_ab_param_budgets/`).
- CSVs include per-seed accuracies and, for matching scripts, the exact model configs and parameter counts.

## Tips

- If you hit out-of-memory (OOM) at higher parameter budgets, reduce `--batch` (e.g., 64 or 32).
- Use `--tiny` for a fast smoke test (~5k train / 1k test) while validating configurations.
- Training precision: scripts set `torch.set_float32_matmul_precision("high")` to speed up MPS; adjust as needed.


