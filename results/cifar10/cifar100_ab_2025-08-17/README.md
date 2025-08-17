# CIFAR-100 A/B — Baseline vs MoP

This folder contains aggregated A/B test results comparing the Baseline model vs **MoP** on CIFAR-100.

## Headline
- **Baseline mean acc:** n/a
- **MoP mean acc:** n/a
- **Δ (B–A):** n/a
- **95% CI:** n/a  
  _Note_: If only one seed was run, the CI shown reflects example-level uncertainty; across-seed CI collapses.

## Files
- `cifar100_ab_acc.png` — bar chart used in the README/PR.
- `aggregate_summary.json` / `aggregate_summary.csv` — summary statistics.
- `multi_seed_results.csv` — per-seed summary table (if >1 seed).
- `history_A.csv` / `history_B.csv` — item-level correctness (optional).
- `plot_ab.py` — script to regenerate the chart.
- `requirements.txt` — minimal deps for the plot script.

## Reproduce
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python plot_ab.py
```

## How to add to the repo
We suggest committing this folder under `results/` with a date-stamped run id:
```
results/
└── cifar100_ab_2025-08-17/
    ├── cifar100_ab_acc.png
    ├── aggregate_summary.json
    ├── aggregate_summary.csv
    ├── multi_seed_results.csv
    ├── history_A.csv
    ├── history_B.csv
    ├── plot_ab.py
    ├── requirements.txt
    └── README.md
```
This directory inherits the repository [LICENSE](../LICENSE). Please ensure any datasets comply with their original licenses.
