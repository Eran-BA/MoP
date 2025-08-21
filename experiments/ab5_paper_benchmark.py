#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np


def read_csv(path: str) -> Tuple[List[str], List[List[float]]]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    headers = [h.strip() for h in lines[0].split(",")]
    rows: List[List[float]] = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        # seed is col 0; rest are floats
        floats: List[float] = []
        for v in parts[1:]:
            try:
                floats.append(float(v))
            except Exception:
                floats.append(float("nan"))
        rows.append(floats)
    return headers, rows


def mean_std(vals: List[float]) -> Tuple[float, float, int]:
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def format_pm(mean: float, std: float, decimals: int = 4) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "-"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def collect_results(
    inputs: List[str], patterns: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    """Return mapping: target -> model_key -> list of seed accuracies"""
    collected: Dict[str, Dict[str, List[float]]] = {}
    files: List[str] = []
    for root in inputs:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(root, pat)))

    for path in sorted(set(files)):
        fn = os.path.basename(path)
        # try extract target from filename patterns used by our scripts
        target = None
        # tournament_target_{int(target)}.csv or cifar100_ab5_target_{int(target)}.csv
        for key in ["tournament_target_", "cifar100_ab5_target_"]:
            if key in fn:
                try:
                    target = fn.split(key, 1)[1].split(".")[0]
                except Exception:
                    target = None
                break
        if target is None:
            # fallback: put all into 'unknown'
            target = "unknown"

        headers, rows = read_csv(path)
        # headers: [seed, acc_A, acc_B, ...]
        model_keys = [h for h in headers[1:]]  # acc_A ...
        # normalize to just A/B/C/D/E labels if present
        norm_keys = []
        for h in model_keys:
            if h.startswith("acc_"):
                norm_keys.append(h.split("acc_", 1)[1])
            else:
                norm_keys.append(h)

        if target not in collected:
            collected[target] = {}
        for col_idx, mk in enumerate(norm_keys):
            vals = [r[col_idx] for r in rows]
            collected[target].setdefault(mk, []).extend(vals)

    return collected


def to_markdown(collected: Dict[str, Dict[str, List[float]]]) -> str:
    lines: List[str] = []
    header = "| Target | A | B | C | D | E |\n|---|---|---|---|---|---|"
    lines.append(header)
    for target in sorted(collected.keys(), key=lambda x: (len(x), x)):
        row = collected[target]
        cells = [target]
        for key in ["A", "B", "C", "D", "E"]:
            mean, std, _ = mean_std(row.get(key, []))
            cells.append(format_pm(mean, std))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(collected: Dict[str, Dict[str, List[float]]]) -> str:
    lines: List[str] = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Target & A & B & C & D & E \\\\")
    lines.append("\\midrule")
    for target in sorted(collected.keys(), key=lambda x: (len(x), x)):
        row = collected[target]
        vals: List[str] = []
        for key in ["A", "B", "C", "D", "E"]:
            mean, std, _ = mean_std(row.get(key, []))
            vals.append(format_pm(mean, std))
        lines.append(f"{target} & " + " & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate A/B/C/D/E results and emit Markdown/LaTeX tables"
    )
    ap.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=[
            "results/ab5_tournament",
            "results/cifar100_ab5_param_budgets",
            "results/imagenet_ab_param_budgets",
        ],
    )
    ap.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=[
            "tournament_target_*.csv",
            "cifar100_ab5_target_*.csv",
            "imagenet_ab_target_*.csv",
        ],
    )
    ap.add_argument("--out_dir", type=str, default="results/paper_benchmark")
    ap.add_argument("--md_name", type=str, default="ab5_benchmark.md")
    ap.add_argument("--tex_name", type=str, default="ab5_benchmark.tex")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    collected = collect_results(args.inputs, args.patterns)

    md = to_markdown(collected)
    tex = to_latex(collected)

    md_path = os.path.join(args.out_dir, args.md_name)
    tex_path = os.path.join(args.out_dir, args.tex_name)
    with open(md_path, "w") as f:
        f.write(md)
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote LaTeX:   {tex_path}")


if __name__ == "__main__":
    main()
