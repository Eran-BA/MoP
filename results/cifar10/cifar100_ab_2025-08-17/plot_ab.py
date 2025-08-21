#!/usr/bin/env python3
import json
import os

import matplotlib.pyplot as plt

here = os.path.dirname(__file__)
agg_path = os.path.join(here, "aggregate_summary.json")

with open(agg_path, "r") as f:
    agg = json.load(f)

# Try multiple key conventions
A = agg.get("baseline_mean") or agg.get("A_mean") or agg.get("baseline")
B = agg.get("mop_mean") or agg.get("B_mean") or agg.get("mop")
delta = (
    agg.get("delta") if "delta" in agg else (None if A is None or B is None else B - A)
)
ci_lo = agg.get("ci_low") or agg.get("ci_lo") or agg.get("ci_lower")
ci_hi = agg.get("ci_high") or agg.get("ci_hi") or agg.get("ci_upper")

title = "CIFAR-100 — mean test acc across seeds"
subtitle = ""
if delta is not None:
    if ci_lo is not None and ci_hi is not None:
        subtitle = f"Δ(B–A)={delta:+.4f} (95% CI {ci_lo:+.4f},{ci_hi:+.4f})"
    else:
        subtitle = f"Δ(B–A)={delta:+.4f}"

fig = plt.figure(figsize=(6, 4))
plt.bar(["Baseline", "MoP"], [A, B])
plt.ylim(0, 1.0)
plt.ylabel("Accuracy")
plt.title(title + ("\n" + subtitle if subtitle else ""))
plt.tight_layout()
out_png = os.path.join(here, "cifar100_ab_acc.png")
plt.savefig(out_png, dpi=200)
print("Wrote", out_png)
