"""Figure 2 -- Split variance dominates seed variance.

Claim: across these development partitions, variance of baseline split means
exceeds the mean within-split baseline variance. This descriptive comparison
is not a decomposition of paired improvements or a random-effects estimate.

Panel a: per-split relative change, 20 splits x 3 tasks, strip/dot plot with
          split-mean and BCa interval overlaid. The k10 outlier on joint is
          shown, not hidden.
Panel b: variance-component bars (between-split vs between-seed) with the
          ratio annotated (8.81 / 6.53 / 1.46).

Data: evidence/r7_multisplit_stats.json
  (per_split_rel_pct, split_variance, seed_variance, split_over_seed_var_ratio,
   cluster_bootstrap_ci_pct_BCa_AUTHORITATIVE, rel_improvement_pct)
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _figstyle_kernel import apply_figure_style, set_frame, panel_letter, goodness_arrow

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"

apply_figure_style(sizes=(8, 7, 6))

TASKS = ["IV", "Tg", "joint"]
TASK_COLOR = {"IV": "#0072B2", "Tg": "#999999", "joint": "#D55E00"}

r7 = json.loads((EV / "r7_multisplit_stats.json").read_text())

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0), width_ratios=[1.75, 1])

# ---- panel a: per-split strip plot ----
rng = np.random.default_rng(12345)
for j, t in enumerate(TASKS):
    per_split = r7[t]["per_split_rel_pct"]
    splits = sorted(per_split, key=lambda s: int(s))
    vals = -np.array([per_split[s] for s in splits])
    xj = j + (rng.random(len(vals)) - 0.5) * 0.34
    axA.scatter(xj, vals, s=13, color=TASK_COLOR[t], alpha=0.75, linewidths=0, zorder=3)
    mean = -r7[t]["rel_improvement_pct"]
    lo, hi = [-v for v in reversed(r7[t]["cluster_bootstrap_ci_pct_BCa_AUTHORITATIVE"])]
    assert lo <= mean <= hi
    assert len(vals) == 20
    axA.errorbar([j], [mean], yerr=[[mean - lo], [hi - mean]], fmt="_", color="black",
                 markersize=18, elinewidth=1.1, capsize=3, zorder=4, markeredgewidth=1.6)
    # flag the k10 outlier on joint explicitly (split index '10' in per_split_rel_pct)
    if t == "joint":
        idx10 = splits.index("10")
        axA.annotate("split 10", (j + (xj[idx10] - j), vals[idx10]),
                     xytext=(j + 0.55, vals[idx10]), fontsize=5.6,
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="black"),
                     ha="left", va="center")

axA.axhline(0, color="black", lw=0.6, zorder=1)
axA.set_xticks(range(len(TASKS)))
axA.set_xticklabels(TASKS)
axA.set_ylabel("per-split relative change (%), positive = worse")
goodness_arrow(axA, text="worse", loc="upper left", axis="y")
set_frame(axA)
axA.set_xlim(-0.6, len(TASKS) - 0.6 + 0.6)

# ---- panel b: variance components ----
xb = np.arange(len(TASKS))
bw = 0.32
split_var = [r7[t]["split_variance"] for t in TASKS]
seed_var = [r7[t]["seed_variance"] for t in TASKS]
ratio = [r7[t]["split_over_seed_var_ratio"] for t in TASKS]

axB.bar(xb - bw / 2, split_var, width=bw * 0.92, color="#444444", edgecolor="none",
        zorder=2, label="between-split")
axB.bar(xb + bw / 2, seed_var, width=bw * 0.92, color="#BBBBBB", edgecolor="none",
        zorder=2, label="between-seed")
axB.set_xticks(xb)
axB.set_xticklabels(TASKS)
axB.set_ylabel("Baseline loss variance")
axB.margins(y=0.25)
for sv, ev, rt in zip(split_var, seed_var, ratio):
    assert np.isclose(sv / ev, rt)
set_frame(axB)
axB.legend(loc="upper left", bbox_to_anchor=(0.0, 1.22), fontsize=6, ncol=1, frameon=False,
           handlelength=0.9)
for i, rt in enumerate(ratio):
    ymax = max(split_var[i], seed_var[i])
    axB.text(i, ymax * 1.08, f"{rt:.2f}\u00d7", ha="center", va="bottom", fontsize=6.3,
              fontweight="bold")

panel_letter(axA, "a")
panel_letter(axB, "b")

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = pathlib.Path(__file__).parent
fig.savefig(out / "fig2_split_vs_seed_variance.pdf")
fig.savefig(out / "fig2_split_vs_seed_variance.png", dpi=300)
print("saved fig2_split_vs_seed_variance.{pdf,png}")
for t in TASKS:
    print(t, "ratio=", round(r7[t]["split_over_seed_var_ratio"], 2),
          "split_var=", r7[t]["split_variance"], "seed_var=", r7[t]["seed_variance"])
