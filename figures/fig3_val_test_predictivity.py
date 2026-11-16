"""Figure 3 -- Paired validation-test associations and held-out outcomes.

Claim: paired validation/test difference correlations have intervals spanning
zero for IV and joint, and a positive association for Tg. These correlations
do not test checkpoint ranking or establish absence of information.

Panel a: per-seed scatter of (delta val) vs (delta test) = best - baseline,
          one subpanel per task, with the fitted line, Pearson r and its
          bootstrap 95% CI, and quadrant shading for sign agreement.
Panel b: paired per-seed slope plot (baseline -> best test loss), one
          subpanel per task. joint uses a log axis given the 18.77 vs 0.152
          magnitude gap (figure-style SS3.3: points/median ticks, never filled
          bars, on a log-scaled value axis).

Data: evidence/r13_val_test_predictivity.json,
      evidence/r8_sealed_test_eval/{IV,Tg,joint}.json
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _figstyle_kernel import apply_figure_style, set_frame, panel_letter

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"

apply_figure_style(sizes=(8, 7, 6))

TASKS = ["IV", "Tg", "joint"]
TASK_COLOR = {"IV": "#0072B2", "Tg": "#999999", "joint": "#D55E00"}

r13 = json.loads((EV / "r13_val_test_predictivity.json").read_text())
sealed = {t: json.loads((EV / "r8_sealed_test_eval" / f"{t}.json").read_text()) for t in TASKS}

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.62, wspace=0.42)
axesA = [fig.add_subplot(gs[0, j]) for j in range(3)]
axesB = [fig.add_subplot(gs[1, j]) for j in range(3)]

for j, t in enumerate(TASKS):
    ax = axesA[j]
    d = sealed[t]
    pb = d["arms"]["baseline"]["per_seed"]
    pf = d["arms"]["best"]["per_seed"]
    common = sorted(set(pb) & set(pf), key=int)
    dval = np.array([pf[s]["validation_loss"] - pb[s]["validation_loss"] for s in common])
    dtest = np.array([pf[s]["test_loss"] - pb[s]["test_loss"] for s in common])

    paired = r13["tasks"][t]["paired"]
    r = paired["pearson_dval_dtest"]
    ci_lo, ci_hi = paired["pearson_dval_dtest_bootstrap_ci95"]
    agree = paired["sign_agreement"]
    n = paired["n_common_seeds"]

    # quadrant shading: agreement quadrants (both same sign) in light grey
    xlim_pad = max(abs(dval).max(), 1e-9) * 1.25
    ylim_pad = max(abs(dtest).max(), 1e-9) * 1.25
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), xlim_pad, ylim_pad, color="#EEEEEE", zorder=0))
    ax.add_patch(Rectangle((-xlim_pad, -ylim_pad), xlim_pad, ylim_pad,
                           color="#EEEEEE", zorder=0))
    assert len(common) == n
    assert np.isclose(np.corrcoef(dval, dtest)[0, 1], r)
    assert int(np.sum(np.sign(dval) == np.sign(dtest))) == agree
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.5, zorder=1)

    ax.scatter(dval, dtest, s=14, color=TASK_COLOR[t], alpha=0.8, linewidths=0, zorder=3)

    # fitted line (OLS), only if there is variance in dval
    if dval.std() > 0:
        slope, intercept = np.polyfit(dval, dtest, 1)
        xs = np.linspace(-xlim_pad, xlim_pad, 50)
        ax.plot(xs, slope * xs + intercept, color="black", lw=0.9, zorder=2)

    ax.set_xlim(-xlim_pad, xlim_pad)
    ax.set_ylim(-ylim_pad, ylim_pad)
    ax.set_xlabel(r"$\Delta$ validation loss", fontsize=6.3)
    if j == 0:
        ax.set_ylabel(r"$\Delta$ test loss", fontsize=6.3)
    set_frame(ax)
    ax.set_title(f"{t}: r={r:+.2f} [{ci_lo:+.2f},{ci_hi:+.2f}]\nsign agree {agree}/{n}",
                 fontsize=6.0, loc="left")

for j, t in enumerate(TASKS):
    ax = axesB[j]
    d = sealed[t]
    pb = d["arms"]["baseline"]["per_seed"]
    pf = d["arms"]["best"]["per_seed"]
    common = sorted(set(pb) & set(pf), key=int)
    base = np.array([pb[s]["test_loss"] for s in common])
    best = np.array([pf[s]["test_loss"] for s in common])

    use_log = (t == "joint")
    for b, s in zip(base, best):
        ax.plot([0, 1], [b, s], color=TASK_COLOR[t], lw=0.6, alpha=0.45, zorder=2)
    ax.scatter(np.zeros_like(base), base, s=10, color="black", alpha=0.7, linewidths=0, zorder=3)
    ax.scatter(np.ones_like(best), best, s=10, color=TASK_COLOR[t], alpha=0.85, linewidths=0, zorder=3)
    # median ticks (figure-style SS3.3: points/median ticks on log axis, never filled bars)
    for xpos, arr in zip([0, 1], [base, best]):
        med = np.median(arr)
        ax.plot([xpos - 0.12, xpos + 0.12], [med, med], color="black", lw=1.6, zorder=4)

    if use_log:
        ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["baseline", "selected"], fontsize=6.3)
    ax.set_ylabel("Test loss" + (" (log scale)" if use_log else ""), fontsize=7)
    set_frame(ax)
    ax.set_title(f"{t}: median {np.median(base):.3g} -> {np.median(best):.3g}", fontsize=6.0, loc="left")

panel_letter(axesA[0], "a")
panel_letter(axesB[0], "b")

fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.91)
out = pathlib.Path(__file__).parent
fig.savefig(out / "fig3_val_test_predictivity.pdf")
fig.savefig(out / "fig3_val_test_predictivity.png", dpi=300)
print("saved fig3_val_test_predictivity.{pdf,png}")
for t in TASKS:
    p = r13["tasks"][t]["paired"]
    print(t, "r=", round(p["pearson_dval_dtest"], 3), "CI=", p["pearson_dval_dtest_bootstrap_ci95"],
          "agree=", p["sign_agreement"], "/", p["n_common_seeds"])
