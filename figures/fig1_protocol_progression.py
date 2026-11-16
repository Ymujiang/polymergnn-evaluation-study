"""Figure 1: descriptive development and held-out estimates, not protocol effects.
Reads saved aggregate statistics only; never imports training or inference code.
Development improvement and its CI are negated to align with test loss change.
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figstyle_kernel import apply_figure_style, set_frame, panel_letter

ROOT = pathlib.Path(__file__).resolve().parent.parent
EV = ROOT / "evidence"
r7 = json.loads((EV / "r7_multisplit_stats.json").read_text())
r8 = json.loads((EV / "r8_test_stats.json").read_text())
TASKS = ["IV", "Tg", "joint"]
COLORS = ["#0072B2", "#777777", "#D55E00"]
apply_figure_style(sizes=(9, 8, 7))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for panel, (ax, source) in enumerate(zip(axes, [r7, r8])):
    for y, (task, color) in enumerate(zip(TASKS, COLORS)):
        row = source[task]
        if panel == 0:
            value = -row["rel_improvement_pct"]
            low, high = [-v for v in reversed(row["cluster_bootstrap_ci_pct_BCa_AUTHORITATIVE"])]
            expected = 100 * (row["best_mean"] / row["baseline_mean"] - 1)
        else:
            value = row["relative_change_pct"]
            low, high = row["bootstrap_ci95"]
            expected = 100 * (row["best_mean_test_loss"] / row["baseline_mean_test_loss"] - 1)
        assert abs(value - expected) < 1e-8, (task, value, expected)
        assert low <= value <= high
        ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o",
                    color=color, capsize=3, markersize=5, linewidth=1.2)
        ax.annotate(f"{value:+.2f}%", (value, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8)
        print(task, "development" if panel == 0 else "test", value, [low, high])
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_yticks(range(3), TASKS)
    ax.set_ylim(2.55, -0.65)
    ax.set_xlabel("Loss change (%)")
    ax.margins(x=0.2)
    set_frame(ax)
    panel_letter(ax, "ab"[panel])
axes[0].set_title("Development: 20 splits × 10 seeds", fontsize=9, pad=16)
axes[1].set_title("Sealed test: 20 / 51 / 23 seeds", fontsize=9, pad=16)
fig.text(0.5, 0.02, "Negative = lower loss; positive = higher loss. Panels use different x-axis scales.",
         ha="center", fontsize=8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
out = pathlib.Path(__file__).parent
fig.savefig(out / "fig1_protocol_progression.pdf")
fig.savefig(out / "fig1_protocol_progression.png", dpi=300)
plt.close(fig)
