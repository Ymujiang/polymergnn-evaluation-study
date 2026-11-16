"""Figure 4 -- Diagnosis of the joint baseline failure.

Claim: the joint baseline's test failure is a real untrained-encoder defect,
not an evaluation artefact.

Panel a: per-sample Tg error, baseline vs fixed, across the 28 joint test
          identifiers (paired by sample). Medians annotated.
Panel b: IV predicted vs measured on the same baseline checkpoint and the
          same samples (MAE, r2), establishing the checkpoint/evaluator are
          sound and the Tg-only failure is architecture-specific.

Data: evidence/r8_joint_baseline_tg_diagnosis_persample.json
  (full per-sample arrays from read-only re-inference on the frozen R8
   checkpoints; extends evidence/r8_joint_baseline_tg_diagnosis.json, whose
   summary numbers -- median 194.93 K / 8.35 K -- are reproduced here within
   expected stochastic-pooling tolerance, see script docstring of
   scripts/diagnose_r8_joint_baseline_persample.py).
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

d = json.loads((EV / "r8_joint_baseline_tg_diagnosis_persample.json").read_text())
old = json.loads((EV / "r8_joint_baseline_tg_diagnosis.json").read_text())

base_recs = {r["resin_id"]: r for r in d["baseline"]["records"]}
best_recs = {r["resin_id"]: r for r in d["best"]["records"]}
ids = sorted(set(base_recs) & set(best_recs), key=lambda s: base_recs[s]["tg_abs_err_K"])

base_err = np.array([base_recs[i]["tg_abs_err_K"] for i in ids])
best_err = np.array([best_recs[i]["tg_abs_err_K"] for i in ids])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1), width_ratios=[1.3, 1])

# ---- panel a: per-sample Tg error, baseline vs fixed ----
x = np.arange(len(ids))
axA.scatter(x, base_err, s=16, color="#D55E00", alpha=0.85, linewidths=0, zorder=3,
            label="baseline (bug)")
axA.scatter(x, best_err, s=16, color="#0072B2", alpha=0.85, linewidths=0, zorder=3,
            label="fixed (IDEA-001)")
for xi, b, f in zip(x, base_err, best_err):
    axA.plot([xi, xi], [f, b], color="#999999", lw=0.5, zorder=2)
med_base, med_best = np.median(base_err), np.median(best_err)
axA.axhline(med_base, color="#D55E00", lw=1.0, linestyle=(0, (3, 1.5)), zorder=1)
axA.axhline(med_best, color="#0072B2", lw=1.0, linestyle=(0, (3, 1.5)), zorder=1)
axA.text(len(ids) - 0.5, med_base, f" median {med_base:.1f} K", color="#D55E00",
          fontsize=6.0, va="bottom", ha="right")
axA.text(len(ids) - 0.5, med_best, f" median {med_best:.1f} K", color="#0072B2",
          fontsize=6.0, va="top", ha="right")
axA.set_xlabel("test sample (sorted by baseline Tg error)", fontsize=6.3)
axA.set_ylabel("Tg absolute error (K)")
set_frame(axA)
axA.legend(loc="upper left", fontsize=6.3, frameon=False, handlelength=0.9)
axA.set_xlim(-1, len(ids))

# ---- panel b: IV predicted vs measured on the SAME baseline checkpoint ----
iv_pred = np.array([base_recs[i]["iv_pred"] for i in ids])
iv_true = np.array([base_recs[i]["iv_true"] for i in ids])
mae = float(np.abs(iv_pred - iv_true).mean())
ss_res = float(((iv_pred - iv_true) ** 2).sum())
ss_tot = float(((iv_true - iv_true.mean()) ** 2).sum())
r2 = 1 - ss_res / ss_tot

lo = min(iv_true.min(), iv_pred.min()) * 0.95
hi = max(iv_true.max(), iv_pred.max()) * 1.05
axB.plot([lo, hi], [lo, hi], color="black", lw=0.7, linestyle=(0, (4, 2)), zorder=1)
axB.scatter(iv_true, iv_pred, s=16, color="#0072B2", alpha=0.85, linewidths=0, zorder=3)
axB.set_xlim(lo, hi)
axB.set_ylim(lo, hi)
axB.set_xlabel("measured IV")
axB.set_ylabel("predicted IV")
set_frame(axB)
axB.text(0.04, 0.94, f"MAE={mae:.3f}\n$r^2$={r2:.3f}", transform=axB.transAxes,
          fontsize=6.3, va="top", ha="left")
axB.set_title("baseline checkpoint, same 28 samples", fontsize=6.3, loc="left")

panel_letter(axA, "a")
panel_letter(axB, "b")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = pathlib.Path(__file__).parent
fig.savefig(out / "fig4_joint_baseline_diagnosis.pdf")
fig.savefig(out / "fig4_joint_baseline_diagnosis.png", dpi=300)
print("saved fig4_joint_baseline_diagnosis.{pdf,png}")
print("median Tg err baseline=", round(med_base, 2), "fixed=", round(med_best, 2))
print("IV MAE=", round(mae, 4), "r2=", round(r2, 4))
print("[ledger reference] baseline median 194.93K, best median 8.35K, IV mae 0.089, r2 0.714")
