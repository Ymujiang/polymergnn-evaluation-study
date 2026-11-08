#!/usr/bin/env python3
"""R9 analysis: did fixed-split search decisions survive 20 splits?

Reference arm = evidence/multisplit/joint/best (byte-identical to IDEA-001).
Candidate arms = evidence/r9_ablation/{gat,sagpool,init}.

Pre-registered decision rule
----------------------------
For each candidate the round-2 search made a call on the fixed seed-2026 split.
We re-measure the same contrast across 20 dev-only re-splits x 10 seeds and
declare a REVERSAL when the 20-split result contradicts the original call:
  - originally discarded as worse, but now significantly better, or
  - originally discarded as worse, but now statistically indistinguishable
    (i.e. the original call was not supported by evidence).
Significance uses the split-level Wilcoxon on the 20 per-split effects, with
Holm correction across the three candidates.
"""
import json, math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_multisplit import (SEEDS, SPLITS, mean, var, cluster_boot_ci,
                                cluster_boot_ci_bca, split_level_t)

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'evidence/multisplit/joint/best'
CAND = ROOT / 'evidence/r9_ablation'
KEY = 'checkpoint_restored_validation_loss'

FIXED_SPLIT_CALL = {
    'gat':     {'idea': 'IDEA-007', 'rel_fixed': -3.73, 'call': 'discarded'},
    'sagpool': {'idea': 'IDEA-011', 'rel_fixed': -2.50, 'call': 'discarded'},
    'init':    {'idea': 'IDEA-010', 'rel_fixed': +5.64, 'call': 'rejected_guardrail'},
}


def load_dir(d):
    vals = {}
    for k in SPLITS:
        for s in SEEDS:
            f = d / f'k{k}_s{s}' / 'run' / 'result.json'
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            v = j.get(KEY) or (j.get('best') or {}).get('validation_loss')
            if v is not None and math.isfinite(v):
                assert j.get('test_evaluated') is False, f'test seal broken {f}'
                vals[(k, s)] = float(v)
    return vals


def holm(raw):
    s = sorted(raw.items(), key=lambda x: x[1])
    m = len(s); prev = 0.0; out = {}
    for i, (k, p) in enumerate(s):
        adj = min(1.0, max(prev, (m - i) * p)); out[k] = adj; prev = adj
    return out


def main():
    try:
        from scipy import stats
    except ImportError:
        stats = None

    ref = load_dir(REF)
    print(f'reference arm (_best == IDEA-001): {len(ref)} cells')

    report, raw_p = {}, {}
    for arm in ('gat', 'sagpool', 'init'):
        cand = load_dir(CAND / arm)
        common = sorted(set(ref) & set(cand))
        if len(common) < 100:
            print(f'[{arm}] INCOMPLETE {len(common)} cells'); continue

        # positive = candidate BETTER than reference (loss went down)
        rel = (mean([ref[c] for c in common]) - mean([cand[c] for c in common])) \
            / mean([ref[c] for c in common]) * 100

        per_split = {}
        for k in SPLITS:
            rr = [ref[(k, s)] for s in SEEDS if (k, s) in ref and (k, s) in cand]
            cc = [cand[(k, s)] for s in SEEDS if (k, s) in ref and (k, s) in cand]
            if rr:
                per_split[k] = (mean(rr) - mean(cc)) / mean(rr) * 100
        v = [per_split[k] for k in sorted(per_split)]

        pairs = {c: (ref[c], cand[c]) for c in common}
        splits = sorted({c[0] for c in common})
        clo, chi = cluster_boot_ci(pairs, splits)
        blo, bhi, z0, acc = cluster_boot_ci_bca(pairs, splits)
        t, pt = split_level_t(per_split)

        pw = stats.wilcoxon(v).pvalue if stats else float('nan')
        psign = stats.binomtest(sum(1 for x in v if x > 0), len(v), 0.5).pvalue if stats else float('nan')
        loo = [split_level_t({kk: vv for kk, vv in per_split.items() if kk != k})[1]
               for k in sorted(per_split)]

        raw_p[arm] = pw
        report[arm] = {
            'idea': FIXED_SPLIT_CALL[arm]['idea'],
            'original_call': FIXED_SPLIT_CALL[arm]['call'],
            'rel_fixed_split_pct': FIXED_SPLIT_CALL[arm]['rel_fixed'],
            'rel_20split_pct': rel,
            'n_cells': len(common),
            'per_split_rel_pct': per_split,
            'splits_favourable': sum(1 for x in v if x > 0),
            'cluster_ci_pct_percentile': [clo, chi],
            'cluster_ci_pct_BCa': [blo, bhi],
            'split_level_t': t, 'split_level_t_p': pt,
            'split_level_wilcoxon_p': pw,
            'sign_test_p': psign,
            'loo_t_p_range': [min(loo), max(loo)],
        }

    adj = holm(raw_p) if raw_p else {}
    for arm, r in report.items():
        r['holm_wilcoxon_p'] = adj.get(arm)

    for arm, r in report.items():
        print(f"\n[{arm}] {r['idea']} (originally {r['original_call']})")
        print(f"   fixed-split: {r['rel_fixed_split_pct']:+.2f}%  ->  "
              f"20-split: {r['rel_20split_pct']:+.2f}%  ({r['n_cells']} cells)")
        print(f"   favourable splits {r['splits_favourable']}/20 | "
              f"BCa CI [{r['cluster_ci_pct_BCa'][0]:+.2f}, {r['cluster_ci_pct_BCa'][1]:+.2f}]")
        print(f"   split t={r['split_level_t']:+.3f} p={r['split_level_t_p']:.4f} | "
              f"Wilcoxon p={r['split_level_wilcoxon_p']:.4f} -> Holm {r['holm_wilcoxon_p']:.4f}")
        print(f"   sign p={r['sign_test_p']:.4f} | LOO t-p range "
              f"[{r['loo_t_p_range'][0]:.4f}, {r['loo_t_p_range'][1]:.4f}]")

    out = ROOT / 'evidence/r9_ablation_stats.json'
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + chr(10))
    print(f'\nwrote {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
