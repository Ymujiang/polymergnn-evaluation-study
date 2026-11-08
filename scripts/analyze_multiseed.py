#!/usr/bin/env python3
"""Aggregate the multi-seed confirmation matrix into per-seed and summary statistics.

Reports, for each task and each metric:
  * per-seed values for baseline and best (never averaged away),
  * mean, sample std (ddof=1), min, max,
  * PAIRED per-seed differences (best - baseline) on the SAME training seed,
    which is the correct unit of analysis because the two variants share the
    frozen split and the seed controls initialization and batch order,
  * a paired two-sided Wilcoxon signed-rank test and a paired t-test,
  * the seed-2026 single-point value previously reported, for comparison.

Nothing here touches the test split.
"""
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/multiseed'
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
TASKS = ['joint', 'IV', 'Tg']

# Previously reported single-point (train seed == split seed == 2026) results.
SEED2026 = {
    'joint': {'baseline': 0.16609559953212738, 'best': 0.10180651396512985},
    'IV':    {'baseline': 0.07721974700689316, 'best': 0.06046447157859802},
    'Tg':    {'baseline': 0.08968862891197205, 'best': 0.05738804116845131},
}


def load(task, variant, seed):
    f = OUT / f'{task}_{variant}_s{seed}/run/result.json'
    if not f.is_file():
        return None
    j = json.loads(f.read_text())
    assert j['test_evaluated'] is False, f'test_evaluated must be False: {f}'
    assert j['protocol_sha256'] == 'dc49ec25644861dd9da73ff14b6e82ffdaf92d467ea2fe834cd7a27955ae4be1', f
    assert j['seed'] == seed, (j['seed'], seed)
    row = {'validation_loss': j['best']['validation_loss']}
    for target, m in j['validation_metrics'].items():
        for k in ('mae', 'rmse', 'r2'):
            row[f'{target}_{k}'] = m[k]
    return row


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1 / d
    h = d
    for m_ in range(1, MAXIT + 1):
        m2 = 2 * m_
        aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1 / d
        h *= d * c
        aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def describe(xs):
    if not xs:
        return None
    return dict(n=len(xs), mean=st.mean(xs),
                std=st.stdev(xs) if len(xs) > 1 else 0.0,
                min=min(xs), max=max(xs))


def wilcoxon_signed_rank_p(diffs):
    """Exact two-sided Wilcoxon signed-rank p for small n, zeros dropped."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    # Exact null distribution over all 2^n sign assignments.
    from itertools import product
    count = 0
    total = 0
    w_obs = min(w_plus, n * (n + 1) / 2 - w_plus)
    for signs in product([0, 1], repeat=n):
        wp = sum(r for r, s in zip(ranks, signs) if s)
        wm = n * (n + 1) / 2 - wp
        total += 1
        if min(wp, wm) <= w_obs:
            count += 1
    return count / total


def paired_t_p(diffs):
    n = len(diffs)
    if n < 2:
        return None
    m = st.mean(diffs)
    s = st.stdev(diffs)
    if s == 0:
        return 0.0 if m != 0 else None
    t = m / (s / math.sqrt(n))
    df = n - 1
    # Two-sided p via incomplete beta (regularized) using continued fraction.
    x = df / (df + t * t)

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1 / d
        h = d
        for m_ in range(1, MAXIT + 1):
            m2 = 2 * m_
            aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1 / d
            h *= d * c
            aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < EPS:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return math.exp(lbeta) * betacf(a, b, x) / a
        return 1 - math.exp(lbeta) * betacf(b, a, 1 - x) / b

    return betai(df / 2, 0.5, x)


def main():
    report = {}
    for task in TASKS:
        rows = {v: {s: load(task, v, s) for s in SEEDS} for v in ('baseline', 'best')}
        metrics = sorted({k for v in rows for s in SEEDS if rows[v][s] for k in rows[v][s]})
        task_out = {'per_seed': {v: {s: rows[v][s] for s in SEEDS} for v in rows}, 'metrics': {}}
        for metric in metrics:
            b = [rows['baseline'][s][metric] for s in SEEDS if rows['baseline'][s]]
            c = [rows['best'][s][metric] for s in SEEDS if rows['best'][s]]
            paired = [(rows['best'][s][metric] - rows['baseline'][s][metric])
                      for s in SEEDS if rows['baseline'][s] and rows['best'][s]]
            entry = {'baseline': describe(b), 'best': describe(c),
                     'paired_diff': describe(paired),
                     'paired_diffs_per_seed': dict(zip([s for s in SEEDS if rows['baseline'][s] and rows['best'][s]], paired))}
            if paired:
                entry['wilcoxon_two_sided_p'] = wilcoxon_signed_rank_p(paired)
                entry['paired_t_two_sided_p'] = paired_t_p(paired)
                entry['n_seeds_improved'] = sum(1 for x in paired if x < 0)
                entry['n_seeds_worse'] = sum(1 for x in paired if x > 0)
                entry['min_attainable_wilcoxon_p'] = min_attainable_wilcoxon_p(len(paired))
                entry['mdes_80pct_power'] = mdes(paired)
                if entry['baseline'] and entry['baseline']['mean']:
                    bm = abs(entry['baseline']['mean'])
                    entry['mdes_as_pct_of_baseline'] = mdes(paired) / bm * 100
                    entry['tost_p_within_10pct'] = tost_equivalence(paired, 0.10 * bm)
                    entry['tost_p_within_20pct'] = tost_equivalence(paired, 0.20 * bm)
            if b and c and describe(b)['mean'] != 0:
                entry['relative_change_of_means_pct'] = (describe(b)['mean'] - describe(c)['mean']) / describe(b)['mean'] * 100
            task_out['metrics'][metric] = entry
        task_out['seed2026_single_point'] = SEED2026.get(task)
        report[task] = task_out

    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    for task in TASKS:
        m = report[task]['metrics'].get('validation_loss')
        if not m or not m['baseline']:
            print(f'{task}: incomplete')
            continue
        print(f'=== {task} validation_loss ===')
        print(f"  baseline  mean={m['baseline']['mean']:.6f} std={m['baseline']['std']:.6f} "
              f"min={m['baseline']['min']:.6f} max={m['baseline']['max']:.6f} n={m['baseline']['n']}")
        print(f"  best      mean={m['best']['mean']:.6f} std={m['best']['std']:.6f} "
              f"min={m['best']['min']:.6f} max={m['best']['max']:.6f} n={m['best']['n']}")
        print(f"  paired diff mean={m['paired_diff']['mean']:+.6f} std={m['paired_diff']['std']:.6f}")
        print(f"  improved {m['n_seeds_improved']}/{m['paired_diff']['n']} seeds; "
              f"wilcoxon p={m['wilcoxon_two_sided_p']}, t p={m['paired_t_two_sided_p']:.4f}")
        print(f"  relative change of means: {m['relative_change_of_means_pct']:+.2f}%")
        print(f"  seed-2026 single point: baseline={SEED2026[task]['baseline']:.6f} best={SEED2026[task]['best']:.6f} "
              f"({(SEED2026[task]['baseline']-SEED2026[task]['best'])/SEED2026[task]['baseline']*100:+.2f}%)")




# ---------------------------------------------------------------------------
# Power / sensitivity analysis appended for the n=10 extension.
# ---------------------------------------------------------------------------

def min_attainable_wilcoxon_p(n):
    """Smallest two-sided exact p reachable at this n (all diffs same sign)."""
    from itertools import product
    ranks = list(range(1, n + 1))
    tot = n * (n + 1) / 2
    cnt = total = 0
    for signs in product([0, 1], repeat=n):
        wp = sum(r for r, s in zip(ranks, signs) if s)
        total += 1
        if min(wp, tot - wp) <= 0:
            cnt += 1
    return cnt / total


def tost_equivalence(diffs, bound):
    """Two one-sided t-tests: is the paired effect within +/- bound?

    Returns the larger of the two one-sided p values; < 0.05 means we can
    positively assert equivalence (not merely fail to find a difference).
    """
    n = len(diffs)
    if n < 2:
        return None
    m = st.mean(diffs)
    s = st.stdev(diffs)
    if s == 0:
        return 0.0 if abs(m) < bound else 1.0
    se = s / math.sqrt(n)
    df = n - 1

    def one_sided_p(t, upper):
        x = df / (df + t * t)
        p2 = _betai(df / 2, 0.5, x)
        return p2 / 2 if (t > 0) == upper else 1 - p2 / 2

    t_lower = (m - (-bound)) / se
    t_upper = (m - bound) / se
    return max(one_sided_p(t_lower, True), one_sided_p(t_upper, False))


def mdes(diffs, alpha=0.05, power=0.80):
    """Minimum detectable effect size (paired t, two-sided) at observed std."""
    n = len(diffs)
    if n < 2:
        return None
    s = st.stdev(diffs)
    # normal approximation: (z_{1-a/2} + z_{1-b}) * s / sqrt(n)
    return (1.959964 + 0.841621) * s / math.sqrt(n)


if __name__ == '__main__':
    main()
