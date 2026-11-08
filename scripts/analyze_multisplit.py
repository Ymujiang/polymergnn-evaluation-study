#!/usr/bin/env python3
"""R7 analysis: decompose variance into DATA-SPLIT vs TRAINING-SEED components.

Reads evidence/multisplit/<task>/<arm>/k<split>_s<seed>/run/result.json and asks the
question that blocks writing: does the joint bugfix effect survive re-splitting?

Outputs evidence/r7_multisplit_stats.json.
"""
import json, itertools, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/multisplit'
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
SPLITS = list(range(20))  # R12: 20 splits
KEY = 'checkpoint_restored_validation_loss'


def load(task, arm):
    vals = {}
    for k in SPLITS:
        for s in SEEDS:
            f = OUT / task / arm / f'k{k}_s{s}' / 'run' / 'result.json'
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            v = d.get(KEY)
            if v is None:
                v = (d.get('best') or {}).get('validation_loss')
            if v is not None and math.isfinite(v):
                assert d.get('test_evaluated') is False, f'test seal broken {f}'
                vals[(k, s)] = float(v)
    return vals


def mean(x):
    return sum(x) / len(x)


def var(x):
    if len(x) < 2:
        return 0.0
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def wilcoxon_p(d):
    """Exact two-sided Wilcoxon signed-rank via full enumeration (n<=20) else normal."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    mags = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    for r, i in enumerate(mags, 1):
        rank[i] = r
    w = sum(rank[i] for i in range(n) if d[i] > 0)
    if n <= 20:
        cnt = tot = 0
        for signs in itertools.product([0, 1], repeat=n):
            s = sum(rank[i] for i in range(n) if signs[i])
            tot += 1
            if abs(s - n * (n + 1) / 4) >= abs(w - n * (n + 1) / 4) - 1e-12:
                cnt += 1
        return cnt / tot
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (abs(w - mu) - 0.5) / sd
    return math.erfc(z / math.sqrt(2))


def boot_ci(pairs, reps=20000, seed=7):
    """Cell-level bootstrap. UNDERSTATES uncertainty: cells within a split are
    correlated, so this treats 50 cells as 50 independent units. Reported only for
    comparison with the round-2 (fixed-split) CI."""
    rng = random.Random(seed)
    keys = list(pairs)
    out = []
    for _ in range(reps):
        samp = [pairs[rng.choice(keys)] for _ in keys]
        b = mean([x[0] for x in samp])
        c = mean([x[1] for x in samp])
        if b > 0:
            out.append((b - c) / b * 100)
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def cluster_boot_ci(pairs, splits, reps=20000, seed=11):
    """CLUSTER bootstrap resampling whole SPLITS. This is the honest interval when the
    question is generalisation to a NEW split: the split is the unit of replication."""
    rng = random.Random(seed)
    by_split = {}
    for (k, s), v in pairs.items():
        by_split.setdefault(k, []).append(v)
    ks = sorted(by_split)
    if len(ks) < 2:
        return None, None
    out = []
    for _ in range(reps):
        chosen = [rng.choice(ks) for _ in ks]
        cells = [c for k in chosen for c in by_split[k]]
        b = mean([x[0] for x in cells])
        c_ = mean([x[1] for x in cells])
        if b > 0:
            out.append((b - c_) / b * 100)
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def cluster_boot_ci_bca(pairs, splits, reps=20000, seed=11):
    """BCa cluster bootstrap. Percentile intervals are anti-conservative when the
    number of clusters is small, so bias-correction and acceleration are applied.
    Returns (lo, hi, z0, accel)."""
    rng = random.Random(seed)
    by_split = {}
    for (k, s), v in pairs.items():
        by_split.setdefault(k, []).append(v)
    ks = sorted(by_split)
    n = len(ks)
    if n < 3:
        return None, None, None, None

    def theta(keys):
        cells = [c for k in keys for c in by_split[k]]
        b = mean([x[0] for x in cells])
        c_ = mean([x[1] for x in cells])
        return (b - c_) / b * 100 if b > 0 else None

    t_hat = theta(ks)
    if t_hat is None:
        return None, None, None, None

    boots = []
    for _ in range(reps):
        v = theta([rng.choice(ks) for _ in ks])
        if v is not None:
            boots.append(v)
    boots.sort()
    if not boots:
        return None, None, None, None

    # bias-correction z0 from the proportion of bootstrap replicates below t_hat
    prop = sum(1 for b in boots if b < t_hat) / len(boots)
    prop = min(max(prop, 1.0 / (2 * len(boots))), 1 - 1.0 / (2 * len(boots)))
    z0 = _ndtri(prop)
    # acceleration from jackknife over CLUSTERS
    jack = [theta([k for k in ks if k != kk]) for kk in ks]
    jack = [j for j in jack if j is not None]
    jm = mean(jack)
    num = sum((jm - j) ** 3 for j in jack)
    den = 6.0 * (sum((jm - j) ** 2 for j in jack) ** 1.5)
    accel = num / den if den != 0 else 0.0

    def endpoint(alpha):
        z = _ndtri(alpha)
        adj = z0 + (z0 + z) / max(1e-12, (1 - accel * (z0 + z)))
        q = _ndtr(adj)
        q = min(max(q, 1.0 / len(boots)), 1 - 1.0 / len(boots))
        return boots[int(q * (len(boots) - 1))]

    return endpoint(0.025), endpoint(0.975), z0, accel


def _ndtr(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ndtri(p):
    """Inverse standard normal CDF (Acklam), refined by one Halley step."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = _ndtr(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def split_level_t(per_split_rel):
    """Paired t over the 5 per-split effects: does the effect generalise across splits?"""
    v = list(per_split_rel.values())
    n = len(v)
    if n < 2:
        return None, None
    m, sd = mean(v), math.sqrt(var(v))
    if sd == 0:
        return m, 0.0
    t = m / (sd / math.sqrt(n))
    df = n - 1
    # Exact two-sided Student-t p. Prefer scipy; fall back to the regularised
    # incomplete beta identity, which is exact rather than quadrature-limited.
    try:
        from scipy import stats as _st
        return t, float(2 * _st.t.sf(abs(t), df))
    except ImportError:
        x = df / (df + t * t)
        return t, min(1.0, _betainc(df / 2.0, 0.5, x))


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b) via continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m_ in range(1, itmax + 1):
        m2 = 2 * m_
        aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def main():
    report = {}
    for task in ('joint', 'IV', 'Tg'):
        base, best = load(task, 'baseline'), load(task, 'best')
        common = sorted(set(base) & set(best))
        if not common:
            report[task] = {'status': 'NO_DATA'}
            continue
        b = [base[c] for c in common]
        c_ = [best[c] for c in common]
        rel = (mean(b) - mean(c_)) / mean(b) * 100

        # --- variance decomposition on the BASELINE arm ---
        per_split_means = [mean([base[(k, s)] for s in SEEDS if (k, s) in base])
                           for k in SPLITS if any((k, s) in base for s in SEEDS)]
        within = [var([base[(k, s)] for s in SEEDS if (k, s) in base])
                  for k in SPLITS if sum((k, s) in base for s in SEEDS) > 1]
        split_var = var(per_split_means)
        seed_var = mean(within) if within else 0.0

        # --- per-split paired effect ---
        per_split_rel = {}
        for k in SPLITS:
            bb = [base[(k, s)] for s in SEEDS if (k, s) in base and (k, s) in best]
            cc = [best[(k, s)] for s in SEEDS if (k, s) in base and (k, s) in best]
            if bb:
                per_split_rel[k] = (mean(bb) - mean(cc)) / mean(bb) * 100

        diffs = [base[k] - best[k] for k in common]
        pairs = {k: (base[k], best[k]) for k in common}
        lo, hi = boot_ci(pairs)
        clo, chi = cluster_boot_ci(pairs, SPLITS)
        blo, bhi, z0, accel = cluster_boot_ci_bca(pairs, SPLITS)
        t_split, p_split = split_level_t(per_split_rel)
        # split-level t CI on the per-split effects (same unit of replication as BCa)
        _v = list(per_split_rel.values())
        _n = len(_v)
        _m, _sd = mean(_v), math.sqrt(var(_v))
        if _n >= 2 and _sd > 0:
            try:
                from scipy import stats as _st2
                _tc = float(_st2.t.ppf(0.975, _n - 1))
            except ImportError:
                _tc = 2.093
            _se = _sd / math.sqrt(_n)
            t_ci = [_m - _tc * _se, _m + _tc * _se]
        else:
            t_ci = [None, None]

        report[task] = {
            'n_cells': len(common),
            'baseline_mean': mean(b), 'best_mean': mean(c_),
            'rel_improvement_pct': rel,
            'bootstrap_ci_pct_cell_level_OPTIMISTIC': [lo, hi],
            'cluster_bootstrap_ci_pct_percentile': [clo, chi],
            'cluster_bootstrap_ci_pct_BCa_AUTHORITATIVE': [blo, bhi],
            'bca_bias_z0': z0, 'bca_acceleration': accel,
            'split_level_t': t_split, 'split_level_p': p_split,
            'split_level_t_ci_pct': t_ci,
            'n_splits': len(per_split_rel),
            'n_improved': sum(1 for d in diffs if d > 0),
            'wilcoxon_p': wilcoxon_p(diffs),
            'split_variance': split_var,
            'seed_variance': seed_var,
            'split_over_seed_var_ratio': (split_var / seed_var) if seed_var else None,
            'per_split_rel_pct': per_split_rel,
            'n_splits_favourable': sum(1 for v in per_split_rel.values() if v > 0),
        }
        _f = lambda x: f'{x:+.2f}' if x is not None else 'NA'
        print(f'   clusterCI_pctile=[{_f(clo)},{_f(chi)}] '
              f'clusterCI_BCa=[{_f(blo)},{_f(bhi)}] (z0={_f(z0)} a={_f(accel)})')
        print(f'   split-level t={_f(t_split)} p={p_split:.4f} '
              f'CI=[{_f(t_ci[0])},{_f(t_ci[1])}] over {len(per_split_rel)} splits')
        print(f'[{task}] n={len(common)} rel={rel:+.2f}% cellCI=[{lo:+.2f},{hi:+.2f}] '
              f'p={report[task]["wilcoxon_p"]:.4f} '
              f'splitvar/seedvar={report[task]["split_over_seed_var_ratio"]} '
              f'splits_favourable={report[task]["n_splits_favourable"]}/{len(per_split_rel)}')
        print(f'   per-split: ' + ', '.join(f'k{k}={v:+.1f}%' for k, v in sorted(per_split_rel.items())))

    (ROOT / 'evidence/r7_multisplit_stats.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + chr(10))
    print('wrote evidence/r7_multisplit_stats.json')


if __name__ == '__main__':
    main()
