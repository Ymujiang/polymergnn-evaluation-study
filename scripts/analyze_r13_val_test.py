#!/usr/bin/env python3
"""R13: 第2轮自审的验证集-测试集预测力分析（只读，不触发任何训练/重评测）。

数据源：evidence/r8_sealed_test_eval/{IV,Tg,joint}.json 中已密封的逐种子记录，
每条同时含 validation_loss 与 test_loss。本脚本不重新评测测试集，
仅对已落盘的数字做二次统计，因此不违反红线规则 3。

本版本新增：配对相关系数的 bootstrap 95% CI（按种子重抽样，20000 次，seed=12345，
与全文其余 bootstrap 一致）；以及按任务的测试集相对改善 leave-one-seed-out 范围
（逐次留一个种子，重算 mean-based relative change，报告 [min, max]）。

产出：evidence/r13_val_test_predictivity.json
"""
import json, pathlib, math, random, statistics as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEALED = ROOT / "evidence" / "r8_sealed_test_eval"
OUT = ROOT / "evidence" / "r13_val_test_predictivity.json"
N_BOOT = 20000
BOOT_SEED = 12345

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    sxx = sum((a-mx)**2 for a in xs); syy = sum((b-my)**2 for b in ys)
    if sxx <= 0 or syy <= 0: return None
    return sxy / math.sqrt(sxx*syy)

def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0]*len(vals); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and vals[order[j+1]] == vals[order[i]]: j += 1
        avg = (i+j)/2.0 + 1.0
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))

def bootstrap_ci_pearson(dval, dtst, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    n = len(dval)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        xs = [dval[i] for i in idx]; ys = [dtst[i] for i in idx]
        r = pearson(xs, ys)
        if r is not None:
            boots.append(r)
    boots.sort()
    lo = boots[int((alpha/2) * len(boots))]
    hi = boots[int((1 - alpha/2) * len(boots)) - 1]
    return [lo, hi], len(boots)

def loo_rel_change_range(baseline_test, best_test):
    n = len(baseline_test)
    assert n == len(best_test)
    vals = []
    for i in range(n):
        b = [x for j, x in enumerate(baseline_test) if j != i]
        s = [x for j, x in enumerate(best_test) if j != i]
        bm, sm = st.mean(b), st.mean(s)
        vals.append((sm - bm) / bm * 100.0)
    return [min(vals), max(vals)]

report = {"note": "read-only re-analysis of sealed R8 per-seed records; no re-evaluation performed",
          "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "method": "percentile, resampling seeds with replacement"},
          "tasks": {}}

for task in ["IV", "Tg", "joint"]:
    d = json.loads((SEALED / f"{task}.json").read_text())
    entry = {"n_test_samples": d["n_test_samples"], "arms": {}, "paired": {}}
    per_arm = {}
    for arm in ["baseline", "best"]:
        ps = d["arms"][arm]["per_seed"]
        seeds = sorted(ps, key=lambda s: int(s))
        val = [ps[s]["validation_loss"] for s in seeds]
        tst = [ps[s]["test_loss"] for s in seeds]
        per_arm[arm] = {"seeds": seeds, "val": val, "test": tst}
        entry["arms"][arm] = {
            "n_seeds": len(seeds),
            "val_mean": st.mean(val), "val_median": st.median(val),
            "test_mean": st.mean(tst), "test_median": st.median(tst),
            "test_sd": st.pstdev(tst) if len(tst) > 1 else 0.0,
            "pearson_val_test": pearson(val, tst),
            "spearman_val_test": spearman(val, tst),
        }
    sb, ss = per_arm["baseline"], per_arm["best"]
    common = [s for s in sb["seeds"] if s in ss["seeds"]]
    ib = {s: i for i, s in enumerate(sb["seeds"])}; isx = {s: i for i, s in enumerate(ss["seeds"])}
    dval = [ss["val"][isx[s]] - sb["val"][ib[s]] for s in common]
    dtst = [ss["test"][isx[s]] - sb["test"][ib[s]] for s in common]
    agree = sum(1 for a, b in zip(dval, dtst) if (a < 0) == (b < 0))
    base_test_common = [sb["test"][ib[s]] for s in common]
    best_test_common = [ss["test"][isx[s]] for s in common]
    bm, sm = st.mean(sb["test"]), st.mean(ss["test"])
    bmed, smed = st.median(sb["test"]), st.median(ss["test"])
    ci, n_valid_boot = bootstrap_ci_pearson(dval, dtst)
    loo_range = loo_rel_change_range(base_test_common, best_test_common)
    entry["paired"] = {
        "n_common_seeds": len(common),
        "pearson_dval_dtest": pearson(dval, dtst),
        "pearson_dval_dtest_bootstrap_ci95": ci,
        "pearson_dval_dtest_bootstrap_n_valid": n_valid_boot,
        "spearman_dval_dtest": spearman(dval, dtst),
        "sign_agreement": agree,
        "sign_agreement_rate": agree/len(common),
        "mean_based_rel_change_pct": (sm-bm)/bm*100.0,
        "median_based_rel_change_pct": (smed-bmed)/bmed*100.0,
        "val_mean_rel_change_pct": (st.mean(ss["val"])-st.mean(sb["val"]))/st.mean(sb["val"])*100.0,
        "test_rel_change_leave_one_seed_out_range_pct": loo_range,
    }
    report["tasks"][task] = entry

OUT.write_text(json.dumps(report, indent=2))
for t, e in report["tasks"].items():
    p = e["paired"]
    print(f"== {t} (n_test={e['n_test_samples']}) ==")
    for arm in ["baseline", "best"]:
        a = e["arms"][arm]
        print(f"  {arm:8s} val_mean={a['val_mean']:.4f} test_mean={a['test_mean']:.4f} "
              f"test_med={a['test_median']:.4f} r(val,test)={a['pearson_val_test']:+.3f} "
              f"rho={a['spearman_val_test']:+.3f}")
    ci_r = [round(x,3) for x in p['pearson_dval_dtest_bootstrap_ci95']]
    loo_r = [round(x,2) for x in p['test_rel_change_leave_one_seed_out_range_pct']]
    print(f"  paired: r(dval,dtest)={p['pearson_dval_dtest']:+.3f} CI95={ci_r} "
          f"rho={p['spearman_dval_dtest']:+.3f} sign_agree={p['sign_agreement']}/{p['n_common_seeds']} ({p['sign_agreement_rate']*100:.0f}%)")
    print(f"  val_rel={p['val_mean_rel_change_pct']:+.2f}%  test_rel(mean)={p['mean_based_rel_change_pct']:+.2f}%  "
          f"test_rel(median)={p['median_based_rel_change_pct']:+.2f}%  LOO_range={loo_r}")
print(f"\nwrote {OUT.relative_to(ROOT)}")
