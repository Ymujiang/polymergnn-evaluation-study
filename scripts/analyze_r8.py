"""R8 一次性测试集评估——配对统计重算脚本。

输入：evidence/r8_sealed_test_eval/{IV,Tg,joint}.json（R8 已密封、不可重跑的结果）
输出：evidence/r8_test_stats.json

统计口径与 scripts/analyze_round2.py（② 轮开发集配对统计）完全一致：
- 相对改善 = (baseline_mean - best_mean) / baseline_mean * 100%（均值之比，非逐种子均值）
- 配对 bootstrap 95% CI：对共同种子索引做 20000 次重抽样
- Wilcoxon 符号秩检验（精确法，双侧）
- Holm-Bonferroni 跨 3 任务校正（对 raw Wilcoxon p）

诊断说明（joint-baseline 异常）：
joint baseline 臂 test_loss 均值 18.77（远超历史 validation_loss 量级 0.08-0.17），
经逐样本重算证实为真实模型失败，非脚本 bug：用 best checkpoint 跑同一批 test 样本，
Tg 预测中位误差从 194.9K 降至 8.4K。该现象与 IDEA-001 修复的
joint_test.py glycol 分支 self.Asage→self.Gsage 拷贝粘贴 bug 完全吻合
（baseline 的 glycol 编码器权重从未被训练）。详见 experiment_ledger.md §10.3。
"""
import json
import numpy as np
from scipy.stats import wilcoxon

RNG_SEED = 12345
N_BOOT = 20000

def load_task(task: str) -> dict:
    with open(f'evidence/r8_sealed_test_eval/{task}.json') as f:
        d = json.load(f)
    base = {int(s): v['test_loss'] for s, v in d['arms']['baseline']['per_seed'].items()}
    best = {int(s): v['test_loss'] for s, v in d['arms']['best']['per_seed'].items()}
    common = sorted(set(base) & set(best))
    b = np.array([base[s] for s in common])
    k = np.array([best[s] for s in common])
    return {'n_test_samples': d['n_test_samples'], 'seeds': common, 'baseline': b, 'best': k}

def paired_bootstrap_ci(b: np.ndarray, k: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    n = len(b)
    rel = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bb, kk = b[idx].mean(), k[idx].mean()
        rel[i] = (kk - bb) / bb * 100
    lo, hi = np.percentile(rel, [2.5, 97.5])
    return float(lo), float(hi)

def main():
    rng = np.random.default_rng(RNG_SEED)
    tasks = ['IV', 'Tg', 'joint']
    results = {}
    for task in tasks:
        data = load_task(task)
        b, k = data['baseline'], data['best']
        rel = float((k.mean() - b.mean()) / b.mean() * 100)
        ci_lo, ci_hi = paired_bootstrap_ci(b, k, rng)
        stat, p = wilcoxon(k - b)
        wins = int((k < b).sum())
        results[task] = {
            'n_seeds': len(data['seeds']),
            'n_test_samples': data['n_test_samples'],
            'baseline_mean_test_loss': float(b.mean()),
            'baseline_sd_test_loss': float(b.std(ddof=1)),
            'best_mean_test_loss': float(k.mean()),
            'best_sd_test_loss': float(k.std(ddof=1)),
            'relative_change_pct': rel,
            'bootstrap_ci95': [ci_lo, ci_hi],
            'wilcoxon_p_raw': float(p),
            'wins_best_lt_baseline': wins,
        }

    items = sorted(((t, results[t]['wilcoxon_p_raw']) for t in tasks), key=lambda x: x[1])
    m = len(items)
    prev = 0.0
    holm = {}
    for i, (t, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        adj = max(adj, prev)
        holm[t] = adj
        prev = adj
    for t in tasks:
        results[t]['holm_p'] = holm[t]

    with open('evidence/r8_test_stats.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
