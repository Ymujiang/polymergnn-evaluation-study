"""R8 diagnostic (read-only, extends diagnose_r8_joint_baseline.py) -- per-sample arrays
for Figure 4.

Reuses the exact same read-only inference approach as
scripts/diagnose_r8_joint_baseline.py (same checkpoints, same frozen 28-sample
test split, no re-sealing, no re-training). That script only persisted
summary statistics (median/max/min Tg error); this script additionally
persists the full per-sample Tg absolute error array (baseline vs best) and
the full per-sample IV predicted-vs-measured arrays on the SAME baseline
checkpoint, which Figure 4 needs for its two panels:
  (a) per-sample Tg error, baseline vs fixed, across the 28 joint test IDs
  (b) IV predicted vs measured on the same baseline checkpoint/samples

Output: evidence/r8_joint_baseline_tg_diagnosis_persample.json
"""
import sys, glob
sys.path.insert(0, 'runtime/compatible')
sys.path.insert(0, 'scripts')
import json
import torch
from torch_geometric.data import Batch
import sealed_test_eval as m
from polymerlearn.models.gnn import PolymerGNN_Joint_test as cls


def predictions_for(arm: str):
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
    recs = []
    for i, row in frame.iterrows():
        groups = []
        for columns in (frame.columns[20:33], frame.columns[34:46]):
            graphs = [graph_fn(c[1:]) for c in columns if row[c] > 0]
            groups.append(Batch.from_data_list(graphs))
        y = torch.tensor(row[targets].to_numpy(dtype=float), dtype=torch.float32)
        with torch.no_grad():
            output = model(*groups, torch.tensor(aux[i], dtype=torch.float32))
        pred = torch.cat([output[t].reshape(-1) for t in targets])
        recs.append({
            'resin_id': str(row.get('Resin#', i)),
            'iv_pred': float(pred[0]),
            'iv_true': float(y[0]),
            'tg_pred': float(pred[1]),
            'tg_true': float(y[1]),
            'tg_abs_err_K': abs(float(pred[1]) - float(y[1])),
        })
    return {'uuid': uuid, 'n': len(recs), 'records': recs}


def main():
    result = {
        'note': 'read-only re-inference on frozen R8 test checkpoints; no re-sealing, '
                 'extends diagnose_r8_joint_baseline.py with full per-sample arrays for Figure 4',
        'baseline': predictions_for('baseline'),
        'best': predictions_for('best'),
    }
    out_path = 'evidence/r8_joint_baseline_tg_diagnosis_persample.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {out_path}')
    print('baseline n =', result['baseline']['n'], ' best n =', result['best']['n'])


if __name__ == '__main__':
    main()
