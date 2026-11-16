"""Independent, non-self-trusting re-scorer for bounded PolymerGNN training runs.

This script is the evaluator side of the isolation design: it must run as a
SEPARATE process launch from the candidate's training run (scripts/bounded_train.py
executed under scripts/sandbox_run.py), hold its own copy of the protocol and
scoring logic, and recompute validation metrics directly from the serialized
model checkpoint and the frozen dataset -- it never trusts the candidate's
self-reported result.json/validation_metrics. The candidate-reported numbers
are read only for a final consistency comparison against this script's own
independently computed numbers; a mismatch beyond tolerance is a FAIL, not a
warning.

Usage:
    scripts/independent_score.py --run-dir evidence/<run>/run \
        --protocol evidence/bounded_protocol/protocol.json

Exit code 0 only if:
  (1) the checkpoint's embedded protocol_sha256 matches the trusted protocol file,
  (2) every provenance hash embedded in the checkpoint matches the CURRENT
      on-disk file (trainer code, dataset, model source, structures), so the
      evaluator is scoring the code/data that actually produced this checkpoint,
  (3) this script's own freshly computed validation metrics agree with the
      candidate's self-reported validation_metrics within tolerance,
This is a baseline consistency checker, not a security boundary for arbitrary
candidate code. Structural acceptance is a separate validate_evaluation.py step.
Host-side source pinning and isolated candidate inference remain required.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPS = ['Mw (PS)', 'AN', 'OHN', '%TMP']


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild_validation_set(metadata, protocol):
    """Reconstruct the exact validation tensors from frozen inputs only,
    independently of any candidate-reported intermediate values."""
    import csv
    import numpy as np
    import torch
    from torch_geometric.data import Batch, Data
    sys.path.insert(0, str(ROOT / 'runtime/compatible'))
    from polymerlearn.utils.graph_prep import convert_xyz_to_mol, get_atom_features

    source = ROOT / 'runtime/upstream/dataset/pub_data.csv'
    with source.open(newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = [row for row in reader if row['Resin#'] in set(protocol['train'] + protocol['validation'])]
    import pandas as pd
    frame = pd.DataFrame(rows, columns=columns)
    for col in frame.columns:
        if col != 'Resin#':
            frame[col] = pd.to_numeric(frame[col], errors='coerce')

    targets = metadata['targets']
    vframe = frame[frame['Resin#'].isin(protocol['validation'])].copy()
    vframe = vframe[vframe[targets].notna().all(axis=1)].reset_index(drop=True)
    if 'Tg' in targets:
        vframe['Tg'] += 273.15
    if vframe['Resin#'].tolist() != metadata['selected_ids']['validation']:
        raise ValueError('Reconstructed validation ID set does not match checkpoint metadata; '
                          'cannot independently verify without exact sample identity')

    # Fit only on the frozen training partition, never on checkpoint statistics.
    train = frame[frame['Resin#'].isin(protocol['train'])].copy()
    train = train[train[targets].notna().all(axis=1)]
    values_train = train[PROPS].to_numpy(dtype=float)
    if np.isinf(values_train).any() or np.isnan(values_train).all(axis=0).any():
        raise ValueError('Invalid training auxiliary values')
    med = np.nanmedian(values_train, axis=0)
    filled = np.where(np.isnan(values_train), med, values_train)
    std = filled.std(axis=0)
    std[std == 0] = 1.
    stats = dict(median=med, mean=filled.mean(axis=0), scale=std)
    for key, value in stats.items():
        reported_value = np.asarray(metadata['auxiliary'][key])
        if reported_value.shape != value.shape or not np.allclose(value, reported_value, rtol=1e-12, atol=1e-12):
            raise ValueError('Checkpoint preprocessing mismatch: ' + key)
    values = vframe[PROPS].to_numpy(dtype=float)
    median, mean, scale = (np.asarray(stats[k]) for k in ('median', 'mean', 'scale'))
    aux = (np.where(np.isnan(values), median, values) - mean) / scale

    cache = {}
    structures = ROOT / 'runtime/upstream/Structures/AG/xyz'

    def graph(name):
        if name not in cache:
            mol = convert_xyz_to_mol(str(structures / (name + '.xyz')))
            cache[name] = Data(x=get_atom_features(mol), edge_index=get_edge_index_local(mol))
        return cache[name].clone()

    from polymerlearn.utils.graph_prep import get_edge_index as get_edge_index_local

    samples = []
    for i, row in vframe.iterrows():
        groups = []
        for columns_slice in (vframe.columns[20:33], vframe.columns[34:46]):
            graphs = [graph(c[1:]) for c in columns_slice if row[c] > 0]
            if not graphs:
                raise ValueError('Empty monomer group during independent reconstruction')
            groups.append(Batch.from_data_list(graphs))
        y = torch.tensor(row[targets].to_numpy(dtype=float), dtype=torch.float32)
        samples.append((*groups, torch.tensor(aux[i], dtype=torch.float32), y))
    return samples


def independent_metrics(samples, targets, state_dict, task):
    import torch
    sys.path.insert(0, str(ROOT / 'runtime/compatible'))
    from polymerlearn.models.gnn import PolymerGNN_Joint_test, PolymerGNN_Tg
    from polymerlearn.models.gnn.ablation_models.iv_mono import PolymerGNN_IVMono
    cls, hidden = {'IV': (PolymerGNN_IVMono, 32), 'Tg': (PolymerGNN_Tg, 32),
                   'joint': (PolymerGNN_Joint_test, 64)}[task]
    model = cls(6, hidden, num_additional=4)
    model.load_state_dict(state_dict)
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for A, G, x, y in samples:
            output = model(A, G, x)
            pred = torch.cat([output[t].reshape(-1) for t in targets]) if isinstance(output, dict) else output.reshape(-1)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite independent prediction')
            predictions.append(pred.double())
            labels.append(y.double())
    predictions, labels = torch.stack(predictions), torch.stack(labels)
    metrics = {}
    for i, target in enumerate(targets):
        error = predictions[:, i] - labels[:, i]
        denominator = (labels[:, i] - labels[:, i].mean()).square().sum()
        metrics[target] = dict(
            mae=float(error.abs().mean()), rmse=float(error.square().mean().sqrt()),
            r2=float(1 - error.square().sum() / denominator) if denominator > 0 else None,
            unit='K' if target == 'Tg' else 'upstream IV units', n=len(labels))
    return metrics


def compare_metrics(independent, reported, rel_tol=1e-4, abs_tol=1e-4):
    for target, imetrics in independent.items():
        rmetrics = reported.get(target)
        if rmetrics is None:
            raise ValueError('Candidate-reported metrics missing target: ' + target)
        for key in ('mae', 'rmse'):
            if not math.isclose(imetrics[key], rmetrics[key], rel_tol=rel_tol, abs_tol=abs_tol):
                raise ValueError(f'Independent/candidate metric mismatch for {target}.{key}: '
                                  f'{imetrics[key]} vs {rmetrics[key]}')
        if imetrics['r2'] is None or rmetrics['r2'] is None:
            if imetrics['r2'] != rmetrics['r2']:
                raise ValueError(f'Independent/candidate R2 definedness mismatch for {target}')
        elif not math.isclose(imetrics['r2'], rmetrics['r2'], rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(f'Independent/candidate metric mismatch for {target}.r2: '
                              f'{imetrics["r2"]} vs {rmetrics["r2"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True,
                    help='Directory containing best.pt and result.json from a bounded_train.py run')
    p.add_argument('--protocol', type=Path, default=ROOT / 'evidence/bounded_protocol/protocol.json')
    args = p.parse_args()

    import torch
    protocol = json.loads(args.protocol.read_text())
    expected_protocol = 'dc49ec25644861dd9da73ff14b6e82ffdaf92d467ea2fe834cd7a27955ae4be1'
    body = {k: v for k, v in protocol.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != expected_protocol or protocol['sha256'] != expected_protocol:
        raise ValueError('Frozen protocol content mismatch')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.run_dir / 'best.pt', map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    reported = json.loads((args.run_dir / 'result.json').read_text())
    expected_targets = {'joint': ['IV', 'Tg'], 'IV': ['IV'], 'Tg': ['Tg']}[metadata['task']]
    if metadata['targets'] != expected_targets:
        raise ValueError('Task/target mismatch')
    required = [ROOT/'scripts/bounded_train.py', ROOT/'runtime/upstream/dataset/pub_data.csv']
    required += sorted((ROOT/'runtime/compatible/polymerlearn').rglob('*.py'))
    required += sorted((ROOT/'runtime/upstream/Structures/AG/xyz').glob('*.xyz'))
    if set(metadata['provenance']) != {str(p.relative_to(ROOT)) for p in required}:
        raise ValueError('Incomplete provenance manifest')

    if metadata['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Checkpoint protocol_sha256 does not match the trusted frozen protocol')

    # The dataset path is special-cased: scripts/sandbox_run.py intentionally
    # bind-mounts evidence/isolation/dev_only_pub_data.csv OVER
    # runtime/upstream/dataset/pub_data.csv inside the sandbox (blanking every
    # column except Resin# for the 45 frozen test rows), so a checkpoint
    # produced by a SANDBOXED training run embeds the sha256 of that blanked
    # overlay file, not of the real on-disk file this script sees when run
    # un-sandboxed on the host. Both files are byte-identical over every
    # protocol['train'] + protocol['validation'] row (verified: 0 diffs across
    # all 251 rows actually used for training/validation; only the 45 held-out
    # test rows, which neither this script nor the trainer reads values from,
    # differ). Accept the dataset hash if it matches either file's CURRENT
    # on-disk hash -- both are read fresh below, never a hardcoded constant --
    # so a silent future edit to either file still fails this check. Every
    # other provenance entry (trainer script, model code, structures) keeps
    # the original single-hash exact match: only this one path has a
    # legitimate dual-context origin (sandboxed vs. direct run).
    DATASET_PATH = 'runtime/upstream/dataset/pub_data.csv'
    DEV_ONLY_OVERLAY = ROOT / 'evidence/isolation/dev_only_pub_data.csv'
    for name, expected_hash in metadata['provenance'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError('Invalid provenance path in checkpoint metadata: ' + name)
        actual_hash = sha256_file(path)
        if actual_hash == expected_hash:
            continue
        if name == DATASET_PATH and DEV_ONLY_OVERLAY.is_file() and sha256_file(DEV_ONLY_OVERLAY) == expected_hash:
            continue
        raise ValueError('Provenance mismatch, code/data changed since checkpoint was produced: ' + name)

    samples = rebuild_validation_set(metadata, protocol)
    own_metrics = independent_metrics(samples, metadata['targets'], checkpoint['state_dict'], metadata['task'])
    compare_metrics(own_metrics, reported['validation_metrics'])

    output = dict(
        status='BASELINE_RESCORE_AGREES',
        task=metadata['task'],
        protocol_sha256=protocol['sha256'],
        independent_validation_metrics=own_metrics,
        candidate_reported_validation_metrics=reported['validation_metrics'],
        agreement='within rel_tol=1e-4, abs_tol=1e-4',
        test_evaluated=False,
        note='Computed by a separate process launch from a fresh checkpoint load; '
             'never trusted the training run\'s self-reported metrics beyond this comparison.',
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
