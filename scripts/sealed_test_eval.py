"""One-shot sealed test-set evaluation (R8).

CRITICAL: This script touches evidence/bounded_protocol/protocol.json's
'test' split (45 frozen Resin# IDs) for the FIRST AND ONLY time. It must
not be re-run against the same task's checkpoints after a real result
file has been written; see the sealing mechanism below.

Design goals (per project policy, see R9/multiseed findings):
  - Evaluate the round-2 (statistically-corrected, multiseed) authoritative
    checkpoints, not the round-1 single-seed-2026 ones, because seed 2026
    was found to be an unusually unfavorable baseline seed.
  - Evaluate EVERY available seed's checkpoint for baseline and for best,
    on the SAME frozen test split, and report the full per-seed
    distribution (mean +/- sd) rather than a single point estimate,
    following the paper's own thesis about fixed-seed unreliability.
  - Pair baseline(seed=s) vs best(seed=s) across the same seed values to
    allow a paired test-set comparison (Wilcoxon signed-rank / paired t),
    matching the methodology already used for R7/R12 on validation splits.
  - Never refit auxiliary standardization or target scaling on test data:
    always reuse the frozen stats saved inside each checkpoint's metadata
    (fit on that run's TRAIN split only, exactly as scripts/bounded_train.py
    does for validation).
  - Verify checkpoint identity via full SHA-256 provenance-dict equality
    (not a single file's hash) before trusting any checkpoint, learned the
    hard way from an earlier false-alarm audit of Tg's graph_prep.py.
  - Refuse to run twice: write a sealed result file per task and abort if
    it already exists.

Usage:
  python scripts/sealed_test_eval.py --task joint --confirm I_UNDERSTAND_THIS_IS_IRREVERSIBLE
  python scripts/sealed_test_eval.py --task all   --confirm I_UNDERSTAND_THIS_IS_IRREVERSIBLE

Output:
  evidence/r8_sealed_test_eval/<task>.json   (refuses to overwrite)
"""
import argparse
import glob
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPS = ['Mw (PS)', 'AN', 'OHN', '%TMP']
PROTOCOL_PATH = ROOT / 'evidence/bounded_protocol/protocol.json'
FROZEN_PROTOCOL_SHA256 = 'dc49ec25644861dd9da73ff14b6e82ffdaf92d467ea2fe834cd7a27955ae4be1'

# Canonical round-2 (authoritative, multiseed) checkpoint groups.
# Each entry is a list of uuid prefixes under evidence/autosota_runs_ms/<task>/
# whose run/result.json full 'provenance' dict is byte-identical (verified by
# full-dict equality, not a single file's hash). Some groups are split across
# multiple uuids because the autosota harness re-ran partial seed sweeps; the
# union of their available seeds gives full coverage with some redundancy.
CANONICAL_GROUPS = {
    'IV': {
        'model_file': 'runtime/compatible/polymerlearn/models/gnn/ablation_models/iv_mono.py',
        'best': {
            'uuids': ['1b9858ffafad430c8b73fdf7ec4138b2',
                      '556b0c2545ad4fdc891751910ace75b4',
                      '747aff0cca494edaaa091fb368a49bd3'],
            'expected_model_hash': '02e531250e',
            'label': 'IDEA-019 (SAGPooling ratio 0.5 -> 0.7), commit 21e8b07',
        },
        'baseline': {
            'uuids': ['2dff30a277374fdb9e2d6b6e6ee21aaf',
                      '34f29245e9524383bd4955ad0f99a810',
                      'fa747ed4fc434accb7afab0a366125d1'],
            'expected_model_hash': '5b4ad1b7f1',
            'label': 'round-2 baseline, commit 6cc294c5',
        },
        'hidden': 32,
    },
    'Tg': {
        'model_file': 'runtime/compatible/polymerlearn/models/gnn/tg.py',
        'best': {
            'uuids': ['460074b019b446f08ffa760f705ca301'],
            'expected_model_hash': '4ce3afff62',
            'label': 'IDEA-003 (mult_factor gate-bias init=1.5), commit ef5a381',
        },
        'baseline': {
            'uuids': ['2c916befad2640d4a7afa95a8464ced9'],
            'expected_model_hash': '4fef20f780',
            'label': 'round-2 baseline, commit a8011cba',
        },
        'hidden': 32,
    },
    'joint': {
        'model_file': 'runtime/compatible/polymerlearn/models/gnn/joint_test.py',
        'best': {
            'uuids': ['0e28fb4e541941899423ddc0ce3f0e48'],
            'expected_model_hash': 'fe96748775',
            'label': 'IDEA-001 (Asage/Gsage bugfix), commit ecf854a',
        },
        'baseline': {
            'uuids': ['6f4a6663b65940f1a2e3bb1488be9ee9'],
            'expected_model_hash': '1e368755ef',
            'label': 'round-2 baseline, commit f8bfe34b',
        },
        'hidden': 64,
    },
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get('sha256') != FROZEN_PROTOCOL_SHA256:
        raise ValueError('Protocol hash does not match the frozen protocol; refusing to proceed')
    return protocol


def read_split(source_csv, ids):
    import csv
    import pandas as pd
    allowed = set(ids)
    with Path(source_csv).open(newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = [row for row in reader if row['Resin#'] in allowed]
    frame = pd.DataFrame(rows, columns=columns)
    for col in frame.columns:
        if col != 'Resin#':
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return frame


def transform_aux(values, stats):
    import numpy as np
    values = np.asarray(values, dtype=float)
    result = (np.where(np.isnan(values), stats['median'], values) - stats['mean']) / stats['scale']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite transformed auxiliaries on test split')
    return result


def collect_checkpoints(task, arm):
    spec = CANONICAL_GROUPS[task][arm]
    results = {}
    for uuid in spec['uuids']:
        for run_dir in sorted(glob.glob(str(ROOT / f'evidence/autosota_runs_ms/{task}/{uuid}_s*'))):
            seed = int(Path(run_dir).name.rsplit('_s', 1)[1])
            ckpt_path = Path(run_dir) / 'run' / 'best.pt'
            result_path = Path(run_dir) / 'run' / 'result.json'
            if not ckpt_path.is_file() or not result_path.is_file():
                continue
            results.setdefault(seed, (ckpt_path, result_path))  # first match wins; groups are redundant copies
    return results


def verify_and_load(task, arm):
    import torch
    spec = CANONICAL_GROUPS[task][arm]
    model_file = CANONICAL_GROUPS[task]['model_file']
    checkpoints = collect_checkpoints(task, arm)
    if not checkpoints:
        raise ValueError(f'No checkpoints found for {task}/{arm}')
    loaded = {}
    reference_provenance = None
    for seed, (ckpt_path, result_path) in sorted(checkpoints.items()):
        result = json.loads(result_path.read_text())
        if result.get('protocol_sha256') != FROZEN_PROTOCOL_SHA256:
            raise ValueError(f'{result_path}: protocol hash mismatch, refusing to trust this checkpoint')
        if result.get('test_evaluated') is not False:
            raise ValueError(f'{result_path}: test_evaluated flag is not False, aborting out of caution')
        model_hash = result.get('provenance', {}).get(model_file, '')[:10]
        if model_hash != spec['expected_model_hash']:
            raise ValueError(
                f'{result_path}: model file hash {model_hash!r} does not match expected '
                f'{spec["expected_model_hash"]!r} for {task}/{arm} ({spec["label"]})')
        provenance = result.get('provenance')
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise ValueError(f'{result_path}: full provenance dict differs within the same canonical group; '
                              f'refusing to mix checkpoints from different code states')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        loaded[seed] = dict(state_dict=ckpt['state_dict'], metadata=ckpt['metadata'],
                             validation_loss=ckpt['validation_loss'])
    return loaded, spec['label']


def build_test_dataset(task, hidden_dim):
    import torch
    from torch_geometric.data import Batch, Data
    sys.path.insert(0, str(ROOT / 'runtime/compatible'))
    from polymerlearn.utils.graph_prep import convert_xyz_to_mol, get_atom_features, get_edge_index

    protocol = load_protocol()
    source = ROOT / 'runtime/upstream/dataset/pub_data.csv'
    frame = read_split(source, protocol['test'])
    targets = ['IV', 'Tg'] if task == 'joint' else [task]
    frame = frame[frame[targets].notna().all(axis=1)].reset_index(drop=True)
    if frame.empty:
        raise ValueError('Empty sealed test split for this task (complete-case filter removed everything)')
    if 'Tg' in targets:
        frame = frame.copy()
        frame['Tg'] += 273.15

    cache = {}
    structures = ROOT / 'runtime/upstream/Structures/AG/xyz'

    def graph(name):
        if name not in cache:
            mol = convert_xyz_to_mol(str(structures / (name + '.xyz')))
            cache[name] = Data(x=get_atom_features(mol), edge_index=get_edge_index(mol))
        return cache[name].clone()

    return frame, targets, graph


def evaluate_one_checkpoint(task, hidden_dim, state_dict, metadata, frame, targets, graph_fn):
    import torch
    from torch_geometric.data import Batch
    if task == 'IV':
        from polymerlearn.models.gnn.ablation_models.iv_mono import PolymerGNN_IVMono as cls
    elif task == 'Tg':
        from polymerlearn.models.gnn import PolymerGNN_Tg as cls
    else:
        from polymerlearn.models.gnn import PolymerGNN_Joint_test as cls

    aux = transform_aux(frame[PROPS].to_numpy(), metadata['auxiliary'])
    samples = []
    for i, row in frame.iterrows():
        groups = []
        for columns in (frame.columns[20:33], frame.columns[34:46]):
            graphs = [graph_fn(c[1:]) for c in columns if row[c] > 0]
            if not graphs:
                raise ValueError('Empty monomer group on sealed test row')
            groups.append(Batch.from_data_list(graphs))
        y = torch.tensor(row[targets].to_numpy(dtype=float), dtype=torch.float32)
        samples.append((*groups, torch.tensor(aux[i], dtype=torch.float32), y))

    model = cls(6, hidden_dim, num_additional=4)
    model.load_state_dict(state_dict)
    model.eval()

    scale = torch.tensor(metadata['target_scale'], dtype=torch.float64)
    predictions, labels = [], []
    with torch.no_grad():
        for A, G, x, y in samples:
            output = model(A, G, x)
            pred = torch.cat([output[t].reshape(-1) for t in targets]) if isinstance(output, dict) else output.reshape(-1)
            predictions.append(pred.double())
            labels.append(y.double())
    predictions, labels = torch.stack(predictions), torch.stack(labels)
    test_loss = float(((predictions - labels) / scale).square().mean())

    metrics = {}
    for i, target in enumerate(targets):
        error = predictions[:, i] - labels[:, i]
        denom = (labels[:, i] - labels[:, i].mean()).square().sum()
        metrics[target] = dict(
            mae=float(error.abs().mean()), rmse=float(error.square().mean().sqrt()),
            r2=float(1 - error.square().sum() / denom) if denom > 0 else None,
            unit='K' if target == 'Tg' else 'upstream IV units', n=len(labels))
    return test_loss, metrics


def run_task(task, output_dir):
    sealed_path = output_dir / f'{task}.json'
    if sealed_path.exists():
        raise SystemExit(f'REFUSING TO RUN: {sealed_path} already exists. '
                          f'R8 is a one-shot evaluation; this task has already been sealed.')

    spec = CANONICAL_GROUPS[task]
    hidden_dim = spec['hidden']
    frame, targets, graph_fn = build_test_dataset(task, hidden_dim)
    n_test = len(frame)

    report = dict(task=task, targets=targets, n_test_samples=n_test,
                   protocol_sha256=FROZEN_PROTOCOL_SHA256, arms={})
    for arm in ('baseline', 'best'):
        loaded, label = verify_and_load(task, arm)
        per_seed = {}
        for seed, payload in sorted(loaded.items()):
            test_loss, metrics = evaluate_one_checkpoint(
                task, hidden_dim, payload['state_dict'], payload['metadata'], frame, targets, graph_fn)
            per_seed[seed] = dict(test_loss=test_loss, metrics=metrics,
                                   validation_loss=payload['validation_loss'])
        losses = [v['test_loss'] for v in per_seed.values()]
        report['arms'][arm] = dict(
            label=label, n_seeds=len(per_seed),
            mean_test_loss=statistics.mean(losses), sd_test_loss=statistics.pstdev(losses),
            min_test_loss=min(losses), max_test_loss=max(losses),
            per_seed={str(s): v for s, v in per_seed.items()})

    output_dir.mkdir(parents=True, exist_ok=True)
    with sealed_path.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'Sealed R8 result written to {sealed_path}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['IV', 'Tg', 'joint', 'all'], required=True)
    p.add_argument('--confirm', required=True,
                   help='Must be exactly I_UNDERSTAND_THIS_IS_IRREVERSIBLE to proceed')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'evidence/r8_sealed_test_eval')
    args = p.parse_args()
    if args.confirm != 'I_UNDERSTAND_THIS_IS_IRREVERSIBLE':
        p.error('--confirm must be exactly I_UNDERSTAND_THIS_IS_IRREVERSIBLE')
    tasks = ['IV', 'Tg', 'joint'] if args.task == 'all' else [args.task]
    for task in tasks:
        run_task(task, args.output_dir)


if __name__ == '__main__':
    main()
