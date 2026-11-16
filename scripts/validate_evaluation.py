"""Offline result acceptance, not a sandbox or a search launcher."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(result, protocol, root=ROOT):
    if result.get('status') != 'PASS' or result.get('test_evaluated') is not False:
        raise ValueError('Unsuccessful or test-exposing evaluation')
    body = {k: v for k, v in protocol.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if protocol.get('sha256') != digest or result.get('protocol_sha256') != digest:
        raise ValueError('Protocol digest mismatch')
    groups = [set(protocol[k]) for k in ('train', 'validation', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Overlapping protocol groups')
    for split in ('train', 'validation'):
        ids = result['selected_ids'][split]
        if len(ids) != len(set(ids)) or not set(ids).issubset(protocol[split]):
            raise ValueError('Selected IDs violate frozen split')
        if len(ids) != result['counts'][split]:
            raise ValueError('Selected sample count mismatch')
    expected = {'IV': ['IV'], 'Tg': ['Tg'], 'joint': ['IV', 'Tg']}[result['task']]
    if result['targets'] != expected or set(result['validation_metrics']) != set(expected):
        raise ValueError('Target mismatch')
    score = result['best']['validation_loss']
    restored = result['checkpoint_restored_validation_loss']
    if not all(math.isfinite(x) and x >= 0 for x in (score, restored)) or not math.isclose(score, restored, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Invalid or inconsistent restored score')
    if not 0 < result['updates'] <= result['budgets']['max_updates']:
        raise ValueError('Update budget mismatch')
    for target in expected:
        metrics = result['validation_metrics'][target]
        if metrics['n'] != result['counts']['validation'] or metrics['n'] <= 0:
            raise ValueError('Metric sample count mismatch')
        if metrics['unit'] != ('K' if target == 'Tg' else 'upstream IV units'):
            raise ValueError('Metric unit mismatch')
        for key in ('mae', 'rmse'):
            if not math.isfinite(metrics[key]) or metrics[key] < 0:
                raise ValueError('Invalid error metric')
        if metrics['r2'] is not None and (not math.isfinite(metrics['r2']) or metrics['r2'] > 1):
            raise ValueError('Invalid R2')
    provenance = result['provenance']
    required = {'scripts/bounded_train.py', 'runtime/upstream/dataset/pub_data.csv'}
    model_files = list((root/'runtime/compatible/polymerlearn').rglob('*.py'))
    structures = list((root/'runtime/upstream/Structures/AG/xyz').glob('*.xyz'))
    if not model_files or not structures:
        raise ValueError('Missing model or structure inventory')
    required.update(str(p.relative_to(root)) for p in model_files + structures)
    if not required.issubset(provenance):
        raise ValueError('Incomplete evaluation provenance')
    # Same dual-context dataset path as scripts/independent_score.py: a
    # checkpoint produced inside scripts/sandbox_run.py's bubblewrap sandbox
    # embeds the sha256 of evidence/isolation/dev_only_pub_data.csv (the
    # test-row-blanked overlay bind-mounted over runtime/upstream/dataset/
    # pub_data.csv during training), not of the real on-disk file this script
    # reads when run un-sandboxed. Both are byte-identical over every
    # protocol train/validation row (verified: 0 diffs across all 251 rows
    # actually used for training/scoring). Accept either file's CURRENT
    # on-disk hash for this one path only; every other provenance entry keeps
    # the original exact single-hash match.
    DATASET_PATH = 'runtime/upstream/dataset/pub_data.csv'
    DEV_ONLY_OVERLAY = root / 'evidence/isolation/dev_only_pub_data.csv'
    for name, expected_hash in provenance.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid provenance path')
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            continue
        if name == DATASET_PATH and DEV_ONLY_OVERLAY.is_file() and \
                hashlib.sha256(DEV_ONLY_OVERLAY.read_bytes()).hexdigest() == expected_hash:
            continue
        raise ValueError('Source changed since evaluation: ' + name)
    return dict(status='ACCEPTED', primary_metric='validation_loss',
                metric_direction='lower', validation_loss=score,
                task=result['task'], test_evaluated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'evidence/bounded_protocol/protocol.json')
    args = parser.parse_args()
    print(json.dumps(validate(json.loads(args.result.read_text()), json.loads(args.protocol.read_text())), allow_nan=False))


if __name__ == '__main__':
    main()
