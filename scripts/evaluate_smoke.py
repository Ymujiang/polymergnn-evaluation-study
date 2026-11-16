"""Offline AutoSOTA-facing smoke evaluation; never launches AutoSOTA or APIs.

Fixed engineering budget, NOT a scientifically converged baseline.
stdout contains exactly one accepted JSON object on success; logs go to disk.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

from validate_evaluation import ROOT, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['IV', 'Tg', 'joint'], default='joint')
    args = parser.parse_args()
    output = ROOT / 'evidence' / 'adapter_runs' / uuid.uuid4().hex
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / 'evidence/bounded_protocol/protocol.json'
    if not protocol.is_file():
        parser.error('Frozen protocol must exist before evaluation')
    command = [sys.executable, str(ROOT/'scripts/bounded_train.py'),
               '--task', args.task, '--epochs', '2', '--max-updates', '8',
               '--timeout-seconds', '120', '--batch-size', '4',
               '--learning-rate', '0.0001', '--protocol', str(protocol),
               '--output', str(output)]
    # Trainer owns the worker process-group timeout and preserves failure logs.
    completed = subprocess.run(command, cwd=ROOT, stdout=sys.stderr, stderr=sys.stderr)
    print('Evaluation artifacts: ' + str(output), file=sys.stderr)
    if completed.returncode:
        raise SystemExit(completed.returncode if completed.returncode > 0 else 1)
    accepted = validate(json.loads((output/'result.json').read_text()),
                        json.loads(protocol.read_text()))
    accepted['evaluation_kind'] = 'engineering_smoke_not_reproduction'
    accepted['artifact_directory'] = str(output.relative_to(ROOT))
    encoded = json.dumps(accepted, allow_nan=False)
    with (output/'acceptance.json').open('x') as handle:
        handle.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
