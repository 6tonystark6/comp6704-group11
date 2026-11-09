"""Run the complete study in a new directory without replacing report results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT_FILES = ('controlled.csv', 'ablation.csv', 'sampling.csv', 'summary.json')
FIGURE_FILES = ('convergence_algorithms.pdf', 'controlled_scaling.pdf',
                'step_ablation.pdf', 'sampling_algorithms.pdf', 'sampling_risk.pdf')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def run_script(folder, filename):
    print(f'Running {filename}', flush=True)
    with (folder / f'{Path(filename).stem}.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(
            [sys.executable, '-u', filename], cwd=folder, stdout=log,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f'{filename} failed. See {folder / (Path(filename).stem + ".log")}')


def compare_csv(reference, rerun):
    """Compare recorded numerical values while excluding machine-dependent times."""
    original, current = read_csv(reference), read_csv(rerun)
    if len(original) != len(current):
        raise AssertionError(f'Unexpected row count in {rerun.name}')
    identity = ('algorithm', 'n', 'kappa_full', 'replication', 'step_fraction')
    differences = []
    counts_match = 0
    for index, (old, new) in enumerate(zip(original, current), 1):
        if old.keys() != new.keys():
            raise AssertionError(f'Unexpected columns in {rerun.name}')
        for key in identity:
            if key in old and old[key] != new[key]:
                raise AssertionError(f'Case ordering changed in {rerun.name}, row {index}')
        counts_match += int(old['iterations'] == new['iterations'])
        if new['converged'] != 'True':
            raise AssertionError(f'Nonconverged run in {rerun.name}, row {index}')
        for key in ('budget_error', 'return_error'):
            if key in new and float(new[key]) >= 1e-12:
                raise AssertionError(f'Feasibility check failed in row {index}')
        for key, value in old.items():
            if key.startswith('seconds_'):
                continue
            try:
                old_number, new_number = float(value), float(new[key])
            except ValueError:
                if value != new[key]:
                    differences.append({'row': index, 'column': key, 'reference': value, 'rerun': new[key]})
                continue
            if not math.isclose(old_number, new_number, rel_tol=1e-7, abs_tol=1e-12):
                differences.append({'row': index, 'column': key, 'reference': old_number, 'rerun': new_number})
    return {'rows': len(current), 'converged': len(current),
            'matching_update_counts': counts_match,
            'non_timing_differences': differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduction_output',
                        help='A new directory. Existing directories are never overwritten.')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output == ROOT or output in ROOT.parents:
        parser.error('Choose a new output directory, separate from the repository root.')
    output.mkdir(parents=True, exist_ok=False)

    reference_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in RESULT_FILES}
    environment = {
        'python': platform.python_version(), 'numpy': np.__version__,
        'matplotlib': matplotlib.__version__, 'system': platform.system(),
        'system_release': platform.release(), 'machine': platform.machine(),
        'thread_environment': {name: os.environ.get(name) for name in
                               ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')},
        'reference_sha256': reference_hashes,
    }
    config = getattr(np.__config__, 'CONFIG', {})
    environment['blas'] = {key: config.get('Build Dependencies', {}).get('blas', {}).get(key)
                           for key in ('name', 'version', 'openblas configuration')}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    for filename in ('run_experiments.py', 'audit_report.py'):
        shutil.copy2(ROOT / filename, output / filename)
    run_script(output, 'run_experiments.py')
    run_script(output, 'audit_report.py')

    comparison = {name: compare_csv(ROOT / name, output / name)
                  for name in RESULT_FILES if name.endswith('.csv')}
    audit = json.loads((output / 'review_20261109' / 'accuracy_audit.json').read_text(encoding='utf-8'))
    if audit['cg_runs'] != 39 or audit['true_gradient_passes'] != 39:
        raise AssertionError('The final true-gradient check must pass all 39 CG cases.')
    if audit['matching_update_counts'] != 39:
        raise AssertionError('CG reruns within this environment changed update counts.')
    for name in FIGURE_FILES:
        if not (output / 'figures' / name).read_bytes().startswith(b'%PDF'):
            raise AssertionError(f'Missing or invalid generated figure {name}')
    for name, digest in reference_hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError('A report result file was modified.')

    comparison['cg_true_gradient_passes'] = audit['true_gradient_passes']
    comparison['generated_figures'] = len(FIGURE_FILES)
    comparison['reference_files_unchanged'] = True
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    print(json.dumps({name: {key: value for key, value in result.items()
                             if key != 'non_timing_differences'}
                      if isinstance(result, dict) else result
                      for name, result in comparison.items()}, indent=2), flush=True)
    differences = sum(len(result.get('non_timing_differences', []))
                      for result in comparison.values() if isinstance(result, dict))
    print(f'Non-timing differences from the recorded CSVs: {differences}', flush=True)
    print(f'Results and logs saved in {output}', flush=True)


if __name__ == '__main__':
    main()
