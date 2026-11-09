"""Export the synthetic matrices used by the experiments as NumPy NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_experiments as exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'generated_data')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for n in (20, 50, 100):
        for condition in (100, 1000, 10000):
            seed = exp.MASTER_SEED + 1000 * n + condition
            problem = exp.make_problem(n, condition, seed)
            filename = f'A_n{n}_k{condition}.npz'
            np.savez_compressed(output / filename, sigma=problem.sigma, mu=problem.mu,
                                target=problem.target, w0=problem.w0, B=problem.B,
                                H=problem.H, c=problem.c)
            records.append({'file': filename, 'dataset': 'A', 'seed': seed, 'n': n,
                            'covariance_condition_number': condition})
    for replication in range(30):
        seed = exp.MASTER_SEED + 10_000 + replication
        sigma = exp.covariance(30, 300.0, seed)
        population = exp.make_problem(30, 300.0, seed, sigma)
        rng = np.random.default_rng(seed + 100_000)
        train = rng.multivariate_normal(population.mu / 252.0, sigma / 252.0, size=756)
        test = rng.multivariate_normal(population.mu / 252.0, sigma / 252.0, size=252)
        sample_covariance = np.cov(train, rowvar=False, ddof=1) * 252.0
        estimate = 0.95 * sample_covariance + 0.05 * np.trace(sample_covariance) / 30 * np.eye(30)
        filename = f'B_rep{replication:02d}.npz'
        np.savez_compressed(output / filename, train=train, test=test,
                            sigma_true=sigma, sigma_estimated=estimate,
                            sigma_test=np.cov(test, rowvar=False, ddof=1) * 252.0,
                            mu=population.mu, target=population.target)
        records.append({'file': filename, 'dataset': 'B', 'replication': replication,
                        'covariance_seed': seed, 'return_seed': seed + 100_000,
                        'train_shape': [756, 30], 'test_shape': [252, 30]})
    (output / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Exported 9 covariance cases and 30 training/test cases to {output}')


if __name__ == '__main__':
    main()
