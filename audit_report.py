"""Audit reported accuracy without overwriting experiment results or timings."""
import csv
import json
from pathlib import Path
import numpy as np
import run_experiments as exp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'review_20261109'
OUT.mkdir(exist_ok=True)

def rows(name):
    with (ROOT / name).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))

controlled, sampling = rows('controlled.csv'), rows('sampling.csv')
checks = []
def check(problem, recorded, case):
    run = exp.conjugate_gradient(problem)
    true_ratio = float(np.linalg.norm(exp.gradient(problem, run['z'])) / max(np.linalg.norm(problem.c), 1e-15))
    checks.append(dict(case=case, iterations=run['iterations'], recorded_iterations=int(recorded['iterations']),
                       recursive_residual_ratio=run['gradient_ratio'], true_gradient_ratio=true_ratio,
                       ratio_difference=abs(true_ratio-run['gradient_ratio']),
                       true_gradient_pass=bool(true_ratio <= exp.GRAD_RTOL)))

for row in controlled:
    if row['algorithm'] == 'CG':
        n, kappa = int(row['n']), float(row['kappa_full'])
        check(exp.make_problem(n, kappa, exp.MASTER_SEED+1000*n+int(kappa)), row, f'A_n{n}_k{int(kappa)}')
for j in range(30):
    seed = exp.MASTER_SEED+10_000+j
    sigma = exp.covariance(30, 300., seed)
    p_true = exp.make_problem(30, 300., seed, sigma)
    rng = np.random.default_rng(seed+100_000)
    train = rng.multivariate_normal(p_true.mu/252., sigma/252., size=756)
    sample_cov = np.cov(train, rowvar=False, ddof=1)*252.
    estimate = .95*sample_cov+.05*np.trace(sample_cov)/30*np.eye(30)
    row = next(r for r in sampling if r['algorithm']=='CG' and int(r['replication'])==j)
    check(exp.make_problem(30, 300., seed, estimate), row, f'B_rep{j}')

result = dict(cg_runs=len(checks), true_gradient_passes=sum(r['true_gradient_pass'] for r in checks),
              matching_update_counts=sum(r['iterations']==r['recorded_iterations'] for r in checks),
              max_true_gradient_ratio=max(r['true_gradient_ratio'] for r in checks),
              max_residual_norm_ratio_difference=max(r['ratio_difference'] for r in checks), cg_checks=checks,
              median_weight_errors={a:float(np.median([float(r['solver_relative_weight_error']) for r in sampling if r['algorithm']==a])) for a in exp.ALGORITHM_ORDER})
for metric in ['population_vol_ratio_vs_oracle','test_vol_ratio_vs_equal']:
    result['max_within_replication_spread_'+metric]=max(np.ptp([float(r[metric]) for r in sampling if int(r['replication'])==j]) for j in range(30))
p = exp.make_problem(50, 1000., exp.MASTER_SEED+50_000+1000)
result['nag_parameter_pairs']=[dict(step_fraction=a, eta=a/p.lam_max, beta=(1-np.sqrt(a*p.lam_min/p.lam_max))/(1+np.sqrt(a*p.lam_min/p.lam_max))) for a in [.25,.5,1.]]
(OUT/'accuracy_audit.json').write_text(json.dumps(result, indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='cg_checks'},indent=2))
