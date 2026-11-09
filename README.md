# Portfolio Risk Minimization

Reproducibility materials for COMP6704 Group 11, Fall 2026.

Four NumPy implementations solve the same minimum-variance portfolio problem. A null-space parametrization eliminates the budget and target-return equalities, leaving an unconstrained, positive-definite quadratic. The methods are gradient descent (GD), Nesterov accelerated gradient (NAG), conjugate gradient (CG), and full-memory BFGS with exact quadratic line search. A direct linear solve supplies the numerical reference.

All data are synthetic and generated locally from fixed seeds. No market-data download, account, or API key is needed.

## Quick start

Use Python 3.11. The recorded software versions are NumPy 1.26.4 and Matplotlib 3.9.0, pinned in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell with `.\.venv\Scripts\Activate.ps1`, or on macOS/Linux with `source .venv/bin/activate`. Then run

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The reproduction command runs all three experiments, regenerates five PDF figures, and performs the separate final-gradient audit for all 39 CG cases. It saves new outputs in `reproduction_output/`. It leaves the committed CSV files and figures unchanged.

For another run, choose a new directory

```bash
python reproduce.py --output-dir reproduction_output_2
```

The output directory must not already exist. The two script logs, `environment.json`, and `comparison.json` record what ran and how it compares with the report results. Runtime depends on hardware, BLAS configuration, threading, and system load. The original timing environment was not recorded, so the new environment record describes only the new run.

## Repository contents

| File | Purpose |
| --- | --- |
| `run_experiments.py` | Data generation, all four solvers, three experiments, statistics, assertions, and plots |
| `audit_report.py` | Separate CG true-gradient checks and numerical/risk summary checks |
| `reproduce.py` | Runs the original scripts in a new directory, records the environment, and compares outputs |
| `export_data.py` | Optional export of all covariance and daily-return matrices |
| `requirements.txt` | Pinned numerical and plotting dependencies |
| `controlled.csv` | The 36 controlled results used in the report |
| `ablation.csv` | The three NAG step/momentum settings used in the report |
| `sampling.csv` | The 120 sampling results used in the report |
| `summary.json` | Experiment configuration, quantiles, bootstrap intervals, and aggregate results |
| `figures/` | The five vector figures used in the report |
| `review_20261109/accuracy_audit.json` | The saved audit supporting the corrected report |

The report and presentation are submitted separately. This repository contains the scientific code and evidence needed to reproduce their experimental results.

## Optimization problem and implementations

The asset weights minimize annual portfolio variance, subject to full investment and a 10% expected annual return. Forecast returns are evenly spaced from 4% to 16%. Short positions are allowed. The forecasts remain known inputs throughout the sampling study.

Writing the weights as `w0 + B @ z` eliminates both equalities. Here `w0` is feasible and the columns of `B` form an orthonormal null-space basis. Every method updates `z` while the reduced Hessian `H` and linear term `c` remain fixed.

| Method | Function in `run_experiments.py` | Implementation |
| --- | --- | --- |
| GD | `gradient_descent` | Fixed step from the smallest and largest eigenvalues of `H` |
| NAG | `accelerated_gradient` | Extrapolated evaluation point, followed by a gradient step |
| CG | `conjugate_gradient` | Linear CG for the reduced positive-definite system |
| BFGS | `bfgs` | Inverse-Hessian approximation with exact quadratic line search |

All runs start at zero, use float64 arithmetic, target a relative tolerance of `1e-8`, and allow at most 200,000 updates. GD, NAG, and BFGS test the directly evaluated gradient. CG tests the recursively updated residual. The CSV column `gradient_ratio` therefore has a different computational definition for CG, although the two quantities agree in exact arithmetic. The separate audit directly evaluates the true gradient at every final CG iterate.

The measured BFGS code directly multiplies dense matrices in its inverse-Hessian update. Its arithmetic cost per update is cubic in the reduced dimension, with quadratic extra storage. An algebraically expanded rank-two update could reduce arithmetic to quadratic cost, but it was not used for the reported measurements. GD and NAG each use two Hessian-vector products per timed loop, including the stopping check. CG uses one. Their extra vector storage excludes the common dense Hessian.

## Experiment 1 · controlled covariance problems

Dataset A crosses asset counts `20, 50, 100` with covariance condition numbers `100, 1000, 10000`. This gives nine problems and 36 solver runs.

Each covariance matrix uses a random orthogonal basis and geometrically spaced positive eigenvalues from `0.09 / condition_number` to `0.09`. This dataset contains covariance matrices, not daily return rows. The reduced Hessian condition number is recorded separately because it need not equal the covariance condition number.

The master seed is `670411`. Each case uses seed `670411 + 1000*n + condition_number`. Every solver is timed five times, and the median loop time is reported. Shared problem construction and spectral preprocessing are excluded.

Outputs include `controlled.csv`, `figures/convergence_algorithms.pdf`, and `figures/controlled_scaling.pdf`.

## Experiment 2 · paired NAG parameters

The problem is fixed at 50 assets and covariance condition number 1,000. The NAG step is one quarter, one half, or one times its standard value. Momentum is recalculated for every step setting.

This compares three matched step/momentum pairs. It does not isolate the effect of the step alone. The recorded update counts are 888, 624, and 439. The vertical axis of `figures/step_ablation.pdf` shows updates to convergence, not objective gaps.

## Experiment 3 · finite training data and independent evaluation

Dataset B contains 30 independent replications. Each replication has 30 synthetic assets, 756 training days, and 252 independent test days. Each row is a day and each column an asset. Returns are decimal simple returns, so `-0.014295` means `-1.4295%`.

Daily returns follow a multivariate Gaussian distribution. The daily mean and covariance are the annual inputs divided by 252. The population covariance condition number is 300. Replication `j`, indexed from zero, uses covariance seed `670411 + 10000 + j` and return seed `covariance_seed + 100000`. Training rows are drawn first and test rows next from the same seeded generator.

The training sample covariance is annualized and shrunk by a fixed 5% toward a scaled identity matrix. The shrinkage coefficient is chosen in advance. It is not estimated by the Ledoit–Wolf procedure. All solvers receive this same estimated covariance. Test observations do not participate in optimization or parameter selection.

The experiment reports three volatility ratios

| Ratio | Evaluation covariance for both numerator and denominator |
| --- | --- |
| Population / oracle | True population covariance |
| Population / equal weights | True population covariance |
| Test / equal weights | Covariance estimated from the same held-out test returns |

The oracle optimizes the true covariance under the same budget and return equalities. Equal weights are feasible under the chosen forecast returns. The study does not report a test-period oracle comparison.

Sampling times use one measured loop per solver and dataset, followed by the median across the 30 datasets. IQR describes the middle half of replication outcomes. Bootstrap intervals use 5,000 resamples to estimate uncertainty in selected median risk ratios. They are not intervals for future daily returns.

Outputs include `sampling.csv`, `summary.json`, `figures/sampling_algorithms.pdf`, and `figures/sampling_risk.pdf`.

## Recorded results

The representative controlled case has 50 assets, covariance condition number 1,000, and reduced Hessian condition number about 899.

| Method | Updates | Recorded median loop time (ms) | Relative weight error |
| --- | ---: | ---: | ---: |
| GD | 7,517 | 31.80 | 1.21e-8 |
| NAG | 439 | 2.30 | 1.18e-6 |
| CG | 76 | 0.36 | 5.32e-7 |
| BFGS | 53 | 2.10 | 6.06e-7 |

Across the sampling study, median update counts are 628, 125, 30, and 29. Median loop times are 12.08, 2.80, 0.44, and 2.56 ms. Weight-error distributions differ across methods, although all four medians are below `1.1e-7`. Within each replication, their portfolio risks are almost identical.

Median population/oracle volatility is 1.083, population/equal-weight volatility is 0.391, and test/equal-weight volatility is 0.390. These figures describe volatility, not investment returns.

The saved CG audit covers nine controlled and 30 sampling cases. All 39 true gradients pass the tolerance, and all update counts match the original records. Its computation is separate from the recorded loop timings.

Floating-point arithmetic can vary across BLAS implementations. Near a strict tolerance, this can change the final update count or tiny numerical errors. `comparison.json` records differences from the committed CSVs, excluding runtime columns, with numeric comparison tolerances of `rtol=1e-7` and `atol=1e-12`. Those comparison tolerances are distinct from the solvers' `1e-8` stopping target. The reproduction command checks convergence, feasibility, the CG true gradients, and figure generation independently.

## Inspect the generated data

The experiment regenerates its data automatically. To save the actual matrices for inspection, run

```bash
python export_data.py
```

This writes nine Dataset A archives and 30 Dataset B archives to `generated_data/`, with seeds and array dimensions in `manifest.json`. Inspect the first training/test replication with

```python
import numpy as np

data = np.load('generated_data/B_rep00.npz')
print(data['train'].shape)  # (756, 30)
print(data['test'].shape)   # (252, 30)
print(data['train'][:2, :5])
print(data['test'][:1, :5])
```

The first five entries of the first training row round to `-0.014295, 0.004190, -0.012861, 0.007954, 0.002834`, matching the report's example.

## Running the original scripts directly

```bash
python run_experiments.py
python audit_report.py
```

These commands write next to the scripts. The first replaces the root CSVs, summary, and figures with the new run. The second writes `review_20261109/accuracy_audit.json`. Use `reproduce.py` when retaining the committed report results is important.
