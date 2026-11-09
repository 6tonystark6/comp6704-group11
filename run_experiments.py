"""Reproducible experiments for Group 11 portfolio risk minimization.

Four independently coded optimization methods solve the same unconstrained
positive-definite quadratic obtained by null-space elimination:

* fixed-step gradient descent (GD),
* strongly convex Nesterov accelerated gradient (NAG),
* linear conjugate gradient (CG), and
* inverse-Hessian BFGS with exact line search on the quadratic.

The dense direct solve is used only as a numerical reference. All market data
in this project are synthetic, and all risk quantities are annualized.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
MASTER_SEED = 670411
MAX_ITER = 200_000
GRAD_RTOL = 1e-8
ALGORITHM_ORDER = ("GD", "NAG", "CG", "BFGS")
ALGORITHM_LABELS = {
    "GD": "Gradient descent",
    "NAG": "Nesterov AG",
    "CG": "Conjugate gradient",
    "BFGS": "BFGS",
}


@dataclass
class Problem:
    n: int
    kappa_requested: float
    mu: np.ndarray
    target: float
    sigma: np.ndarray
    w0: np.ndarray
    B: np.ndarray
    H: np.ndarray
    c: np.ndarray
    lam_min: float
    lam_max: float


def covariance(n: int, kappa: float, seed: int) -> np.ndarray:
    """Return an SPD covariance with a prescribed spectral condition number."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    q = q @ np.diag(np.sign(np.diag(r)))
    eigenvalues = np.geomspace(0.09 / kappa, 0.09, n)
    return (q * eigenvalues) @ q.T


def make_problem(
    n: int,
    kappa: float,
    seed: int,
    sigma: np.ndarray | None = None,
) -> Problem:
    """Construct the null-space reduced portfolio problem."""
    mu = np.linspace(0.04, 0.16, n)
    target = 0.10
    A = np.vstack((np.ones(n), mu))
    b = np.array([1.0, target])
    _, singular_values, vt = np.linalg.svd(A, full_matrices=True)
    assert singular_values[-1] > 1e-12
    B = vt[2:].T
    w0 = A.T @ np.linalg.solve(A @ A.T, b)
    if sigma is None:
        sigma = covariance(n, kappa, seed)
    H = 2.0 * B.T @ sigma @ B
    c = 2.0 * B.T @ sigma @ w0
    eigenvalues = np.linalg.eigvalsh(H)
    assert eigenvalues[0] > 0.0
    return Problem(
        n=n,
        kappa_requested=kappa,
        mu=mu,
        target=target,
        sigma=sigma,
        w0=w0,
        B=B,
        H=H,
        c=c,
        lam_min=float(eigenvalues[0]),
        lam_max=float(eigenvalues[-1]),
    )


def exact_reference(p: Problem) -> tuple[np.ndarray, np.ndarray]:
    """Solve the reduced normal equations; this is not a team algorithm."""
    z_star = np.linalg.solve(p.H, -p.c)
    return z_star, p.w0 + p.B @ z_star


def objective(p: Problem, z: np.ndarray) -> float:
    w = p.w0 + p.B @ z
    return float(w @ p.sigma @ w)


def gradient(p: Problem, z: np.ndarray) -> np.ndarray:
    return p.H @ z + p.c


def finish_result(
    p: Problem,
    z: np.ndarray,
    iterations: int,
    grad_norm: float,
    initial_grad_norm: float,
    elapsed: float,
    history: list[float],
    **metadata: float,
) -> dict:
    threshold_scale = max(initial_grad_norm, 1e-15)
    return {
        "z": z,
        "w": p.w0 + p.B @ z,
        "iterations": iterations,
        "converged": grad_norm <= GRAD_RTOL * threshold_scale,
        "gradient_ratio": float(grad_norm / threshold_scale),
        "seconds_iterations_only": float(elapsed),
        "history": history,
        **metadata,
    }


def gradient_descent(
    p: Problem,
    rtol: float = GRAD_RTOL,
    max_iter: int = MAX_ITER,
    trace: bool = False,
) -> dict:
    """Fixed-step GD with the optimal spectral step for an SPD quadratic."""
    eta = 2.0 / (p.lam_min + p.lam_max)
    z = np.zeros(p.n - 2)
    initial_grad_norm = float(np.linalg.norm(p.c))
    threshold = rtol * max(initial_grad_norm, 1e-15)
    history = [objective(p, z)] if trace else []
    grad_norm = initial_grad_norm
    start = perf_counter()
    for k in range(1, max_iter + 1):
        z -= eta * gradient(p, z)
        grad_norm = float(np.linalg.norm(gradient(p, z)))
        if trace:
            history.append(objective(p, z))
        if grad_norm <= threshold:
            break
    elapsed = perf_counter() - start
    return finish_result(
        p, z, k, grad_norm, initial_grad_norm, elapsed, history, eta=float(eta)
    )


def accelerated_gradient(
    p: Problem,
    step_fraction: float = 1.0,
    rtol: float = GRAD_RTOL,
    max_iter: int = MAX_ITER,
    trace: bool = False,
) -> dict:
    """Strongly convex Nesterov accelerated gradient."""
    eta = step_fraction / p.lam_max
    beta = (1.0 - np.sqrt(eta * p.lam_min)) / (
        1.0 + np.sqrt(eta * p.lam_min)
    )
    z = np.zeros(p.n - 2)
    z_previous = z.copy()
    initial_grad_norm = float(np.linalg.norm(p.c))
    threshold = rtol * max(initial_grad_norm, 1e-15)
    history = [objective(p, z)] if trace else []
    grad_norm = initial_grad_norm
    start = perf_counter()
    for k in range(1, max_iter + 1):
        y = z + beta * (z - z_previous)
        z_new = y - eta * gradient(p, y)
        z_previous, z = z, z_new
        grad_norm = float(np.linalg.norm(gradient(p, z)))
        if trace:
            history.append(objective(p, z))
        if grad_norm <= threshold:
            break
    elapsed = perf_counter() - start
    return finish_result(
        p,
        z,
        k,
        grad_norm,
        initial_grad_norm,
        elapsed,
        history,
        eta=float(eta),
        beta=float(beta),
    )


def conjugate_gradient(
    p: Problem,
    rtol: float = GRAD_RTOL,
    max_iter: int = MAX_ITER,
    trace: bool = False,
) -> dict:
    """Linear CG applied to H z = -c, equivalent to minimizing F(z)."""
    z = np.zeros(p.n - 2)
    residual = -p.c.copy()
    direction = residual.copy()
    residual_sq = float(residual @ residual)
    initial_grad_norm = float(np.sqrt(residual_sq))
    threshold = rtol * max(initial_grad_norm, 1e-15)
    history = [objective(p, z)] if trace else []
    grad_norm = initial_grad_norm
    start = perf_counter()
    for k in range(1, max_iter + 1):
        h_direction = p.H @ direction
        curvature = float(direction @ h_direction)
        if curvature <= 0.0:
            raise FloatingPointError("CG encountered non-positive curvature")
        alpha = residual_sq / curvature
        z += alpha * direction
        residual_new = residual - alpha * h_direction
        residual_sq_new = float(residual_new @ residual_new)
        grad_norm = float(np.sqrt(residual_sq_new))
        if trace:
            history.append(objective(p, z))
        if grad_norm <= threshold:
            residual = residual_new
            residual_sq = residual_sq_new
            break
        beta = residual_sq_new / residual_sq
        direction = residual_new + beta * direction
        residual = residual_new
        residual_sq = residual_sq_new
    elapsed = perf_counter() - start
    return finish_result(
        p, z, k, grad_norm, initial_grad_norm, elapsed, history
    )


def bfgs(
    p: Problem,
    rtol: float = GRAD_RTOL,
    max_iter: int = MAX_ITER,
    trace: bool = False,
) -> dict:
    """Full-memory inverse BFGS with exact line search for this quadratic."""
    dimension = p.n - 2
    identity = np.eye(dimension)
    inverse_hessian = identity.copy()
    z = np.zeros(dimension)
    grad = p.c.copy()
    initial_grad_norm = float(np.linalg.norm(grad))
    threshold = rtol * max(initial_grad_norm, 1e-15)
    history = [objective(p, z)] if trace else []
    grad_norm = initial_grad_norm
    start = perf_counter()
    for k in range(1, max_iter + 1):
        direction = -inverse_hessian @ grad
        directional_derivative = float(grad @ direction)
        if directional_derivative >= 0.0:
            inverse_hessian = identity.copy()
            direction = -grad
            directional_derivative = -float(grad @ grad)
        curvature = float(direction @ (p.H @ direction))
        alpha = -directional_derivative / curvature
        step = alpha * direction
        z_new = z + step
        grad_new = gradient(p, z_new)
        y = grad_new - grad
        y_step = float(y @ step)
        if y_step <= np.finfo(float).eps * np.linalg.norm(y) * np.linalg.norm(step):
            inverse_hessian = identity.copy()
        else:
            rho = 1.0 / y_step
            left = identity - rho * np.outer(step, y)
            inverse_hessian = (
                left @ inverse_hessian @ left.T + rho * np.outer(step, step)
            )
        z, grad = z_new, grad_new
        grad_norm = float(np.linalg.norm(grad))
        if trace:
            history.append(objective(p, z))
        if grad_norm <= threshold:
            break
    elapsed = perf_counter() - start
    return finish_result(
        p, z, k, grad_norm, initial_grad_norm, elapsed, history
    )


SOLVERS: dict[str, Callable[..., dict]] = {
    "GD": gradient_descent,
    "NAG": accelerated_gradient,
    "CG": conjugate_gradient,
    "BFGS": bfgs,
}


def benchmark_solver(
    solver: Callable[..., dict], p: Problem, repeats: int = 5
) -> tuple[dict, float]:
    """Return a deterministic result and median iteration-loop time."""
    runs = [solver(p, trace=False) for _ in range(repeats)]
    assert len({run["iterations"] for run in runs}) == 1
    median_seconds = float(
        np.median([run["seconds_iterations_only"] for run in runs])
    )
    return runs[0], median_seconds


def risk(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(max(float(w @ sigma @ w), 0.0)))


def quantiles(values: list[float]) -> dict[str, float]:
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"q25": float(q25), "median": float(median), "q75": float(q75)}


def bootstrap_median_ci(
    values: list[float], seed: int, draws: int = 5000
) -> list[float]:
    x = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(draws, len(x)))]
    return [float(v) for v in np.quantile(np.median(samples, axis=1), [0.025, 0.975])]


def write_csv(filename: str, rows: list[dict]) -> None:
    with (OUT / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def relative_gap(p: Problem, z: np.ndarray, z_star: np.ndarray) -> float:
    numerator = objective(p, z) - objective(p, z_star)
    denominator = objective(p, np.zeros(p.n - 2)) - objective(p, z_star)
    return float(max(0.0, numerator / max(denominator, 1e-30)))


def make_controlled_figures(controlled: list[dict]) -> None:
    """Create representative convergence and scaling figures."""
    p = make_problem(50, 1000.0, MASTER_SEED + 50_000 + 1000)
    z_star, _ = exact_reference(p)
    f_star = objective(p, z_star)
    convergence_styles = {
        "GD": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8},
        "NAG": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.8},
        "BFGS": {
            "color": "#d62728",
            "linestyle": "-.",
            "linewidth": 1.8,
            "marker": "s",
            "markevery": 8,
            "markersize": 3.0,
        },
        "CG": {
            "color": "#2ca02c",
            "linestyle": ":",
            "linewidth": 2.4,
            "marker": "o",
            "markevery": 8,
            "markersize": 3.2,
            "zorder": 5,
        },
    }
    histories: dict[str, np.ndarray] = {}
    for name in ("GD", "NAG", "BFGS", "CG"):
        run = SOLVERS[name](p, trace=True)
        values = np.asarray(run["history"])
        gap = np.maximum(values - f_star, 1e-20)
        histories[name] = gap / gap[0]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.05), sharey=True)
    for name in ("GD", "NAG", "BFGS", "CG"):
        history = histories[name]
        axes[0].semilogy(
            np.arange(len(history)),
            history,
            label=ALGORITHM_LABELS[name],
            **convergence_styles[name],
        )
        zoom_end = min(len(history), 151)
        axes[1].semilogy(
            np.arange(zoom_end),
            history[:zoom_end],
            label=ALGORITHM_LABELS[name],
            **convergence_styles[name],
        )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Relative objective gap")
    axes[0].set_title("Full convergence horizon")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("First 150 iterations (zoom)")
    axes[1].set_xlim(0, 150)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
        axis.set_ylim(1e-17, 2.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(r"Representative case: $n=50$, $\kappa_\Sigma=10^3$", y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(FIG / "convergence_algorithms.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.05))
    for name in ALGORITHM_ORDER:
        selected = sorted(
            [r for r in controlled if r["algorithm"] == name and r["n"] == 50],
            key=lambda r: r["kappa_reduced"],
        )
        axes[0].loglog(
            [r["kappa_reduced"] for r in selected],
            [r["iterations"] for r in selected],
            marker="o",
            label=ALGORITHM_LABELS[name],
        )
        selected = sorted(
            [
                r
                for r in controlled
                if r["algorithm"] == name and r["kappa_full"] == 1000.0
            ],
            key=lambda r: r["n"],
        )
        axes[1].plot(
            [r["n"] for r in selected],
            [1000.0 * r["seconds_iterations_only_median"] for r in selected],
            marker="o",
            label=ALGORITHM_LABELS[name],
        )
    axes[0].set_xlabel(r"Reduced condition number $\kappa_H$")
    axes[0].set_ylabel("Iterations")
    axes[0].set_title("Conditioning sensitivity ($n=50$)")
    axes[1].set_xlabel("Number of assets $n$")
    axes[1].set_ylabel("Median loop time (ms)")
    axes[1].set_title(r"Dimension scaling ($\kappa_\Sigma=10^3$)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "controlled_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def make_sampling_figures(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.05))
    iteration_data = [
        [r["iterations"] for r in rows if r["algorithm"] == name]
        for name in ALGORITHM_ORDER
    ]
    error_data = [
        [r["solver_relative_weight_error"] for r in rows if r["algorithm"] == name]
        for name in ALGORITHM_ORDER
    ]
    axes[0].boxplot(iteration_data, tick_labels=ALGORITHM_ORDER)
    axes[0].set_ylabel("Iterations")
    axes[0].set_title("Numerical effort over 30 replications")
    axes[1].boxplot(error_data, tick_labels=ALGORITHM_ORDER)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Relative weight error")
    axes[1].set_title("Agreement with direct reference")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "sampling_algorithms.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.05))
    population_data = [
        [r["population_vol_ratio_vs_oracle"] for r in rows if r["algorithm"] == name]
        for name in ALGORITHM_ORDER
    ]
    test_data = [
        [r["test_vol_ratio_vs_equal"] for r in rows if r["algorithm"] == name]
        for name in ALGORITHM_ORDER
    ]
    axes[0].boxplot(population_data, tick_labels=ALGORITHM_ORDER)
    axes[0].set_ylabel("Volatility ratio")
    axes[0].set_title("Population risk / oracle risk")
    axes[1].boxplot(test_data, tick_labels=ALGORITHM_ORDER)
    axes[1].set_ylabel("Volatility ratio")
    axes[1].set_title("Held-out risk / equal-weight risk")
    for axis in axes:
        axis.axhline(1.0, color="black", lw=0.8, ls="--")
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "sampling_risk.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

    controlled: list[dict] = []
    for n in (20, 50, 100):
        for kappa in (100.0, 1000.0, 10000.0):
            seed = MASTER_SEED + 1000 * n + int(kappa)
            p = make_problem(n, kappa, seed)
            z_star, w_star = exact_reference(p)
            for name in ALGORITHM_ORDER:
                run, median_seconds = benchmark_solver(SOLVERS[name], p)
                controlled.append(
                    {
                        "algorithm": name,
                        "n": n,
                        "kappa_full": kappa,
                        "kappa_reduced": p.lam_max / p.lam_min,
                        "iterations": run["iterations"],
                        "gradient_ratio": run["gradient_ratio"],
                        "relative_objective_gap": relative_gap(p, run["z"], z_star),
                        "relative_weight_error": float(
                            np.linalg.norm(run["w"] - w_star)
                            / max(np.linalg.norm(w_star), 1e-15)
                        ),
                        "budget_error": abs(float(np.sum(run["w"]) - 1.0)),
                        "return_error": abs(float(p.mu @ run["w"] - p.target)),
                        "seconds_iterations_only_median": median_seconds,
                        "converged": run["converged"],
                    }
                )
    make_controlled_figures(controlled)

    ablation_problem = make_problem(50, 1000.0, MASTER_SEED + 50_000 + 1000)
    ablation: list[dict] = []
    for alpha in (0.25, 0.5, 1.0):
        run = accelerated_gradient(ablation_problem, step_fraction=alpha)
        ablation.append(
            {
                "algorithm": "NAG",
                "step_fraction": alpha,
                "iterations": run["iterations"],
                "gradient_ratio": run["gradient_ratio"],
                "converged": run["converged"],
            }
        )
    plt.figure(figsize=(4.4, 3.1))
    plt.plot(
        [row["step_fraction"] for row in ablation],
        [row["iterations"] for row in ablation],
        marker="o",
    )
    plt.xlabel(r"Step fraction $\alpha$ in $\eta=\alpha/L$")
    plt.ylabel("Iterations")
    plt.title("NAG with matched momentum")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG / "step_ablation.pdf", bbox_inches="tight")
    plt.close()

    sampling: list[dict] = []
    n, kappa, train_days, test_days, shrinkage = 30, 300.0, 756, 252, 0.05
    for replication in range(30):
        seed = MASTER_SEED + 10_000 + replication
        sigma_true = covariance(n, kappa, seed)
        p_true = make_problem(n, kappa, seed, sigma_true)
        rng = np.random.default_rng(seed + 100_000)
        train = rng.multivariate_normal(
            p_true.mu / 252.0, sigma_true / 252.0, size=train_days
        )
        test = rng.multivariate_normal(
            p_true.mu / 252.0, sigma_true / 252.0, size=test_days
        )
        sample_covariance = np.cov(train, rowvar=False, ddof=1) * 252.0
        sigma_estimated = (
            (1.0 - shrinkage) * sample_covariance
            + shrinkage * np.trace(sample_covariance) / n * np.eye(n)
        )
        p_estimated = make_problem(n, kappa, seed, sigma_estimated)
        _, w_estimated_reference = exact_reference(p_estimated)
        _, w_oracle = exact_reference(p_true)
        w_equal = np.full(n, 1.0 / n)
        test_covariance = np.cov(test, rowvar=False, ddof=1) * 252.0
        for name in ALGORITHM_ORDER:
            run = SOLVERS[name](p_estimated)
            population_solver = risk(run["w"], sigma_true)
            population_oracle = risk(w_oracle, sigma_true)
            population_equal = risk(w_equal, sigma_true)
            test_solver = risk(run["w"], test_covariance)
            test_equal = risk(w_equal, test_covariance)
            sampling.append(
                {
                    "replication": replication,
                    "algorithm": name,
                    "iterations": run["iterations"],
                    "gradient_ratio": run["gradient_ratio"],
                    "seconds_iterations_only": run["seconds_iterations_only"],
                    "solver_relative_weight_error": float(
                        np.linalg.norm(run["w"] - w_estimated_reference)
                        / max(np.linalg.norm(w_estimated_reference), 1e-15)
                    ),
                    "budget_error": abs(float(np.sum(run["w"]) - 1.0)),
                    "return_error": abs(
                        float(p_true.mu @ run["w"] - p_true.target)
                    ),
                    "population_vol_solver": population_solver,
                    "population_vol_oracle": population_oracle,
                    "population_vol_equal": population_equal,
                    "test_vol_solver": test_solver,
                    "test_vol_equal": test_equal,
                    "population_vol_ratio_vs_oracle": population_solver
                    / population_oracle,
                    "population_vol_ratio_vs_equal": population_solver
                    / population_equal,
                    "test_vol_ratio_vs_equal": test_solver / test_equal,
                    "gross_exposure": float(np.sum(np.abs(run["w"]))),
                    "converged": run["converged"],
                }
            )
    make_sampling_figures(sampling)

    sampling_summary: dict[str, dict] = {}
    for algorithm_index, name in enumerate(ALGORITHM_ORDER):
        rows = [row for row in sampling if row["algorithm"] == name]
        population_oracle = [row["population_vol_ratio_vs_oracle"] for row in rows]
        population_equal = [row["population_vol_ratio_vs_equal"] for row in rows]
        test_equal = [row["test_vol_ratio_vs_equal"] for row in rows]
        sampling_summary[name] = {
            "iterations": quantiles([row["iterations"] for row in rows]),
            "seconds_iterations_only": quantiles(
                [row["seconds_iterations_only"] for row in rows]
            ),
            "solver_relative_weight_error": quantiles(
                [row["solver_relative_weight_error"] for row in rows]
            ),
            "population_vol_ratio_vs_oracle": quantiles(population_oracle),
            "population_vol_ratio_vs_equal": quantiles(population_equal),
            "test_vol_ratio_vs_equal": quantiles(test_equal),
            "bootstrap_95pct_ci_median_population_vs_oracle": bootstrap_median_ci(
                population_oracle, MASTER_SEED + 100 + algorithm_index
            ),
            "bootstrap_95pct_ci_median_test_vs_equal": bootstrap_median_ci(
                test_equal, MASTER_SEED + 200 + algorithm_index
            ),
            "count_test_ratio_below_one": sum(value < 1.0 for value in test_equal),
            "gross_exposure": quantiles([row["gross_exposure"] for row in rows]),
            "max_budget_error": max(row["budget_error"] for row in rows),
            "max_return_error": max(row["return_error"] for row in rows),
            "convergence_count": sum(bool(row["converged"]) for row in rows),
        }

    summary = {
        "configuration": {
            "master_seed": MASTER_SEED,
            "gradient_relative_tolerance": GRAD_RTOL,
            "max_iterations": MAX_ITER,
            "algorithms": list(ALGORITHM_ORDER),
            "controlled_n": [20, 50, 100],
            "controlled_kappa": [100, 1000, 10000],
            "timing_repeats_controlled": 5,
            "sampling_repetitions_per_algorithm": 30,
            "sampling_n": n,
            "sampling_kappa": kappa,
            "train_days": train_days,
            "test_days": test_days,
            "shrinkage": shrinkage,
        },
        "controlled": controlled,
        "ablation": ablation,
        "sampling_by_algorithm": sampling_summary,
    }

    assert len(controlled) == 36
    assert all(bool(row["converged"]) for row in controlled)
    assert len(sampling) == 120
    assert all(bool(row["converged"]) for row in sampling)
    assert max(row["budget_error"] for row in sampling) < 1e-12
    assert max(row["return_error"] for row in sampling) < 1e-12
    assert max(row["solver_relative_weight_error"] for row in sampling) < 1e-5

    write_csv("controlled.csv", controlled)
    write_csv("ablation.csv", ablation)
    write_csv("sampling.csv", sampling)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
