"""Diagnostics for AdamW clipping variants on noisy synthetic quadratics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from .adaptive_optimizers import AdaptiveAdamW, RESIDUAL_CLIP_ADAMW_M
from .quadratics import make_eigs, make_log_steps, make_seed_inputs, objective


DEFAULT_METHODS = ("adamw_uncut", "adamw_clip", RESIDUAL_CLIP_ADAMW_M)
DEFAULT_THRESHOLDS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


@dataclass(frozen=True)
class AdaptiveQuadraticRun:
    method: str
    threshold: float
    lr: float

    @property
    def label(self) -> str:
        if self.method == "adamw_uncut":
            return self.method
        return f"{self.method}_C_{self.threshold:g}"


@dataclass(frozen=True)
class AdaptiveQuadraticSuite:
    d: int = 32
    steps: int = 3_000
    num_seeds: int = 10
    seed_start: int = 0
    x0_norm: float = 5.0
    eig_min: float = 1.0
    eig_max: float = 100.0
    tail_indices: tuple[float, ...] = (1.2, 1.5, 2.5)
    noise_scales: tuple[float, ...] = (1.0,)
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    lrs: tuple[float, ...] = (0.01,)
    methods: tuple[str, ...] = DEFAULT_METHODS
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    clipping_scope: str = "global"
    log_every: int = 100
    tail_fraction: float = 0.2
    divergence_objective: float = 1e12
    dtype: str = "float64"


def torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_runs(
    methods: Iterable[str],
    thresholds: Iterable[float],
    lrs: Iterable[float],
) -> list[AdaptiveQuadraticRun]:
    runs: list[AdaptiveQuadraticRun] = []
    for lr in lrs:
        for method in methods:
            if method == "adamw_uncut":
                runs.append(AdaptiveQuadraticRun(method, math.inf, lr))
            else:
                runs.extend(AdaptiveQuadraticRun(method, float(threshold), lr) for threshold in thresholds)
    return runs


def _make_optimizer(
    parameter: torch.nn.Parameter,
    run: AdaptiveQuadraticRun,
    suite: AdaptiveQuadraticSuite,
) -> AdaptiveAdamW:
    return AdaptiveAdamW(
        [parameter],
        optimizer_name=run.method,
        beta1=suite.beta1,
        beta2=suite.beta2,
        eps=suite.eps,
        weight_decay=0.0,
        clip_threshold=run.threshold,
        clipping_scope=suite.clipping_scope,
        correct_bias=True,
    )


def run_constant_gradient_diagnostic(
    suite: AdaptiveQuadraticSuite,
    *,
    gradient: float = 10.0,
    steps: int = 1_000,
) -> pd.DataFrame:
    """Expose numerator/denominator scale coupling without parameter dynamics."""
    dtype = torch_dtype(suite.dtype)
    interesting_steps = make_log_steps(steps, max(1, steps // 20))
    interesting_steps.update({1, 2, 5, 10, 20, 50, 100, 200, 500, steps})
    records: list[dict[str, Any]] = []

    for run in build_runs(suite.methods, suite.thresholds, (0.0,)):
        parameter = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        optimizer = _make_optimizer(parameter, run, suite)
        grad = torch.full_like(parameter, gradient)
        for step in range(1, steps + 1):
            diagnostics = optimizer.step([grad], lr=0.0, collect_diagnostics=step in interesting_steps)
            if step not in interesting_steps:
                continue
            m_hat = optimizer.exp_avg[0] / max(1.0 - suite.beta1**step, 1e-12)
            v_hat = optimizer.exp_avg_sq[0] / max(1.0 - suite.beta2**step, 1e-12)
            denominator = v_hat.sqrt().add(suite.eps)
            records.append(
                {
                    "method": run.method,
                    "method_label": run.label,
                    "threshold": run.threshold,
                    "step": step,
                    "gradient": gradient,
                    "m_hat": float(m_hat.item()),
                    "sqrt_v_hat": float(denominator.item()),
                    "normalized_update": float((m_hat / denominator).item()),
                    "pseudo_grad_norm": float(diagnostics["pseudo_grad_global_norm"]),
                    "residual_norm": float(diagnostics["residual_global_norm"]),
                    "grad_clip_scale": float(diagnostics["v_clipping_scale"]),
                    "residual_clip_scale": float(diagnostics["clipping_scale"]),
                }
            )
    return pd.DataFrame(records)


def _rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return math.nan
    return float(pd.Series(left).rank().corr(pd.Series(right).rank()))


def run_noisy_quadratic(
    *,
    suite: AdaptiveQuadraticSuite,
    run: AdaptiveQuadraticRun,
    eigs: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    pareto_alpha: float,
    noise_scale: float,
    seed: int,
    log_steps: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parameter = torch.nn.Parameter(x0.clone())
    optimizer = _make_optimizer(parameter, run, suite)
    initial_objective = float(objective(eigs, parameter.detach()))
    best_objective = initial_objective
    maximum_objective = initial_objective
    previous_objective = initial_objective
    objectives: list[float] = []
    update_norms: list[float] = []
    noise_norms: list[float] = []
    descent_dots: list[float] = []
    curve_records: list[dict[str, Any]] = []
    objective_spikes = 0
    diverged = False
    steps_run = 0
    largest_noise_norm = -math.inf
    largest_noise_step = 0
    largest_noise_update_norm = math.nan

    for step in range(1, suite.steps + 1):
        value_before = parameter.detach().clone()
        true_grad = eigs * value_before
        step_noise = noise[step - 1]
        stochastic_grad = true_grad + step_noise
        should_log = step in log_steps
        diagnostics = optimizer.step([stochastic_grad], lr=run.lr, collect_diagnostics=should_log)
        update = (value_before - parameter.detach()) / run.lr
        update_norm = float(torch.linalg.vector_norm(update))
        noise_norm = float(torch.linalg.vector_norm(step_noise))
        descent_dot = float(torch.dot(true_grad, update))
        current_objective = float(objective(eigs, parameter.detach()))
        steps_run = step

        objectives.append(current_objective)
        update_norms.append(update_norm)
        noise_norms.append(noise_norm)
        descent_dots.append(descent_dot)
        best_objective = min(best_objective, current_objective)
        maximum_objective = max(maximum_objective, current_objective)
        objective_spikes += int(current_objective > 2.0 * max(previous_objective, 1e-300))
        previous_objective = current_objective
        if noise_norm > largest_noise_norm:
            largest_noise_norm = noise_norm
            largest_noise_step = step
            largest_noise_update_norm = update_norm

        if should_log:
            curve_records.append(
                {
                    "pareto_alpha": pareto_alpha,
                    "noise_scale": noise_scale,
                    "seed": seed,
                    "step": step,
                    "method": run.method,
                    "method_label": run.label,
                    "threshold": run.threshold,
                    "lr": run.lr,
                    "objective": current_objective,
                    "true_grad_norm": float(torch.linalg.vector_norm(true_grad)),
                    "stochastic_grad_norm": float(diagnostics["grad/norm"]),
                    "noise_norm": noise_norm,
                    "center_norm": float(diagnostics["center/norm"]),
                    "residual_norm": float(diagnostics["residual/norm"]),
                    "pseudo_grad_norm": float(diagnostics["pseudo_grad/norm"]),
                    "m_hat_norm": float(diagnostics["adam/m_hat_norm"]),
                    "sqrt_v_hat_norm": float(diagnostics["adam/sqrt_v_hat_norm"]),
                    "normalized_update_norm": float(diagnostics["update/norm"]),
                    "grad_clip_scale": float(diagnostics["clipping/grad_scale"]),
                    "residual_clip_scale": float(diagnostics["clipping/residual_scale"]),
                    "grad_clip_rate": float(diagnostics["clipping/grad_activation_rate"]),
                    "residual_clip_rate": float(diagnostics["clipping/residual_activation_rate"]),
                    "descent_dot": descent_dot,
                }
            )

        if not math.isfinite(current_objective) or current_objective >= suite.divergence_objective:
            diverged = True
            break

    tail_count = max(1, int(len(objectives) * suite.tail_fraction))
    tail = np.asarray(objectives[-tail_count:], dtype=np.float64)
    updates = np.asarray(update_norms, dtype=np.float64)
    summary = {
        "pareto_alpha": pareto_alpha,
        "noise_scale": noise_scale,
        "seed": seed,
        "method": run.method,
        "method_label": run.label,
        "threshold": run.threshold,
        "lr": run.lr,
        "initial_objective": initial_objective,
        "final_objective": objectives[-1],
        "best_objective": best_objective,
        "tail_median_objective": float(np.median(tail)),
        "tail_mean_objective": float(np.mean(tail)),
        "log10_objective_auc": float(np.mean(np.log10(np.maximum(objectives, 1e-300)))),
        "diverged": float(diverged),
        "steps_run": steps_run,
        "max_objective_ratio": maximum_objective / max(initial_objective, 1e-300),
        "objective_spike_rate": objective_spikes / steps_run,
        "median_update_norm": float(np.median(updates)),
        "p99_update_norm": float(np.quantile(updates, 0.99)),
        "max_update_norm": float(np.max(updates)),
        "outlier_update_amplification": largest_noise_update_norm / max(float(np.median(updates)), 1e-300),
        "largest_noise_norm": largest_noise_norm,
        "largest_noise_step": largest_noise_step,
        "largest_noise_update_norm": largest_noise_update_norm,
        "noise_update_rank_correlation": _rank_correlation(noise_norms, update_norms),
        "descent_alignment_rate": float(np.mean(np.asarray(descent_dots) > 0.0)),
        "grad_clip_rate": optimizer.grad_clipping_activation_count / steps_run,
        "residual_clip_rate": optimizer.residual_clipping_activation_count / steps_run,
    }
    return curve_records, summary


def aggregate_seed_summaries(seed_summaries: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["pareto_alpha", "noise_scale", "method", "method_label", "threshold", "lr"]
    metrics = [
        "final_objective",
        "best_objective",
        "tail_median_objective",
        "tail_mean_objective",
        "log10_objective_auc",
        "diverged",
        "max_objective_ratio",
        "objective_spike_rate",
        "median_update_norm",
        "p99_update_norm",
        "max_update_norm",
        "outlier_update_amplification",
        "noise_update_rank_correlation",
        "descent_alignment_rate",
        "grad_clip_rate",
        "residual_clip_rate",
    ]
    grouped = seed_summaries.groupby(group_columns, dropna=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    median = grouped.median().add_suffix("_median")
    std = grouped.std().add_suffix("_std")
    return pd.concat([mean, median, std], axis=1).reset_index()


def threshold_sensitivity(aggregate: pd.DataFrame) -> pd.DataFrame:
    clipped = aggregate[np.isfinite(aggregate["threshold"])].copy()
    records: list[dict[str, Any]] = []
    group_columns = ["pareto_alpha", "noise_scale", "method", "lr"]
    for keys, group in clipped.groupby(group_columns, dropna=False):
        values = group["tail_median_objective_median"].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        best = float(np.min(finite))
        worst = float(np.max(finite))
        positive_best = max(best, 1e-300)
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "num_thresholds": int(finite.size),
                "best_tail_objective": best,
                "worst_tail_objective": worst,
                "log10_threshold_spread": math.log10(max(worst, 1e-300) / positive_best),
                "fraction_within_2x_best": float(np.mean(finite <= 2.0 * positive_best)),
                "fraction_within_10x_best": float(np.mean(finite <= 10.0 * positive_best)),
                "divergence_rate_across_thresholds": float(group["diverged_mean"].mean()),
                "median_update_range_ratio": float(
                    group["median_update_norm_median"].max()
                    / max(group["median_update_norm_median"].min(), 1e-300)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def run_suite(suite: AdaptiveQuadraticSuite, output_dir: Path) -> dict[str, pd.DataFrame]:
    if not 0.0 < suite.tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch_dtype(suite.dtype)
    eigs = make_eigs(suite.d, suite.eig_min, suite.eig_max, dtype)
    log_steps = make_log_steps(suite.steps, suite.log_every)
    runs = build_runs(suite.methods, suite.thresholds, suite.lrs)
    curves: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    constant_gradient = run_constant_gradient_diagnostic(suite)
    total = len(suite.tail_indices) * len(suite.noise_scales) * suite.num_seeds * len(runs)
    completed = 0
    for pareto_alpha in suite.tail_indices:
        for noise_scale in suite.noise_scales:
            for seed in range(suite.seed_start, suite.seed_start + suite.num_seeds):
                x0, noise = make_seed_inputs(
                    steps=suite.steps,
                    d=suite.d,
                    x0_norm=suite.x0_norm,
                    pareto_alpha=pareto_alpha,
                    r_min=1.0,
                    noise_scale=noise_scale,
                    seed=seed,
                    dtype=dtype,
                )
                for run in runs:
                    run_curves, run_summary = run_noisy_quadratic(
                        suite=suite,
                        run=run,
                        eigs=eigs,
                        x0=x0,
                        noise=noise,
                        pareto_alpha=pareto_alpha,
                        noise_scale=noise_scale,
                        seed=seed,
                        log_steps=log_steps,
                    )
                    curves.extend(run_curves)
                    seed_summaries.append(run_summary)
                    completed += 1
                    if completed == 1 or completed == total or completed % max(1, total // 20) == 0:
                        print(f"[adaptive quadratics] {completed}/{total} runs", flush=True)

    frames = {
        "constant_gradient": constant_gradient,
        "curves": pd.DataFrame(curves),
        "seed_summaries": pd.DataFrame(seed_summaries),
    }
    frames["aggregate"] = aggregate_seed_summaries(frames["seed_summaries"])
    frames["threshold_sensitivity"] = threshold_sensitivity(frames["aggregate"])
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(suite), handle, indent=2)
    return frames
