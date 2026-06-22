"""Hyperparameter selection and plots for adaptive quadratic experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adaptive_quadratics import (
    AdaptiveQuadraticSuite,
    build_runs,
    run_noisy_quadratic,
    torch_dtype,
)
from .adaptive_optimizers import RESIDUAL_CLIP_ADAMW_M
from .quadratics import make_eigs, make_log_steps, make_seed_inputs


SWEEP_METHODS = ("adamw_uncut", "adamw_clip", RESIDUAL_CLIP_ADAMW_M)
METHOD_LABELS = {
    "adamw_uncut": "AdamW",
    "adamw_clip": "AdamWClip",
    RESIDUAL_CLIP_ADAMW_M: "ResidualClipAdamW-M",
}
METHOD_COLORS = {
    "adamw_uncut": "#4C78A8",
    "adamw_clip": "#F58518",
    RESIDUAL_CLIP_ADAMW_M: "#54A24B",
}
SELECTION_METRIC = "tail_median_objective_mean"


def aggregate_sweep(seed_summaries: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "pareto_alpha",
        "noise_scale",
        "method",
        "threshold",
        "lr",
        "beta1",
    ]
    metrics = [
        "final_objective",
        "best_objective",
        "tail_median_objective",
        "tail_mean_objective",
        "log10_objective_auc",
        "diverged",
        "median_update_norm",
        "p99_update_norm",
        "max_update_norm",
        "descent_alignment_rate",
        "grad_clip_rate",
        "residual_clip_rate",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in seed_summaries.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            record[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        record["num_seeds"] = len(group)
        records.append(record)
    return pd.DataFrame(records)


def _best_row(frame: pd.DataFrame) -> pd.Series:
    ranked = frame.sort_values(
        ["diverged_mean", SELECTION_METRIC, "log10_objective_auc_median"],
        ascending=[True, True, True],
    )
    return ranked.iloc[0]


def select_best_by_threshold(
    aggregate: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Select LR and beta1 per method/threshold using mean loss across seeds."""
    thresholds = list(thresholds)
    records: list[dict[str, Any]] = []
    for method in SWEEP_METHODS:
        method_frame = aggregate[aggregate["method"] == method]
        if method_frame.empty:
            continue
        if method == "adamw_uncut":
            best = _best_row(method_frame)
            for threshold in thresholds:
                record = best.to_dict()
                record["plot_threshold"] = threshold
                record["selected_threshold"] = math.inf
                records.append(record)
            continue
        for threshold in thresholds:
            candidates = method_frame[np.isclose(method_frame["threshold"], threshold)]
            if candidates.empty:
                continue
            best = _best_row(candidates)
            record = best.to_dict()
            record["plot_threshold"] = threshold
            record["selected_threshold"] = float(best["threshold"])
            records.append(record)
    selected = pd.DataFrame(records)
    if not selected.empty:
        selected["display_name"] = selected["method"].map(METHOD_LABELS)
        selected = selected.sort_values(["method", "plot_threshold"]).reset_index(drop=True)
    return selected


def select_global_best(aggregate: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method in SWEEP_METHODS:
        candidates = aggregate[aggregate["method"] == method]
        if candidates.empty:
            continue
        record = _best_row(candidates).to_dict()
        record["display_name"] = METHOD_LABELS[method]
        records.append(record)
    return pd.DataFrame(records)


def summarize_best_trajectories(
    curves: pd.DataFrame,
    best_configs: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, config in best_configs.iterrows():
        threshold_mask = (
            np.isinf(curves["threshold"])
            if math.isinf(float(config["threshold"]))
            else np.isclose(curves["threshold"], float(config["threshold"]))
        )
        selected = curves[
            (curves["method"] == config["method"])
            & threshold_mask
            & np.isclose(curves["lr"], float(config["lr"]))
            & np.isclose(curves["beta1"], float(config["beta1"]))
        ]
        for step, group in selected.groupby("step"):
            values = group["objective"].to_numpy(dtype=np.float64)
            records.append(
                {
                    "method": config["method"],
                    "display_name": config["display_name"],
                    "threshold": config["threshold"],
                    "lr": config["lr"],
                    "beta1": config["beta1"],
                    "step": int(step),
                    "objective_mean": float(np.mean(values)),
                    "objective_median": float(np.median(values)),
                    "objective_q25": float(np.quantile(values, 0.25)),
                    "objective_q75": float(np.quantile(values, 0.75)),
                    "num_seeds": len(values),
                }
            )
    return pd.DataFrame(records)


def plot_best_loss_vs_threshold(selected: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in SWEEP_METHODS:
        group = selected[selected["method"] == method].sort_values("plot_threshold")
        if group.empty:
            continue
        x = group["plot_threshold"].to_numpy(dtype=np.float64)
        y = group[SELECTION_METRIC].to_numpy(dtype=np.float64)
        q25 = group["tail_median_objective_q25"].to_numpy(dtype=np.float64)
        q75 = group["tail_median_objective_q75"].to_numpy(dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, q25, q75, color=METHOD_COLORS[method], alpha=0.16)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Clipping threshold")
    ax.set_ylabel("Best mean tail loss across seeds")
    ax.set_title("Best loss after learning-rate and momentum selection\n(lines: mean; bands: seed IQR)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_best_trajectories(trajectories: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in SWEEP_METHODS:
        group = trajectories[trajectories["method"] == method].sort_values("step")
        if group.empty:
            continue
        x = group["step"].to_numpy(dtype=np.float64)
        y = group["objective_mean"].to_numpy(dtype=np.float64)
        q25 = group["objective_q25"].to_numpy(dtype=np.float64)
        q75 = group["objective_q75"].to_numpy(dtype=np.float64)
        config = group.iloc[0]
        threshold = "inf" if math.isinf(float(config["threshold"])) else f"{config['threshold']:g}"
        label = (
            f"{METHOD_LABELS[method]} "
            f"(C={threshold}, lr={config['lr']:g}, beta1={config['beta1']:g})"
        )
        ax.plot(x, y, linewidth=2.0, color=METHOD_COLORS[method], label=label)
        ax.fill_between(x, q25, q75, color=METHOD_COLORS[method], alpha=0.16)
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Objective")
    ax.set_title("Best configuration for each method\n(lines: mean; bands: seed IQR)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def run_hyperparameter_sweep(
    *,
    suite: AdaptiveQuadraticSuite,
    beta1_values: Iterable[float],
    output_dir: Path,
    figure_dir: Path,
) -> dict[str, pd.DataFrame]:
    beta1_values = tuple(float(value) for value in beta1_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch_dtype(suite.dtype)
    eigs = make_eigs(suite.d, suite.eig_min, suite.eig_max, dtype)
    log_steps = make_log_steps(suite.steps, suite.log_every)
    runs = build_runs(suite.methods, suite.thresholds, suite.lrs)
    total = len(beta1_values) * suite.num_seeds * len(runs)
    completed = 0
    curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for beta1 in beta1_values:
        beta_suite = replace(suite, beta1=beta1)
        for seed in range(suite.seed_start, suite.seed_start + suite.num_seeds):
            x0, noise = make_seed_inputs(
                steps=suite.steps,
                d=suite.d,
                x0_norm=suite.x0_norm,
                pareto_alpha=suite.tail_indices[0],
                r_min=1.0,
                noise_scale=suite.noise_scales[0],
                seed=seed,
                dtype=dtype,
            )
            for run in runs:
                run_curves, run_summary = run_noisy_quadratic(
                    suite=beta_suite,
                    run=run,
                    eigs=eigs,
                    x0=x0,
                    noise=noise,
                    pareto_alpha=suite.tail_indices[0],
                    noise_scale=suite.noise_scales[0],
                    seed=seed,
                    log_steps=log_steps,
                )
                for record in run_curves:
                    record["beta1"] = beta1
                run_summary["beta1"] = beta1
                curves.extend(run_curves)
                summaries.append(run_summary)
                completed += 1
                if completed == 1 or completed == total or completed % max(1, total // 25) == 0:
                    print(f"[adaptive sweep] {completed}/{total} runs", flush=True)

    curve_frame = pd.DataFrame(curves)
    summary_frame = pd.DataFrame(summaries)
    aggregate = aggregate_sweep(summary_frame)
    best_by_threshold = select_best_by_threshold(aggregate, suite.thresholds)
    best_configs = select_global_best(aggregate)
    best_trajectories = summarize_best_trajectories(curve_frame, best_configs)
    frames = {
        "sweep_curves": curve_frame,
        "sweep_seed_summaries": summary_frame,
        "sweep_aggregate": aggregate,
        "best_by_threshold": best_by_threshold,
        "best_configs": best_configs,
        "best_trajectories": best_trajectories,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "sweep_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "suite": asdict(suite),
                "beta1_values": beta1_values,
                "selection_metric": SELECTION_METRIC,
            },
            handle,
            indent=2,
        )

    plot_best_loss_vs_threshold(best_by_threshold, figure_dir / "best_loss_vs_threshold")
    plot_best_trajectories(best_trajectories, figure_dir / "best_run_trajectories")
    return frames
