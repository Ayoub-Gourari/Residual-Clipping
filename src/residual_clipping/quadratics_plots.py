"""Plotting helpers for synthetic quadratics outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .quadratics import SearchConfig, aggregate_curves, method_label, run_search_curve


METHOD_ORDER = ["unclipped", "standard", "residual"]
CLIPPED_METHOD_ORDER = ["standard", "residual"]
METHOD_COLORS = {
    "unclipped": "#54A24B",
    "standard": "#4C78A8",
    "residual": "#F58518",
}


def save_figure(fig: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    png_path = destination.with_suffix(".png")
    pdf_path = destination.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(
    dataframe: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for label, group in dataframe.groupby("method_label", dropna=False):
        group = group.sort_values("step")
        mean = np.clip(group[f"{metric}_mean"].to_numpy(dtype=float), 1e-300, None)
        ax.plot(group["step"], mean, linewidth=1.8, label=label)
        std_column = f"{metric}_std"
        if std_column in group.columns:
            std = np.nan_to_num(group[std_column].to_numpy(dtype=float), nan=0.0)
            low = np.clip(mean - std, 1e-300, None)
            high = np.clip(mean + std, 1e-300, None)
            ax.fill_between(group["step"], low, high, alpha=0.12)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, destination)


def pick_best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for method_family in METHOD_ORDER:
        group = summary[summary["method_family"] == method_family]
        if group.empty:
            continue
        ordered = group.sort_values(
            ["diverged_mean", "final_objective_mean", "target_step_or_max_mean"],
            ascending=[True, True, True],
        )
        picked.append(ordered.iloc[0])
    if not picked:
        raise ValueError("No matching methods found in search summary.")
    return pd.DataFrame(picked).reset_index(drop=True)


def rerun_best_curves(
    best_rows: pd.DataFrame,
    eigs,
    make_seed_inputs_fn,
    *,
    steps: int,
    divergence_objective: float,
    curve_log_every: int,
    seed_start: int,
    num_seeds: int,
):
    log_steps = set(range(curve_log_every, steps + 1, curve_log_every))
    log_steps.add(1)
    log_steps.add(steps)
    curve_records = []

    configs = []
    for _, row in best_rows.iterrows():
        clip_value = None if pd.isna(row["clip_value"]) else float(row["clip_value"])
        configs.append(
            SearchConfig(
                method_family=str(row["method_family"]),
                clip_value=clip_value,
                lr=float(row["lr"]),
                beta=float(row["beta"]),
            )
        )

    for seed in range(seed_start, seed_start + num_seeds):
        x0, noise = make_seed_inputs_fn(seed)
        for config in configs:
            curve_records.extend(
                run_search_curve(
                    eigs=eigs,
                    x0=x0,
                    noise=noise,
                    config=config,
                    steps=steps,
                    divergence_objective=divergence_objective,
                    log_steps=log_steps,
                    seed=seed,
                )
            )

    curves = pd.DataFrame(curve_records)
    return best_rows, curves, aggregate_curves(curves)


def _curve_band_from_seed_curves(seed_curves: pd.DataFrame, config_id: str, floor: float) -> pd.DataFrame:
    group = seed_curves[seed_curves["config_id"] == config_id]
    if group.empty:
        return pd.DataFrame()
    clipped = group.copy()
    clipped["objective"] = np.clip(clipped["objective"].to_numpy(dtype=float), floor, None)
    clipped["log_objective"] = np.log10(clipped["objective"])
    band = clipped.groupby("step", dropna=False)["log_objective"].agg(
        log_median="median",
        log_q25=lambda values: float(np.quantile(values, 0.25)),
        log_q75=lambda values: float(np.quantile(values, 0.75)),
        run_count="count",
    ).reset_index()
    for column in ["log_median", "log_q25", "log_q75"]:
        band[column.replace("log_", "")] = np.power(10.0, band[column].to_numpy(dtype=float))
    return band.sort_values("step").reset_index(drop=True)


def plot_best_trajectories(best_rows: pd.DataFrame, seed_curves: pd.DataFrame, destination: Path) -> None:
    floor = 1e-16
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    observed_values = []
    for method_family in METHOD_ORDER:
        row = best_rows[best_rows["method_family"] == method_family]
        if row.empty:
            continue
        row = row.iloc[0]
        band = _curve_band_from_seed_curves(seed_curves, str(row["config_id"]), floor=floor)
        if band.empty:
            continue
        step = band["step"].to_numpy(dtype=float)
        median = band["median"].to_numpy(dtype=float)
        low = band["q25"].to_numpy(dtype=float)
        high = band["q75"].to_numpy(dtype=float)
        observed_values.extend(median.tolist())
        observed_values.extend(low.tolist())
        observed_values.extend(high.tolist())
        label = row["method_family"]
        if pd.notna(row["clip_value"]):
            label = f"{label} (C={float(row['clip_value']):g}, lr={float(row['lr']):g}, beta={float(row['beta']):g})"
        else:
            label = f"{label} (lr={float(row['lr']):g}, beta={float(row['beta']):g})"
        ax.plot(step, median, linewidth=2.1, label=label, color=METHOD_COLORS[method_family])
        if np.any(np.abs(np.log10(high) - np.log10(low)) > 1e-12):
            ax.fill_between(step, low, high, color=METHOD_COLORS[method_family], alpha=0.14)
    ax.set_xlabel("step")
    ax.set_ylabel("objective")
    ax.set_title("Best Tuned Loss Trajectories")
    ax.set_yscale("log")
    finite_observed = np.asarray([value for value in observed_values if np.isfinite(value) and value > 0])
    if finite_observed.size:
        upper = float(np.nanmax(finite_observed))
        lower = max(floor, min(float(np.nanmin(finite_observed)) * 0.5, upper / 100.0))
        ax.set_ylim(lower, upper * 2.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, destination)


def plot_best_final_loss_vs_threshold(summary: pd.DataFrame, destination: Path) -> pd.DataFrame:
    rows = []
    for method_family in CLIPPED_METHOD_ORDER:
        method_df = summary[summary["method_family"] == method_family]
        for clip_value, group in method_df.groupby("clip_value", dropna=False):
            ordered = group.sort_values(
                ["diverged_mean", "final_objective_mean", "target_step_or_max_mean"],
                ascending=[True, True, True],
            )
            best = ordered.iloc[0]
            rows.append(
                {
                    "method_family": method_family,
                    "clip_value": clip_value,
                    "best_config_id": best["config_id"],
                    "best_lr": best["lr"],
                    "best_beta": best["beta"],
                    "best_final_objective_mean": best["final_objective_mean"],
                    "best_final_objective_std": best["final_objective_std"],
                    "best_target_hit_mean": best["target_hit_mean"],
                    "best_target_step_or_max_mean": best["target_step_or_max_mean"],
                    "best_diverged_mean": best["diverged_mean"],
                }
            )
    result = pd.DataFrame(rows).sort_values(["method_family", "clip_value"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for method_family in CLIPPED_METHOD_ORDER:
        group = result[result["method_family"] == method_family]
        ax.plot(
            group["clip_value"],
            np.clip(group["best_final_objective_mean"], 1e-300, None),
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method_family],
            label=method_family,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clipping threshold")
    ax.set_ylabel("best final objective")
    ax.set_title("Best Final Loss vs Clipping Threshold")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, destination)
    return result


def plot_threshold_heatmaps(summary: pd.DataFrame, destination: Path) -> None:
    clipped = summary[
        summary["method_family"].isin(CLIPPED_METHOD_ORDER)
        & summary["clip_value"].notna()
    ].copy()
    betas = sorted(float(value) for value in clipped["beta"].dropna().unique())
    lrs = sorted(float(value) for value in clipped["lr"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    images = []
    for axis, method_family in zip(axes, CLIPPED_METHOD_ORDER):
        method_df = clipped[clipped["method_family"] == method_family]
        matrix = np.full((len(betas), len(lrs)), np.nan)
        labels = [["" for _ in lrs] for _ in betas]
        for beta_index, beta in enumerate(betas):
            for lr_index, lr in enumerate(lrs):
                group = method_df[
                    np.isclose(method_df["beta"], beta) & np.isclose(method_df["lr"], lr)
                ]
                if group.empty:
                    continue
                ordered = group.sort_values(
                    ["diverged_mean", "final_objective_mean", "target_step_or_max_mean"],
                    ascending=[True, True, True],
                )
                best = ordered.iloc[0]
                matrix[beta_index, lr_index] = best["final_objective_mean"]
                labels[beta_index][lr_index] = f"C={float(best['clip_value']):g}"
        image = axis.imshow(np.log10(np.clip(matrix, 1e-300, None)), aspect="auto", cmap="viridis")
        images.append(image)
        axis.set_title(f"{method_family.capitalize()} Best Threshold Heatmap")
        axis.set_xticks(np.arange(len(lrs)))
        axis.set_xticklabels([f"{lr:g}" for lr in lrs], rotation=35, ha="right")
        axis.set_yticks(np.arange(len(betas)))
        axis.set_yticklabels([f"{beta:g}" for beta in betas])
        axis.set_xlabel("learning rate")
        axis.set_ylabel("beta")
        for beta_index in range(len(betas)):
            for lr_index in range(len(lrs)):
                if labels[beta_index][lr_index]:
                    axis.text(lr_index, beta_index, labels[beta_index][lr_index], ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.88)
    colorbar.set_label("log10(best final objective)")
    save_figure(fig, destination)
