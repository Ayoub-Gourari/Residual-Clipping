"""Plotting helpers for centralized WikiText-2 experiment outputs."""

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


METHOD_COLORS = {
    "sgd_momentum": "#54A24B",
    "clipped_momentum": "#4C78A8",
    "residual_clipped_momentum": "#F58518",
}

METHOD_LINESTYLES = {
    "sgd_momentum": ":",
    "clipped_momentum": "-",
    "residual_clipped_momentum": "--",
}

METHOD_MARKERS = {
    "sgd_momentum": "^",
    "clipped_momentum": "o",
    "residual_clipped_momentum": "s",
}


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_best_perplexity_vs_threshold(
    summary: pd.DataFrame,
    path: Path,
    metric: str = "best_perplexity",
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    baseline = summary[summary["optimizer_mode"] == "sgd_momentum"].copy()
    clipped = summary[summary["optimizer_mode"] != "sgd_momentum"].copy()
    if not baseline.empty:
        baseline_value = float(pd.to_numeric(baseline[metric], errors="coerce").min())
        ax.axhline(
            baseline_value,
            linestyle="--",
            linewidth=1.6,
            color=METHOD_COLORS["sgd_momentum"],
            label="sgd_momentum",
            alpha=0.95,
        )
    for method, group in clipped.groupby("optimizer_mode", dropna=False):
        group = group.sort_values("clip_threshold")
        x_values = pd.to_numeric(group["clip_threshold"], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not finite_mask.any():
            continue
        ax.plot(
            x_values[finite_mask],
            y_values[finite_mask],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xlabel("clipping threshold")
    ax.set_ylabel("best validation perplexity")
    ax.set_title("Best Perplexity vs Clipping Threshold")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path)


def plot_best_trajectories(
    summary: pd.DataFrame,
    path: Path,
    metric: str = "validation_perplexity_mean",
    title: str = "Best Validation Perplexity Trajectories",
    include_initial_evaluation: bool = False,
    logarithmic_y: bool = False,
) -> None:
    if summary.empty:
        raise ValueError("Cannot plot best trajectories because the aggregated trajectory dataframe is empty.")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    std_metric = metric.replace("_mean", "_std") if metric.endswith("_mean") else None
    plotted_any = False
    for method, group in summary.groupby("optimizer_mode", dropna=False):
        group = group.sort_values("train/global_step")
        x_values = pd.to_numeric(group["train/global_step"], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[finite_mask]
        y_values = y_values[finite_mask]
        std_values = None
        if std_metric is not None and std_metric in group.columns:
            std_values = pd.to_numeric(group[std_metric], errors="coerce").to_numpy(dtype=float)[finite_mask]
            std_values = np.nan_to_num(std_values, nan=0.0)
        if not include_initial_evaluation and np.any(x_values > 0):
            trained_mask = x_values > 0
            x_values = x_values[trained_mask]
            y_values = y_values[trained_mask]
            if std_values is not None:
                std_values = std_values[trained_mask]
        if logarithmic_y:
            positive_mask = y_values > 0
            x_values = x_values[positive_mask]
            y_values = y_values[positive_mask]
            if std_values is not None:
                std_values = std_values[positive_mask]
        if x_values.size == 0:
            continue
        marker_interval = max(1, x_values.size // 10)
        ax.plot(
            x_values,
            y_values,
            linewidth=2.0,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=4.0,
            markevery=marker_interval,
            label=method,
        )
        plotted_any = True
        if std_values is not None:
            if np.any(std_values > 0):
                ax.fill_between(
                    x_values,
                    y_values - std_values,
                    y_values + std_values,
                    color=METHOD_COLORS[method],
                    alpha=0.14,
                )
    ax.set_xlabel("global step")
    ax.set_ylabel(metric.replace("_", " "))
    if logarithmic_y:
        ax.set_yscale("log")
    ax.margins(y=0.08)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if not plotted_any:
        raise ValueError(
            f"Could not plot any trajectory lines for metric '{metric}'. "
            "The aggregated dataframe contains no finite values for that metric."
        )
    ax.legend()
    save_figure(fig, path)
