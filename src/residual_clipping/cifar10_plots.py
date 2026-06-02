"""Plotting helpers for centralized CIFAR-10 experiment outputs."""

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


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_best_accuracy_vs_threshold(summary: pd.DataFrame, path: Path, metric: str = "best_accuracy") -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    clipped = summary[summary["optimizer_mode"] != "sgd_momentum"].copy()
    for method, group in clipped.groupby("optimizer_mode", dropna=False):
        group = group.sort_values("clip_threshold")
        ax.plot(
            group["clip_threshold"],
            group[metric],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xlabel("clipping threshold")
    ax.set_ylabel("best validation accuracy")
    ax.set_title("Best Accuracy vs Clipping Threshold")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path)


def plot_best_trajectories(summary: pd.DataFrame, path: Path, metric: str = "train_accuracy_mean", title: str = "Best Trajectories") -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for method, group in summary.groupby("optimizer_mode", dropna=False):
        group = group.sort_values("train/global_step")
        ax.plot(
            group["train/global_step"],
            group[metric],
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=method,
        )
    ax.set_xlabel("global step")
    ax.set_ylabel(metric.replace("_", " "))
    if "loss" in metric:
        ax.set_ylim(bottom=0)
    elif "accuracy" in metric:
        ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path)
