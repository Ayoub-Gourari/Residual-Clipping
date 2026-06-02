"""Generate centralized CIFAR-10 sweep plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from residual_clipping.cifar10_plots import (
    aggregate_trajectories,
    plot_best_accuracy_vs_threshold,
    plot_best_trajectories,
)


def load_metrics_frame(run_dir: Path) -> pd.DataFrame:
    metrics_file = run_dir / "metrics.jsonl"
    summary_file = run_dir / "summary.json"
    if not metrics_file.exists() or not summary_file.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    frame.attrs["run_name"] = summary["run_name"]
    frame.attrs["optimizer_mode"] = summary["optimizer_mode"]
    frame.attrs["model"] = summary["model"]
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot centralized CIFAR-10 sweep outputs.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs") / "cifar10_sweeps")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.sweep_dir / "threshold_summary.csv")
    plot_best_accuracy_vs_threshold(summary, args.figure_dir / "cifar10_best_accuracy_vs_threshold")

    run_summaries = pd.read_csv(args.sweep_dir / "run_summaries.csv")
    best_rows = []
    for method, group in run_summaries.groupby("optimizer_mode", dropna=False):
        best_rows.append(group.sort_values("best_validation_accuracy", ascending=False).iloc[0])
    metrics_frames = [load_metrics_frame(args.runs_root / row["run_name"]) for row in best_rows]
    aggregated = aggregate_trajectories(metrics_frames)
    plot_best_trajectories(
        aggregated,
        args.figure_dir / "cifar10_best_trajectories",
        metric="train_accuracy_mean",
        title="Best Training Accuracy Trajectories",
    )


if __name__ == "__main__":
    main()
