"""Generate centralized CIFAR-10 sweep plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from residual_clipping.cifar10_plots import plot_best_accuracy_vs_threshold, plot_best_trajectories
from residual_clipping.cifar10_reports import collect_best_runs, collect_best_trajectory_records, summarize_best_trajectories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot centralized CIFAR-10 sweep outputs.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs") / "cifar10_sweeps")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--figure-prefix", type=str, default="cifar10")
    parser.add_argument(
        "--trajectory-metric",
        choices=("train_accuracy_mean", "train_loss_mean", "validation_accuracy_mean", "validation_loss_mean"),
        default="validation_accuracy_mean",
    )
    parser.add_argument("--trajectory-title", type=str, default="Best Validation Accuracy Trajectories")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.report_dir if args.report_dir is not None else args.sweep_dir

    summary = pd.read_csv(summary_dir / "threshold_summary.csv")
    plot_best_accuracy_vs_threshold(summary, args.figure_dir / f"{args.figure_prefix}_best_accuracy_vs_threshold")

    precomputed_trajectory_summary = summary_dir / "best_trajectory_summary.csv"
    if precomputed_trajectory_summary.exists():
        aggregated = pd.read_csv(precomputed_trajectory_summary)
    else:
        run_summaries = pd.read_csv(summary_dir / "run_summaries.csv")
        best_runs = collect_best_runs(run_summaries)
        records = collect_best_trajectory_records(best_runs, args.runs_root)
        aggregated = summarize_best_trajectories(records)
    plot_best_trajectories(
        aggregated,
        args.figure_dir / f"{args.figure_prefix}_best_trajectories",
        metric=args.trajectory_metric,
        title=args.trajectory_title,
    )


if __name__ == "__main__":
    main()
