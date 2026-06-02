"""Collect compact CSV artifacts from centralized CIFAR-10 sweep outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from residual_clipping.cifar10_reports import (
    collect_best_runs,
    collect_best_trajectory_records,
    expand_run_diagnostics,
    summarize_best_trajectories,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect centralized CIFAR-10 sweep summaries.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs") / "cifar10_sweeps")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports") / "cifar10")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = pd.read_csv(args.sweep_dir / "run_summaries.csv")
    threshold_summary = pd.read_csv(args.sweep_dir / "threshold_summary.csv")
    run_diagnostics = expand_run_diagnostics(run_summaries)
    best_runs = collect_best_runs(run_summaries)
    best_trajectory_records = collect_best_trajectory_records(best_runs, args.runs_root)
    best_trajectory_summary = summarize_best_trajectories(best_trajectory_records)

    threshold_summary.to_csv(args.report_dir / "threshold_summary.csv", index=False)
    run_summaries.to_csv(args.report_dir / "run_summaries.csv", index=False)
    run_diagnostics.to_csv(args.report_dir / "run_diagnostics.csv", index=False)
    best_runs.to_csv(args.report_dir / "best_runs.csv", index=False)
    best_trajectory_records.to_csv(args.report_dir / "best_trajectory_records.csv", index=False)
    best_trajectory_summary.to_csv(args.report_dir / "best_trajectory_summary.csv", index=False)


if __name__ == "__main__":
    main()
