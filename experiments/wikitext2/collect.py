"""Collect compact CSV artifacts from centralized WikiText-2 sweep outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from residual_clipping.logging_utils import atomic_write_json
from residual_clipping.wikitext2_reports import (
    build_threshold_summary,
    collect_best_runs,
    collect_best_trajectory_records,
    combine_run_summaries,
    discover_run_summaries,
    expand_run_diagnostics,
    summarize_best_trajectories,
    summarize_wandb_context,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect centralized WikiText-2 sweep summaries.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs") / "wikitext2" / "wikitext2_sweeps")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs") / "wikitext2")
    parser.add_argument("--report-dir", type=Path, default=Path("reports") / "wikitext2" / "lstm")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    sweep_run_summaries_path = args.sweep_dir / "run_summaries.csv"
    sweep_threshold_summary_path = args.sweep_dir / "threshold_summary.csv"
    sweep_run_summaries = (
        pd.read_csv(sweep_run_summaries_path)
        if sweep_run_summaries_path.exists()
        else pd.DataFrame()
    )
    discovered_run_summaries = discover_run_summaries(args.runs_root)
    run_summaries = combine_run_summaries(sweep_run_summaries, discovered_run_summaries)
    if run_summaries.empty:
        raise FileNotFoundError(
            f"No run summaries found in {sweep_run_summaries_path} or under {args.runs_root}/*/summary.json."
        )
    threshold_summary = (
        pd.read_csv(sweep_threshold_summary_path)
        if sweep_threshold_summary_path.exists()
        else build_threshold_summary(run_summaries)
    )
    if discovered_run_summaries is not None and not discovered_run_summaries.empty:
        threshold_summary = build_threshold_summary(run_summaries)
    run_diagnostics = expand_run_diagnostics(run_summaries)
    best_runs = collect_best_runs(run_summaries)
    best_trajectory_records = collect_best_trajectory_records(best_runs, args.runs_root)
    best_trajectory_summary = summarize_best_trajectories(best_trajectory_records)
    wandb_context = summarize_wandb_context(run_summaries)

    threshold_summary.to_csv(args.report_dir / "threshold_summary.csv", index=False)
    run_summaries.to_csv(args.report_dir / "run_summaries.csv", index=False)
    run_diagnostics.to_csv(args.report_dir / "run_diagnostics.csv", index=False)
    best_runs.to_csv(args.report_dir / "best_runs.csv", index=False)
    best_trajectory_records.to_csv(args.report_dir / "best_trajectory_records.csv", index=False)
    best_trajectory_summary.to_csv(args.report_dir / "best_trajectory_summary.csv", index=False)
    atomic_write_json(args.report_dir / "wandb_context.json", wandb_context)


if __name__ == "__main__":
    main()
