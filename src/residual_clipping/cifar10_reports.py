"""Aggregation helpers for centralized CIFAR-10 sweep outputs."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


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


def _parse_diagnostics(value: object) -> dict[str, float]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def expand_run_diagnostics(run_summaries: pd.DataFrame) -> pd.DataFrame:
    if run_summaries.empty:
        return run_summaries.copy()
    records = []
    for _, row in run_summaries.iterrows():
        record = row.to_dict()
        diagnostics = _parse_diagnostics(record.pop("diagnostics", {}))
        record.update(diagnostics)
        record["clip_threshold"] = record.get("clip_c")
        if pd.isna(record["clip_threshold"]):
            record["clip_threshold"] = record.get("clip_c_res")
        records.append(record)
    return pd.DataFrame(records)


def collect_best_runs(run_summaries: pd.DataFrame) -> pd.DataFrame:
    if run_summaries.empty:
        return run_summaries.copy()
    best_rows = []
    for (_, optimizer_mode), group in run_summaries.groupby(["model", "optimizer_mode"], dropna=False):
        best_rows.append(group.sort_values("best_validation_accuracy", ascending=False).iloc[0].to_dict())
    return pd.DataFrame(best_rows).sort_values(["model", "optimizer_mode"]).reset_index(drop=True)


def collect_best_trajectory_records(best_runs: pd.DataFrame, runs_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in best_runs.iterrows():
        metrics = load_metrics_frame(runs_root / row["run_name"])
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["run_name"] = row["run_name"]
        metrics["model"] = row["model"]
        metrics["optimizer_mode"] = row["optimizer_mode"]
        metrics["best_validation_accuracy"] = row["best_validation_accuracy"]
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_best_trajectories(trajectory_records: pd.DataFrame) -> pd.DataFrame:
    if trajectory_records.empty:
        return trajectory_records.copy()
    group_cols = ["model", "optimizer_mode", "train/global_step"]
    aggregated = trajectory_records.groupby(group_cols, dropna=False).agg(
        train_loss_mean=("train/loss", "mean"),
        train_loss_std=("train/loss", "std"),
        train_accuracy_mean=("train/accuracy", "mean"),
        train_accuracy_std=("train/accuracy", "std"),
        validation_loss_mean=("validation/loss", "mean"),
        validation_loss_std=("validation/loss", "std"),
        validation_accuracy_mean=("validation/accuracy", "mean"),
        validation_accuracy_std=("validation/accuracy", "std"),
        run_count=("run_name", "nunique"),
    ).reset_index()
    return aggregated.sort_values(group_cols).reset_index(drop=True)


def summarize_wandb_context(run_summaries: pd.DataFrame) -> dict[str, object]:
    if run_summaries.empty:
        return {"run_count": 0, "wandb_groups": [], "wandb_projects": [], "wandb_entities": []}
    def unique_non_null(column: str) -> list[str]:
        if column not in run_summaries.columns:
            return []
        values = [str(value) for value in run_summaries[column].dropna().unique().tolist() if str(value).strip()]
        return sorted(values)

    return {
        "run_count": int(len(run_summaries)),
        "wandb_groups": unique_non_null("wandb_group"),
        "wandb_projects": unique_non_null("wandb_project"),
        "wandb_entities": unique_non_null("wandb_entity"),
        "wandb_job_types": unique_non_null("wandb_job_type"),
    }
