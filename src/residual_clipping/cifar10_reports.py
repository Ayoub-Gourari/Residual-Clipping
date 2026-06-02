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


def discover_run_summaries(runs_root: Path) -> pd.DataFrame:
    rows = []
    for summary_file in sorted(runs_root.glob("*/summary.json")):
        if summary_file.parent.name == "cifar10_sweeps":
            continue
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
        if "run_name" not in payload:
            payload["run_name"] = summary_file.parent.name
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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


def combine_run_summaries(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True, sort=False)
    if "run_name" in combined.columns:
        combined = combined.drop_duplicates(subset=["run_name"], keep="last")
    return combined.reset_index(drop=True)


def build_threshold_summary(run_summaries: pd.DataFrame) -> pd.DataFrame:
    if run_summaries.empty:
        return pd.DataFrame()
    results = run_summaries.copy()
    for column in ["clip_c", "clip_c_res", "best_validation_accuracy"]:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    results["clip_threshold"] = results.get("clip_c", pd.Series(dtype=float)).fillna(
        results.get("clip_c_res", pd.Series(dtype=float))
    )
    group_cols = ["model", "optimizer_mode", "clip_threshold"]
    summary = results.groupby(group_cols, dropna=False).agg(
        best_accuracy=("best_validation_accuracy", "max"),
        mean_best_accuracy=("best_validation_accuracy", "mean"),
        run_count=("run_name", "count"),
    ).reset_index()
    return summary.sort_values(["model", "optimizer_mode", "clip_threshold"]).reset_index(drop=True)


def collect_best_runs(run_summaries: pd.DataFrame) -> pd.DataFrame:
    if run_summaries.empty:
        return run_summaries.copy()
    best_rows = []
    for (_, optimizer_mode), group in run_summaries.groupby(["model", "optimizer_mode"], dropna=False):
        best_rows.append(group.sort_values("best_validation_accuracy", ascending=False).iloc[0].to_dict())
    return pd.DataFrame(best_rows).sort_values(["model", "optimizer_mode"]).reset_index(drop=True)


def collect_best_trajectory_records(best_runs: pd.DataFrame, runs_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing = []
    for _, row in best_runs.iterrows():
        run_dir = runs_root / row["run_name"]
        metrics = load_metrics_frame(run_dir)
        if metrics.empty:
            missing.append(str(run_dir))
            continue
        metrics = metrics.copy()
        metrics["run_name"] = row["run_name"]
        metrics["model"] = row["model"]
        metrics["optimizer_mode"] = row["optimizer_mode"]
        metrics["best_validation_accuracy"] = row["best_validation_accuracy"]
        frames.append(metrics)
    if not frames:
        message = "No trajectory metrics found for selected best runs."
        if missing:
            message += " Missing run directories or metrics files:\n" + "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(message)
    result = pd.concat(frames, ignore_index=True)
    if missing:
        result.attrs["missing_trajectory_runs"] = missing
    return result


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
