#!/usr/bin/env python3
"""Aggregate ALBERT/RTE reproduction validation curves across seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/albert_rte_reproduction"))
    parser.add_argument("--csv", type=Path, nargs="*", default=None, help="Optional W&B-exported CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--plots", action="store_true", default=False, help="Also write PNG plots when matplotlib works.")
    return parser


def _first_available(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def records_from_local(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(input_dir.glob("**/summary.json")):
        run_dir = summary_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            val_loss = _first_available(row, ["eval/loss", "val_loss", "validation/loss"])
            val_accuracy = _first_available(row, ["eval/accuracy", "val_accuracy", "validation/accuracy"])
            if val_loss is None and val_accuracy is None:
                continue
            step = _first_available(row, ["global_step", "validation/global_step", "train/global_step", "Step"])
            records.append(
                {
                    "optimizer_name": summary.get("optimizer_name"),
                    "clip_threshold": summary.get("clip_threshold"),
                    "clipping_scope": summary.get("clipping_scope"),
                    "seed": summary.get("seed"),
                    "global_step": int(step or 0),
                    "val_loss": None if val_loss is None else float(val_loss),
                    "val_accuracy": None if val_accuracy is None else float(val_accuracy),
                    "best_loss": summary.get("best_validation_loss"),
                    "best_accuracy": summary.get("best_validation_accuracy"),
                    "run_name": summary.get("run_name", run_dir.name),
                }
            )
    return records


def records_from_csv(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            val_loss = _first_available(row_dict, ["eval/loss", "val_loss", "validation/loss", "validation_loss"])
            val_accuracy = _first_available(
                row_dict,
                ["eval/accuracy", "val_accuracy", "validation/accuracy", "validation_accuracy"],
            )
            if pd.isna(val_loss) and pd.isna(val_accuracy):
                continue
            step = _first_available(row_dict, ["global_step", "validation/global_step", "Step", "_step"])
            records.append(
                {
                    "optimizer_name": _first_available(row_dict, ["optimizer_name", "run/optimizer_name"]),
                    "clip_threshold": _first_available(row_dict, ["clip_threshold", "run/clip_threshold"]),
                    "clipping_scope": _first_available(row_dict, ["clipping_scope", "run/clipping_scope"]),
                    "seed": _first_available(row_dict, ["seed", "run/seed"]),
                    "global_step": int(0 if pd.isna(step) else step),
                    "val_loss": None if pd.isna(val_loss) else float(val_loss),
                    "val_accuracy": None if pd.isna(val_accuracy) else float(val_accuracy),
                    "best_loss": _first_available(
                        row_dict,
                        ["eval/best_loss", "best_validation_loss"],
                    ),
                    "best_accuracy": _first_available(
                        row_dict,
                        ["eval/best_accuracy", "best_validation_accuracy"],
                    ),
                    "run_name": _first_available(row_dict, ["run_name", "Name", "name"]),
                }
            )
    return records


def summarize(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    filtered = frame.dropna(subset=[value_col]).copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=["optimizer_name", "clip_threshold", "clipping_scope", "global_step", "median", "q05", "q95", "run_count"]
        )
    grouped = filtered.groupby(["optimizer_name", "clip_threshold", "clipping_scope", "global_step"], dropna=False)[value_col]
    return grouped.agg(
        median="median",
        q05=lambda values: values.quantile(0.05),
        q95=lambda values: values.quantile(0.95),
        run_count="count",
    ).reset_index()


def run_accuracy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["optimizer_name", "clip_threshold", "clipping_scope", "seed", "run_name", "accuracy"]
        )
    rows: list[dict[str, Any]] = []
    group_cols = ["optimizer_name", "clip_threshold", "clipping_scope", "seed", "run_name"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        best_values = pd.to_numeric(group.get("best_accuracy"), errors="coerce").dropna()
        accuracy_values = pd.to_numeric(group["val_accuracy"], errors="coerce").dropna()
        if best_values.empty and accuracy_values.empty:
            continue
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "accuracy": float(best_values.max() if not best_values.empty else accuracy_values.max()),
            }
        )
    return pd.DataFrame(rows)


def run_loss_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["optimizer_name", "clip_threshold", "clipping_scope", "seed", "run_name", "loss"]
        )
    rows: list[dict[str, Any]] = []
    group_cols = ["optimizer_name", "clip_threshold", "clipping_scope", "seed", "run_name"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        best_source = group["best_loss"] if "best_loss" in group else pd.Series(dtype=float)
        best_values = pd.to_numeric(best_source, errors="coerce").dropna()
        loss_values = pd.to_numeric(group["val_loss"], errors="coerce").dropna()
        if best_values.empty and loss_values.empty:
            continue
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "loss": float(best_values.min() if not best_values.empty else loss_values.min()),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "optimizer_name",
        "clip_threshold",
        "clipping_scope",
        "threshold/accuracy_mean",
        "threshold/accuracy_std",
        "threshold/gap_to_best",
        "threshold/best_accuracy",
        "threshold/worst_accuracy",
        "threshold/median_accuracy",
        "threshold/spread",
        "threshold/auc_gap",
        "seed/accuracy_mean",
        "seed/accuracy_std",
        "seed/accuracy_min",
        "seed/accuracy_max",
        "seed/accuracy_median",
        "seed/accuracy_range",
        "seed/count",
    ]
    if run_frame.empty:
        return pd.DataFrame(columns=columns)

    grouped = run_frame.groupby(
        ["optimizer_name", "clip_threshold", "clipping_scope"],
        dropna=False,
    )["accuracy"]
    summary = grouped.agg(
        **{
            "seed/accuracy_mean": "mean",
            "seed/accuracy_std": "std",
            "seed/accuracy_min": "min",
            "seed/accuracy_max": "max",
            "seed/accuracy_median": "median",
            "seed/count": "count",
        }
    ).reset_index()
    summary["seed/accuracy_std"] = summary["seed/accuracy_std"].fillna(0.0)
    summary["seed/accuracy_range"] = summary["seed/accuracy_max"] - summary["seed/accuracy_min"]
    summary["threshold/accuracy_mean"] = summary["seed/accuracy_mean"]
    summary["threshold/accuracy_std"] = summary["seed/accuracy_std"]

    method_cols = ["optimizer_name", "clipping_scope"]
    summary["threshold/best_accuracy"] = summary.groupby(method_cols, dropna=False)[
        "threshold/accuracy_mean"
    ].transform("max")
    summary["threshold/worst_accuracy"] = summary.groupby(method_cols, dropna=False)[
        "threshold/accuracy_mean"
    ].transform("min")
    summary["threshold/median_accuracy"] = summary.groupby(method_cols, dropna=False)[
        "threshold/accuracy_mean"
    ].transform("median")
    summary["threshold/gap_to_best"] = (
        summary["threshold/best_accuracy"] - summary["threshold/accuracy_mean"]
    )
    summary["threshold/spread"] = summary["threshold/best_accuracy"] - summary["threshold/worst_accuracy"]
    summary["threshold/auc_gap"] = summary.groupby(method_cols, dropna=False)[
        "threshold/gap_to_best"
    ].transform("mean")
    return summary[columns]


def threshold_loss_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "optimizer_name",
        "clip_threshold",
        "clipping_scope",
        "threshold/loss_mean",
        "threshold/loss_std",
        "threshold/gap_to_best_loss",
        "threshold/best_loss",
        "threshold/worst_loss",
        "threshold/median_loss",
        "threshold/loss_spread",
        "seed/loss_mean",
        "seed/loss_std",
        "seed/loss_min",
        "seed/loss_max",
        "seed/loss_median",
        "seed/count",
    ]
    if run_frame.empty:
        return pd.DataFrame(columns=columns)

    grouped = run_frame.groupby(
        ["optimizer_name", "clip_threshold", "clipping_scope"],
        dropna=False,
    )["loss"]
    summary = grouped.agg(
        **{
            "seed/loss_mean": "mean",
            "seed/loss_std": "std",
            "seed/loss_min": "min",
            "seed/loss_max": "max",
            "seed/loss_median": "median",
            "seed/count": "count",
        }
    ).reset_index()
    summary["seed/loss_std"] = summary["seed/loss_std"].fillna(0.0)
    summary["threshold/loss_mean"] = summary["seed/loss_mean"]
    summary["threshold/loss_std"] = summary["seed/loss_std"]
    method_cols = ["optimizer_name", "clipping_scope"]
    summary["threshold/best_loss"] = summary.groupby(method_cols, dropna=False)[
        "threshold/loss_mean"
    ].transform("min")
    summary["threshold/worst_loss"] = summary.groupby(method_cols, dropna=False)[
        "threshold/loss_mean"
    ].transform("max")
    summary["threshold/median_loss"] = summary.groupby(method_cols, dropna=False)[
        "threshold/loss_mean"
    ].transform("median")
    summary["threshold/gap_to_best_loss"] = (
        summary["threshold/loss_mean"] - summary["threshold/best_loss"]
    )
    summary["threshold/loss_spread"] = (
        summary["threshold/worst_loss"] - summary["threshold/best_loss"]
    )
    return summary[columns]


def select_best_loss_curves(
    loss_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> pd.DataFrame:
    if loss_summary.empty or threshold_summary.empty:
        return loss_summary.iloc[0:0].copy()
    pieces = []
    method_cols = ["optimizer_name", "clipping_scope"]
    for keys, candidates in threshold_summary.groupby(method_cols, dropna=False):
        best = candidates.sort_values(
            ["threshold/loss_mean", "clip_threshold"],
            ascending=[True, True],
        ).iloc[0]
        optimizer_name, clipping_scope = keys
        threshold = float(best["clip_threshold"])
        threshold_values = pd.to_numeric(loss_summary["clip_threshold"], errors="coerce")
        threshold_mask = np.isinf(threshold_values) if np.isinf(threshold) else np.isclose(threshold_values, threshold)
        selected = loss_summary[
            (loss_summary["optimizer_name"] == optimizer_name)
            & (loss_summary["clipping_scope"] == clipping_scope)
            & threshold_mask
        ].copy()
        selected["selected_by"] = "mean_best_validation_loss"
        pieces.append(selected)
    if not pieces:
        return loss_summary.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


DIAGNOSTIC_KEYS = (
    "clipped_grad/relative_norm",
    "pseudo_grad/relative_norm",
    "residual/relative_norm",
    "update/norm",
    "clipping/grad_activation_rate",
    "clipping/residual_activation_rate",
)


def diagnostic_records_from_local(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(input_dir.glob("**/summary.json")):
        run_dir = summary_path.parent
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not any(key in row for key in DIAGNOSTIC_KEYS):
                continue
            record = {
                "optimizer_name": summary.get("optimizer_name"),
                "clip_threshold": summary.get("clip_threshold"),
                "clipping_scope": summary.get("clipping_scope"),
                "seed": summary.get("seed"),
                "run_name": summary.get("run_name", run_dir.name),
                "global_step": int(_first_available(row, ["train/global_step", "global_step"]) or 0),
            }
            for key in DIAGNOSTIC_KEYS:
                record[key] = row.get(key)
            records.append(record)
    return records


def summarize_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["optimizer_name", "clip_threshold", "clipping_scope", *DIAGNOSTIC_KEYS])
    aggregations = {key: "mean" for key in DIAGNOSTIC_KEYS if key in frame.columns}
    return (
        frame.groupby(["optimizer_name", "clip_threshold", "clipping_scope"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )


def write_plot(summary: pd.DataFrame, *, value_name: str, output_path: Path) -> None:
    if summary.empty:
        return
    mpl_config_dir = Path(tempfile.gettempdir()) / "residual_clipping_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {output_path.name}: matplotlib is unavailable ({exc}).", flush=True)
        return

    try:
        fig, axis = plt.subplots(figsize=(7, 4))
        for label_values, group in summary.groupby(["optimizer_name", "clip_threshold"], dropna=False):
            optimizer_name, clip_threshold = label_values
            group = group.sort_values("global_step")
            x_values = pd.to_numeric(group["global_step"], errors="coerce").to_numpy(dtype=float)
            median = pd.to_numeric(group["median"], errors="coerce").to_numpy(dtype=float)
            q05 = pd.to_numeric(group["q05"], errors="coerce").to_numpy(dtype=float)
            q95 = pd.to_numeric(group["q95"], errors="coerce").to_numpy(dtype=float)
            if pd.isna(clip_threshold):
                label = str(optimizer_name)
            else:
                label = f"{optimizer_name}, C={clip_threshold}"
            axis.plot(x_values, median, label=label)
            axis.fill_between(x_values, q05, q95, alpha=0.2)
        axis.set_xlabel("Validation step")
        axis.set_ylabel(value_name)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"Skipping {output_path.name}: plotting failed ({exc}).", flush=True)


def write_threshold_plot(
    frame: pd.DataFrame,
    *,
    value_col: str,
    output_path: Path,
    ylabel: str,
    error_col: str | None = None,
) -> None:
    if frame.empty or value_col not in frame:
        return
    mpl_config_dir = Path(tempfile.gettempdir()) / "residual_clipping_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {output_path.name}: matplotlib is unavailable ({exc}).", flush=True)
        return

    fig, axis = plt.subplots(figsize=(7, 4))
    plotted = False
    all_thresholds = pd.to_numeric(frame["clip_threshold"], errors="coerce").to_numpy(dtype=float)
    finite_thresholds = np.unique(all_thresholds[np.isfinite(all_thresholds) & (all_thresholds > 0.0)])
    for optimizer_name, group in frame.groupby("optimizer_name", dropna=False):
        x = pd.to_numeric(group["clip_threshold"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
        if not valid.any():
            finite_y = y[np.isfinite(y)]
            if finite_y.size and finite_thresholds.size:
                y_value = float(finite_y[0])
                axis.plot(
                    finite_thresholds,
                    np.full_like(finite_thresholds, y_value),
                    linestyle="--",
                    label=str(optimizer_name),
                )
                if error_col is not None and error_col in group:
                    errors = pd.to_numeric(group[error_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    finite_errors = errors[np.isfinite(y)]
                    error = float(finite_errors[0]) if finite_errors.size else 0.0
                    axis.fill_between(finite_thresholds, y_value - error, y_value + error, alpha=0.15)
                plotted = True
            continue
        order = np.argsort(x[valid])
        x_values = x[valid][order]
        y_values = y[valid][order]
        if error_col is None or error_col not in group:
            axis.plot(x_values, y_values, marker="o", label=str(optimizer_name))
        else:
            errors = pd.to_numeric(group[error_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            axis.errorbar(
                x_values,
                y_values,
                yerr=errors[valid][order],
                marker="o",
                capsize=3,
                label=str(optimizer_name),
            )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xscale("log")
    axis.set_xlabel("Clipping threshold C")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_paired_threshold_plot(
    frame: pd.DataFrame,
    *,
    value_cols: tuple[str, str],
    output_path: Path,
    ylabel: str,
) -> None:
    if frame.empty:
        return
    mpl_config_dir = Path(tempfile.gettempdir()) / "residual_clipping_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {output_path.name}: matplotlib is unavailable ({exc}).", flush=True)
        return

    fig, axis = plt.subplots(figsize=(7, 4))
    plotted = False
    for optimizer_name, group in frame.groupby("optimizer_name", dropna=False):
        x = pd.to_numeric(group["clip_threshold"], errors="coerce").to_numpy(dtype=float)
        for value_col in value_cols:
            if value_col not in group:
                continue
            y = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
            if not valid.any():
                continue
            order = np.argsort(x[valid])
            metric_name = value_col.rsplit("/", maxsplit=1)[-1].replace("_", " ")
            axis.plot(x[valid][order], y[valid][order], marker="o", label=f"{optimizer_name}: {metric_name}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xscale("log")
    axis.set_xlabel("Clipping threshold C")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_residual_training_plot(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty or "residual/relative_norm" not in frame:
        return
    mpl_config_dir = Path(tempfile.gettempdir()) / "residual_clipping_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {output_path.name}: matplotlib is unavailable ({exc}).", flush=True)
        return

    grouped = (
        frame.dropna(subset=["residual/relative_norm"])
        .groupby(["optimizer_name", "clip_threshold", "global_step"], dropna=False)["residual/relative_norm"]
        .mean()
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    for keys, group in grouped.groupby(["optimizer_name", "clip_threshold"], dropna=False):
        optimizer_name, threshold = keys
        group = group.sort_values("global_step")
        axis.plot(group["global_step"], group["residual/relative_norm"], label=f"{optimizer_name}, C={threshold}")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Residual relative norm")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    records = records_from_csv(args.csv) if args.csv else records_from_local(args.input_dir)
    if not records:
        raise SystemExit("No validation records found.")

    frame = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_summary = summarize(frame, "val_loss")
    accuracy_summary = summarize(frame, "val_accuracy")
    run_accuracy = run_accuracy_frame(frame)
    run_loss = run_loss_frame(frame)
    threshold_summary = threshold_sensitivity_summary(run_accuracy)
    loss_threshold_summary = threshold_loss_summary(run_loss)
    best_loss_curves = select_best_loss_curves(loss_summary, loss_threshold_summary)
    diagnostic_frame = pd.DataFrame(diagnostic_records_from_local(args.input_dir)) if not args.csv else pd.DataFrame()
    diagnostic_summary = summarize_diagnostics(diagnostic_frame)
    loss_path = args.output_dir / "rte_reproduction_val_loss_summary.csv"
    accuracy_path = args.output_dir / "rte_reproduction_val_accuracy_summary.csv"
    threshold_path = args.output_dir / "rte_threshold_sensitivity_summary.csv"
    loss_threshold_path = args.output_dir / "rte_threshold_loss_summary.csv"
    best_loss_curves_path = args.output_dir / "rte_best_loss_trajectories.csv"
    diagnostic_path = args.output_dir / "rte_optimizer_diagnostics_summary.csv"
    loss_summary.to_csv(loss_path, index=False)
    accuracy_summary.to_csv(accuracy_path, index=False)
    threshold_summary.to_csv(threshold_path, index=False)
    loss_threshold_summary.to_csv(loss_threshold_path, index=False)
    best_loss_curves.to_csv(best_loss_curves_path, index=False)
    diagnostic_summary.to_csv(diagnostic_path, index=False)
    if args.plots:
        write_plot(loss_summary, value_name="Validation loss", output_path=args.output_dir / "rte_reproduction_val_loss.png")
        write_plot(
            accuracy_summary,
            value_name="Validation accuracy",
            output_path=args.output_dir / "rte_reproduction_val_accuracy.png",
        )
        write_threshold_plot(
            threshold_summary,
            value_col="threshold/accuracy_mean",
            error_col="threshold/accuracy_std",
            ylabel="Best validation accuracy",
            output_path=args.output_dir / "rte_threshold_accuracy.png",
        )
        write_threshold_plot(
            threshold_summary,
            value_col="threshold/gap_to_best",
            ylabel="Gap to best validation accuracy",
            output_path=args.output_dir / "rte_threshold_gap_to_best.png",
        )
        write_threshold_plot(
            threshold_summary,
            value_col="seed/accuracy_mean",
            error_col="seed/accuracy_std",
            ylabel="Validation accuracy (mean +/- std)",
            output_path=args.output_dir / "rte_seed_accuracy.png",
        )
        write_threshold_plot(
            loss_threshold_summary,
            value_col="threshold/loss_mean",
            error_col="threshold/loss_std",
            ylabel="Best validation loss (mean +/- std)",
            output_path=args.output_dir / "rte_threshold_best_loss.png",
        )
        write_plot(
            best_loss_curves,
            value_name="Validation loss",
            output_path=args.output_dir / "rte_best_loss_trajectories.png",
        )
        write_paired_threshold_plot(
            diagnostic_summary,
            value_cols=("clipped_grad/relative_norm", "pseudo_grad/relative_norm"),
            ylabel="Relative norm",
            output_path=args.output_dir / "rte_clipped_vs_pseudo_relative_norm.png",
        )
        write_threshold_plot(
            diagnostic_summary,
            value_col="update/norm",
            ylabel="Adaptive update norm",
            output_path=args.output_dir / "rte_update_norm.png",
        )
        write_paired_threshold_plot(
            diagnostic_summary,
            value_cols=("clipping/grad_activation_rate", "clipping/residual_activation_rate"),
            ylabel="Clipping activation rate",
            output_path=args.output_dir / "rte_clipping_activation_rates.png",
        )
        write_residual_training_plot(
            diagnostic_frame,
            args.output_dir / "rte_residual_relative_norm_over_training.png",
        )
    print(f"Wrote {loss_path}", flush=True)
    print(f"Wrote {accuracy_path}", flush=True)
    print(f"Wrote {threshold_path}", flush=True)
    print(f"Wrote {loss_threshold_path}", flush=True)
    print(f"Wrote {best_loss_curves_path}", flush=True)
    print(f"Wrote {diagnostic_path}", flush=True)


if __name__ == "__main__":
    main()
