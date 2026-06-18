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
            val_loss = _first_available(row, ["val_loss", "validation/loss"])
            val_accuracy = _first_available(row, ["val_accuracy", "validation/accuracy"])
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
            val_loss = _first_available(row_dict, ["val_loss", "validation/loss", "validation_loss"])
            val_accuracy = _first_available(row_dict, ["val_accuracy", "validation/accuracy", "validation_accuracy"])
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


def write_plot(summary: pd.DataFrame, *, value_name: str, output_path: Path) -> None:
    if summary.empty:
        return
    mpl_config_dir = Path(tempfile.gettempdir()) / "residual_clipping_mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
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


def main() -> None:
    args = build_parser().parse_args()
    records = records_from_csv(args.csv) if args.csv else records_from_local(args.input_dir)
    if not records:
        raise SystemExit("No validation records found.")

    frame = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_summary = summarize(frame, "val_loss")
    accuracy_summary = summarize(frame, "val_accuracy")
    loss_path = args.output_dir / "rte_reproduction_val_loss_summary.csv"
    accuracy_path = args.output_dir / "rte_reproduction_val_accuracy_summary.csv"
    loss_summary.to_csv(loss_path, index=False)
    accuracy_summary.to_csv(accuracy_path, index=False)
    if args.plots:
        write_plot(loss_summary, value_name="Validation loss", output_path=args.output_dir / "rte_reproduction_val_loss.png")
        write_plot(
            accuracy_summary,
            value_name="Validation accuracy",
            output_path=args.output_dir / "rte_reproduction_val_accuracy.png",
        )
    print(f"Wrote {loss_path}", flush=True)
    print(f"Wrote {accuracy_path}", flush=True)


if __name__ == "__main__":
    main()
