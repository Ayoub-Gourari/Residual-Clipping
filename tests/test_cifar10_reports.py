import json
import tempfile
from pathlib import Path

import pandas as pd

from residual_clipping.cifar10_pipeline import summary_indicates_complete, write_summary
from residual_clipping.cifar10_reports import (
    collect_best_runs,
    collect_best_trajectory_records,
    expand_run_diagnostics,
    load_metrics_frame,
    summarize_best_trajectories,
)


def test_expand_run_diagnostics_extracts_metrics_and_threshold():
    run_summaries = pd.DataFrame(
        [
            {
                "run_name": "run-a",
                "model": "resnet20",
                "optimizer_mode": "clipped_momentum",
                "best_validation_accuracy": 91.0,
                "clip_c": 0.3,
                "clip_c_res": float("nan"),
                "diagnostics": "{'clip_fraction_mean': 0.5, 'grad_norm_mean': 2.0}",
            },
            {
                "run_name": "run-b",
                "model": "resnet20",
                "optimizer_mode": "residual_clipped_momentum",
                "best_validation_accuracy": 92.0,
                "clip_c": float("nan"),
                "clip_c_res": 0.1,
                "diagnostics": "{'residual_clip_fraction_mean': 0.25, 'cos_grad_center_mean': 0.8}",
            },
        ]
    )
    expanded = expand_run_diagnostics(run_summaries)
    assert list(expanded["clip_threshold"]) == [0.3, 0.1]
    assert float(expanded.loc[0, "clip_fraction_mean"]) == 0.5
    assert float(expanded.loc[1, "cos_grad_center_mean"]) == 0.8


def test_collect_best_runs_picks_best_per_model_and_method():
    run_summaries = pd.DataFrame(
        [
            {"run_name": "a", "model": "resnet20", "optimizer_mode": "clipped_momentum", "best_validation_accuracy": 90.0},
            {"run_name": "b", "model": "resnet20", "optimizer_mode": "clipped_momentum", "best_validation_accuracy": 92.0},
            {"run_name": "c", "model": "resnet20", "optimizer_mode": "sgd_momentum", "best_validation_accuracy": 89.0},
            {"run_name": "d", "model": "resnet18", "optimizer_mode": "sgd_momentum", "best_validation_accuracy": 87.0},
        ]
    )
    best = collect_best_runs(run_summaries)
    assert list(best["run_name"]) == ["d", "b", "c"]


def test_metrics_loading_and_trajectory_summary():
    with tempfile.TemporaryDirectory() as tmp_dir:
        runs_root = Path(tmp_dir)
        run_dir = runs_root / "resnet20-clipped"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "train/global_step": 0,
                            "validation/loss": 1.8,
                            "validation/accuracy": 40.0,
                        }
                    ),
                    json.dumps(
                        {
                            "train/global_step": 1,
                            "train/loss": 1.2,
                            "train/accuracy": 55.0,
                            "validation/loss": 1.6,
                            "validation/accuracy": 48.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_name": "resnet20-clipped",
                    "optimizer_mode": "clipped_momentum",
                    "model": "resnet20",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        frame = load_metrics_frame(run_dir)
        assert frame.attrs["run_name"] == "resnet20-clipped"
        assert frame.attrs["optimizer_mode"] == "clipped_momentum"

        best_runs = pd.DataFrame(
            [
                {
                    "run_name": "resnet20-clipped",
                    "model": "resnet20",
                    "optimizer_mode": "clipped_momentum",
                    "best_validation_accuracy": 48.0,
                }
            ]
        )
        records = collect_best_trajectory_records(best_runs, runs_root)
        assert list(records["run_name"].unique()) == ["resnet20-clipped"]

        summary = summarize_best_trajectories(records)
        step_zero = summary[summary["train/global_step"] == 0].iloc[0]
        step_one = summary[summary["train/global_step"] == 1].iloc[0]
        assert float(step_zero["validation_accuracy_mean"]) == 40.0
        assert float(step_one["train_accuracy_mean"]) == 55.0


def test_summary_indicates_complete_checks_epoch_and_flag():
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir) / "run"
        write_summary(run_dir, {"epochs": 10, "completed": 1})
        assert summary_indicates_complete(run_dir, epochs=10) is True
        assert summary_indicates_complete(run_dir, epochs=11) is False

        write_summary(run_dir, {"epochs": 12, "completed": 0})
        assert summary_indicates_complete(run_dir, epochs=10) is False
