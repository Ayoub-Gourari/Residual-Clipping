import json
import subprocess
import sys
import tempfile
import csv
from pathlib import Path


def write_fake_run(run_dir: Path, *, run_name: str, model: str, optimizer_mode: str, best_accuracy: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        {"train/global_step": 0, "validation/loss": 2.0, "validation/accuracy": best_accuracy - 10.0},
        {
            "train/global_step": 1,
            "train/loss": 1.4,
            "train/accuracy": best_accuracy - 5.0,
            "validation/loss": 1.5,
            "validation/accuracy": best_accuracy,
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in metrics) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "model": model,
                "optimizer_mode": optimizer_mode,
                "best_validation_accuracy": best_accuracy,
                "wandb_group": "demo-group",
                "wandb_project": "demo-project",
                "wandb_job_type": "sweep",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_list_experiments_cli_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/list_experiments.py", "--kind", "sweep"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cifar10-resnet20-sweep" in result.stdout
    assert "cifar100-vgg16-sweep" in result.stdout
    assert "wikitext2-lstm-sweep" in result.stdout


def test_registered_report_cli_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        runs_root = tmp_root / "runs"
        sweep_dir = tmp_root / "sweeps"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "run_name": "resnet20-sgd",
                "model": "resnet20",
                "optimizer_mode": "sgd_momentum",
                "best_validation_accuracy": 80.0,
                "clip_c": "",
                "clip_c_res": "",
                "wandb_group": "demo-group",
                "wandb_project": "demo-project",
                "wandb_job_type": "sweep",
                "diagnostics": "{}",
            },
            {
                "run_name": "resnet20-standard",
                "model": "resnet20",
                "optimizer_mode": "clipped_momentum",
                "best_validation_accuracy": 81.0,
                "clip_c": 0.3,
                "clip_c_res": "",
                "wandb_group": "demo-group",
                "wandb_project": "demo-project",
                "wandb_job_type": "sweep",
                "diagnostics": "{'clip_fraction_mean': 0.5}",
            },
            {
                "run_name": "resnet20-residual",
                "model": "resnet20",
                "optimizer_mode": "residual_clipped_momentum",
                "best_validation_accuracy": 82.0,
                "clip_c": "",
                "clip_c_res": 0.1,
                "wandb_group": "demo-group",
                "wandb_project": "demo-project",
                "wandb_job_type": "sweep",
                "diagnostics": "{'residual_clip_fraction_mean': 0.25}",
            },
        ]
        with (sweep_dir / "run_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (sweep_dir / "threshold_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "optimizer_mode",
                    "clip_threshold",
                    "best_accuracy",
                    "mean_best_accuracy",
                    "run_count",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"model": "resnet20", "optimizer_mode": "clipped_momentum", "clip_threshold": 0.3, "best_accuracy": 81.0, "mean_best_accuracy": 81.0, "run_count": 1},
                    {"model": "resnet20", "optimizer_mode": "residual_clipped_momentum", "clip_threshold": 0.1, "best_accuracy": 82.0, "mean_best_accuracy": 82.0, "run_count": 1},
                    {"model": "resnet20", "optimizer_mode": "sgd_momentum", "clip_threshold": "", "best_accuracy": 80.0, "mean_best_accuracy": 80.0, "run_count": 1},
                ]
            )

        write_fake_run(runs_root / "resnet20-sgd", run_name="resnet20-sgd", model="resnet20", optimizer_mode="sgd_momentum", best_accuracy=80.0)
        write_fake_run(runs_root / "resnet20-standard", run_name="resnet20-standard", model="resnet20", optimizer_mode="clipped_momentum", best_accuracy=81.0)
        write_fake_run(runs_root / "resnet20-residual", run_name="resnet20-residual", model="resnet20", optimizer_mode="residual_clipped_momentum", best_accuracy=82.0)

        report_dir = tmp_root / "reports"
        figure_dir = tmp_root / "figures"
        subprocess.run(
            [
                sys.executable,
                "scripts/run_registered_experiment.py",
                "--name",
                "cifar10-resnet20-report",
                "--",
                "--sweep-dir",
                str(sweep_dir),
                "--runs-root",
                str(runs_root),
                "--report-dir",
                str(report_dir),
                "--figure-dir",
                str(figure_dir),
            ],
            cwd=repo_root,
            check=True,
        )

        assert (report_dir / "best_runs.csv").exists()
        assert (report_dir / "wandb_context.json").exists()
        assert (figure_dir / "cifar10_best_accuracy_vs_threshold.png").exists()
        assert (figure_dir / "cifar10_best_trajectories.pdf").exists()
