"""Run centralized CIFAR sweeps across models, methods, lrs, and thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_clipping.cli import add_resume_args, add_wandb_args
from residual_clipping.cifar10_data import DATASET_NAMES
from residual_clipping.cifar10_pipeline import (
    OPTIMIZER_MODES,
    run_sweep,
    summarize_sweep_results,
    sweep_output_dir,
)
from residual_clipping.quadratics import parse_float_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run centralized CIFAR sweeps.")
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument("--models", type=lambda value: [item.strip() for item in value.split(",") if item.strip()], default=["resnet20"])
    parser.add_argument("--optimizer-mode", choices=OPTIMIZER_MODES, default=None)
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="cifar10")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--lrs", type=parse_float_list, default=[0.05, 0.1])
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clip-values", type=parse_float_list, default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--res-clip-values", type=parse_float_list, default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--lr-milestones", type=str, default="75")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--fake-train-size", type=int, default=512)
    parser.add_argument("--fake-test-size", type=int, default=128)
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_sweep(args)
    sweep_dir = sweep_output_dir(args.output_dir, args.dataset)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(sweep_dir / "run_summaries.csv", index=False)
    summarize_sweep_results(results).to_csv(sweep_dir / "threshold_summary.csv", index=False)


if __name__ == "__main__":
    main()
