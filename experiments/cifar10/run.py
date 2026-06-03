"""Run centralized CIFAR clipping experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_clipping.cli import add_resume_args, add_wandb_args
from residual_clipping.cifar10_data import DATASET_NAMES
from residual_clipping.cifar10_pipeline import OPTIMIZER_MODES, run_cifar10_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run centralized CIFAR clipping experiments.")
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument(
        "--model",
        choices=("resnet20", "resnet18", "vgg16"),
        default="resnet20",
        help="Model architecture.",
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=OPTIMIZER_MODES,
        required=True,
        help="Optimization mode.",
    )
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.9, help="EMA momentum coefficient.")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clip-c", "--clip-threshold", dest="clip_c", type=float, default=None)
    parser.add_argument("--clip-c-res", "--res-clip-threshold", dest="clip_c_res", type=float, default=None)
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
    if not 0 <= args.beta < 1:
        raise ValueError(f"--beta must be in [0,1). Got {args.beta}.")
    if args.optimizer_mode == "clipped_momentum" and args.clip_c is None:
        raise ValueError("--clip-c is required for clipped_momentum.")
    if args.optimizer_mode == "residual_clipped_momentum" and args.clip_c_res is None:
        raise ValueError("--clip-c-res is required for residual_clipped_momentum.")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1.")
    run_cifar10_experiment(args)


if __name__ == "__main__":
    main()
