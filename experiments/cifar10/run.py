"""CIFAR-10 entrypoint placeholder built on shared CLI conventions."""

from __future__ import annotations

import argparse

from residual_clipping.cli import add_resume_args, add_wandb_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run centralized CIFAR-10 clipping experiments.")
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
        choices=("sgd_momentum", "clipped_momentum", "residual_clipped_momentum"),
        required=True,
        help="Optimization mode.",
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.9, help="EMA momentum coefficient.")
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise SystemExit("CIFAR-10 implementation has not been added yet.")


if __name__ == "__main__":
    main()
