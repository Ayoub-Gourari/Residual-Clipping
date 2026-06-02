"""Print three-machine launch commands for centralized CIFAR-10 methods."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print three-machine CIFAR-10 commands.")
    parser.add_argument("--model", default="resnet20")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--common-args", default="--wandb-mode online")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    commands = [
        ("machine 1", "sgd_momentum"),
        ("machine 2", "clipped_momentum"),
        ("machine 3", "residual_clipped_momentum"),
    ]
    for machine, mode in commands:
        print(
            f"{machine}: python3 -m experiments.cifar10.sweep "
            f"--models {args.model} "
            f"--optimizer-mode {mode} "
            f"--wandb-group {args.wandb_group} "
            f"{args.common_args}"
        )


if __name__ == "__main__":
    main()
