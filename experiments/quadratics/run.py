"""Quadratics entrypoint placeholder built on shared CLI conventions."""

from __future__ import annotations

import argparse

from residual_clipping.cli import add_resume_args, add_wandb_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic quadratics experiments.")
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument("--steps", type=int, default=20000, help="Number of optimization steps.")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of random seeds.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.9, help="EMA momentum coefficient.")
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise SystemExit("Quadratics implementation has not been added yet.")


if __name__ == "__main__":
    main()
