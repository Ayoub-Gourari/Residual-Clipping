"""Run WikiText-2 language-modeling clipping experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_clipping.cli import add_resume_args, add_wandb_args
from residual_clipping.cifar10_pipeline import OPTIMIZER_MODES
from residual_clipping.wikitext2_pipeline import DATASET_NAMES, run_wikitext2_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WikiText-2 language-modeling clipping experiments.")
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument("--model", choices=("lstm",), default="lstm")
    parser.add_argument("--optimizer-mode", choices=OPTIMIZER_MODES, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="wikitext2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--bptt", type=int, default=35)
    parser.add_argument("--embedding-size", type=int, default=650)
    parser.add_argument("--hidden-size", type=int, default=650)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--tie-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=20.0)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-c", "--clip-threshold", dest="clip_c", type=float, default=None)
    parser.add_argument("--clip-c-res", "--res-clip-threshold", dest="clip_c_res", type=float, default=None)
    parser.add_argument("--lr-milestones", type=str, default="")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--fake-token-count", type=int, default=512)
    parser.add_argument("--fake-vocab-size", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def validate_args(args) -> None:
    if not 0 <= args.beta < 1:
        raise ValueError(f"--beta must be in [0,1). Got {args.beta}.")
    if args.optimizer_mode == "clipped_momentum" and args.clip_c is None:
        raise ValueError("--clip-c is required for clipped_momentum.")
    if args.optimizer_mode == "residual_clipped_momentum" and args.clip_c_res is None:
        raise ValueError("--clip-c-res is required for residual_clipped_momentum.")
    if args.tie_weights and args.embedding_size != args.hidden_size:
        raise ValueError("--tie-weights requires --embedding-size to equal --hidden-size.")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run_wikitext2_experiment(args)


if __name__ == "__main__":
    main()
