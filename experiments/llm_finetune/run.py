"""Run causal language-model fine-tuning experiments with adaptive optimizers."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_clipping.adaptive_optimizers import ADAPTIVE_OPTIMIZER_NAMES, CLIPPING_SCOPES
from residual_clipping.cli import add_resume_args, add_wandb_args
from residual_clipping.llm_finetune_pipeline import (
    DATASET_SOURCES,
    MODEL_SOURCES,
    TASK_TYPES,
    run_llm_finetune_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run transformer fine-tuning experiments.")
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument("--task-type", choices=TASK_TYPES, default="causal_lm")
    parser.add_argument("--model-source", choices=MODEL_SOURCES, default="fake")
    parser.add_argument("--model-name", type=str, default="tiny-causal-lm")
    parser.add_argument(
        "--model-checkpoint",
        dest="model_name",
        type=str,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Optional Hugging Face model revision, branch, tag, or commit hash.",
    )
    parser.add_argument("--dataset-source", choices=DATASET_SOURCES, default="fake")
    parser.add_argument("--dataset-name", type=str, default="fake_causal_lm")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--sentence1-column", type=str, default="sentence1")
    parser.add_argument("--sentence2-column", type=str, default="sentence2")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--validation-file", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Shared Hugging Face cache root for model/tokenizer and dataset downloads.",
    )
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer-name", choices=ADAPTIVE_OPTIMIZER_NAMES, required=True)
    parser.add_argument("--optimizer-mode", dest="optimizer_name", choices=ADAPTIVE_OPTIMIZER_NAMES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--learning-rate", dest="lr", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-threshold", type=float, default=float("inf"))
    parser.add_argument("--clipping-scope", choices=CLIPPING_SCOPES, default="local")
    parser.add_argument("--correct-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--classifier-dropout", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-check-interval", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-eval-sequences", type=int, default=None)
    parser.add_argument("--fake-train-sequences", type=int, default=64)
    parser.add_argument("--fake-eval-sequences", type=int, default=16)
    parser.add_argument("--fake-vocab-size", type=int, default=128)
    parser.add_argument("--fake-hidden-size", type=int, default=64)
    parser.add_argument("--fake-num-layers", type=int, default=1)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="Optional namespace added to run names and the derived W&B group.",
    )
    return parser


def validate_args(args, *, sweep: bool = False) -> None:
    if not sweep and args.optimizer_name is None:
        raise ValueError("--optimizer-name is required.")
    if args.model_source == "fake" and args.dataset_source != "fake":
        raise ValueError("--model-source fake requires --dataset-source fake.")
    if args.model_source == "hf" and args.model_name == "tiny-causal-lm":
        raise ValueError("--model-source hf requires a Hugging Face --model-name.")
    if args.task_type == "sequence_classification" and args.dataset_source == "text":
        raise ValueError("--task-type sequence_classification supports --dataset-source fake or hf.")
    if args.task_type == "sequence_classification" and args.num_labels < 2:
        raise ValueError("--num-labels must be >= 2 for sequence classification.")
    if args.dataset_source == "text" and args.train_file is None:
        raise ValueError("--train-file is required with --dataset-source text.")
    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be >= 1.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("--batch-size and --eval-batch-size must be >= 1.")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1.")
    if args.lr is not None and args.lr < 0:
        raise ValueError("--lr must be non-negative.")
    if not 0.0 <= args.adam_beta1 < 1.0:
        raise ValueError("--adam-beta1 must be in [0,1).")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("--adam-beta2 must be in [0,1).")
    if args.adam_eps <= 0.0:
        raise ValueError("--adam-eps must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.clip_threshold < 0.0:
        raise ValueError("--clip-threshold must be non-negative.")
    if args.classifier_dropout < 0.0 or args.classifier_dropout >= 1.0:
        raise ValueError("--classifier-dropout must be in [0,1).")
    if args.val_check_interval is not None and args.val_check_interval < 1:
        raise ValueError("--val-check-interval must be >= 1 when provided.")
    if args.fake_vocab_size < 2:
        raise ValueError("--fake-vocab-size must be >= 2.")
    if args.fake_train_sequences < 1 or args.fake_eval_sequences < 1:
        raise ValueError("--fake-train-sequences and --fake-eval-sequences must be >= 1.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run_llm_finetune_experiment(args)


if __name__ == "__main__":
    main()
