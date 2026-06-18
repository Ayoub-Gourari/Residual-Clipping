"""Run causal language-model fine-tuning sweeps."""

from __future__ import annotations

import argparse

from residual_clipping.adaptive_optimizers import ADAPTIVE_OPTIMIZER_MODES
from residual_clipping.llm_finetune_pipeline import run_sweep, summarize_sweep_results, sweep_output_dir
from residual_clipping.quadratics import parse_float_list

from .run import build_parser as build_run_parser, validate_args


def parse_string_list(text: str | None) -> list[str]:
    if text is None or text == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = build_run_parser()
    parser.description = "Run transformer fine-tuning sweeps."
    for action in parser._actions:
        if action.dest == "optimizer_mode":
            action.required = False
            action.default = None
        if action.dest == "lr":
            action.default = None
    parser.add_argument("--optimizer-modes", type=parse_string_list, default=list(ADAPTIVE_OPTIMIZER_MODES))
    parser.add_argument("--lrs", type=parse_float_list, default=[5e-4])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    return parser


def validate_sweep_args(args) -> None:
    validate_args(args, sweep=True)
    if not args.optimizer_modes:
        raise ValueError("--optimizer-modes must contain at least one mode.")
    unknown = sorted(set(args.optimizer_modes) - set(ADAPTIVE_OPTIMIZER_MODES))
    if unknown:
        raise ValueError(f"Unknown optimizer modes: {', '.join(unknown)}")
    if not args.lrs:
        raise ValueError("--lrs must contain at least one learning rate.")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_sweep_args(args)
    results = run_sweep(args)
    sweep_dir = sweep_output_dir(args.output_dir, args.dataset_name)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(sweep_dir / "run_summaries.csv", index=False)
    summarize_sweep_results(results).to_csv(sweep_dir / "threshold_summary.csv", index=False)


if __name__ == "__main__":
    main()
