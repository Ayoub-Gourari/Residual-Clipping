"""Run WikiText-2 language-modeling sweeps."""

from __future__ import annotations

import argparse

from residual_clipping.cifar10_pipeline import OPTIMIZER_MODES
from residual_clipping.quadratics import parse_float_list
from residual_clipping.wikitext2_pipeline import run_sweep, summarize_sweep_results, sweep_output_dir

from .run import build_parser as build_run_parser, validate_args


def build_parser() -> argparse.ArgumentParser:
    parser = build_run_parser()
    parser.description = "Run WikiText-2 language-modeling sweeps."
    for action in parser._actions:
        if action.dest == "optimizer_mode":
            action.required = False
            action.default = None
        if action.dest == "lr":
            action.default = None
    parser.add_argument("--lrs", type=parse_float_list, default=[10.0, 20.0, 30.0])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--clip-values", type=parse_float_list, default=[0.1, 0.3, 1.0, 3.0])
    parser.add_argument("--res-clip-values", type=parse_float_list, default=[0.03, 0.1, 0.3, 1.0])
    parser._option_string_actions["--optimizer-mode"].choices = (*OPTIMIZER_MODES,)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    results = run_sweep(args)
    sweep_dir = sweep_output_dir(args.output_dir, args.dataset)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(sweep_dir / "run_summaries.csv", index=False)
    summarize_sweep_results(results).to_csv(sweep_dir / "threshold_summary.csv", index=False)


if __name__ == "__main__":
    main()
