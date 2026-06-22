"""Run AdamW clipping variants on heavy-tailed synthetic quadratics."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_clipping.adaptive_quadratics import (
    DEFAULT_METHODS,
    DEFAULT_THRESHOLDS,
    AdaptiveQuadraticSuite,
    run_suite,
)
from residual_clipping.quadratics import parse_float_list


def parse_text_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose adaptive clipping under heavy-tailed noise on quadratics."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adaptive_quadratics"))
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--x0-norm", type=float, default=5.0)
    parser.add_argument("--eig-min", type=float, default=1.0)
    parser.add_argument("--eig-max", type=float, default=100.0)
    parser.add_argument("--tail-indices", type=parse_float_list, default=list((1.2, 1.5, 2.5)))
    parser.add_argument("--noise-scales", type=parse_float_list, default=[1.0])
    parser.add_argument("--thresholds", type=parse_float_list, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--lrs", type=parse_float_list, default=[0.01])
    parser.add_argument("--methods", type=parse_text_list, default=list(DEFAULT_METHODS))
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--clipping-scope", choices=("global", "local", "elementwise"), default="global")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--divergence-objective", type=float, default=1e12)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite = AdaptiveQuadraticSuite(
        d=args.d,
        steps=args.steps,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        x0_norm=args.x0_norm,
        eig_min=args.eig_min,
        eig_max=args.eig_max,
        tail_indices=tuple(args.tail_indices),
        noise_scales=tuple(args.noise_scales),
        thresholds=tuple(args.thresholds),
        lrs=tuple(args.lrs),
        methods=tuple(args.methods),
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        clipping_scope=args.clipping_scope,
        log_every=args.log_every,
        tail_fraction=args.tail_fraction,
        divergence_objective=args.divergence_objective,
        dtype=args.dtype,
    )
    frames = run_suite(suite, args.output_dir)
    for name, frame in frames.items():
        print(f"Wrote {args.output_dir / f'{name}.csv'} ({len(frame)} rows)", flush=True)


if __name__ == "__main__":
    main()
