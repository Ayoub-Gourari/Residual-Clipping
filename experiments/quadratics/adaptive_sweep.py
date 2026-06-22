"""Sweep AdamW clipping methods over thresholds, learning rates, and momenta."""

from __future__ import annotations

import argparse
from pathlib import Path

from residual_clipping.adaptive_quadratics import AdaptiveQuadraticSuite
from residual_clipping.adaptive_quadratics_sweep import SWEEP_METHODS, run_hyperparameter_sweep
from residual_clipping.quadratics import parse_float_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adaptive_quadratics/hparam_sweep"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/adaptive_quadratics/hparam_sweep"))
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--x0-norm", type=float, default=5.0)
    parser.add_argument("--eig-min", type=float, default=1.0)
    parser.add_argument("--eig-max", type=float, default=100.0)
    parser.add_argument("--pareto-alpha", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--thresholds", type=parse_float_list, default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--lrs", type=parse_float_list, default=[0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--beta1-values", type=parse_float_list, default=[0.5, 0.7, 0.9, 0.95])
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--divergence-objective", type=float, default=1e12)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.thresholds or not args.lrs or not args.beta1_values:
        raise ValueError("thresholds, lrs, and beta1-values must not be empty.")
    suite = AdaptiveQuadraticSuite(
        d=args.d,
        steps=args.steps,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        x0_norm=args.x0_norm,
        eig_min=args.eig_min,
        eig_max=args.eig_max,
        tail_indices=(args.pareto_alpha,),
        noise_scales=(args.noise_scale,),
        thresholds=tuple(args.thresholds),
        lrs=tuple(args.lrs),
        methods=SWEEP_METHODS,
        beta2=args.beta2,
        eps=args.eps,
        clipping_scope="global",
        log_every=args.log_every,
        tail_fraction=args.tail_fraction,
        divergence_objective=args.divergence_objective,
        dtype=args.dtype,
    )
    frames = run_hyperparameter_sweep(
        suite=suite,
        beta1_values=args.beta1_values,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    )
    for name, frame in frames.items():
        print(f"Wrote {args.output_dir / f'{name}.csv'} ({len(frame)} rows)", flush=True)
    print(f"Wrote plots under {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
