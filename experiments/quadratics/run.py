"""Run the base synthetic quadratics experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from residual_clipping.cli import add_resume_args, add_wandb_args
from residual_clipping.quadratics import (
    DEFAULT_CLIP_VALUES,
    DEFAULT_RES_CLIP_VALUES,
    aggregate,
    aggregate_final,
    dtype_from_name,
    log_dataframe_table,
    make_eigs,
    make_fixed_mu,
    make_log_steps,
    make_seed_inputs,
    maybe_init_wandb,
    parse_float_list,
    run_fixed_mu,
    run_quadratic_method,
    finalize_wandb,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic quadratics experiments.")
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--x0-norm", type=float, default=10.0)
    parser.add_argument("--eig-min", type=float, default=1.0)
    parser.add_argument("--eig-max", type=float, default=100.0)
    parser.add_argument("--pareto-alpha", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=1.0)
    parser.add_argument(
        "--clip-values",
        type=parse_float_list,
        default=DEFAULT_CLIP_VALUES,
        help="Comma-separated thresholds for standard clipped momentum.",
    )
    parser.add_argument(
        "--res-clip-values",
        type=parse_float_list,
        default=DEFAULT_RES_CLIP_VALUES,
        help="Comma-separated thresholds for residual-clipped momentum.",
    )
    parser.add_argument(
        "--fixed-clip-values",
        type=parse_float_list,
        default=None,
        help="Comma-separated fixed-mu thresholds. Defaults to --clip-values.",
    )
    parser.add_argument("--fixed-mu-norm", type=float, default=1.0)
    parser.add_argument("--fixed-seed-offset", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser


def maybe_skip_existing(output_dir: Path, resume: bool, overwrite: bool) -> bool:
    summary_path = output_dir / "final_summary.csv"
    if overwrite:
        return False
    if resume and summary_path.exists():
        print(f"Found existing summary at {summary_path}; skipping rerun because --resume was set.")
        return True
    return False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.fixed_clip_values is None:
        args.fixed_clip_values = list(args.clip_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if maybe_skip_existing(output_dir, args.resume, args.overwrite):
        return

    dtype = dtype_from_name(args.dtype)
    eigs = make_eigs(args.d, args.eig_min, args.eig_max, dtype)
    log_steps = make_log_steps(args.steps, args.log_every)
    all_records = []

    print("Running quadratic experiment suite...", flush=True)
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        print(
            f"[quadratic] seed {seed - args.seed_start + 1}/{args.num_seeds} "
            f"| lr={args.lr:g} | beta={args.beta:g}",
            flush=True,
        )
        x0, noise = make_seed_inputs(
            steps=args.steps,
            d=args.d,
            x0_norm=args.x0_norm,
            pareto_alpha=args.pareto_alpha,
            r_min=args.r_min,
            noise_scale=args.noise_scale,
            seed=seed,
            dtype=dtype,
        )
        all_records.extend(
            run_quadratic_method(
                eigs=eigs,
                x0=x0,
                noise=noise,
                method_family="unclipped",
                clip_value=None,
                lr=args.lr,
                beta=args.beta,
                steps=args.steps,
                log_steps=log_steps,
                seed=seed,
            )
        )
        for clip_value in args.clip_values:
            all_records.extend(
                run_quadratic_method(
                    eigs=eigs,
                    x0=x0,
                    noise=noise,
                    method_family="standard",
                    clip_value=float(clip_value),
                    lr=args.lr,
                    beta=args.beta,
                    steps=args.steps,
                    log_steps=log_steps,
                    seed=seed,
                )
            )
        for clip_value in args.res_clip_values:
            all_records.extend(
                run_quadratic_method(
                    eigs=eigs,
                    x0=x0,
                    noise=noise,
                    method_family="residual",
                    clip_value=float(clip_value),
                    lr=args.lr,
                    beta=args.beta,
                    steps=args.steps,
                    log_steps=log_steps,
                    seed=seed,
                )
            )

    quadratic = pd.DataFrame(all_records)
    quadratic_mean = aggregate(quadratic)
    quadratic_final = aggregate_final(quadratic)

    print("Running fixed-mu diagnostic...", flush=True)
    fixed_mu = run_fixed_mu(
        d=args.d,
        steps=args.steps,
        beta=args.beta,
        clip_values=list(args.fixed_clip_values),
        fixed_mu_norm=args.fixed_mu_norm,
        fixed_seed_offset=args.fixed_seed_offset,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        pareto_alpha=args.pareto_alpha,
        r_min=args.r_min,
        noise_scale=args.noise_scale,
        log_steps=log_steps,
        dtype=dtype,
    )
    fixed_mean = aggregate(fixed_mu)
    fixed_final = aggregate_final(fixed_mu)

    final_summary = pd.concat(
        [
            quadratic_final.assign(source="quadratic_final"),
            fixed_final.assign(source="fixed_mu_final"),
        ],
        ignore_index=True,
        sort=False,
    )

    quadratic_path = output_dir / "quadratic_seed_metrics.csv"
    quadratic_mean_path = output_dir / "quadratic_mean_metrics.csv"
    fixed_path = output_dir / "fixed_mu_seed_metrics.csv"
    fixed_mean_path = output_dir / "fixed_mu_mean_metrics.csv"
    summary_path = output_dir / "final_summary.csv"

    quadratic.to_csv(quadratic_path, index=False)
    quadratic_mean.to_csv(quadratic_mean_path, index=False)
    fixed_mu.to_csv(fixed_path, index=False)
    fixed_mean.to_csv(fixed_mean_path, index=False)
    final_summary.to_csv(summary_path, index=False)

    run = maybe_init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        run_id=args.run_id,
        config={**vars(args), "output_dir": str(output_dir)},
        resume=args.resume,
    )
    log_dataframe_table(run, "quadratic_mean_table", quadratic_mean)
    log_dataframe_table(run, "fixed_mu_mean_table", fixed_mean)
    log_dataframe_table(run, "final_summary_table", final_summary)
    finalize_wandb(run)

    print(f"Wrote {quadratic_path}", flush=True)
    print(f"Wrote {quadratic_mean_path}", flush=True)
    print(f"Wrote {fixed_path}", flush=True)
    print(f"Wrote {fixed_mean_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
