"""Run a resumable hyperparameter search for synthetic quadratics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from residual_clipping.cli import add_resume_args, add_wandb_args
from residual_clipping.quadratics import (
    DEFAULT_BETAS,
    DEFAULT_CLIP_VALUES,
    DEFAULT_LRS,
    DEFAULT_RES_CLIP_VALUES,
    aggregate_curves,
    aggregate_search,
    append_dataframe_row,
    build_search_configs,
    completed_search_pairs,
    dtype_from_name,
    finalize_wandb,
    log_dataframe_table,
    make_eigs,
    make_log_steps,
    make_seed_inputs,
    maybe_init_wandb,
    maybe_load_csv,
    parse_float_list,
    print_method_progress,
    run_search_config,
    run_search_curve,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact hyperparameter search for clipped momentum variants."
    )
    add_wandb_args(parser)
    add_resume_args(parser)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--x0-norm", type=float, default=10.0)
    parser.add_argument("--eig-min", type=float, default=1.0)
    parser.add_argument("--eig-max", type=float, default=100.0)
    parser.add_argument("--pareto-alpha", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=1.0)
    parser.add_argument(
        "--betas",
        type=parse_float_list,
        default=DEFAULT_BETAS,
        help="Comma-separated momentum beta values.",
    )
    parser.add_argument(
        "--lrs",
        type=parse_float_list,
        default=DEFAULT_LRS,
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--clip-values",
        type=parse_float_list,
        default=DEFAULT_CLIP_VALUES,
        help="Comma-separated standard clipping thresholds.",
    )
    parser.add_argument(
        "--res-clip-values",
        type=parse_float_list,
        default=DEFAULT_RES_CLIP_VALUES + [10.0],
        help="Comma-separated residual clipping thresholds.",
    )
    parser.add_argument("--target-objective", type=float, default=1e-2)
    parser.add_argument("--divergence-objective", type=float, default=1e12)
    parser.add_argument("--top-k-curves", type=int, default=8)
    parser.add_argument("--curve-log-every", type=int, default=200)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser


def rerun_top_curves(args, summary: pd.DataFrame):
    if args.top_k_curves <= 0:
        return pd.DataFrame(), pd.DataFrame()

    dtype = dtype_from_name(args.dtype)
    eigs = make_eigs(args.d, args.eig_min, args.eig_max, dtype)
    log_steps = make_log_steps(args.steps, args.curve_log_every)
    top = summary.head(args.top_k_curves)
    records = []

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
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
        for _, row in top.iterrows():
            clip_value = None if pd.isna(row["clip_value"]) else float(row["clip_value"])
            records.extend(
                run_search_curve(
                    eigs=eigs,
                    x0=x0,
                    noise=noise,
                    config=type("Cfg", (), {
                        "method_family": str(row["method_family"]),
                        "clip_value": clip_value,
                        "lr": float(row["lr"]),
                        "beta": float(row["beta"]),
                        "config_id": str(row["config_id"]),
                        "method_label": str(row["method_label"]),
                    })(),
                    steps=args.steps,
                    divergence_objective=args.divergence_objective,
                    log_steps=log_steps,
                    seed=seed,
                )
            )

    curves = pd.DataFrame(records)
    if curves.empty:
        return curves, curves
    return curves, aggregate_curves(curves)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "hparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "search_seed_metrics.csv"
    summary_path = output_dir / "search_summary.csv"
    top_path = output_dir / "top_curves_seed_metrics.csv"
    top_mean_path = output_dir / "top_curves_mean_metrics.csv"

    if args.overwrite:
        for path in [seed_path, summary_path, top_path, top_mean_path]:
            if path.exists():
                path.unlink()

    existing_seed_metrics = maybe_load_csv(seed_path) if args.resume else pd.DataFrame()
    completed_pairs = completed_search_pairs(existing_seed_metrics)

    dtype = dtype_from_name(args.dtype)
    eigs = make_eigs(args.d, args.eig_min, args.eig_max, dtype)
    configs = build_search_configs(args.betas, args.lrs, args.clip_values, args.res_clip_values)
    grouped_configs = {
        method_family: [config for config in configs if config.method_family == method_family]
        for method_family in ("unclipped", "standard", "residual")
    }

    run = maybe_init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        run_id=args.run_id,
        config={**vars(args), "output_dir": str(output_dir)},
        resume=args.resume,
    )

    started = time.time()
    all_seed_metrics = []
    if not existing_seed_metrics.empty:
        all_seed_metrics.extend(existing_seed_metrics.to_dict(orient="records"))

    for seed_offset, seed in enumerate(range(args.seed_start, args.seed_start + args.num_seeds), start=1):
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
        for method_family, method_configs in grouped_configs.items():
            best_final = float("inf")
            completed = 0
            total = len(method_configs)
            report_every = max(1, total // 10)
            for config in method_configs:
                key = (seed, config.config_id)
                if key in completed_pairs:
                    completed += 1
                    existing_row = existing_seed_metrics[
                        (existing_seed_metrics["seed"] == seed)
                        & (existing_seed_metrics["config_id"] == config.config_id)
                    ]
                    if not existing_row.empty:
                        best_final = min(best_final, float(existing_row.iloc[0]["final_objective"]))
                    continue

                result = run_search_config(
                    eigs=eigs,
                    x0=x0,
                    noise=noise,
                    config=config,
                    steps=args.steps,
                    divergence_objective=args.divergence_objective,
                    target_objective=args.target_objective,
                    seed=seed,
                )
                append_dataframe_row(seed_path, result)
                all_seed_metrics.append(result)
                completed_pairs.add(key)
                completed += 1
                best_final = min(best_final, float(result["final_objective"]))
                if completed == 1 or completed == total or completed % report_every == 0:
                    print_method_progress(
                        method_family=method_family,
                        completed=completed,
                        total=total,
                        current_seed_index=seed_offset,
                        total_seeds=args.num_seeds,
                        elapsed_seconds=time.time() - started,
                        current_final_loss=float(result["final_objective"]),
                        best_final_loss=best_final,
                    )

    seed_metrics = pd.DataFrame(all_seed_metrics)
    summary = aggregate_search(seed_metrics)
    top_curves, top_curves_mean = rerun_top_curves(args, summary)

    summary.to_csv(summary_path, index=False)
    top_curves.to_csv(top_path, index=False)
    top_curves_mean.to_csv(top_mean_path, index=False)

    log_dataframe_table(run, "search_summary", summary)
    log_dataframe_table(run, "top_curves_mean", top_curves_mean)
    finalize_wandb(run)

    print(f"Wrote {seed_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {top_path}", flush=True)
    print(f"Wrote {top_mean_path}", flush=True)


if __name__ == "__main__":
    main()
