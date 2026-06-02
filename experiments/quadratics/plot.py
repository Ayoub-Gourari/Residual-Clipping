"""Generate paper-quality figures from quadratics outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from residual_clipping.quadratics import dtype_from_name, make_eigs, make_seed_inputs
from residual_clipping.quadratics_plots import (
    pick_best_rows,
    plot_best_final_loss_vs_threshold,
    plot_best_trajectories,
    plot_metric_curves,
    plot_threshold_heatmaps,
    rerun_best_curves,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot synthetic quadratics outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--search-output-dir", type=Path, default=Path("outputs") / "hparam_search")
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--x0-norm", type=float, default=10.0)
    parser.add_argument("--eig-min", type=float, default=1.0)
    parser.add_argument("--eig-max", type=float, default=100.0)
    parser.add_argument("--pareto-alpha", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=1.0)
    parser.add_argument("--divergence-objective", type=float, default=1e12)
    parser.add_argument("--curve-log-every", type=int, default=200)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)

    quadratic_mean = pd.read_csv(args.output_dir / "quadratic_mean_metrics.csv")
    fixed_mu_mean = pd.read_csv(args.output_dir / "fixed_mu_mean_metrics.csv")
    plot_metric_curves(
        quadratic_mean,
        metric="objective",
        ylabel="mean objective",
        title="Quadratic Objective",
        destination=args.figure_dir / "quadratic_objective",
    )
    plot_metric_curves(
        fixed_mu_mean,
        metric="estimator_error",
        ylabel="mean estimator error",
        title="Fixed-Mu Estimator Error",
        destination=args.figure_dir / "fixed_mu_estimator_error",
    )

    summary = pd.read_csv(args.search_output_dir / "search_summary.csv")
    best_rows = pick_best_rows(summary)
    dtype = dtype_from_name(args.dtype)
    eigs = make_eigs(args.d, args.eig_min, args.eig_max, dtype)

    def seed_inputs(seed: int):
        return make_seed_inputs(
            steps=args.steps,
            d=args.d,
            x0_norm=args.x0_norm,
            pareto_alpha=args.pareto_alpha,
            r_min=args.r_min,
            noise_scale=args.noise_scale,
            seed=seed,
            dtype=dtype,
        )

    _, curves, mean_curves = rerun_best_curves(
        best_rows,
        eigs,
        seed_inputs,
        steps=args.steps,
        divergence_objective=args.divergence_objective,
        curve_log_every=args.curve_log_every,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
    )
    curves.to_csv(args.search_output_dir / "best_by_method_curves_seed_metrics.csv", index=False)
    mean_curves.to_csv(args.search_output_dir / "best_by_method_curves_mean_metrics.csv", index=False)
    best_rows.to_csv(args.search_output_dir / "best_by_method_configs.csv", index=False)

    plot_best_trajectories(best_rows, mean_curves, args.figure_dir / "best_by_method_curves")
    threshold_table = plot_best_final_loss_vs_threshold(
        summary, args.figure_dir / "threshold_best_final_error_comparison"
    )
    threshold_table.to_csv(
        args.figure_dir / "threshold_best_final_error_comparison_table.csv",
        index=False,
    )
    plot_threshold_heatmaps(summary, args.figure_dir / "threshold_robustness_heatmaps")

    print(f"Wrote figures in {args.figure_dir}", flush=True)


if __name__ == "__main__":
    main()
