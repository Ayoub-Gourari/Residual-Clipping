import math

import torch

from residual_clipping.adaptive_optimizers import RESIDUAL_CLIP_ADAMW_M
from residual_clipping.adaptive_quadratics import (
    AdaptiveQuadraticRun,
    AdaptiveQuadraticSuite,
    build_runs,
    run_constant_gradient_diagnostic,
    run_noisy_quadratic,
)
from residual_clipping.quadratics import make_eigs, make_log_steps, make_seed_inputs


def test_build_runs_only_duplicates_thresholded_methods():
    runs = build_runs(("adamw_uncut", "adamw_clip"), (0.1, 1.0), (0.01, 0.02))

    assert len(runs) == 6
    assert sum(math.isinf(run.threshold) for run in runs) == 2


def test_constant_gradient_resclip_m_keeps_coupled_normalized_update_bounded():
    suite = AdaptiveQuadraticSuite(
        methods=("adamw_clip", RESIDUAL_CLIP_ADAMW_M),
        thresholds=(0.1,),
        dtype="float64",
    )
    frame = run_constant_gradient_diagnostic(suite, gradient=10.0, steps=200)
    final = frame[frame["step"] == 200].set_index("method")

    assert abs(final.loc["adamw_clip", "normalized_update"] - 1.0) < 1e-6
    assert final.loc[RESIDUAL_CLIP_ADAMW_M, "normalized_update"] < 2.0
    assert final.loc[RESIDUAL_CLIP_ADAMW_M, "sqrt_v_hat"] > 1.0


def test_noisy_quadratic_records_curves_and_summary():
    suite = AdaptiveQuadraticSuite(
        d=4,
        steps=8,
        num_seeds=1,
        tail_indices=(1.5,),
        thresholds=(1.0,),
        methods=("adamw_clip",),
        log_every=4,
    )
    dtype = torch.float64
    eigs = make_eigs(suite.d, suite.eig_min, suite.eig_max, dtype)
    x0, noise = make_seed_inputs(
        suite.steps, suite.d, suite.x0_norm, 1.5, 1.0, 1.0, 0, dtype
    )
    curves, summary = run_noisy_quadratic(
        suite=suite,
        run=AdaptiveQuadraticRun("adamw_clip", 1.0, 0.01),
        eigs=eigs,
        x0=x0,
        noise=noise,
        pareto_alpha=1.5,
        noise_scale=1.0,
        seed=0,
        log_steps=make_log_steps(suite.steps, suite.log_every),
    )

    assert [row["step"] for row in curves] == [1, 4, 8]
    assert summary["steps_run"] == 8
    assert math.isfinite(summary["tail_median_objective"])
