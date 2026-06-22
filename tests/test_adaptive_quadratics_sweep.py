import math

import pandas as pd

from residual_clipping.adaptive_optimizers import RESIDUAL_CLIP_ADAMW_M
from residual_clipping.adaptive_quadratics_sweep import (
    select_best_by_threshold,
    select_global_best,
)


def _row(method, threshold, lr, beta1, loss, diverged=0.0):
    return {
        "pareto_alpha": 1.5,
        "noise_scale": 1.0,
        "method": method,
        "threshold": threshold,
        "lr": lr,
        "beta1": beta1,
        "tail_median_objective_mean": loss,
        "tail_median_objective_median": loss,
        "tail_median_objective_q25": 0.8 * loss,
        "tail_median_objective_q75": 1.2 * loss,
        "log10_objective_auc_median": loss,
        "diverged_mean": diverged,
    }


def test_select_best_by_threshold_optimizes_lr_and_momentum():
    aggregate = pd.DataFrame(
        [
            _row("adamw_uncut", math.inf, 0.01, 0.9, 2.0),
            _row("adamw_uncut", math.inf, 0.003, 0.7, 1.0),
            _row("adamw_clip", 0.1, 0.01, 0.9, 3.0),
            _row("adamw_clip", 0.1, 0.003, 0.7, 0.5),
            _row("adamw_clip", 1.0, 0.01, 0.9, 0.7),
            _row(RESIDUAL_CLIP_ADAMW_M, 0.1, 0.01, 0.9, 0.6),
            _row(RESIDUAL_CLIP_ADAMW_M, 1.0, 0.003, 0.7, 0.4),
        ]
    )

    selected = select_best_by_threshold(aggregate, (0.1, 1.0))

    uncut = selected[selected["method"] == "adamw_uncut"]
    assert len(uncut) == 2
    assert set(uncut["lr"]) == {0.003}
    clipped = selected[
        (selected["method"] == "adamw_clip") & (selected["plot_threshold"] == 0.1)
    ].iloc[0]
    assert clipped["lr"] == 0.003
    assert clipped["beta1"] == 0.7


def test_select_global_best_prefers_nondiverged_configuration():
    aggregate = pd.DataFrame(
        [
            _row("adamw_uncut", math.inf, 0.01, 0.9, 1.0),
            _row("adamw_clip", 0.1, 0.01, 0.9, 0.01, diverged=1.0),
            _row("adamw_clip", 1.0, 0.003, 0.7, 0.5),
            _row(RESIDUAL_CLIP_ADAMW_M, 1.0, 0.003, 0.7, 0.4),
        ]
    )

    best = select_global_best(aggregate).set_index("method")

    assert best.loc["adamw_clip", "threshold"] == 1.0
    assert best.loc[RESIDUAL_CLIP_ADAMW_M, "tail_median_objective_median"] == 0.4
