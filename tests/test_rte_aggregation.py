import math

import pandas as pd

from scripts.aggregate_rte_reproduction import (
    best_loss_by_lr_summary,
    run_accuracy_frame,
    run_loss_frame,
    select_best_loss_curves,
    summarize,
    threshold_loss_summary,
    threshold_sensitivity_summary,
)


def test_threshold_sensitivity_and_seed_statistics():
    records = []
    accuracies = {
        ("adamw_clip", 0.5): [70.0, 72.0],
        ("adamw_clip", 1.0): [75.0, 77.0],
        ("ResidualClipAdamW-M", 0.5): [76.0, 78.0],
        ("ResidualClipAdamW-M", 1.0): [77.0, 79.0],
    }
    for (optimizer_name, threshold), seed_accuracies in accuracies.items():
        for seed, accuracy in enumerate(seed_accuracies):
            records.append(
                {
                    "optimizer_name": optimizer_name,
                    "lr": 1e-5,
                    "clip_threshold": threshold,
                    "clipping_scope": "global",
                    "seed": seed,
                    "run_name": f"{optimizer_name}-{threshold}-{seed}",
                    "val_accuracy": accuracy - 1.0,
                    "best_accuracy": accuracy,
                }
            )

    run_frame = run_accuracy_frame(pd.DataFrame(records))
    summary = threshold_sensitivity_summary(run_frame)

    clipped_small = summary[
        (summary["optimizer_name"] == "adamw_clip") & (summary["clip_threshold"] == 0.5)
    ].iloc[0]
    residual_small = summary[
        (summary["optimizer_name"] == "ResidualClipAdamW-M") & (summary["clip_threshold"] == 0.5)
    ].iloc[0]
    assert clipped_small["seed/accuracy_mean"] == 71.0
    assert clipped_small["threshold/gap_to_best"] == 5.0
    assert clipped_small["threshold/spread"] == 5.0
    assert clipped_small["threshold/auc_gap"] == 2.5
    assert residual_small["threshold/gap_to_best"] == 1.0
    assert residual_small["threshold/auc_gap"] == 0.5


def test_threshold_loss_summary_and_best_trajectory_selection():
    records = []
    losses = {
        ("adamw_uncut", math.inf): [0.70, 0.72],
        ("adamw_clip", 1.0): [0.60, 0.62],
        ("adamw_clip", 2.0): [0.55, 0.57],
        ("ResidualClipAdamW-M", 1.0): [0.58, 0.59],
        ("ResidualClipAdamW-M", 2.0): [0.54, 0.56],
    }
    for (optimizer_name, threshold), seed_losses in losses.items():
        for seed, best_loss in enumerate(seed_losses):
            for step, offset in ((0, 0.2), (12, 0.1), (24, 0.0)):
                records.append(
                    {
                        "optimizer_name": optimizer_name,
                        "lr": 1e-5,
                        "clip_threshold": threshold,
                        "clipping_scope": "local",
                        "seed": seed,
                        "run_name": f"{optimizer_name}-{threshold}-{seed}",
                        "global_step": step,
                        "val_loss": best_loss + offset,
                        "best_loss": best_loss,
                    }
                )

    frame = pd.DataFrame(records)
    run_losses = run_loss_frame(frame)
    threshold_losses = threshold_loss_summary(run_losses)
    loss_curves = summarize(frame, "val_loss")
    best_curves = select_best_loss_curves(loss_curves, threshold_losses)

    clipped_best = threshold_losses[
        (threshold_losses["optimizer_name"] == "adamw_clip")
        & (threshold_losses["clip_threshold"] == 2.0)
    ].iloc[0]
    assert clipped_best["threshold/loss_mean"] == 0.56
    assert clipped_best["threshold/gap_to_best_loss"] == 0.0
    assert set(best_curves["optimizer_name"]) == {
        "adamw_uncut",
        "adamw_clip",
        "ResidualClipAdamW-M",
    }
    assert set(best_curves[best_curves["optimizer_name"] == "adamw_clip"]["clip_threshold"]) == {2.0}


def test_best_loss_by_lr_and_joint_trajectory_selection():
    records = []
    losses = {
        (1e-5, 8.0): [0.60, 0.62],
        (1e-5, 16.0): [0.58, 0.60],
        (2e-5, 8.0): [0.55, 0.57],
        (2e-5, 16.0): [0.52, 0.54],
    }
    for (lr, threshold), seed_losses in losses.items():
        for seed, best_loss in enumerate(seed_losses):
            for step, offset in ((0, 0.1), (12, 0.0)):
                records.append(
                    {
                        "optimizer_name": "ResidualClipAdamW-MPost",
                        "lr": lr,
                        "clip_threshold": threshold,
                        "clipping_scope": "local",
                        "seed": seed,
                        "run_name": f"run-{lr}-{threshold}-{seed}",
                        "global_step": step,
                        "val_loss": best_loss + offset,
                        "best_loss": best_loss,
                    }
                )

    frame = pd.DataFrame(records)
    threshold_summary = threshold_loss_summary(run_loss_frame(frame))
    by_lr = best_loss_by_lr_summary(threshold_summary)
    curves = select_best_loss_curves(summarize(frame, "val_loss"), threshold_summary)

    assert list(by_lr["clip_threshold"]) == [16.0, 16.0]
    assert list(by_lr["threshold/loss_mean"]) == [0.59, 0.53]
    assert set(curves["lr"]) == {2e-5}
    assert set(curves["clip_threshold"]) == {16.0}
