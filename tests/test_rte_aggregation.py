import pandas as pd

from scripts.aggregate_rte_reproduction import run_accuracy_frame, threshold_sensitivity_summary


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
