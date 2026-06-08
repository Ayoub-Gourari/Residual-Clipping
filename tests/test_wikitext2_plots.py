from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import residual_clipping.wikitext2_plots as plots


def trajectory_summary() -> pd.DataFrame:
    rows = []
    values = {
        "sgd_momentum": [33000.0, 300.0, 98.0],
        "clipped_momentum": [33000.0, 280.0, 96.0],
        "residual_clipped_momentum": [33000.0, 260.0, 94.0],
    }
    for method, perplexities in values.items():
        for step, perplexity in zip([0, 1000, 2000], perplexities):
            rows.append(
                {
                    "optimizer_mode": method,
                    "train/global_step": step,
                    "validation_perplexity_mean": perplexity,
                    "validation_perplexity_std": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_best_trajectory_plot_omits_untrained_point_by_default(monkeypatch, tmp_path: Path):
    captured = {}

    def capture_figure(fig, path):
        captured["figure"] = fig

    monkeypatch.setattr(plots, "save_figure", capture_figure)
    plots.plot_best_trajectories(trajectory_summary(), tmp_path / "trajectories")

    figure = captured["figure"]
    axis = figure.axes[0]
    method_lines = axis.lines
    assert len(method_lines) == 3
    assert all(0 not in line.get_xdata() for line in method_lines)
    assert axis.get_ylim()[1] < 1000
    assert len({line.get_linestyle() for line in method_lines}) == 3
    plt.close(figure)


def test_best_trajectory_plot_can_include_untrained_point(monkeypatch, tmp_path: Path):
    captured = {}

    def capture_figure(fig, path):
        captured["figure"] = fig

    monkeypatch.setattr(plots, "save_figure", capture_figure)
    plots.plot_best_trajectories(
        trajectory_summary(),
        tmp_path / "trajectories",
        include_initial_evaluation=True,
    )

    figure = captured["figure"]
    assert all(0 in line.get_xdata() for line in figure.axes[0].lines)
    assert figure.axes[0].get_ylim()[1] > 30000
    plt.close(figure)


def test_best_trajectory_plot_supports_logarithmic_y_axis(monkeypatch, tmp_path: Path):
    captured = {}

    def capture_figure(fig, path):
        captured["figure"] = fig

    monkeypatch.setattr(plots, "save_figure", capture_figure)
    plots.plot_best_trajectories(
        trajectory_summary(),
        tmp_path / "trajectories",
        logarithmic_y=True,
    )

    figure = captured["figure"]
    assert figure.axes[0].get_yscale() == "log"
    assert all((line.get_ydata() > 0).all() for line in figure.axes[0].lines)
    plt.close(figure)
