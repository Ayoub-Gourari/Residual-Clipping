"""Synthetic quadratics experiment helpers and runners."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_CLIP_VALUES = [0.1, 0.3, 1.0, 3.0, 10.0]
DEFAULT_RES_CLIP_VALUES = [0.03, 0.1, 0.3, 1.0, 3.0]
DEFAULT_BETAS = [0.5, 0.7, 0.9, 0.95, 0.99]
DEFAULT_LRS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]


@dataclass(frozen=True)
class SearchConfig:
    method_family: str
    clip_value: float | None
    lr: float
    beta: float

    @property
    def config_id(self) -> str:
        return config_label(self.method_family, self.clip_value, self.lr, self.beta)

    @property
    def method_label(self) -> str:
        return method_label(self.method_family, self.clip_value)


def parse_float_list(text: str | None) -> list[float]:
    if text is None or text == "":
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def method_label(method_family: str, clip_value: float | None) -> str:
    if method_family == "unclipped":
        return "unclipped"
    if clip_value is None:
        raise ValueError(f"{method_family} requires a clip value.")
    return f"{method_family}_C_{float_tag(clip_value)}"


def config_label(
    method_family: str,
    clip_value: float | None,
    lr: float,
    beta: float,
) -> str:
    return f"{method_label(method_family, clip_value)}_lr_{float_tag(lr)}_b_{float_tag(beta)}"


def make_log_steps(steps: int, log_every: int) -> set[int]:
    log_steps = set(range(log_every, steps + 1, log_every))
    log_steps.add(1)
    log_steps.add(steps)
    return log_steps


def global_clip(v: torch.Tensor, threshold: float, eps: float = 1e-12) -> tuple[torch.Tensor, bool]:
    norm = torch.linalg.vector_norm(v)
    norm_value = float(norm)
    scale = min(1.0, threshold / (norm_value + eps))
    return v * scale, norm_value > threshold


def sample_symmetric_pareto_noise(
    steps: int,
    d: int,
    pareto_alpha: float,
    r_min: float,
    noise_scale: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    u = torch.rand(steps, generator=generator, dtype=dtype).clamp_min(
        torch.finfo(dtype).tiny
    )
    radii = r_min * u.pow(-1.0 / pareto_alpha)

    z = torch.randn(steps, d, generator=generator, dtype=dtype)
    directions = z / (torch.linalg.vector_norm(z, dim=1, keepdim=True) + 1e-12)

    signs = torch.randint(0, 2, (steps,), generator=generator)
    signs = signs.to(dtype=dtype).mul(2.0).sub(1.0)

    return signs[:, None] * noise_scale * radii[:, None] * directions


def make_x0(d: int, x0_norm: float, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(d, generator=generator, dtype=dtype)
    return x0_norm * x / (torch.linalg.vector_norm(x) + 1e-12)


def make_fixed_mu(d: int, fixed_mu_norm: float, dtype: torch.dtype) -> torch.Tensor:
    mu = torch.ones(d, dtype=dtype)
    return fixed_mu_norm * mu / (torch.linalg.vector_norm(mu) + 1e-12)


def make_eigs(d: int, eig_min: float, eig_max: float, dtype: torch.dtype) -> torch.Tensor:
    return torch.logspace(math.log10(eig_min), math.log10(eig_max), d, dtype=dtype)


def objective(eigs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.dot(eigs * x, x)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        column
        for column in [
            "objective",
            "x_norm",
            "grad_norm",
            "momentum_norm",
            "clip_rate",
            "estimator_error",
        ]
        if column in df.columns
    ]
    group_cols = [
        "experiment",
        "step",
        "method_family",
        "method_label",
        "clip_value",
    ]
    grouped = df.groupby(group_cols, dropna=False)
    pieces = []
    for metric in metric_cols:
        stats = grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        pieces.append(stats)
    return pd.concat(pieces, axis=1).reset_index()


def aggregate_final(df: pd.DataFrame) -> pd.DataFrame:
    final_step = int(df["step"].max())
    return aggregate(df[df["step"] == final_step]).drop(columns=["step"])


def aggregate_search(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "config_id",
        "method_family",
        "method_label",
        "clip_value",
        "lr",
        "beta",
    ]
    metric_cols = [
        "final_objective",
        "best_objective",
        "target_hit",
        "target_step_or_max",
        "diverged",
        "steps_run",
        "x_norm",
        "grad_norm",
        "momentum_norm",
        "clip_rate",
    ]
    grouped = seed_metrics.groupby(group_cols, dropna=False)
    pieces = []
    for metric in metric_cols:
        stats = grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        pieces.append(stats)
    summary = pd.concat(pieces, axis=1).reset_index()
    summary = summary.sort_values(
        ["diverged_mean", "final_objective_mean", "target_step_or_max_mean"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary["rank_final"] = np.arange(1, len(summary) + 1)
    return summary


def aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "config_id",
        "step",
        "method_family",
        "method_label",
        "clip_value",
        "lr",
        "beta",
    ]
    grouped = curves.groupby(group_cols, dropna=False)
    pieces = []
    for metric in ["objective", "clip_rate"]:
        stats = grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        pieces.append(stats)
    return pd.concat(pieces, axis=1).reset_index()


def append_dataframe_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def maybe_load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_search_configs(
    betas: list[float],
    lrs: list[float],
    clip_values: list[float],
    res_clip_values: list[float],
) -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    for beta in betas:
        for lr in lrs:
            configs.append(SearchConfig("unclipped", None, lr, beta))
            for clip_value in clip_values:
                configs.append(SearchConfig("standard", clip_value, lr, beta))
            for clip_value in res_clip_values:
                configs.append(SearchConfig("residual", clip_value, lr, beta))
    return configs


def make_seed_inputs(
    steps: int,
    d: int,
    x0_norm: float,
    pareto_alpha: float,
    r_min: float,
    noise_scale: float,
    seed: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x0 = make_x0(d, x0_norm, generator, dtype)
    noise = sample_symmetric_pareto_noise(
        steps, d, pareto_alpha, r_min, noise_scale, generator, dtype
    )
    return x0, noise


def run_quadratic_method(
    eigs: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    method_family: str,
    clip_value: float | None,
    lr: float,
    beta: float,
    steps: int,
    log_steps: set[int],
    seed: int,
) -> list[dict[str, Any]]:
    x = x0.clone()
    m = torch.zeros_like(x)
    records = []
    active_count = 0
    label = method_label(method_family, clip_value)

    for step in range(1, steps + 1):
        grad_true = eigs * x
        g = grad_true + noise[step - 1]

        if method_family == "unclipped":
            m = beta * m + (1.0 - beta) * g
        elif method_family == "standard":
            assert clip_value is not None
            clipped_g, active = global_clip(g, clip_value)
            active_count += int(active)
            m = beta * m + (1.0 - beta) * clipped_g
        elif method_family == "residual":
            assert clip_value is not None
            residual = g - m
            clipped_residual, active = global_clip(residual, clip_value)
            active_count += int(active)
            m = m + (1.0 - beta) * clipped_residual
        else:
            raise ValueError(f"Unknown method family: {method_family}")

        x = x - lr * m

        if step in log_steps:
            current_grad = eigs * x
            records.append(
                {
                    "experiment": "quadratic",
                    "seed": seed,
                    "step": step,
                    "method_family": method_family,
                    "method_label": label,
                    "clip_value": np.nan if clip_value is None else clip_value,
                    "objective": float(objective(eigs, x)),
                    "x_norm": float(torch.linalg.vector_norm(x)),
                    "grad_norm": float(torch.linalg.vector_norm(current_grad)),
                    "momentum_norm": float(torch.linalg.vector_norm(m)),
                    "clip_rate": np.nan
                    if method_family == "unclipped"
                    else active_count / step,
                }
            )
    return records


def run_fixed_mu(
    d: int,
    steps: int,
    beta: float,
    clip_values: list[float],
    fixed_mu_norm: float,
    fixed_seed_offset: int,
    num_seeds: int,
    seed_start: int,
    pareto_alpha: float,
    r_min: float,
    noise_scale: float,
    log_steps: set[int],
    dtype: torch.dtype,
) -> pd.DataFrame:
    mu = make_fixed_mu(d, fixed_mu_norm, dtype)
    all_records: list[dict[str, Any]] = []

    for seed in range(seed_start, seed_start + num_seeds):
        generator = torch.Generator().manual_seed(seed + fixed_seed_offset)
        noise = sample_symmetric_pareto_noise(
            steps, d, pareto_alpha, r_min, noise_scale, generator, dtype
        )
        for clip_value in clip_values:
            standard_m = torch.zeros(d, dtype=dtype)
            residual_m = torch.zeros(d, dtype=dtype)
            standard_active = 0
            residual_active = 0

            for step in range(1, steps + 1):
                g = mu + noise[step - 1]
                clipped_g, active = global_clip(g, clip_value)
                standard_active += int(active)
                standard_m = beta * standard_m + (1.0 - beta) * clipped_g

                residual = g - residual_m
                clipped_residual, active = global_clip(residual, clip_value)
                residual_active += int(active)
                residual_m = residual_m + (1.0 - beta) * clipped_residual

                if step in log_steps:
                    all_records.append(
                        {
                            "experiment": "fixed_mu",
                            "seed": seed,
                            "step": step,
                            "method_family": "standard",
                            "method_label": method_label("standard", clip_value),
                            "clip_value": clip_value,
                            "estimator_error": float(torch.linalg.vector_norm(standard_m - mu)),
                            "momentum_norm": float(torch.linalg.vector_norm(standard_m)),
                            "clip_rate": standard_active / step,
                        }
                    )
                    all_records.append(
                        {
                            "experiment": "fixed_mu",
                            "seed": seed,
                            "step": step,
                            "method_family": "residual",
                            "method_label": method_label("residual", clip_value),
                            "clip_value": clip_value,
                            "estimator_error": float(torch.linalg.vector_norm(residual_m - mu)),
                            "momentum_norm": float(torch.linalg.vector_norm(residual_m)),
                            "clip_rate": residual_active / step,
                        }
                    )
    return pd.DataFrame(all_records)


def run_search_config(
    eigs: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    config: SearchConfig,
    steps: int,
    divergence_objective: float,
    target_objective: float,
    seed: int,
) -> dict[str, Any]:
    x = x0.clone()
    m = torch.zeros_like(x)
    active_count = 0
    diverged = False
    initial_objective = float(objective(eigs, x))
    best_objective = initial_objective
    final_objective = initial_objective
    step_to_target = 0 if initial_objective <= target_objective else None
    steps_run = 0

    for step in range(1, steps + 1):
        grad_true = eigs * x
        g = grad_true + noise[step - 1]

        if config.method_family == "unclipped":
            m = config.beta * m + (1.0 - config.beta) * g
        elif config.method_family == "standard":
            assert config.clip_value is not None
            clipped_g, active = global_clip(g, config.clip_value)
            active_count += int(active)
            m = config.beta * m + (1.0 - config.beta) * clipped_g
        elif config.method_family == "residual":
            assert config.clip_value is not None
            residual = g - m
            clipped_residual, active = global_clip(residual, config.clip_value)
            active_count += int(active)
            m = m + (1.0 - config.beta) * clipped_residual
        else:
            raise ValueError(f"Unknown method family: {config.method_family}")

        x = x - config.lr * m
        steps_run = step
        obj = float(objective(eigs, x))

        if not math.isfinite(obj) or obj > divergence_objective:
            final_objective = divergence_objective
            diverged = True
            break

        final_objective = obj
        if obj < best_objective:
            best_objective = obj
        if step_to_target is None and obj <= target_objective:
            step_to_target = step

    return {
        "seed": seed,
        "config_id": config.config_id,
        "method_family": config.method_family,
        "method_label": config.method_label,
        "clip_value": np.nan if config.clip_value is None else config.clip_value,
        "lr": config.lr,
        "beta": config.beta,
        "final_objective": final_objective,
        "best_objective": best_objective,
        "target_hit": float(step_to_target is not None),
        "step_to_target": np.nan if step_to_target is None else step_to_target,
        "target_step_or_max": step_to_target if step_to_target is not None else steps + 1,
        "diverged": float(diverged),
        "steps_run": steps_run,
        "x_norm": float(torch.linalg.vector_norm(x)),
        "grad_norm": float(torch.linalg.vector_norm(eigs * x)),
        "momentum_norm": float(torch.linalg.vector_norm(m)),
        "clip_rate": np.nan if config.method_family == "unclipped" or steps_run == 0 else active_count / steps_run,
    }


def run_search_curve(
    eigs: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    config: SearchConfig,
    steps: int,
    divergence_objective: float,
    log_steps: set[int],
    seed: int,
) -> list[dict[str, Any]]:
    x = x0.clone()
    m = torch.zeros_like(x)
    active_count = 0
    records = []

    for step in range(1, steps + 1):
        grad_true = eigs * x
        g = grad_true + noise[step - 1]

        if config.method_family == "unclipped":
            m = config.beta * m + (1.0 - config.beta) * g
        elif config.method_family == "standard":
            assert config.clip_value is not None
            clipped_g, active = global_clip(g, config.clip_value)
            active_count += int(active)
            m = config.beta * m + (1.0 - config.beta) * clipped_g
        elif config.method_family == "residual":
            assert config.clip_value is not None
            residual = g - m
            clipped_residual, active = global_clip(residual, config.clip_value)
            active_count += int(active)
            m = m + (1.0 - config.beta) * clipped_residual
        else:
            raise ValueError(f"Unknown method family: {config.method_family}")

        x = x - config.lr * m
        obj = float(objective(eigs, x))
        if not math.isfinite(obj) or obj > divergence_objective:
            obj = divergence_objective

        if step in log_steps:
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "config_id": config.config_id,
                    "method_family": config.method_family,
                    "method_label": config.method_label,
                    "clip_value": np.nan if config.clip_value is None else config.clip_value,
                    "lr": config.lr,
                    "beta": config.beta,
                    "objective": obj,
                    "clip_rate": np.nan if config.method_family == "unclipped" else active_count / step,
                }
            )

        if obj >= divergence_objective:
            break
    return records


def methodwise_configs(configs: list[SearchConfig]) -> dict[str, list[SearchConfig]]:
    grouped = {"unclipped": [], "standard": [], "residual": []}
    for config in configs:
        grouped[config.method_family].append(config)
    return grouped


def completed_search_pairs(existing: pd.DataFrame) -> set[tuple[int, str]]:
    if existing.empty:
        return set()
    return {
        (int(row["seed"]), str(row["config_id"]))
        for _, row in existing[["seed", "config_id"]].iterrows()
    }


def maybe_init_wandb(
    mode: str,
    project: str | None,
    entity: str | None,
    run_name: str | None,
    run_id: str | None,
    config: dict[str, Any],
    resume: bool,
):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping W&B logging.")
        return None

    resume_mode = "allow" if resume else None
    return wandb.init(
        mode=mode,
        project=project,
        entity=entity,
        name=run_name,
        id=run_id,
        resume=resume_mode,
        config=config,
    )


def finalize_wandb(run: Any) -> None:
    if run is not None:
        run.finish()


def log_dataframe_table(run: Any, key: str, dataframe: pd.DataFrame) -> None:
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return
    run.log({key: wandb.Table(dataframe=dataframe)})


def print_method_progress(
    method_family: str,
    completed: int,
    total: int,
    current_seed_index: int,
    total_seeds: int,
    elapsed_seconds: float,
    current_final_loss: float,
    best_final_loss: float,
) -> None:
    print(
        f"[{method_family}] config {completed}/{total} | "
        f"seed {current_seed_index}/{total_seeds} | "
        f"elapsed {elapsed_seconds:.1f}s | "
        f"final {current_final_loss:.3e} | best {best_final_loss:.3e}",
        flush=True,
    )
