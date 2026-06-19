#!/usr/bin/env python3
"""Tiny smoke test for all AdamW clipping variants."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from residual_clipping.adaptive_optimizers import ADAPTIVE_OPTIMIZER_NAMES, AdaptiveAdamW


REQUIRED_KEYS = {
    "optimizer_name",
    "clip_threshold",
    "clipping_scope",
    "grad_global_norm",
    "pseudo_grad_global_norm",
    "clipping_scale",
    "residual_global_norm",
    "metric_residual_global_norm",
    "v_pseudo_grad_global_norm",
    "v_clipping_scale",
    "update_global_norm",
    "adam_m_global_norm",
    "adam_v_global_norm",
    "learning_rate",
}


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values.")


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def make_param_groups(model: nn.Sequential) -> list[dict[str, object]]:
    first = list(model[0].parameters())
    second = list(model[2].parameters())
    return [
        {"params": first, "weight_decay": 0.01},
        {"params": second, "weight_decay": 0.0},
    ]


def run_case(optimizer_name: str, clip_threshold: float) -> None:
    torch.manual_seed(123)
    model = make_model()
    optimizer = AdaptiveAdamW(
        make_param_groups(model),
        optimizer_name=optimizer_name,
        beta1=0.9,
        beta2=0.999,
        eps=1e-6,
        weight_decay=0.0,
        clip_threshold=clip_threshold,
        clipping_scope="local",
        correct_bias=False,
    )
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
    params = [param for group in optimizer.param_groups for param in group["params"]]

    diagnostics = None
    for step in range(3):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not math.isfinite(float(loss.item())):
            raise RuntimeError(f"Non-finite loss for {optimizer_name} at step {step}.")
        loss.backward()
        grads = [param.grad.detach().clone() if param.grad is not None else None for param in params]
        diagnostics = optimizer.step(grads, lr=1e-4)

        for index, param in enumerate(params):
            assert_finite_tensor(f"{optimizer_name} parameter {index}", param.detach())
        for index, value in enumerate(optimizer.exp_avg):
            assert_finite_tensor(f"{optimizer_name} exp_avg {index}", value)
        for index, value in enumerate(optimizer.exp_avg_sq):
            assert_finite_tensor(f"{optimizer_name} exp_avg_sq {index}", value)

    if diagnostics is None:
        raise RuntimeError(f"No diagnostics produced for {optimizer_name}.")
    missing = sorted(REQUIRED_KEYS - set(diagnostics))
    if missing:
        raise RuntimeError(f"Missing diagnostics for {optimizer_name}: {', '.join(missing)}")
    for key, value in diagnostics.items():
        if isinstance(value, (int, float)) and math.isnan(float(value)):
            raise RuntimeError(f"NaN diagnostic for {optimizer_name}: {key}={value}")
    print(f"{optimizer_name} C={clip_threshold}: ok", flush=True)


def main() -> None:
    for optimizer_name in ADAPTIVE_OPTIMIZER_NAMES:
        threshold = float("inf") if optimizer_name == "adamw_uncut" else 0.5
        run_case(optimizer_name, threshold)
    run_case("adamw_clip", float("inf"))


if __name__ == "__main__":
    main()
