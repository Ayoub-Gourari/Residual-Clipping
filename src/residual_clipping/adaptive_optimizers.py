"""Adaptive AdamW optimizers for transformer fine-tuning experiments."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from .clipping import tensor_list_global_norm


ADAPTIVE_OPTIMIZER_NAMES = (
    "adamw_uncut",
    "adamw_clip",
    "adamw_resclip_euclidean",
    "adamw_resclip_metric",
)
ADAPTIVE_OPTIMIZER_MODES = ADAPTIVE_OPTIMIZER_NAMES
_EPS_NORM = 1e-12


def collect_trainable_gradients(
    params: Sequence[torch.nn.Parameter],
    *,
    clone: bool = False,
) -> list[torch.Tensor]:
    """Collect gradients for trainable parameters, filling missing gradients with zeros."""
    grads: list[torch.Tensor] = []
    for param in params:
        if param.grad is None:
            grad = torch.zeros_like(param)
        else:
            grad = param.grad.detach()
            if clone:
                grad = grad.clone()
        grads.append(grad)
    return grads


def _validate_hyperparameters(
    *,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    clip_threshold: float,
) -> None:
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"beta1 must be in [0,1), got {beta1}.")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"beta2 must be in [0,1), got {beta2}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")
    if clip_threshold < 0.0:
        raise ValueError(f"clip_threshold must be non-negative, got {clip_threshold}.")


def _clip_scale(norm: float, threshold: float) -> float:
    if not math.isfinite(threshold):
        return 1.0
    if norm <= 0.0 or not math.isfinite(norm):
        return 1.0
    return min(1.0, float(threshold) / (norm + _EPS_NORM))


def _normalize_param_groups(
    params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    materialized = list(params)
    if not materialized:
        raise ValueError("AdaptiveAdamW received an empty parameter list.")

    if isinstance(materialized[0], dict):
        groups = []
        for raw_group in materialized:
            if not isinstance(raw_group, dict):
                raise TypeError("Parameter groups must either all be dicts or all be parameters.")
            group = {**defaults, **raw_group}
            group["params"] = list(group["params"])
            if not group["params"]:
                raise ValueError("Parameter groups must not be empty.")
            groups.append(group)
    else:
        groups = [{**defaults, "params": materialized}]

    for group in groups:
        _validate_hyperparameters(
            beta1=float(group["beta1"]),
            beta2=float(group["beta2"]),
            eps=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
            clip_threshold=float(group["clip_threshold"]),
        )
    return groups


class AdaptiveAdamW:
    """AdamW-style optimizer variants with global clipping diagnostics.

    The class accepts either a flat parameter iterable or PyTorch-style
    parameter groups. Clipping is applied only to gradients or residuals; weight
    decay remains decoupled exactly as in AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        *,
        optimizer_name: str | None = None,
        mode: str | None = None,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        clip_threshold: float = math.inf,
    ) -> None:
        name = optimizer_name or mode
        if name not in ADAPTIVE_OPTIMIZER_NAMES:
            raise ValueError(
                f"Unsupported optimizer_name={name!r}. "
                f"Known optimizers: {', '.join(ADAPTIVE_OPTIMIZER_NAMES)}"
            )
        defaults = {
            "lr": None,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
            "clip_threshold": clip_threshold,
        }
        self.param_groups = _normalize_param_groups(params, defaults)
        self.params = [param for group in self.param_groups for param in group["params"]]
        self.optimizer_name = str(name)
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(param) for param in self.params]
        self.exp_avg_sq = [torch.zeros_like(param) for param in self.params]

    @property
    def mode(self) -> str:
        return self.optimizer_name

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "mode": self.optimizer_name,
            "step_count": self.step_count,
            "param_groups": [
                {key: value for key, value in group.items() if key != "params"} for group in self.param_groups
            ],
            "exp_avg": [value.detach().cpu() for value in self.exp_avg],
            "exp_avg_sq": [value.detach().cpu() for value in self.exp_avg_sq],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step_count = int(state_dict["step_count"])
        self.exp_avg = [tensor.to(param.device) for tensor, param in zip(state_dict["exp_avg"], self.params)]
        self.exp_avg_sq = [
            tensor.to(param.device) for tensor, param in zip(state_dict["exp_avg_sq"], self.params)
        ]
        group_options = state_dict.get("param_groups")
        if group_options is not None and len(group_options) == len(self.param_groups):
            for group, options in zip(self.param_groups, group_options):
                for key, value in options.items():
                    group[key] = value

    def _flat_group_items(
        self,
        grads: Sequence[torch.Tensor],
    ) -> Iterable[tuple[dict[str, Any], torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]]:
        index = 0
        for group in self.param_groups:
            for param in group["params"]:
                yield group, param, grads[index], self.exp_avg[index], self.exp_avg_sq[index]
                index += 1

    def _pseudo_gradients(
        self,
        grads: list[torch.Tensor],
        *,
        next_step: int,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        grad_norm = tensor_list_global_norm(grads).item()
        threshold = max(float(group["clip_threshold"]) for group in self.param_groups)

        if self.optimizer_name == "adamw_uncut":
            return list(grads), {
                "grad_global_norm": grad_norm,
                "pseudo_grad_global_norm": grad_norm,
                "clipping_scale": 1.0,
            }

        if self.optimizer_name == "adamw_clip":
            scale = _clip_scale(grad_norm, threshold)
            pseudo = [grad * scale for grad in grads]
            return pseudo, {
                "grad_global_norm": grad_norm,
                "pseudo_grad_global_norm": tensor_list_global_norm(pseudo).item(),
                "clipping_scale": scale,
            }

        centers = [value.detach().clone() for value in self.exp_avg]
        residuals = [grad - center for grad, center in zip(grads, centers)]
        residual_norm = tensor_list_global_norm(residuals).item()

        if self.optimizer_name == "adamw_resclip_euclidean" or next_step == 1:
            scale = _clip_scale(residual_norm, threshold)
            pseudo = [center + residual * scale for center, residual in zip(centers, residuals)]
            diagnostics = {
                "grad_global_norm": grad_norm,
                "pseudo_grad_global_norm": tensor_list_global_norm(pseudo).item(),
                "clipping_scale": scale,
                "residual_global_norm": residual_norm,
            }
            if self.optimizer_name == "adamw_resclip_metric":
                diagnostics["metric_residual_global_norm"] = residual_norm
            return pseudo, diagnostics

        if self.optimizer_name != "adamw_resclip_metric":
            raise ValueError(f"Unsupported optimizer_name={self.optimizer_name!r}.")

        metric_residuals: list[torch.Tensor] = []
        for group, _param, _grad, _exp_avg, exp_avg_sq in self._flat_group_items(grads):
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            bias_correction2_prev = max(1.0 - beta2**self.step_count, _EPS_NORM)
            v_hat_prev = exp_avg_sq.detach() / bias_correction2_prev
            residual = residuals[len(metric_residuals)]
            metric_residuals.append(residual / (v_hat_prev.sqrt().add(eps)))

        metric_norm = tensor_list_global_norm(metric_residuals).item()
        scale = _clip_scale(metric_norm, threshold)
        pseudo = [center + residual * scale for center, residual in zip(centers, residuals)]
        return pseudo, {
            "grad_global_norm": grad_norm,
            "pseudo_grad_global_norm": tensor_list_global_norm(pseudo).item(),
            "clipping_scale": scale,
            "residual_global_norm": residual_norm,
            "metric_residual_global_norm": metric_norm,
        }

    def step(
        self,
        grads: Sequence[torch.Tensor],
        lr: float,
        *,
        collect_diagnostics: bool = True,
    ) -> dict[str, float | str]:
        if len(grads) != len(self.params):
            raise ValueError(f"Expected {len(self.params)} gradients, got {len(grads)}.")
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}.")

        grads = [grad.detach() for grad in grads]
        next_step = self.step_count + 1
        pseudo_grads, diagnostics = self._pseudo_gradients(grads, next_step=next_step)
        updates: list[torch.Tensor] = []

        with torch.no_grad():
            for group, param, pseudo_grad, exp_avg, exp_avg_sq in self._flat_group_items(pseudo_grads):
                beta1 = float(group["beta1"])
                beta2 = float(group["beta2"])
                eps = float(group["eps"])
                group_lr = lr if group.get("lr") is None else float(group["lr"])
                weight_decay = float(group["weight_decay"])

                exp_avg.mul_(beta1).add_(pseudo_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(pseudo_grad, pseudo_grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**next_step
                bias_correction2 = 1.0 - beta2**next_step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                update = m_hat / (v_hat.sqrt().add(eps))
                updates.append(update)

                if weight_decay != 0.0:
                    param.mul_(1.0 - group_lr * weight_decay)
                param.add_(update, alpha=-group_lr)

        self.step_count = next_step

        if not collect_diagnostics:
            return {}

        update_norm = tensor_list_global_norm(updates).item()
        threshold = max(float(group["clip_threshold"]) for group in self.param_groups)
        diagnostics.update(
            {
                "optimizer_name": self.optimizer_name,
                "clip_threshold": threshold,
                "update_global_norm": update_norm,
                "adam_m_global_norm": tensor_list_global_norm(self.exp_avg).item(),
                "adam_v_global_norm": tensor_list_global_norm(self.exp_avg_sq).item(),
                "learning_rate": lr,
                "adam_step_norm": lr * update_norm,
                "adam_update_norm": update_norm,
                "grad_norm": diagnostics["grad_global_norm"],
            }
        )
        return diagnostics
