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
CLIPPING_SCOPES = ("global", "local", "elementwise")
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
    clipping_scope: str,
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
    if clipping_scope not in CLIPPING_SCOPES:
        raise ValueError(
            f"Unsupported clipping_scope={clipping_scope!r}. "
            f"Known scopes: {', '.join(CLIPPING_SCOPES)}"
        )


def _safe_norm(tensors: Sequence[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    return float(tensor_list_global_norm(list(tensors)).item())


def _scale_from_norm(norm: float, threshold: float) -> float:
    if not math.isfinite(threshold):
        return 1.0
    if norm <= 0.0:
        return 1.0
    return min(1.0, float(threshold) / (norm + _EPS_NORM))


def _norm_ratio(after: float, before: float) -> float:
    if before <= 0.0:
        return 1.0
    return min(1.0, float(after) / (float(before) + _EPS_NORM))


def _elementwise_clip(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    if not math.isfinite(threshold):
        return tensor.detach().clone()
    return tensor.clamp(min=-float(threshold), max=float(threshold))


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
            clipping_scope=str(group["clipping_scope"]),
        )
        group["correct_bias"] = bool(group["correct_bias"])
    return groups


class AdaptiveAdamW:
    """AdamW variants with clipping scopes and scalar diagnostics.

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
        clipping_scope: str = "local",
        correct_bias: bool = False,
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
            "clipping_scope": clipping_scope,
            "correct_bias": correct_bias,
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

    def _group_indices(self) -> Iterable[tuple[dict[str, Any], list[int]]]:
        index = 0
        for group in self.param_groups:
            count = len(group["params"])
            yield group, list(range(index, index + count))
            index += count

    def _flat_group_items(
        self,
        values: Sequence[torch.Tensor],
    ) -> Iterable[tuple[dict[str, Any], torch.nn.Parameter, torch.Tensor, torch.Tensor, torch.Tensor]]:
        index = 0
        for group in self.param_groups:
            for param in group["params"]:
                yield group, param, values[index], self.exp_avg[index], self.exp_avg_sq[index]
                index += 1

    def _clip_values(self, values: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], float, float]:
        clipped: list[torch.Tensor | None] = [None] * len(values)
        before_norm = _safe_norm(values)

        for group, indices in self._group_indices():
            threshold = float(group["clip_threshold"])
            scope = str(group["clipping_scope"])
            group_values = [values[index] for index in indices]

            if scope == "global":
                group_norm = _safe_norm(group_values)
                scale = _scale_from_norm(group_norm, threshold)
                for index in indices:
                    clipped[index] = values[index] * scale
            elif scope == "local":
                for index in indices:
                    scale = _scale_from_norm(float(torch.linalg.vector_norm(values[index]).item()), threshold)
                    clipped[index] = values[index] * scale
            elif scope == "elementwise":
                for index in indices:
                    clipped[index] = _elementwise_clip(values[index], threshold)
            else:
                raise ValueError(f"Unsupported clipping_scope={scope!r}.")

        materialized = [value for value in clipped if value is not None]
        after_norm = _safe_norm(materialized)
        return materialized, after_norm, _norm_ratio(after_norm, before_norm)

    def _clip_metric_residuals(
        self,
        residuals: Sequence[torch.Tensor],
        metric_residuals: Sequence[torch.Tensor],
        denominators: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], float, float]:
        clipped_residuals: list[torch.Tensor | None] = [None] * len(residuals)
        metric_norm = _safe_norm(metric_residuals)
        clipped_metric_values: list[torch.Tensor] = []

        for group, indices in self._group_indices():
            threshold = float(group["clip_threshold"])
            scope = str(group["clipping_scope"])
            group_metric_values = [metric_residuals[index] for index in indices]

            if scope == "global":
                group_metric_norm = _safe_norm(group_metric_values)
                scale = _scale_from_norm(group_metric_norm, threshold)
                for index in indices:
                    clipped_residuals[index] = residuals[index] * scale
                    clipped_metric_values.append(metric_residuals[index] * scale)
            elif scope == "local":
                for index in indices:
                    scale = _scale_from_norm(
                        float(torch.linalg.vector_norm(metric_residuals[index]).item()),
                        threshold,
                    )
                    clipped_residuals[index] = residuals[index] * scale
                    clipped_metric_values.append(metric_residuals[index] * scale)
            elif scope == "elementwise":
                for index in indices:
                    clipped_metric = _elementwise_clip(metric_residuals[index], threshold)
                    clipped_residuals[index] = clipped_metric * denominators[index]
                    clipped_metric_values.append(clipped_metric)
            else:
                raise ValueError(f"Unsupported clipping_scope={scope!r}.")

        materialized = [value for value in clipped_residuals if value is not None]
        clipped_metric_norm = _safe_norm(clipped_metric_values)
        return materialized, metric_norm, _norm_ratio(clipped_metric_norm, metric_norm)

    def _pseudo_gradients(
        self,
        grads: list[torch.Tensor],
        *,
        next_step: int,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        grad_norm = _safe_norm(grads)

        if self.optimizer_name == "adamw_uncut":
            return list(grads), {
                "grad_global_norm": grad_norm,
                "pseudo_grad_global_norm": grad_norm,
                "clipping_scale": 1.0,
                "residual_global_norm": 0.0,
                "metric_residual_global_norm": 0.0,
            }

        if self.optimizer_name == "adamw_clip":
            pseudo, _pseudo_norm, scale = self._clip_values(grads)
            return pseudo, {
                "grad_global_norm": grad_norm,
                "pseudo_grad_global_norm": _safe_norm(pseudo),
                "clipping_scale": scale,
                "residual_global_norm": 0.0,
                "metric_residual_global_norm": 0.0,
            }

        centers = [value.detach().clone() for value in self.exp_avg]
        residuals = [grad - center for grad, center in zip(grads, centers)]
        residual_norm = _safe_norm(residuals)

        if self.optimizer_name == "adamw_resclip_euclidean" or next_step == 1:
            clipped_residuals, _clipped_norm, scale = self._clip_values(residuals)
            pseudo = [center + residual for center, residual in zip(centers, clipped_residuals)]
            return pseudo, {
                "grad_global_norm": grad_norm,
                "pseudo_grad_global_norm": _safe_norm(pseudo),
                "clipping_scale": scale,
                "residual_global_norm": residual_norm,
                "metric_residual_global_norm": 0.0,
            }

        if self.optimizer_name != "adamw_resclip_metric":
            raise ValueError(f"Unsupported optimizer_name={self.optimizer_name!r}.")

        metric_residuals: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        for group, indices in self._group_indices():
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            bias_correction2_prev = max(1.0 - beta2 ** self.step_count, _EPS_NORM)
            for index in indices:
                v_hat_prev = self.exp_avg_sq[index].detach() / bias_correction2_prev
                denominator = v_hat_prev.sqrt().add(eps)
                denominators.append(denominator)
                metric_residuals.append(residuals[index] / denominator)

        clipped_residuals, metric_norm, scale = self._clip_metric_residuals(
            residuals,
            metric_residuals,
            denominators,
        )
        pseudo = [center + residual for center, residual in zip(centers, clipped_residuals)]
        return pseudo, {
            "grad_global_norm": grad_norm,
            "pseudo_grad_global_norm": _safe_norm(pseudo),
            "clipping_scale": scale,
            "residual_global_norm": residual_norm,
            "metric_residual_global_norm": metric_norm,
        }

    def step(
        self,
        grads: Sequence[torch.Tensor | None],
        lr: float,
        *,
        collect_diagnostics: bool = True,
    ) -> dict[str, float | str]:
        if len(grads) != len(self.params):
            raise ValueError(f"Expected {len(self.params)} gradients, got {len(grads)}.")
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}.")

        materialized_grads = [
            torch.zeros_like(param) if grad is None else grad.detach()
            for param, grad in zip(self.params, grads)
        ]
        next_step = self.step_count + 1
        pseudo_grads, diagnostics = self._pseudo_gradients(materialized_grads, next_step=next_step)
        updates: list[torch.Tensor] = []

        with torch.no_grad():
            for group, param, pseudo_grad, exp_avg, exp_avg_sq in self._flat_group_items(pseudo_grads):
                beta1 = float(group["beta1"])
                beta2 = float(group["beta2"])
                eps = float(group["eps"])
                group_lr = lr if group.get("lr") is None else float(group["lr"])
                weight_decay = float(group["weight_decay"])
                correct_bias = bool(group["correct_bias"])

                exp_avg.mul_(beta1).add_(pseudo_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(pseudo_grad, pseudo_grad, value=1.0 - beta2)

                if correct_bias:
                    bias_correction1 = max(1.0 - beta1 ** next_step, _EPS_NORM)
                    bias_correction2 = max(1.0 - beta2 ** next_step, _EPS_NORM)
                    numerator = exp_avg / bias_correction1
                    denominator = (exp_avg_sq / bias_correction2).sqrt().add(eps)
                else:
                    numerator = exp_avg
                    denominator = exp_avg_sq.sqrt().add(eps)

                update = numerator / denominator
                updates.append(update.detach().clone())

                if weight_decay != 0.0:
                    param.mul_(1.0 - group_lr * weight_decay)
                param.add_(update, alpha=-group_lr)

        self.step_count = next_step

        if not collect_diagnostics:
            return {}

        update_norm = _safe_norm(updates)
        threshold = max(float(group["clip_threshold"]) for group in self.param_groups)
        scopes = sorted({str(group["clipping_scope"]) for group in self.param_groups})
        correct_bias_values = {bool(group["correct_bias"]) for group in self.param_groups}
        diagnostics.update(
            {
                "optimizer_name": self.optimizer_name,
                "clip_threshold": threshold,
                "clipping_scope": scopes[0] if len(scopes) == 1 else "mixed",
                "correct_bias": float(next(iter(correct_bias_values))) if len(correct_bias_values) == 1 else -1.0,
                "update_global_norm": update_norm,
                "adam_m_global_norm": _safe_norm(self.exp_avg),
                "adam_v_global_norm": _safe_norm(self.exp_avg_sq),
                "learning_rate": lr,
                "adam_step_norm": lr * update_norm,
                "adam_update_norm": update_norm,
                "grad_norm": diagnostics["grad_global_norm"],
            }
        )
        return diagnostics
