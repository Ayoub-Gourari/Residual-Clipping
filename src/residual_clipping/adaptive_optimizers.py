"""Adaptive AdamW optimizers for transformer fine-tuning experiments."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from .clipping import tensor_list_global_norm


RESIDUAL_CLIP_ADAMW_M = "ResidualClipAdamW-M"


ADAPTIVE_OPTIMIZER_NAMES = (
    "adamw_uncut",
    "adamw_clip",
    "adamw_resclip_euclidean",
    "adamw_resclip_euclidean_vclip",
    "adamw_resclip_euclidean_vclip_varalpha",
    "adamw_resclip_metric",
    "adamw_resclip_metric_vclip",
    RESIDUAL_CLIP_ADAMW_M,
)
ADAPTIVE_OPTIMIZER_MODES = ADAPTIVE_OPTIMIZER_NAMES
CLIPPING_SCOPES = ("global", "local", "layerwise", "elementwise")
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


def _mean_tensor_value(tensor: torch.Tensor) -> float:
    return float(tensor.detach().mean().item())


def _mean_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    left_norm = _safe_norm(left)
    right_norm = _safe_norm(right)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    dot = sum(float((lhs.detach() * rhs.detach()).sum().item()) for lhs, rhs in zip(left, right))
    return max(-1.0, min(1.0, dot / (left_norm * right_norm + _EPS_NORM)))


def _relative_norm(value: float, reference: float) -> float:
    return float(value) / (float(reference) + _EPS_NORM)


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
        self._enforce_algorithm_options()
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(param) for param in self.params]
        self.exp_avg_sq = [torch.zeros_like(param) for param in self.params]
        self.exp_avg_bias_mass = [torch.zeros_like(param) for param in self.params]
        self.grad_clipping_activation_count = 0
        self.residual_clipping_activation_count = 0

    @property
    def mode(self) -> str:
        return self.optimizer_name

    def _enforce_algorithm_options(self) -> None:
        if self.optimizer_name == RESIDUAL_CLIP_ADAMW_M:
            # Bias corrections are part of this named algorithm, not an optional Adam setting.
            for group in self.param_groups:
                group["correct_bias"] = True

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
            "exp_avg_bias_mass": [value.detach().cpu() for value in self.exp_avg_bias_mass],
            "grad_clipping_activation_count": self.grad_clipping_activation_count,
            "residual_clipping_activation_count": self.residual_clipping_activation_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step_count = int(state_dict["step_count"])
        self.exp_avg = [tensor.to(param.device) for tensor, param in zip(state_dict["exp_avg"], self.params)]
        self.exp_avg_sq = [
            tensor.to(param.device) for tensor, param in zip(state_dict["exp_avg_sq"], self.params)
        ]
        bias_mass_values = state_dict.get("exp_avg_bias_mass")
        if bias_mass_values is not None:
            self.exp_avg_bias_mass = [
                tensor.to(param.device) for tensor, param in zip(bias_mass_values, self.params)
            ]
        self.grad_clipping_activation_count = int(state_dict.get("grad_clipping_activation_count", 0))
        self.residual_clipping_activation_count = int(state_dict.get("residual_clipping_activation_count", 0))
        group_options = state_dict.get("param_groups")
        if group_options is not None and len(group_options) == len(self.param_groups):
            for group, options in zip(self.param_groups, group_options):
                for key, value in options.items():
                    group[key] = value
        self._enforce_algorithm_options()

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

    def _first_moment_centers(self, *, next_step: int) -> list[torch.Tensor]:
        if self.optimizer_name != RESIDUAL_CLIP_ADAMW_M or next_step == 1:
            return [value.detach().clone() for value in self.exp_avg]

        centers: list[torch.Tensor] = []
        for group, indices in self._group_indices():
            beta1 = float(group["beta1"])
            bias_correction1_prev = max(1.0 - beta1 ** (next_step - 1), _EPS_NORM)
            centers.extend(self.exp_avg[index].detach() / bias_correction1_prev for index in indices)
        return centers

    def _record_clipping_activations(
        self,
        *,
        next_step: int,
        grad_scale: float,
        residual_scale: float,
    ) -> tuple[float, float, float, float]:
        tolerance = 1e-7
        grad_activated = float(grad_scale < 1.0 - tolerance)
        residual_activated = float(residual_scale < 1.0 - tolerance)
        self.grad_clipping_activation_count += int(grad_activated)
        self.residual_clipping_activation_count += int(residual_activated)
        return (
            grad_activated,
            self.grad_clipping_activation_count / float(next_step),
            residual_activated,
            self.residual_clipping_activation_count / float(next_step),
        )

    def _canonical_diagnostics(
        self,
        *,
        grads: Sequence[torch.Tensor],
        centers: Sequence[torch.Tensor],
        residuals: Sequence[torch.Tensor],
        clipped_grads: Sequence[torch.Tensor],
        pseudo_grads: Sequence[torch.Tensor],
        m_hats: Sequence[torch.Tensor],
        v_hats: Sequence[torch.Tensor],
        sqrt_v_hats: Sequence[torch.Tensor],
        m_hat_updates: Sequence[torch.Tensor],
        updates: Sequence[torch.Tensor],
        lr_scaled_updates: Sequence[torch.Tensor],
        weight_decay_steps: Sequence[torch.Tensor],
        total_steps: Sequence[torch.Tensor],
        param_norm_before: float,
        grad_scale: float,
        residual_scale: float,
        grad_activated: float,
        grad_activation_rate: float,
        residual_activated: float,
        residual_activation_rate: float,
        lr: float,
    ) -> dict[str, float | str]:
        grad_norm = _safe_norm(grads)
        center_norm = _safe_norm(centers)
        residual_norm = _safe_norm(residuals)
        clipped_grad_norm = _safe_norm(clipped_grads)
        pseudo_grad_norm = _safe_norm(pseudo_grads)
        update_norm = _safe_norm(updates)
        beta1_values = [float(group["beta1"]) for group in self.param_groups]
        beta2_values = [float(group["beta2"]) for group in self.param_groups]
        weight_decay_values = [float(group["weight_decay"]) for group in self.param_groups]
        threshold = max(float(group["clip_threshold"]) for group in self.param_groups)
        return {
            "grad/norm": grad_norm,
            "center/norm": center_norm,
            "residual/norm": residual_norm,
            "residual/relative_norm": _relative_norm(residual_norm, grad_norm),
            "clipped_grad/norm": clipped_grad_norm,
            "pseudo_grad/norm": pseudo_grad_norm,
            "clipped_grad/relative_norm": _relative_norm(clipped_grad_norm, grad_norm),
            "pseudo_grad/relative_norm": _relative_norm(pseudo_grad_norm, grad_norm),
            "clipping/grad_activated": grad_activated,
            "clipping/grad_activation_rate": grad_activation_rate,
            "clipping/residual_activated": residual_activated,
            "clipping/residual_activation_rate": residual_activation_rate,
            "clipping/grad_scale": grad_scale,
            "clipping/residual_scale": residual_scale,
            "cosine/grad_center": _safe_cosine(grads, centers),
            "cosine/grad_pseudo_grad": _safe_cosine(grads, pseudo_grads),
            "cosine/center_pseudo_grad": _safe_cosine(centers, pseudo_grads),
            "cosine/clipped_grad_pseudo_grad": _safe_cosine(clipped_grads, pseudo_grads),
            "adam/m_norm": _safe_norm(self.exp_avg),
            "adam/m_hat_norm": _safe_norm(m_hats),
            "adam/v_norm": _safe_norm(self.exp_avg_sq),
            "adam/v_hat_norm": _safe_norm(v_hats),
            "adam/sqrt_v_hat_norm": _safe_norm(sqrt_v_hats),
            "adam/m_hat_over_sqrt_v_hat_norm": _safe_norm(m_hat_updates),
            "update/norm": update_norm,
            "update/lr_scaled_norm": _safe_norm(lr_scaled_updates),
            "update/relative_param_norm": _relative_norm(_safe_norm(lr_scaled_updates), param_norm_before),
            "update/weight_decay_norm": _safe_norm(weight_decay_steps),
            "update/total_step_norm": _safe_norm(total_steps),
            "optimizer/name": self.optimizer_name,
            "optimizer/lr": lr,
            "optimizer/beta1": _mean_float(beta1_values),
            "optimizer/beta2": _mean_float(beta2_values),
            "optimizer/weight_decay": _mean_float(weight_decay_values),
            "optimizer/clip_threshold": threshold,
        }

    def _clip_values_with_scales(
        self,
        values: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], float, float, list[torch.Tensor]]:
        clipped: list[torch.Tensor | None] = [None] * len(values)
        scales: list[torch.Tensor | None] = [None] * len(values)
        before_norm = _safe_norm(values)

        for group, indices in self._group_indices():
            threshold = float(group["clip_threshold"])
            scope = str(group["clipping_scope"])
            group_values = [values[index] for index in indices]

            if scope in {"global", "layerwise"}:
                group_norm = _safe_norm(group_values)
                scale = _scale_from_norm(group_norm, threshold)
                for index in indices:
                    clipped[index] = values[index] * scale
                    scales[index] = torch.full_like(values[index], scale)
            elif scope == "local":
                for index in indices:
                    scale = _scale_from_norm(float(torch.linalg.vector_norm(values[index]).item()), threshold)
                    clipped[index] = values[index] * scale
                    scales[index] = torch.full_like(values[index], scale)
            elif scope == "elementwise":
                for index in indices:
                    if math.isfinite(threshold):
                        abs_value = values[index].abs()
                        scale_tensor = torch.where(
                            abs_value > float(threshold),
                            torch.full_like(values[index], float(threshold)) / (abs_value + _EPS_NORM),
                            torch.ones_like(values[index]),
                        )
                    else:
                        scale_tensor = torch.ones_like(values[index])
                    clipped[index] = values[index] * scale_tensor
                    scales[index] = scale_tensor
            else:
                raise ValueError(f"Unsupported clipping_scope={scope!r}.")

        materialized = [value for value in clipped if value is not None]
        materialized_scales = [value for value in scales if value is not None]
        after_norm = _safe_norm(materialized)
        return materialized, after_norm, _norm_ratio(after_norm, before_norm), materialized_scales

    def _clip_values(self, values: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], float, float]:
        clipped, after_norm, scale, _scales = self._clip_values_with_scales(values)
        return clipped, after_norm, scale

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

            if scope in {"global", "layerwise"}:
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

        centers = self._first_moment_centers(next_step=next_step)
        residuals = [grad - center for grad, center in zip(grads, centers)]
        residual_norm = _safe_norm(residuals)

        if self.optimizer_name in {
            "adamw_resclip_euclidean",
            "adamw_resclip_euclidean_vclip",
            "adamw_resclip_euclidean_vclip_varalpha",
            RESIDUAL_CLIP_ADAMW_M,
        } or next_step == 1:
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

    def _step_metric_vclip(
        self,
        materialized_grads: list[torch.Tensor],
        lr: float,
        *,
        next_step: int,
        collect_diagnostics: bool,
    ) -> dict[str, float | str]:
        grad_norm = _safe_norm(materialized_grads)
        centers = [value.detach().clone() for value in self.exp_avg]
        residuals = [grad - center for grad, center in zip(materialized_grads, centers)]
        residual_norm = _safe_norm(residuals)
        second_moment_grads, _second_moment_norm, v_scale = self._clip_values(materialized_grads)

        denominators: list[torch.Tensor | None] = [None] * len(self.params)
        with torch.no_grad():
            for index, (group, _param, second_moment_grad, _exp_avg, exp_avg_sq) in enumerate(
                self._flat_group_items(second_moment_grads)
            ):
                beta2 = float(group["beta2"])
                eps = float(group["eps"])
                correct_bias = bool(group["correct_bias"])

                exp_avg_sq.mul_(beta2).addcmul_(second_moment_grad, second_moment_grad, value=1.0 - beta2)
                if correct_bias:
                    bias_correction2 = max(1.0 - beta2 ** next_step, _EPS_NORM)
                    denominator = (exp_avg_sq / bias_correction2).sqrt().add(eps)
                else:
                    denominator = exp_avg_sq.sqrt().add(eps)
                denominators[index] = denominator

        if any(value is None for value in denominators):
            raise RuntimeError("Failed to materialize Adam denominators for metric residual clipping.")
        materialized_denominators = [value for value in denominators if value is not None]
        metric_residuals = [
            residual / denominator for residual, denominator in zip(residuals, materialized_denominators)
        ]
        clipped_residuals, metric_norm, scale = self._clip_metric_residuals(
            residuals,
            metric_residuals,
            materialized_denominators,
        )
        pseudo_grads = [center + residual for center, residual in zip(centers, clipped_residuals)]

        updates: list[torch.Tensor] = []
        effective_beta_values: list[float] = []
        bias_mass_values: list[float] = []

        with torch.no_grad():
            for index, (group, param, clipped_residual, exp_avg, _exp_avg_sq) in enumerate(
                self._flat_group_items(clipped_residuals)
            ):
                beta1 = float(group["beta1"])
                group_lr = lr if group.get("lr") is None else float(group["lr"])
                weight_decay = float(group["weight_decay"])
                correct_bias = bool(group["correct_bias"])

                exp_avg.add_(clipped_residual, alpha=1.0 - beta1)
                bias_mass = self.exp_avg_bias_mass[index]
                bias_mass.mul_(beta1).add_(1.0 - beta1)
                if correct_bias:
                    bias_correction1 = max(1.0 - beta1 ** next_step, _EPS_NORM)
                    numerator = exp_avg / bias_correction1
                else:
                    numerator = exp_avg
                effective_beta_values.append(beta1)
                bias_mass_values.append(_mean_tensor_value(bias_mass))

                update = numerator / materialized_denominators[index]
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
        return {
            "grad_global_norm": grad_norm,
            "pseudo_grad_global_norm": _safe_norm(pseudo_grads),
            "clipping_scale": scale,
            "residual_global_norm": residual_norm,
            "metric_residual_global_norm": metric_norm,
            "v_pseudo_grad_global_norm": _safe_norm(second_moment_grads),
            "v_clipping_scale": v_scale,
            "optimizer_name": self.optimizer_name,
            "clip_threshold": threshold,
            "clipping_scope": scopes[0] if len(scopes) == 1 else "mixed",
            "correct_bias": float(next(iter(correct_bias_values))) if len(correct_bias_values) == 1 else -1.0,
            "effective_beta_mean": _mean_float(effective_beta_values),
            "first_moment_bias_mass_mean": _mean_float(bias_mass_values),
            "update_global_norm": update_norm,
            "adam_m_global_norm": _safe_norm(self.exp_avg),
            "adam_v_global_norm": _safe_norm(self.exp_avg_sq),
            "learning_rate": lr,
            "adam_step_norm": lr * update_norm,
            "adam_update_norm": update_norm,
            "grad_norm": grad_norm,
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
        if self.optimizer_name == "adamw_resclip_metric_vclip":
            return self._step_metric_vclip(
                materialized_grads,
                lr,
                next_step=next_step,
                collect_diagnostics=collect_diagnostics,
            )

        centers = self._first_moment_centers(next_step=next_step)
        residuals = [grad - center for grad, center in zip(materialized_grads, centers)]
        param_norm_before = _safe_norm(self.params) if collect_diagnostics else 0.0
        pseudo_grads, diagnostics = self._pseudo_gradients(materialized_grads, next_step=next_step)
        variable_alpha_residual = self.optimizer_name == "adamw_resclip_euclidean_vclip_varalpha"
        residual_scales: list[torch.Tensor] | None = None
        if variable_alpha_residual:
            clipped_residuals, _clipped_norm, scale, residual_scales = self._clip_values_with_scales(residuals)
            pseudo_grads = [center + residual for center, residual in zip(centers, clipped_residuals)]
            diagnostics.update(
                {
                    "pseudo_grad_global_norm": _safe_norm(pseudo_grads),
                    "clipping_scale": scale,
                    "residual_global_norm": _safe_norm(residuals),
                    "metric_residual_global_norm": 0.0,
                }
            )

        if self.optimizer_name in {
            "adamw_resclip_euclidean_vclip",
            "adamw_resclip_euclidean_vclip_varalpha",
        }:
            second_moment_grads, _second_moment_norm, v_scale = self._clip_values(materialized_grads)
        else:
            second_moment_grads = pseudo_grads
            v_scale = diagnostics["clipping_scale"]
        diagnostics["v_pseudo_grad_global_norm"] = _safe_norm(second_moment_grads)
        diagnostics["v_clipping_scale"] = v_scale
        raw_gradient_clipped_modes = {
            "adamw_clip",
            "adamw_resclip_euclidean_vclip",
            "adamw_resclip_euclidean_vclip_varalpha",
        }
        if self.optimizer_name in raw_gradient_clipped_modes:
            clipped_grads = second_moment_grads
            grad_scale = float(v_scale)
        else:
            clipped_grads, _clipped_grad_norm, grad_scale = self._clip_values(materialized_grads)
        residual_scale = (
            float(diagnostics["clipping_scale"])
            if self.optimizer_name not in {"adamw_uncut", "adamw_clip"}
            else 1.0
        )
        (
            grad_activated,
            grad_activation_rate,
            residual_activated,
            residual_activation_rate,
        ) = self._record_clipping_activations(
            next_step=next_step,
            grad_scale=grad_scale,
            residual_scale=residual_scale,
        )
        updates: list[torch.Tensor] = []
        m_hats: list[torch.Tensor] = []
        v_hats: list[torch.Tensor] = []
        sqrt_v_hats: list[torch.Tensor] = []
        m_hat_updates: list[torch.Tensor] = []
        lr_scaled_updates: list[torch.Tensor] = []
        weight_decay_steps: list[torch.Tensor] = []
        total_steps: list[torch.Tensor] = []
        effective_beta_values: list[float] = []
        bias_mass_values: list[float] = []

        with torch.no_grad():
            for index, (group, param, pseudo_grad, exp_avg, exp_avg_sq) in enumerate(
                self._flat_group_items(pseudo_grads)
            ):
                beta1 = float(group["beta1"])
                beta2 = float(group["beta2"])
                eps = float(group["eps"])
                group_lr = lr if group.get("lr") is None else float(group["lr"])
                weight_decay = float(group["weight_decay"])
                correct_bias = bool(group["correct_bias"])

                bias_mass = self.exp_avg_bias_mass[index]
                if variable_alpha_residual:
                    assert residual_scales is not None
                    alpha = residual_scales[index] * (1.0 - beta1)
                    exp_avg.add_((materialized_grads[index] - exp_avg) * alpha)
                    bias_mass.add_((1.0 - bias_mass) * alpha)
                    numerator = exp_avg / bias_mass.clamp_min(_EPS_NORM) if correct_bias else exp_avg
                    if collect_diagnostics:
                        effective_beta_values.append(_mean_tensor_value(1.0 - alpha))
                else:
                    exp_avg.mul_(beta1).add_(pseudo_grad, alpha=1.0 - beta1)
                    bias_mass.mul_(beta1).add_(1.0 - beta1)
                    bias_correction1 = max(1.0 - beta1 ** next_step, _EPS_NORM)
                    if correct_bias:
                        numerator = exp_avg / bias_correction1
                    else:
                        numerator = exp_avg
                    if collect_diagnostics:
                        effective_beta_values.append(beta1)
                if collect_diagnostics:
                    bias_mass_values.append(_mean_tensor_value(bias_mass))

                bias_correction1 = max(1.0 - beta1 ** next_step, _EPS_NORM)
                m_hat = exp_avg / bias_correction1
                if collect_diagnostics:
                    m_hats.append(m_hat.detach().clone())

                second_moment_grad = second_moment_grads[index]
                exp_avg_sq.mul_(beta2).addcmul_(second_moment_grad, second_moment_grad, value=1.0 - beta2)

                bias_correction2 = max(1.0 - beta2 ** next_step, _EPS_NORM)
                v_hat = exp_avg_sq / bias_correction2
                sqrt_v_hat = v_hat.sqrt()
                if collect_diagnostics:
                    v_hats.append(v_hat.detach().clone())
                    sqrt_v_hats.append(sqrt_v_hat.detach().clone())
                    m_hat_updates.append((m_hat / sqrt_v_hat.add(eps)).detach().clone())
                if correct_bias:
                    denominator = sqrt_v_hat.add(eps)
                else:
                    denominator = exp_avg_sq.sqrt().add(eps)

                update = numerator / denominator
                if collect_diagnostics:
                    updates.append(update.detach().clone())
                    lr_scaled_update = update * group_lr
                    weight_decay_step = param.detach() * (group_lr * weight_decay)
                    lr_scaled_updates.append(lr_scaled_update.detach().clone())
                    weight_decay_steps.append(weight_decay_step.detach().clone())
                    total_steps.append((lr_scaled_update + weight_decay_step).detach().clone())

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
                "effective_beta_mean": _mean_float(effective_beta_values),
                "first_moment_bias_mass_mean": _mean_float(bias_mass_values),
                "update_global_norm": update_norm,
                "adam_m_global_norm": _safe_norm(self.exp_avg),
                "adam_v_global_norm": _safe_norm(self.exp_avg_sq),
                "learning_rate": lr,
                "adam_step_norm": lr * update_norm,
                "adam_update_norm": update_norm,
                "grad_norm": diagnostics["grad_global_norm"],
            }
        )
        diagnostics.update(
            self._canonical_diagnostics(
                grads=materialized_grads,
                centers=centers,
                residuals=residuals,
                clipped_grads=clipped_grads,
                pseudo_grads=pseudo_grads,
                m_hats=m_hats,
                v_hats=v_hats,
                sqrt_v_hats=sqrt_v_hats,
                m_hat_updates=m_hat_updates,
                updates=updates,
                lr_scaled_updates=lr_scaled_updates,
                weight_decay_steps=weight_decay_steps,
                total_steps=total_steps,
                param_norm_before=param_norm_before,
                grad_scale=grad_scale,
                residual_scale=residual_scale,
                grad_activated=grad_activated,
                grad_activation_rate=grad_activation_rate,
                residual_activated=residual_activated,
                residual_activation_rate=residual_activation_rate,
                lr=lr,
            )
        )
        return diagnostics
