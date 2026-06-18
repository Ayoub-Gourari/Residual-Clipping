"""Adaptive optimizer helpers for language-model fine-tuning experiments."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .clipping import tensor_list_global_norm


ADAPTIVE_OPTIMIZER_MODES = ("adamw",)


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


class AdaptiveAdamW:
    """AdamW optimizer with diagnostics and a single extension point for clipped variants.

    The initial LLM fine-tuning scaffold only implements the unclipped AdamW baseline.
    Clipped adaptive rules should be added by extending ``_transform_update`` rather
    than changing the experiment runner.
    """

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        mode: str,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        if mode not in ADAPTIVE_OPTIMIZER_MODES:
            raise ValueError(
                f"Unsupported adaptive optimizer mode={mode!r}. "
                f"Known modes: {', '.join(ADAPTIVE_OPTIMIZER_MODES)}"
            )
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0,1), got {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0,1), got {beta2}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}.")

        self.params = list(params)
        self.mode = mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.exp_avg = [torch.zeros_like(param) for param in self.params]
        self.exp_avg_sq = [torch.zeros_like(param) for param in self.params]

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
            "exp_avg": [value.detach().cpu() for value in self.exp_avg],
            "exp_avg_sq": [value.detach().cpu() for value in self.exp_avg_sq],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step_count = int(state_dict["step_count"])
        self.exp_avg = [tensor.to(param.device) for tensor, param in zip(state_dict["exp_avg"], self.params)]
        self.exp_avg_sq = [
            tensor.to(param.device) for tensor, param in zip(state_dict["exp_avg_sq"], self.params)
        ]

    def _transform_update(
        self,
        updates: list[torch.Tensor],
        *,
        collect_diagnostics: bool,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        if self.mode == "adamw":
            return updates, {}
        raise ValueError(f"Unsupported adaptive optimizer mode={self.mode!r}.")

    def step(
        self,
        grads: Sequence[torch.Tensor],
        lr: float,
        *,
        collect_diagnostics: bool = True,
    ) -> dict[str, float]:
        if len(grads) != len(self.params):
            raise ValueError(f"Expected {len(self.params)} gradients, got {len(grads)}.")
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}.")

        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count
        updates: list[torch.Tensor] = []

        with torch.no_grad():
            for exp_avg, exp_avg_sq, grad in zip(self.exp_avg, self.exp_avg_sq, grads):
                grad = grad.detach()
                exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                corrected_avg = exp_avg / bias_correction1
                corrected_sq = exp_avg_sq / bias_correction2
                updates.append(corrected_avg / (corrected_sq.sqrt().add(self.eps)))

            updates, extra_diagnostics = self._transform_update(
                updates,
                collect_diagnostics=collect_diagnostics,
            )

            for param, update in zip(self.params, updates):
                if self.weight_decay != 0.0:
                    param.mul_(1.0 - lr * self.weight_decay)
                param.add_(update, alpha=-lr)

        if not collect_diagnostics:
            return {}

        update_norm = tensor_list_global_norm(updates).item()
        diagnostics = {
            "grad_norm": tensor_list_global_norm(list(grads)).item(),
            "adam_first_moment_norm": tensor_list_global_norm(self.exp_avg).item(),
            "adam_second_moment_norm": tensor_list_global_norm(self.exp_avg_sq).item(),
            "adam_update_norm": update_norm,
            "adam_step_norm": lr * update_norm,
        }
        diagnostics.update(extra_diagnostics)
        return diagnostics
