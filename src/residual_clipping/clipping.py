"""Shared clipping helpers for centralized experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def tensor_list_global_norm(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute a shared global norm across a list of tensors."""
    if not tensors:
        return torch.tensor(0.0)

    total = torch.zeros((), device=tensors[0].device, dtype=tensors[0].dtype)
    for tensor in tensors:
        total = total + tensor.detach().pow(2).sum()
    return total.sqrt()


def clip_tensor(tensor: torch.Tensor, threshold: float | None, eps: float = 1e-12) -> tuple[torch.Tensor, float, float]:
    """Clip a single tensor by its global norm."""
    norm = torch.linalg.vector_norm(tensor).item()
    if threshold is None:
        return tensor.detach().clone(), norm, 1.0
    scale = min(1.0, float(threshold) / (norm + eps))
    return tensor * scale, norm, scale


def clip_tensor_list(
    tensors: Iterable[torch.Tensor],
    threshold: float | None,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor], float, float]:
    """Clip a tensor list by its shared global norm."""
    materialized = [tensor.detach().clone() for tensor in tensors]
    norm = tensor_list_global_norm(materialized).item()
    if threshold is None:
        return materialized, norm, 1.0
    scale = min(1.0, float(threshold) / (norm + eps))
    return [tensor * scale for tensor in materialized], norm, scale
