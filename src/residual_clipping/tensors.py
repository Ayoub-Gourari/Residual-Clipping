"""Tensor reshaping helpers shared across experiment modules."""

from __future__ import annotations

from typing import Iterable

import torch


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten a tensor collection into one vector."""
    materialized = [tensor.reshape(-1) for tensor in tensors]
    if not materialized:
        return torch.empty(0)
    return torch.cat(materialized)


def unflatten_like(vector: torch.Tensor, reference_tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Split a vector into tensors shaped like a reference collection."""
    outputs: list[torch.Tensor] = []
    offset = 0
    for reference in reference_tensors:
        size = reference.numel()
        chunk = vector[offset : offset + size].reshape_as(reference)
        outputs.append(chunk)
        offset += size
    if offset != vector.numel():
        raise ValueError("Vector size does not match reference tensors.")
    return outputs
