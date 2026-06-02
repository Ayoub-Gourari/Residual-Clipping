"""Shared utilities for residual clipping experiments."""

from .cli import add_resume_args, add_wandb_args
from .cifar10_models import get_model
from .clipping import clip_tensor, clip_tensor_list, tensor_list_global_norm
from .logging_utils import append_jsonl, atomic_write_json
from .quadratics import parse_float_list
from .randomness import capture_rng_state, restore_rng_state, set_global_seed

__all__ = [
    "add_resume_args",
    "add_wandb_args",
    "append_jsonl",
    "atomic_write_json",
    "capture_rng_state",
    "get_model",
    "clip_tensor",
    "clip_tensor_list",
    "parse_float_list",
    "restore_rng_state",
    "set_global_seed",
    "tensor_list_global_norm",
]
