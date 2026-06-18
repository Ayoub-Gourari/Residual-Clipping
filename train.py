#!/usr/bin/env python3
"""Storage-conscious ALBERT/RTE training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.llm_finetune.run import build_parser, validate_args
from residual_clipping.configs import load_yaml_config
from residual_clipping.llm_finetune_pipeline import run_llm_finetune_experiment


BOOL_FLAGS = {
    "--correct-bias",
    "--save-checkpoints",
    "--save-final-model",
    "--wandb-log-model",
    "--download",
    "--resume",
    "--overwrite",
    "--use-cuda",
}

ALIASES = {
    "--model-checkpoint": "--model-checkpoint",
    "--model_checkpoint": "--model-checkpoint",
    "--optimizer_name": "--optimizer-name",
    "--clip_threshold": "--clip-threshold",
    "--clipping_scope": "--clipping-scope",
    "--max_epochs": "--epochs",
    "--max-epochs": "--epochs",
    "--learning_rate": "--lr",
    "--learning-rate": "--lr",
    "--batch_size": "--batch-size",
    "--eval_batch_size": "--eval-batch-size",
    "--weight_decay": "--weight-decay",
    "--correct_bias": "--correct-bias",
    "--classifier_dropout": "--classifier-dropout",
    "--val_check_interval": "--val-check-interval",
    "--save_checkpoints": "--save-checkpoints",
    "--save_final_model": "--save-final-model",
    "--wandb_log_model": "--wandb-log-model",
    "--wandb_mode": "--wandb-mode",
    "--wandb_project": "--wandb-project",
    "--wandb_entity": "--wandb-entity",
    "--wandb_group": "--wandb-group",
    "--wandb_run_name": "--wandb-run-name",
    "--wandb_job_type": "--wandb-job-type",
    "--wandb_tags": "--wandb-tags",
    "--wandb_notes": "--wandb-notes",
    "--adam_beta1": "--adam-beta1",
    "--adam_beta2": "--adam-beta2",
    "--adam_eps": "--adam-eps",
    "--eps": "--adam-eps",
}

RTE_DEFAULTS = [
    "--task-type",
    "sequence_classification",
    "--model-source",
    "hf",
    "--model-name",
    "albert-base-v2",
    "--dataset-source",
    "hf",
    "--dataset-name",
    "glue",
    "--dataset-config",
    "rte",
    "--train-split",
    "train",
    "--validation-split",
    "validation",
    "--test-split",
    "validation",
    "--sentence1-column",
    "sentence1",
    "--sentence2-column",
    "sentence2",
    "--label-column",
    "label",
    "--num-labels",
    "2",
    "--epochs",
    "1",
    "--batch-size",
    "8",
    "--eval-batch-size",
    "8",
    "--sequence-length",
    "128",
    "--lr",
    "1e-5",
    "--adam-beta1",
    "0.9",
    "--adam-beta2",
    "0.999",
    "--adam-eps",
    "1e-6",
    "--weight-decay",
    "0.0",
    "--clip-threshold",
    "1.0",
    "--clipping-scope",
    "local",
    "--no-correct-bias",
    "--classifier-dropout",
    "0.0",
    "--log-interval",
    "20",
    "--val-check-interval",
    "12",
    "--seed",
    "123",
    "--download",
    "--no-save-checkpoints",
    "--no-save-final-model",
    "--no-wandb-log-model",
    "--wandb-mode",
    "disabled",
    "--output-dir",
    "outputs/albert_rte_resclip",
]


def _as_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {text!r}.")


def _append_config_value(argv: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if key == "betas":
        beta1, beta2 = value
        argv.extend(["--adam-beta1", str(beta1), "--adam-beta2", str(beta2)])
        return

    flag = ALIASES.get(f"--{key}", f"--{key.replace('_', '-')}")
    if flag in BOOL_FLAGS:
        argv.append(flag if bool(value) else f"--no-{flag[2:]}")
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    if isinstance(value, list):
        argv.extend([flag, ",".join(str(item) for item in value)])
        return
    argv.extend([flag, str(value)])


def _extract_config(argv: list[str]) -> tuple[list[str], list[str]]:
    remaining: list[str] = []
    config_argv: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config requires a path.")
            config = load_yaml_config(argv[index + 1])
            for key, value in config.items():
                _append_config_value(config_argv, key, value)
            index += 2
        else:
            remaining.append(argv[index])
            index += 1
    return config_argv, remaining


def _normalize_user_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    task = "rte"
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--task":
            if index + 1 >= len(argv):
                raise ValueError("--task requires a value.")
            task = argv[index + 1].lower()
            index += 2
            continue

        if token == "--betas":
            if index + 1 >= len(argv):
                raise ValueError("--betas requires beta1,beta2.")
            beta1, beta2 = [item.strip() for item in argv[index + 1].split(",", maxsplit=1)]
            normalized.extend(["--adam-beta1", beta1, "--adam-beta2", beta2])
            index += 2
            continue

        flag, separator, inline_value = token.partition("=")
        mapped = ALIASES.get(flag, flag.replace("_", "-") if flag.startswith("--") else flag)
        if separator:
            if mapped in BOOL_FLAGS:
                value = _as_bool(inline_value)
                normalized.append(mapped if value else f"--no-{mapped[2:]}")
            else:
                normalized.append(f"{mapped}={inline_value}")
            index += 1
            continue

        if mapped in BOOL_FLAGS and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            value = _as_bool(argv[index + 1])
            normalized.append(mapped if value else f"--no-{mapped[2:]}")
            index += 2
            continue

        normalized.append(mapped)
        index += 1

    if task != "rte":
        raise ValueError("This storage-conscious entrypoint currently supports only --task rte.")
    return [*RTE_DEFAULTS, *normalized]


def main() -> None:
    config_argv, user_argv = _extract_config(sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args(_normalize_user_argv([*config_argv, *user_argv]))
    validate_args(args)
    run_llm_finetune_experiment(args)


if __name__ == "__main__":
    main()
