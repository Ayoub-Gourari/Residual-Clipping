"""Helpers for loading tracked experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}, got {type(payload).__name__}.")
    return payload


def config_to_argv(config: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None or value is False:
            continue
        if isinstance(value, bool):
            argv.append(flag)
            continue
        if isinstance(value, list):
            serialized = ",".join(str(item) for item in value)
            argv.extend([flag, serialized])
            continue
        argv.extend([flag, str(value)])
    return argv
