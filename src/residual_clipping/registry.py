"""Named experiment registry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .configs import load_yaml_config


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    kind: str
    family: str
    config: str
    description: str


def load_registry(path: str | Path) -> list[RegistryEntry]:
    payload = load_yaml_config(path)
    items = payload.get("experiments", [])
    if not isinstance(items, list):
        raise ValueError(f"Registry at {path} must define an 'experiments' list.")

    entries: list[RegistryEntry] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Registry entry must be a mapping, got {type(item).__name__}.")
        entries.append(
            RegistryEntry(
                name=str(item["name"]),
                kind=str(item["kind"]),
                family=str(item["family"]),
                config=str(item["config"]),
                description=str(item.get("description", "")),
            )
        )
    return entries


def registry_frame(path: str | Path):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for entry in load_registry(path):
        rows.append(
            {
                "name": entry.name,
                "kind": entry.kind,
                "family": entry.family,
                "config": entry.config,
                "description": entry.description,
            }
        )
    return pd.DataFrame(rows)


def get_registry_entry(path: str | Path, name: str) -> RegistryEntry:
    for entry in load_registry(path):
        if entry.name == name:
            return entry
    raise KeyError(f"Unknown registry entry: {name}")
