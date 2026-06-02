"""Lightweight logging and atomic file-write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to reduce corruption during interrupted runs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=destination.parent, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(destination)
