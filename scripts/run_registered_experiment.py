"""Run a named tracked sweep or report entry from the registry."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from residual_clipping.registry import get_registry_entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a named tracked experiment or report entry.")
    parser.add_argument("--registry", default="configs/cifar10/registry.yaml")
    parser.add_argument("--name", required=True, help="Registry entry name.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional CLI overrides. Prefix with '--', for example: -- --resume --wandb-group demo",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    entry = get_registry_entry(args.registry, args.name)

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    if entry.kind == "sweep" and entry.family.startswith("cifar"):
        command = ["python3", "scripts/run_cifar10_config.py", "--config", entry.config]
    elif entry.kind == "sweep" and entry.family == "wikitext2":
        command = ["python3", "scripts/run_wikitext2_config.py", "--config", entry.config]
    elif entry.kind == "sweep" and entry.family == "llm_finetune":
        command = ["python3", "scripts/run_llm_finetune_config.py", "--config", entry.config]
    elif entry.kind == "report" and entry.family.startswith("cifar"):
        command = ["python3", "scripts/plot_cifar10_config.py", "--config", entry.config]
    elif entry.kind == "report" and entry.family == "wikitext2":
        command = ["python3", "scripts/plot_wikitext2_config.py", "--config", entry.config]
    else:
        raise ValueError(f"Unsupported registry entry kind/family: {entry.kind}/{entry.family}")

    if overrides:
        command.append("--")
        command.extend(overrides)

    raise SystemExit(subprocess.call(command, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    main()
