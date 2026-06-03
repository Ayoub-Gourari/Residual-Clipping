"""Launch a WikiText-2 language-modeling sweep from a tracked YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.wikitext2.sweep import main as sweep_main
from residual_clipping.configs import config_to_argv, load_yaml_config


def build_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tracked WikiText-2 language-modeling sweep config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional CLI overrides. Prefix with '--', for example: -- --resume",
    )
    return parser


def main() -> None:
    parser = build_config_parser()
    args = parser.parse_args()

    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    config = load_yaml_config(args.config)
    config_argv = config_to_argv(config)

    sys.argv = [sys.argv[0], *(config_argv + overrides)]
    sweep_main()


if __name__ == "__main__":
    main()
