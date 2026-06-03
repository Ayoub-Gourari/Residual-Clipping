"""Launch centralized WikiText-2 collection and plotting from tracked YAML configs."""

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

from experiments.wikitext2.collect import build_parser as build_collect_parser, main as collect_main
from experiments.wikitext2.plot import build_parser as build_plot_parser, main as plot_main
from residual_clipping.configs import config_to_argv, load_yaml_config


def filter_overrides(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    option_map = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_map[option] = action

    filtered: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        action = option_map.get(token)
        if action is None:
            index += 1
            continue
        filtered.append(token)
        if action.nargs == 0:
            index += 1
            continue
        if index + 1 < len(argv):
            filtered.append(argv[index + 1])
        index += 2
    return filtered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and plot centralized WikiText-2 results from YAML configs.")
    parser.add_argument("--config", required=True, help="Path to a YAML plotting config.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional CLI overrides. Prefix with '--', for example: -- --figure-dir figures/custom",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    overrides = list(args.overrides)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    config = load_yaml_config(args.config)
    collect_parser = build_collect_parser()
    plot_parser = build_plot_parser()
    collect_known = {action.dest for action in collect_parser._actions if action.dest not in {"help"}}
    plot_known = {action.dest for action in plot_parser._actions if action.dest not in {"help"}}

    collect_config = {key: value for key, value in config.items() if key in collect_known}
    plot_config = {key: value for key, value in config.items() if key in plot_known}

    collect_argv = config_to_argv(collect_config)
    plot_argv = config_to_argv(plot_config)
    collect_overrides = filter_overrides(collect_parser, overrides)
    plot_overrides = filter_overrides(plot_parser, overrides)

    sys.argv = [sys.argv[0], *(collect_argv + collect_overrides)]
    collect_main()
    sys.argv = [sys.argv[0], *(plot_argv + plot_overrides)]
    plot_main()


if __name__ == "__main__":
    main()
