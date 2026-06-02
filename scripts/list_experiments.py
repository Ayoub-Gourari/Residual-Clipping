"""List named tracked experiment and report entries."""

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

from residual_clipping.registry import registry_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List named experiment registry entries.")
    parser.add_argument("--registry", default="configs/cifar10/registry.yaml")
    parser.add_argument("--family", default=None)
    parser.add_argument("--kind", choices=("sweep", "report"), default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    frame = registry_frame(args.registry)
    if args.family is not None:
        frame = frame[frame["family"] == args.family]
    if args.kind is not None:
        frame = frame[frame["kind"] == args.kind]
    if frame.empty:
        print("No registry entries found.")
        return
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
