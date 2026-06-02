"""Common CLI helpers shared across experiment families."""

from __future__ import annotations

import argparse


def add_wandb_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach a standard W&B interface without embedding account defaults."""
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="W&B logging mode.",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-job-type", type=str, default=None, help="W&B job type label.")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags.",
    )
    parser.add_argument("--wandb-notes", type=str, default=None, help="Optional W&B notes.")
    return parser


def add_resume_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach a shared resume/overwrite interface."""
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing outputs or checkpoints when possible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing outputs instead of resuming or skipping them.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable run identifier for resumable local and W&B runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where metrics, summaries, and checkpoints are written.",
    )
    return parser
