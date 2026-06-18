#!/usr/bin/env python3
"""Download ALBERT-base-v2 and GLUE/RTE once into a shared Hugging Face cache."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def default_cache_dir() -> Path:
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()


def prepare_cache_dir(path: Path) -> Path:
    cache_root = path.expanduser().resolve()
    if str(cache_root) == "/path" or str(cache_root).startswith("/path/"):
        raise SystemExit(
            "Replace /path/to/shared/hf-cache with a real writable directory, for example:\n"
            "  export HF_HOME=\"$HOME/.cache/huggingface\"\n"
            "  python scripts/prefetch_albert_rte.py --cache-dir \"$HF_HOME\""
        )
    try:
        (cache_root / "transformers").mkdir(parents=True, exist_ok=True)
        (cache_root / "datasets").mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise SystemExit(
            f"Cannot write to cache directory {cache_root}.\n"
            "Choose a directory owned by your user, for example:\n"
            "  export HF_HOME=\"$HOME/.cache/huggingface\"\n"
            "  python scripts/prefetch_albert_rte.py --cache-dir \"$HF_HOME\""
        ) from exc
    return cache_root


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError:
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--model-checkpoint", type=str, default="albert-base-v2")
    parser.add_argument("--model-revision", type=str, default="8e2f239")
    parser.add_argument("--dataset-name", type=str, default="glue")
    parser.add_argument("--dataset-config", type=str, default="rte")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--classifier-dropout", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache_root = prepare_cache_dir(args.cache_dir)
    transformers_cache = cache_root / "transformers"
    datasets_cache = cache_root / "datasets"

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))

    with file_lock(cache_root / ".albert_rte_prefetch.lock"):
        try:
            from datasets import load_dataset
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Prefetching requires the optional LLM dependencies. "
                "Install them with: python3 -m pip install -e '.[llm]'"
            ) from exc

        print(f"Using shared HF cache root: {cache_root}", flush=True)
        print(f"Downloading tokenizer: {args.model_checkpoint}@{args.model_revision}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_checkpoint,
            revision=args.model_revision,
            cache_dir=str(transformers_cache),
        )
        tokenizer("A first sentence.", "A second sentence.", truncation=True, max_length=128)

        print(f"Downloading model weights: {args.model_checkpoint}@{args.model_revision}", flush=True)
        AutoModelForSequenceClassification.from_pretrained(
            args.model_checkpoint,
            revision=args.model_revision,
            cache_dir=str(transformers_cache),
            num_labels=args.num_labels,
            classifier_dropout_prob=args.classifier_dropout,
        )

        print(f"Downloading dataset: {args.dataset_name}/{args.dataset_config}", flush=True)
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            cache_dir=str(datasets_cache),
        )
        split_sizes = {split: len(values) for split, values in dataset.items()}
        print(f"Cached dataset splits: {split_sizes}", flush=True)
        print("Done. Reuse this cache with:", flush=True)
        print(f"  export HF_HOME={cache_root}", flush=True)
        print(f"  export TRANSFORMERS_CACHE={transformers_cache}", flush=True)
        print(f"  export HF_DATASETS_CACHE={datasets_cache}", flush=True)
        print(f"  python train.py --config configs/albert_rte_resclip_default.yaml --hf-cache-dir {cache_root} --no-download", flush=True)


if __name__ == "__main__":
    main()
