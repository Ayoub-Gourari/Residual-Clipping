"""Transformer fine-tuning helpers for adaptive clipping experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_optimizers import (
    ADAPTIVE_OPTIMIZER_MODES,
    RESIDUAL_CLIP_ADAMW_M,
    RESIDUAL_CLIP_ADAMW_M_POST,
    AdaptiveAdamW,
    collect_trainable_gradients,
)
from .cifar10_pipeline import (
    RunningDiagnostics,
    log_wandb,
    maybe_init_wandb,
    metrics_path,
    replay_wandb_history,
    resolved_wandb_group,
)
from .logging_utils import append_jsonl, atomic_write_json
from .randomness import capture_rng_state, restore_rng_state, set_global_seed


TASK_TYPES = ("causal_lm", "sequence_classification")
MODEL_SOURCES = ("fake", "hf")
DATASET_SOURCES = ("fake", "hf", "text")


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_latest.pt"


def summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def sweep_output_dir(output_dir: str | Path, dataset_name: str) -> Path:
    return Path(output_dir) / f"{dataset_name}_llm_sweeps"


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 100.0))


def _tag(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return sanitized or "unnamed"


def _expanded_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _hf_cache_root(args) -> Path | None:
    return _expanded_optional_path(getattr(args, "hf_cache_dir", None))


def _transformers_cache_dir(args) -> str | None:
    root = _hf_cache_root(args)
    if root is None:
        return None
    return str(root / "transformers")


def _datasets_cache_dir(args) -> str | None:
    root = _hf_cache_root(args)
    if root is not None:
        return str(root / "datasets")
    return str(args.data_dir) if getattr(args, "data_dir", None) is not None else None


def _hf_from_pretrained_kwargs(args) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "local_files_only": not args.download,
    }
    cache_dir = _transformers_cache_dir(args)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return kwargs


def resolved_run_name(args) -> str:
    if args.run_name is not None:
        run_name = args.run_name
    else:
        parts = [
            _tag(args.model_name),
            args.optimizer_name,
            f"scope{_tag(getattr(args, 'clipping_scope', 'local'))}",
            f"C{str(args.clip_threshold).replace('.', 'p')}",
            f"lr{str(args.lr).replace('.', 'p')}",
            f"b1{str(args.adam_beta1).replace('.', 'p')}",
            f"b2{str(args.adam_beta2).replace('.', 'p')}",
            f"seed{args.seed}",
        ]
        run_name = args.wandb_run_name or "-".join(parts)

    experiment_tag = getattr(args, "experiment_tag", None)
    if experiment_tag:
        return f"{experiment_tag}-{run_name}"
    return run_name


def normalize_shared_run_attrs(args) -> None:
    """Populate shared attribute names used by cross-family logging helpers."""
    if not hasattr(args, "optimizer_name") and hasattr(args, "optimizer_mode"):
        args.optimizer_name = args.optimizer_mode
    if not hasattr(args, "clip_threshold"):
        args.clip_threshold = math.inf
    if not hasattr(args, "clipping_scope"):
        args.clipping_scope = "local"
    if not hasattr(args, "correct_bias"):
        args.correct_bias = False
    if args.optimizer_name in {RESIDUAL_CLIP_ADAMW_M, RESIDUAL_CLIP_ADAMW_M_POST}:
        args.correct_bias = True
    if not hasattr(args, "classifier_dropout"):
        args.classifier_dropout = getattr(args, "dropout", 0.0)
    if not hasattr(args, "val_check_interval"):
        args.val_check_interval = None
    if not hasattr(args, "warmup_ratio"):
        args.warmup_ratio = 0.0
    if not hasattr(args, "hf_cache_dir"):
        args.hf_cache_dir = None
    if not hasattr(args, "save_checkpoints"):
        args.save_checkpoints = False
    if not hasattr(args, "save_final_model"):
        args.save_final_model = False
    if not hasattr(args, "wandb_log_model"):
        args.wandb_log_model = False
    if not hasattr(args, "model"):
        args.model = args.model_name
    if not hasattr(args, "model_revision"):
        args.model_revision = None
    if not hasattr(args, "dataset"):
        args.dataset = args.dataset_name


def summary_indicates_complete(run_dir: Path, epochs: int) -> bool:
    path = summary_path(run_dir)
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return int(summary.get("epochs", -1)) >= epochs and int(summary.get("completed", 0)) == 1


class TinyCausalLM(nn.Module):
    """Small local causal LM used for smoke tests and architecture checks."""

    def __init__(self, *, vocab_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        hidden = self.embedding(input_ids)
        hidden, _ = self.rnn(hidden)
        hidden = self.dropout(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


class TinySequenceClassifier(nn.Module):
    """Small local sentence-pair classifier used for smoke tests."""

    def __init__(self, *, vocab_size: int, hidden_size: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        hidden = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


@dataclass(frozen=True)
class FineTuneData:
    task_type: str
    train: torch.Tensor | dict[str, torch.Tensor]
    validation: torch.Tensor | dict[str, torch.Tensor]
    test: torch.Tensor | dict[str, torch.Tensor]
    vocab_size: int | None = None
    num_labels: int | None = None


def _make_fake_sequences(
    *,
    num_sequences: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, vocab_size, (num_sequences, 1), generator=generator)
    offsets = torch.arange(sequence_length + 1).view(1, -1)
    noise = torch.randint(0, 3, (num_sequences, sequence_length + 1), generator=generator)
    return (starts + offsets + noise) % vocab_size


def _make_fake_classification_examples(
    *,
    num_examples: int,
    sequence_length: int,
    vocab_size: int,
    num_labels: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (num_examples, sequence_length), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    labels = (input_ids[:, : sequence_length // 2].sum(dim=1) % num_labels).to(dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM fine-tuning with --model-source hf requires the optional 'llm' dependencies. "
            "Install them with: python3 -m pip install -e '.[llm]'"
        ) from exc
    return AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "LLM fine-tuning with --dataset-source hf requires the optional 'llm' dependencies. "
            "Install them with: python3 -m pip install -e '.[llm]'"
        ) from exc
    return load_dataset


def _texts_from_text_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or [text]


def _texts_from_hf_dataset(args, split: str) -> list[str]:
    load_dataset = _require_datasets()
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=split,
        cache_dir=_datasets_cache_dir(args),
    )
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Column {args.text_column!r} not found in {args.dataset_name!r} split {split!r}. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )
    values = dataset[args.text_column]
    return [str(value) for value in values if str(value).strip()]


def _sequences_from_texts(
    tokenizer,
    texts: Iterable[str],
    *,
    sequence_length: int,
    max_sequences: int | None,
) -> torch.Tensor:
    token_ids: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True)
        token_ids.extend(int(value) for value in encoded["input_ids"])
    block_length = sequence_length + 1
    usable = (len(token_ids) // block_length) * block_length
    if usable == 0:
        raise ValueError(
            f"Not enough tokens to build one sequence of length {block_length}. "
            "Use more text or lower --sequence-length."
        )
    sequences = torch.tensor(token_ids[:usable], dtype=torch.long).view(-1, block_length)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    return sequences


def _classification_tensors_from_hf_dataset(
    args,
    split: str,
    tokenizer,
    *,
    max_examples: int | None,
) -> dict[str, torch.Tensor]:
    load_dataset = _require_datasets()
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=split,
        cache_dir=_datasets_cache_dir(args),
    )
    required_columns = [args.sentence1_column, args.label_column]
    if args.sentence2_column:
        required_columns.append(args.sentence2_column)
    missing = [column for column in required_columns if column not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Missing columns in {args.dataset_name!r} split {split!r}: {', '.join(missing)}. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )

    text1_values: list[str] = []
    text2_values: list[str] | None = [] if args.sentence2_column else None
    labels: list[int] = []
    for row in dataset:
        label = int(row[args.label_column])
        if label < 0:
            continue
        text1_values.append(str(row[args.sentence1_column]))
        if text2_values is not None:
            text2_values.append(str(row[args.sentence2_column]))
        labels.append(label)
        if max_examples is not None and len(labels) >= max_examples:
            break
    if not labels:
        raise ValueError(
            f"No labeled examples found in split {split!r}. "
            "For GLUE-style datasets, use a validation split when test labels are hidden."
        )

    tokenizer_args = [text1_values] if text2_values is None else [text1_values, text2_values]
    encoded = tokenizer(
        *tokenizer_args,
        truncation=True,
        padding="max_length",
        max_length=args.sequence_length,
        return_tensors="pt",
    )
    tensors = {key: value for key, value in encoded.items()}
    tensors["labels"] = torch.tensor(labels, dtype=torch.long)
    return tensors


def load_model_and_data(args, device: torch.device) -> tuple[nn.Module, FineTuneData]:
    if args.task_type == "causal_lm":
        if args.model_source == "fake":
            model = TinyCausalLM(
                vocab_size=args.fake_vocab_size,
                hidden_size=args.fake_hidden_size,
                num_layers=args.fake_num_layers,
                dropout=args.dropout,
            ).to(device)
            data = FineTuneData(
                task_type="causal_lm",
                train=_make_fake_sequences(
                    num_sequences=args.fake_train_sequences,
                    sequence_length=args.sequence_length,
                    vocab_size=args.fake_vocab_size,
                    seed=args.seed,
                ),
                validation=_make_fake_sequences(
                    num_sequences=args.fake_eval_sequences,
                    sequence_length=args.sequence_length,
                    vocab_size=args.fake_vocab_size,
                    seed=args.seed + 10_000,
                ),
                test=_make_fake_sequences(
                    num_sequences=args.fake_eval_sequences,
                    sequence_length=args.sequence_length,
                    vocab_size=args.fake_vocab_size,
                    seed=args.seed + 20_000,
                ),
                vocab_size=args.fake_vocab_size,
            )
            return model, data

        AutoModelForCausalLM, _, AutoTokenizer = _require_transformers()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, **_hf_from_pretrained_kwargs(args))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **_hf_from_pretrained_kwargs(args)).to(device)

        if args.dataset_source == "fake":
            data = FineTuneData(
                task_type="causal_lm",
                train=_make_fake_sequences(
                    num_sequences=args.fake_train_sequences,
                    sequence_length=args.sequence_length,
                    vocab_size=int(model.config.vocab_size),
                    seed=args.seed,
                ),
                validation=_make_fake_sequences(
                    num_sequences=args.fake_eval_sequences,
                    sequence_length=args.sequence_length,
                    vocab_size=int(model.config.vocab_size),
                    seed=args.seed + 10_000,
                ),
                test=_make_fake_sequences(
                    num_sequences=args.fake_eval_sequences,
                    sequence_length=args.sequence_length,
                    vocab_size=int(model.config.vocab_size),
                    seed=args.seed + 20_000,
                ),
                vocab_size=int(model.config.vocab_size),
            )
            return model, data

        if args.dataset_source == "text":
            if args.train_file is None:
                raise ValueError("--train-file is required with --dataset-source text.")
            train_texts = _texts_from_text_file(args.train_file)
            if args.validation_file is None:
                split = max(1, int(0.9 * len(train_texts)))
                validation_texts = train_texts[split:] or train_texts[-1:]
                train_texts = train_texts[:split]
            else:
                validation_texts = _texts_from_text_file(args.validation_file)
            test_texts = validation_texts
        else:
            train_texts = _texts_from_hf_dataset(args, args.train_split)
            validation_texts = _texts_from_hf_dataset(args, args.validation_split)
            test_texts = _texts_from_hf_dataset(args, args.test_split)

        data = FineTuneData(
            task_type="causal_lm",
            train=_sequences_from_texts(
                tokenizer,
                train_texts,
                sequence_length=args.sequence_length,
                max_sequences=args.max_train_sequences,
            ),
            validation=_sequences_from_texts(
                tokenizer,
                validation_texts,
                sequence_length=args.sequence_length,
                max_sequences=args.max_eval_sequences,
            ),
            test=_sequences_from_texts(
                tokenizer,
                test_texts,
                sequence_length=args.sequence_length,
                max_sequences=args.max_eval_sequences,
            ),
            vocab_size=int(model.config.vocab_size),
        )
        return model, data

    if args.task_type != "sequence_classification":
        raise ValueError(f"Unsupported task_type={args.task_type!r}.")

    if args.model_source == "fake":
        model = TinySequenceClassifier(
            vocab_size=args.fake_vocab_size,
            hidden_size=args.fake_hidden_size,
            num_labels=args.num_labels,
            dropout=args.classifier_dropout,
        ).to(device)
        data = FineTuneData(
            task_type="sequence_classification",
            train=_make_fake_classification_examples(
                num_examples=args.fake_train_sequences,
                sequence_length=args.sequence_length,
                vocab_size=args.fake_vocab_size,
                num_labels=args.num_labels,
                seed=args.seed,
            ),
            validation=_make_fake_classification_examples(
                num_examples=args.fake_eval_sequences,
                sequence_length=args.sequence_length,
                vocab_size=args.fake_vocab_size,
                num_labels=args.num_labels,
                seed=args.seed + 10_000,
            ),
            test=_make_fake_classification_examples(
                num_examples=args.fake_eval_sequences,
                sequence_length=args.sequence_length,
                vocab_size=args.fake_vocab_size,
                num_labels=args.num_labels,
                seed=args.seed + 20_000,
            ),
            vocab_size=args.fake_vocab_size,
            num_labels=args.num_labels,
        )
        return model, data

    _, AutoModelForSequenceClassification, AutoTokenizer = _require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **_hf_from_pretrained_kwargs(args))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        classifier_dropout_prob=args.classifier_dropout,
        **_hf_from_pretrained_kwargs(args),
    ).to(device)

    if args.dataset_source == "fake":
        vocab_size = int(getattr(model.config, "vocab_size", args.fake_vocab_size))
        data = FineTuneData(
            task_type="sequence_classification",
            train=_make_fake_classification_examples(
                num_examples=args.fake_train_sequences,
                sequence_length=args.sequence_length,
                vocab_size=vocab_size,
                num_labels=args.num_labels,
                seed=args.seed,
            ),
            validation=_make_fake_classification_examples(
                num_examples=args.fake_eval_sequences,
                sequence_length=args.sequence_length,
                vocab_size=vocab_size,
                num_labels=args.num_labels,
                seed=args.seed + 10_000,
            ),
            test=_make_fake_classification_examples(
                num_examples=args.fake_eval_sequences,
                sequence_length=args.sequence_length,
                vocab_size=vocab_size,
                num_labels=args.num_labels,
                seed=args.seed + 20_000,
            ),
            vocab_size=vocab_size,
            num_labels=args.num_labels,
        )
        return model, data

    if args.dataset_source != "hf":
        raise ValueError("--task-type sequence_classification currently supports --dataset-source fake or hf.")

    data = FineTuneData(
        task_type="sequence_classification",
        train=_classification_tensors_from_hf_dataset(
            args,
            args.train_split,
            tokenizer,
            max_examples=args.max_train_sequences,
        ),
        validation=_classification_tensors_from_hf_dataset(
            args,
            args.validation_split,
            tokenizer,
            max_examples=args.max_eval_sequences,
        ),
        test=_classification_tensors_from_hf_dataset(
            args,
            args.test_split,
            tokenizer,
            max_examples=args.max_eval_sequences,
        ),
        vocab_size=int(getattr(model.config, "vocab_size", 0)),
        num_labels=args.num_labels,
    )
    return model, data


def iter_batches(
    sequences: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(sequences.size(0), generator=generator)
    else:
        indices = torch.arange(sequences.size(0))

    for start in range(0, sequences.size(0), batch_size):
        batch = sequences[indices[start : start + batch_size]].to(device)
        yield batch[:, :-1], batch[:, 1:]


def iter_classification_batches(
    examples: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> Iterable[dict[str, torch.Tensor]]:
    labels = examples["labels"]
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(labels.size(0), generator=generator)
    else:
        indices = torch.arange(labels.size(0))

    for start in range(0, labels.size(0), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield {key: value[batch_indices].to(device) for key, value in examples.items()}


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdaptiveAdamW,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_validation_accuracy: float | None,
    best_epoch: int,
    tracker: RunningDiagnostics,
    args_dict: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            "best_epoch": best_epoch,
            "tracker_values": tracker.values,
            "rng_state": capture_rng_state(),
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdaptiveAdamW,
    tracker: RunningDiagnostics,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    tracker.values = payload.get("tracker_values", {})
    restore_rng_state(payload["rng_state"])
    return payload


def evaluate(
    model: nn.Module,
    eval_data: torch.Tensor | dict[str, torch.Tensor],
    *,
    task_type: str,
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0.0
    with torch.no_grad():
        if task_type == "causal_lm":
            assert isinstance(eval_data, torch.Tensor)
            for batch_idx, (input_ids, labels) in enumerate(
                iter_batches(eval_data, batch_size=batch_size, device=device, shuffle=False, seed=0)
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                output = model(input_ids=input_ids, labels=labels)
                tokens = labels.numel()
                total_loss += float(output.loss.item()) * tokens
                total_items += tokens
            loss = total_loss / max(total_items, 1)
            return {"loss": loss, "perplexity": perplexity(loss), "tokens": float(total_items)}

        if task_type == "sequence_classification":
            assert isinstance(eval_data, dict)
            for batch_idx, batch in enumerate(
                iter_classification_batches(eval_data, batch_size=batch_size, device=device, shuffle=False, seed=0)
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                output = model(**batch)
                labels = batch["labels"]
                examples = labels.numel()
                total_loss += float(output.loss.item()) * examples
                total_items += examples
                correct += float(output.logits.argmax(dim=-1).eq(labels).sum().item())
            loss = total_loss / max(total_items, 1)
            return {
                "loss": loss,
                "accuracy": 100.0 * correct / max(total_items, 1),
                "examples": float(total_items),
            }

    raise ValueError(f"Unsupported task_type={task_type!r}.")


def num_training_examples(train_data: torch.Tensor | dict[str, torch.Tensor]) -> int:
    if isinstance(train_data, torch.Tensor):
        return int(train_data.size(0))
    return int(train_data["labels"].size(0))


def steps_per_epoch(
    train_data: torch.Tensor | dict[str, torch.Tensor],
    *,
    batch_size: int,
    max_train_batches: int | None,
) -> int:
    steps = math.ceil(num_training_examples(train_data) / batch_size)
    if max_train_batches is not None:
        steps = min(steps, max_train_batches)
    return max(steps, 1)


def scheduled_learning_rate(base_lr: float, step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = int(math.ceil(max(total_steps, 1) * warmup_ratio))
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, float(step) / float(warmup_steps))


def optimizer_log_metadata(args, lr: float) -> dict[str, float | int | str]:
    return {
        "optimizer/name": args.optimizer_name,
        "optimizer/lr": lr,
        "optimizer/beta1": args.adam_beta1,
        "optimizer/beta2": args.adam_beta2,
        "optimizer/weight_decay": args.weight_decay,
        "optimizer/clip_threshold": args.clip_threshold,
        "run/seed": args.seed,
    }


def _layerwise_group_name(parameter_name: str) -> str:
    """Best-effort module grouping for true layer-wise clipping experiments."""
    albert_layer = re.match(
        r"^(albert\.encoder\.albert_layer_groups\.\d+\.albert_layers\.\d+)",
        parameter_name,
    )
    if albert_layer is not None:
        return albert_layer.group(1)
    for prefix in (
        "albert.embeddings",
        "albert.encoder.embedding_hidden_mapping_in",
        "albert.pooler",
        "classifier",
        "embedding",
    ):
        if parameter_name.startswith(f"{prefix}.") or parameter_name == prefix:
            return prefix
    if "." in parameter_name:
        return parameter_name.rsplit(".", maxsplit=1)[0]
    return parameter_name


def optimizer_parameters(
    model: nn.Module,
    *,
    clipping_scope: str,
) -> tuple[list[nn.Parameter], Iterable[nn.Parameter] | list[dict[str, Any]]]:
    flat_params = [param for param in model.parameters() if param.requires_grad]
    if clipping_scope != "layerwise":
        return flat_params, flat_params

    groups: dict[str, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups.setdefault(_layerwise_group_name(name), []).append(param)

    return flat_params, [
        {"params": params, "clipping_scope": "layerwise", "layer_name": layer_name}
        for layer_name, params in groups.items()
    ]


def train_one_epoch(
    *,
    args,
    model: nn.Module,
    train_data: torch.Tensor | dict[str, torch.Tensor],
    validation_data: torch.Tensor | dict[str, torch.Tensor],
    task_type: str,
    optimizer: AdaptiveAdamW,
    epoch: int,
    global_step: int,
    tracker: RunningDiagnostics,
    run_dir: Path,
    device: torch.device,
    total_training_steps: int,
    params: list[nn.Parameter],
    best_state: dict[str, float | int | None],
) -> int:
    model.train()
    if task_type == "causal_lm":
        assert isinstance(train_data, torch.Tensor)
        batches = (
            {"input_ids": input_ids, "labels": labels}
            for input_ids, labels in iter_batches(
                train_data,
                batch_size=args.batch_size,
                device=device,
                shuffle=True,
                seed=args.seed + epoch,
            )
        )
    elif task_type == "sequence_classification":
        assert isinstance(train_data, dict)
        batches = iter_classification_batches(
            train_data,
            batch_size=args.batch_size,
            device=device,
            shuffle=True,
            seed=args.seed + epoch,
        )
    else:
        raise ValueError(f"Unsupported task_type={task_type!r}.")

    for batch_idx, batch in enumerate(batches):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        should_log = batch_idx % args.log_interval == 0
        grads = collect_trainable_gradients(params, clone=False)
        step_lr = scheduled_learning_rate(args.lr, global_step + 1, total_training_steps, args.warmup_ratio)
        diagnostics = optimizer.step(grads, step_lr, collect_diagnostics=should_log)
        tracker.update(diagnostics)

        global_step += 1
        if should_log:
            payload = {
                "train/global_step": global_step,
                "train/epoch": epoch,
                "train/batch_idx": batch_idx,
                "train/loss": loss.item(),
                "train/lr": step_lr,
                "global_step": global_step,
                "epoch": epoch,
                "train_loss": loss.item(),
                "learning_rate": step_lr,
                **optimizer_log_metadata(args, step_lr),
            }
            if task_type == "causal_lm":
                payload["train/perplexity"] = perplexity(loss.item())
            else:
                labels = batch["labels"]
                payload["train/accuracy"] = 100.0 * float(output.logits.argmax(dim=-1).eq(labels).float().mean().item())
            payload.update(diagnostics)
            payload.update({f"train/{key}": value for key, value in diagnostics.items() if "/" not in key})
            append_jsonl(metrics_path(run_dir), payload)
            log_wandb(payload)
            print(
                " ".join(
                    [
                        f"E={epoch}",
                        f"B={batch_idx}",
                        f"step={global_step}",
                        f"loss={loss.item():.4f}",
                        f"metric={payload.get('train/perplexity', payload.get('train/accuracy')):.2f}",
                    ]
                ),
                flush=True,
            )

        if args.val_check_interval is not None and global_step % args.val_check_interval == 0:
            validation = evaluate(
                model,
                validation_data,
                task_type=task_type,
                batch_size=args.eval_batch_size,
                max_batches=args.max_eval_batches,
                device=device,
            )
            val_payload = {
                "train/global_step": global_step,
                "validation/global_step": global_step,
                "validation/epoch": epoch,
                "validation/loss": validation["loss"],
                "global_step": global_step,
                "epoch": epoch,
                "val_loss": validation["loss"],
                "eval/global_step": global_step,
                "eval/epoch": epoch,
                "eval/loss": validation["loss"],
                **optimizer_log_metadata(
                    args,
                    scheduled_learning_rate(args.lr, global_step, total_training_steps, args.warmup_ratio),
                ),
            }
            if task_type == "causal_lm":
                if validation["loss"] < float(best_state["loss"]):
                    best_state["loss"] = validation["loss"]
                    best_state["epoch"] = epoch
                val_payload["validation/perplexity"] = validation["perplexity"]
                val_payload["validation/tokens"] = validation["tokens"]
            else:
                best_accuracy = best_state["accuracy"]
                if best_accuracy is None or validation["accuracy"] > float(best_accuracy):
                    best_state["accuracy"] = validation["accuracy"]
                    best_state["loss"] = validation["loss"]
                    best_state["epoch"] = epoch
                val_payload["validation/accuracy"] = validation["accuracy"]
                val_payload["validation/examples"] = validation["examples"]
                val_payload["val_accuracy"] = validation["accuracy"]
                val_payload["eval/accuracy"] = validation["accuracy"]
                val_payload["eval/best_accuracy"] = best_state["accuracy"]
            append_jsonl(metrics_path(run_dir), val_payload)
            log_wandb(val_payload)
            model.train()
    return global_step


def run_llm_finetune_experiment(args) -> dict[str, Any]:
    require_cuda = bool(getattr(args, "require_cuda", False))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is false.")
    if require_cuda:
        args.use_cuda = True
    elif args.use_cuda and not torch.cuda.is_available():
        print("Requested CUDA, but no CUDA device is available; using CPU.", flush=True)
        args.use_cuda = False
    normalize_shared_run_attrs(args)
    device = torch.device("cuda" if args.use_cuda else "cpu")

    run_name = resolved_run_name(args)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and args.resume and summary_indicates_complete(run_dir, args.epochs):
        summary = json.loads(summary_path(run_dir).read_text(encoding="utf-8"))
        run = maybe_init_wandb(args, run_name)
        if run is not None:
            uploaded = replay_wandb_history(run, metrics_path(run_dir))
            print(f"Uploaded {uploaded} existing metric rows to W&B.", flush=True)
            run.finish()
        print(f"Skipping completed run at {run_dir}", flush=True)
        return summary
    if not args.resume and not args.overwrite:
        if summary_path(run_dir).exists() or checkpoint_path(run_dir).exists():
            raise FileExistsError(
                f"Found existing outputs in {run_dir}. Use --resume to continue or --overwrite to replace."
            )
    if args.overwrite:
        for path in [metrics_path(run_dir), summary_path(run_dir), checkpoint_path(run_dir)]:
            if path.exists():
                path.unlink()

    run = maybe_init_wandb(args, run_name)

    set_global_seed(args.seed)
    model, data = load_model_and_data(args, device)
    total_training_steps = steps_per_epoch(
        data.train,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
    ) * args.epochs
    params, optimizer_param_groups = optimizer_parameters(model, clipping_scope=args.clipping_scope)
    optimizer = AdaptiveAdamW(
        optimizer_param_groups,
        optimizer_name=args.optimizer_name,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        clip_threshold=args.clip_threshold,
        clipping_scope=args.clipping_scope,
        correct_bias=args.correct_bias,
    )
    args.correct_bias = all(bool(group["correct_bias"]) for group in optimizer.param_groups)
    tracker = RunningDiagnostics()
    global_step = 0
    start_epoch = 1
    best_validation_loss = math.inf
    best_validation_accuracy: float | None = None
    best_epoch = 0
    initial_payload = None

    if args.resume and checkpoint_path(run_dir).exists():
        checkpoint = load_checkpoint(
            checkpoint_path(run_dir),
            model=model,
            optimizer=optimizer,
            tracker=tracker,
        )
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        checkpoint_accuracy = checkpoint.get("best_validation_accuracy")
        best_validation_accuracy = None if checkpoint_accuracy is None else float(checkpoint_accuracy)
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming {run_name} from epoch {start_epoch}", flush=True)
    else:
        initial_eval = evaluate(
            model,
            data.validation,
            task_type=data.task_type,
            batch_size=args.eval_batch_size,
            max_batches=args.max_eval_batches,
            device=device,
        )
        initial_payload = {
            "train/global_step": 0,
            "validation/global_step": 0,
            "validation/epoch": 0,
            "validation/loss": initial_eval["loss"],
            "global_step": 0,
            "epoch": 0,
            "val_loss": initial_eval["loss"],
            "eval/global_step": 0,
            "eval/epoch": 0,
            "eval/loss": initial_eval["loss"],
            **optimizer_log_metadata(args, args.lr),
            **{
                f"run/{key}": value
                for key, value in vars(args).items()
                if isinstance(value, (int, float, str, bool)) or value is None
            },
        }
        if data.task_type == "causal_lm":
            initial_payload["validation/perplexity"] = initial_eval["perplexity"]
            initial_payload["validation/tokens"] = initial_eval["tokens"]
        else:
            initial_payload["validation/accuracy"] = initial_eval["accuracy"]
            initial_payload["validation/examples"] = initial_eval["examples"]
            initial_payload["val_accuracy"] = initial_eval["accuracy"]
            initial_payload["eval/accuracy"] = initial_eval["accuracy"]
            initial_payload["eval/best_accuracy"] = initial_eval["accuracy"]
            best_validation_accuracy = initial_eval["accuracy"]
        append_jsonl(metrics_path(run_dir), initial_payload)
        best_validation_loss = initial_eval["loss"]

    best_state: dict[str, float | int | None] = {
        "loss": best_validation_loss,
        "accuracy": best_validation_accuracy,
        "epoch": best_epoch,
    }

    if run is not None and start_epoch == 1 and initial_payload is not None:
        log_wandb(initial_payload)
    elif run is not None and start_epoch > 1:
        uploaded = replay_wandb_history(run, metrics_path(run_dir))
        if uploaded:
            print(f"Uploaded {uploaded} existing metric rows to W&B before resuming.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        global_step = train_one_epoch(
            args=args,
            model=model,
            train_data=data.train,
            validation_data=data.validation,
            task_type=data.task_type,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            tracker=tracker,
            run_dir=run_dir,
            device=device,
            total_training_steps=total_training_steps,
            params=params,
            best_state=best_state,
        )
        best_validation_loss = float(best_state["loss"])
        best_validation_accuracy = (
            None if best_state["accuracy"] is None else float(best_state["accuracy"])
        )
        best_epoch = int(best_state["epoch"])
        validation = evaluate(
            model,
            data.validation,
            task_type=data.task_type,
            batch_size=args.eval_batch_size,
            max_batches=args.max_eval_batches,
            device=device,
        )
        if data.task_type == "causal_lm":
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                best_epoch = epoch
        else:
            if best_validation_accuracy is None or validation["accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation["accuracy"]
                best_validation_loss = validation["loss"]
                best_epoch = epoch
        best_state.update(
            {
                "loss": best_validation_loss,
                "accuracy": best_validation_accuracy,
                "epoch": best_epoch,
            }
        )
        payload = {
            "train/global_step": global_step,
            "validation/global_step": global_step,
            "validation/epoch": epoch,
            "validation/loss": validation["loss"],
            "validation/best_loss": best_validation_loss,
            "validation/best_epoch": best_epoch,
            "train/lr": scheduled_learning_rate(args.lr, global_step, total_training_steps, args.warmup_ratio),
            "global_step": global_step,
            "epoch": epoch,
            "val_loss": validation["loss"],
            "eval/global_step": global_step,
            "eval/epoch": epoch,
            "eval/loss": validation["loss"],
            **optimizer_log_metadata(
                args,
                scheduled_learning_rate(args.lr, global_step, total_training_steps, args.warmup_ratio),
            ),
        }
        if data.task_type == "causal_lm":
            payload["validation/perplexity"] = validation["perplexity"]
            payload["validation/best_perplexity"] = perplexity(best_validation_loss)
            payload["validation/tokens"] = validation["tokens"]
        else:
            payload["validation/accuracy"] = validation["accuracy"]
            payload["validation/best_accuracy"] = best_validation_accuracy
            payload["validation/examples"] = validation["examples"]
            payload["val_accuracy"] = validation["accuracy"]
            payload["eval/accuracy"] = validation["accuracy"]
            payload["eval/best_accuracy"] = best_validation_accuracy
        append_jsonl(metrics_path(run_dir), payload)
        log_wandb(payload)
        if data.task_type == "causal_lm":
            print(
                f"E={epoch} validation_loss={validation['loss']:.4f} "
                f"validation_ppl={validation['perplexity']:.2f} "
                f"best_ppl={perplexity(best_validation_loss):.2f}@{best_epoch} lr={args.lr}",
                flush=True,
            )
        else:
            print(
                f"E={epoch} validation_loss={validation['loss']:.4f} "
                f"validation_acc={validation['accuracy']:.2f} "
                f"best_acc={best_validation_accuracy:.2f}@{best_epoch} lr={args.lr}",
                flush=True,
            )
        if args.save_checkpoints:
            save_checkpoint(
                checkpoint_path(run_dir),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                best_validation_accuracy=best_validation_accuracy,
                best_epoch=best_epoch,
                tracker=tracker,
                args_dict={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            )

    test = evaluate(
        model,
        data.test,
        task_type=data.task_type,
        batch_size=args.eval_batch_size,
        max_batches=args.max_eval_batches,
        device=device,
    )
    summary = {
        "run_name": run_name,
        "model": args.model_name,
        "model_revision": args.model_revision,
        "model_source": args.model_source,
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_source": args.dataset_source,
        "task_type": data.task_type,
        "optimizer_name": args.optimizer_name,
        "optimizer_mode": args.optimizer_name,
        "clip_threshold": args.clip_threshold,
        "clipping_scope": args.clipping_scope,
        "correct_bias": args.correct_bias,
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": perplexity(best_validation_loss) if data.task_type == "causal_lm" else None,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_epoch": best_epoch,
        "test_loss": test["loss"],
        "test_perplexity": test["perplexity"] if data.task_type == "causal_lm" else None,
        "test_accuracy": test.get("accuracy"),
        "global_step": global_step,
        "epochs": args.epochs,
        "seed": args.seed,
        "lr": args.lr,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "total_training_steps": total_training_steps,
        "classifier_dropout": args.classifier_dropout,
        "val_check_interval": args.val_check_interval,
        "save_checkpoints": args.save_checkpoints,
        "save_final_model": args.save_final_model,
        "wandb_log_model": args.wandb_log_model,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "vocab_size": data.vocab_size,
        "num_labels": data.num_labels,
        "experiment_tag": getattr(args, "experiment_tag", None),
        "wandb_group": resolved_wandb_group(args),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_job_type": args.wandb_job_type,
        "completed": 1,
        "diagnostics": tracker.summarize(),
    }
    if args.save_final_model:
        torch.save(model.state_dict(), run_dir / "final_model.pt")
    atomic_write_json(summary_path(run_dir), summary)
    if run is not None:
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


@dataclass(frozen=True)
class SweepConfig:
    optimizer_name: str
    lr: float
    clip_threshold: float
    seed: int

    @property
    def config_id(self) -> str:
        clip_tag = str(self.clip_threshold).replace(".", "p")
        return f"llm-{self.optimizer_name}-C{clip_tag}-lr{str(self.lr).replace('.', 'p')}-seed{self.seed}"


def build_sweep_configs(args) -> list[SweepConfig]:
    configs = []
    for optimizer_name in args.optimizer_names:
        thresholds = [float("inf")] if optimizer_name == "adamw_uncut" else args.clip_thresholds
        for lr in args.lrs:
            for clip_threshold in thresholds:
                for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                    configs.append(SweepConfig(optimizer_name, float(lr), float(clip_threshold), seed))
    return configs


def run_sweep(args) -> pd.DataFrame:
    configs = build_sweep_configs(args)
    results = []
    completed = 0
    for config in configs:
        run_args = argparse.Namespace(**vars(args))
        run_args.optimizer_name = config.optimizer_name
        run_args.lr = config.lr
        run_args.clip_threshold = config.clip_threshold
        run_args.seed = config.seed
        dataset_tag = _tag(args.dataset_config or args.dataset_name)
        run_args.run_name = f"{_tag(args.model_name)}-{dataset_tag}-{_tag(args.task_type)}-{config.config_id}"
        summary = run_llm_finetune_experiment(run_args)
        results.append(summary)
        completed += 1
        print(f"[sweep] completed {completed}/{len(configs)}: {config.config_id}", flush=True)
    return pd.DataFrame(results)


def summarize_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    results = results.copy()
    for column in ["lr", "clip_threshold", "best_validation_loss", "best_validation_perplexity", "best_validation_accuracy"]:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    group_cols = [
        "model",
        "dataset",
        "dataset_config",
        "task_type",
        "optimizer_name",
        "clipping_scope",
        "correct_bias",
        "clip_threshold",
        "lr",
    ]
    grouped = results.groupby(group_cols, dropna=False)
    aggregations: dict[str, tuple[str, str]] = {
        "best_loss": ("best_validation_loss", "min"),
        "run_count": ("run_name", "count"),
    }
    if "best_validation_perplexity" in results.columns:
        aggregations["best_perplexity"] = ("best_validation_perplexity", "min")
        aggregations["mean_best_perplexity"] = ("best_validation_perplexity", "mean")
    if "best_validation_accuracy" in results.columns:
        aggregations["best_accuracy"] = ("best_validation_accuracy", "max")
        aggregations["mean_best_accuracy"] = ("best_validation_accuracy", "mean")
    summary = grouped.agg(**aggregations).reset_index()
    return summary.sort_values(group_cols).reset_index(drop=True)
