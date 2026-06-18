"""Smoke-test all LLM AdamW optimizer variants on a tiny local classifier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from residual_clipping.adaptive_optimizers import ADAPTIVE_OPTIMIZER_NAMES
from residual_clipping.llm_finetune_pipeline import run_llm_finetune_experiment


REQUIRED_LOG_KEYS = {
    "train/optimizer_name",
    "train/clip_threshold",
    "train/clipping_scope",
    "train/grad_global_norm",
    "train/pseudo_grad_global_norm",
    "train/clipping_scale",
    "train/residual_global_norm",
    "train/metric_residual_global_norm",
    "train/update_global_norm",
    "train/adam_m_global_norm",
    "train/adam_v_global_norm",
    "train/learning_rate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local smoke checks for all LLM optimizer variants.")
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/residual-clipping-llm-optimizer-smoke"))
    parser.add_argument("--clip-threshold", type=float, default=0.5)
    return parser


def make_args(output_dir: Path, optimizer_name: str, clip_threshold: float) -> argparse.Namespace:
    return argparse.Namespace(
        wandb_mode="disabled",
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_run_name=None,
        wandb_job_type=None,
        wandb_tags=None,
        wandb_notes=None,
        resume=False,
        overwrite=True,
        run_id=None,
        output_dir=str(output_dir),
        task_type="sequence_classification",
        model_source="fake",
        model_name="tiny-sequence-classifier",
        model_revision=None,
        dataset_source="fake",
        dataset_name="fake_rte",
        dataset_config=None,
        train_split="train",
        validation_split="validation",
        test_split="validation",
        text_column="text",
        sentence1_column="sentence1",
        sentence2_column="sentence2",
        label_column="label",
        num_labels=2,
        train_file=None,
        validation_file=None,
        data_dir=output_dir / "datasets",
        hf_cache_dir=None,
        download=False,
        optimizer_name=optimizer_name,
        seed=0,
        use_cuda=False,
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        sequence_length=8,
        lr=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_eps=1e-8,
        weight_decay=0.01,
        clip_threshold=clip_threshold,
        clipping_scope="local",
        correct_bias=False,
        dropout=0.0,
        classifier_dropout=0.0,
        log_interval=1,
        val_check_interval=None,
        max_train_batches=2,
        max_eval_batches=1,
        max_train_sequences=None,
        max_eval_sequences=None,
        fake_train_sequences=8,
        fake_eval_sequences=4,
        fake_vocab_size=16,
        fake_hidden_size=8,
        fake_num_layers=1,
        run_name=f"smoke-{optimizer_name}",
        save_checkpoints=False,
        save_final_model=False,
        wandb_log_model=False,
        experiment_tag=None,
    )


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for optimizer_name in ADAPTIVE_OPTIMIZER_NAMES:
        threshold = float("inf") if optimizer_name == "adamw_uncut" else args.clip_threshold
        run_args = make_args(args.output_dir, optimizer_name, threshold)
        summary = run_llm_finetune_experiment(run_args)
        if summary["completed"] != 1:
            raise RuntimeError(f"Smoke run did not complete for {optimizer_name}.")
        metrics_path = args.output_dir / f"smoke-{optimizer_name}" / "metrics.jsonl"
        train_rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "train/loss" in json.loads(line)
        ]
        if not train_rows:
            raise RuntimeError(f"No train metrics found for {optimizer_name}.")
        missing = sorted(REQUIRED_LOG_KEYS - set(train_rows[-1]))
        if missing:
            raise RuntimeError(f"Missing log keys for {optimizer_name}: {', '.join(missing)}")
        for key, value in train_rows[-1].items():
            if isinstance(value, (int, float)) and math.isnan(float(value)):
                raise RuntimeError(f"NaN log value for {optimizer_name}: {key}")
        print(f"{optimizer_name}: ok", flush=True)


if __name__ == "__main__":
    main()
