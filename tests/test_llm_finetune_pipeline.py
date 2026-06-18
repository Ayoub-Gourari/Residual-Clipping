import argparse
import json

from residual_clipping.llm_finetune_pipeline import run_llm_finetune_experiment, run_sweep, sweep_output_dir


def make_args(tmp_path, **overrides):
    values = {
        "wandb_mode": "disabled",
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_group": None,
        "wandb_run_name": None,
        "wandb_job_type": None,
        "wandb_tags": None,
        "wandb_notes": None,
        "resume": False,
        "overwrite": False,
        "run_id": None,
        "output_dir": str(tmp_path),
        "task_type": "causal_lm",
        "model_source": "fake",
        "model_name": "tiny-causal-lm",
        "dataset_source": "fake",
        "dataset_name": "fake_causal_lm",
        "dataset_config": None,
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "text_column": "text",
        "sentence1_column": "sentence1",
        "sentence2_column": "sentence2",
        "label_column": "label",
        "num_labels": 2,
        "train_file": None,
        "validation_file": None,
        "data_dir": tmp_path / "datasets",
        "download": False,
        "optimizer_name": "adamw_uncut",
        "clip_threshold": float("inf"),
        "seed": 0,
        "use_cuda": False,
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "sequence_length": 8,
        "lr": 0.001,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "log_interval": 1,
        "max_train_batches": 2,
        "max_eval_batches": 1,
        "max_train_sequences": None,
        "max_eval_sequences": None,
        "fake_train_sequences": 6,
        "fake_eval_sequences": 4,
        "fake_vocab_size": 16,
        "fake_hidden_size": 8,
        "fake_num_layers": 1,
        "run_name": "fake-llm-smoke",
        "experiment_tag": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fake_llm_finetune_writes_summary_and_metrics(tmp_path):
    args = make_args(tmp_path)

    summary = run_llm_finetune_experiment(args)

    run_dir = tmp_path / "fake-llm-smoke"
    assert summary["completed"] == 1
    assert summary["optimizer_name"] == "adamw_uncut"
    assert summary["global_step"] == 2
    assert (run_dir / "metrics.jsonl").exists()
    written_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert written_summary["run_name"] == "fake-llm-smoke"
    assert "update_global_norm_mean" in written_summary["diagnostics"]


def test_fake_llm_sweep_writes_expected_rows(tmp_path):
    args = make_args(
        tmp_path,
        optimizer_names=["adamw_uncut"],
        clip_thresholds=[float("inf")],
        lrs=[0.001, 0.0005],
        seed_start=0,
        num_seeds=1,
        run_name=None,
    )

    results = run_sweep(args)

    assert len(results) == 2
    assert sorted(results["lr"].tolist()) == [0.0005, 0.001]
    assert sweep_output_dir(tmp_path, "fake_causal_lm") == tmp_path / "fake_causal_lm_llm_sweeps"


def test_fake_sequence_classification_logs_accuracy(tmp_path):
    args = make_args(
        tmp_path,
        task_type="sequence_classification",
        run_name="fake-rte-smoke",
        sequence_length=8,
        fake_train_sequences=8,
        fake_eval_sequences=4,
    )

    summary = run_llm_finetune_experiment(args)

    assert summary["task_type"] == "sequence_classification"
    assert summary["best_validation_accuracy"] is not None
    assert summary["test_accuracy"] is not None
    assert summary["num_labels"] == 2
