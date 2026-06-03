import json
import subprocess
import sys
from pathlib import Path

import torch

from residual_clipping.wikitext2_data import batchify, get_batch, load_corpus
from residual_clipping.wikitext2_models import make_lstm_language_model


def write_fake_wikitext_run(
    run_dir: Path,
    *,
    run_name: str,
    optimizer_mode: str,
    best_perplexity: float,
    clip_c: float | None = None,
    clip_c_res: float | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        {"train/global_step": 0, "validation/loss": 5.4, "validation/perplexity": best_perplexity + 5.0},
        {
            "train/global_step": 1,
            "train/loss": 5.1,
            "train/perplexity": 164.0,
            "train/grad_norm": 0.2,
            "train/lr": 10.0,
        },
        {
            "train/global_step": 2,
            "validation/loss": 4.8,
            "validation/perplexity": best_perplexity,
            "validation/best_loss": 4.8,
            "validation/best_perplexity": best_perplexity,
            "train/lr": 10.0,
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in metrics) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "dataset": "wikitext2",
                "model": "lstm",
                "optimizer_mode": optimizer_mode,
                "best_validation_loss": 4.8,
                "best_validation_perplexity": best_perplexity,
                "test_perplexity": best_perplexity + 1.0,
                "clip_c": clip_c,
                "clip_c_res": clip_c_res,
                "wandb_group": "demo-group",
                "wandb_project": "demo-project",
                "wandb_job_type": "sweep",
                "diagnostics": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_fake_wikitext2_corpus_batches_and_tied_lstm(tmp_path: Path):
    corpus = load_corpus(
        "fake_wikitext2",
        tmp_path,
        download=False,
        fake_token_count=120,
        fake_vocab_size=12,
    )
    data = batchify(corpus.train, batch_size=4, device=torch.device("cpu"))
    inputs, targets = get_batch(data, index=0, bptt=5)
    model = make_lstm_language_model(
        vocab_size=corpus.vocab_size,
        embedding_size=8,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
        tie_weights=True,
    )
    hidden = model.init_hidden(batch_size=4, device=torch.device("cpu"))
    output, _ = model(inputs, hidden)

    assert output.shape == torch.Size([targets.numel(), corpus.vocab_size])
    assert model.decoder.weight is model.encoder.weight


def test_registered_wikitext2_sweep_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_registered_experiment.py",
            "--name",
            "wikitext2-lstm-sweep",
            "--",
            "--dataset",
            "fake_wikitext2",
            "--optimizer-mode",
            "sgd_momentum",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--eval-batch-size",
            "2",
            "--bptt",
            "5",
            "--embedding-size",
            "8",
            "--hidden-size",
            "8",
            "--dropout",
            "0.1",
            "--lrs",
            "1.0",
            "--fake-token-count",
            "120",
            "--fake-vocab-size",
            "12",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        cwd=repo_root,
        check=True,
    )
    summary_path = output_dir / "lstm-sgd_momentum-lr1p0-b0p9-seed0" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["dataset"] == "fake_wikitext2"
    assert summary["model"] == "lstm"
    assert summary["beta"] == 0.9
    assert (output_dir / "fake_wikitext2_sweeps" / "run_summaries.csv").exists()


def test_registered_wikitext2_report_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    runs_root = tmp_path / "runs"
    sweep_dir = tmp_path / "sweeps"
    report_dir = tmp_path / "reports"
    figure_dir = tmp_path / "figures"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("lstm-sgd", "sgd_momentum", 130.0, None, None),
        ("lstm-clipped", "clipped_momentum", 125.0, 0.3, None),
        ("lstm-residual", "residual_clipped_momentum", 120.0, None, 0.1),
    ]
    rows = []
    for run_name, optimizer_mode, best_perplexity, clip_c, clip_c_res in runs:
        write_fake_wikitext_run(
            runs_root / run_name,
            run_name=run_name,
            optimizer_mode=optimizer_mode,
            best_perplexity=best_perplexity,
            clip_c=clip_c,
            clip_c_res=clip_c_res,
        )
        rows.append(
            {
                "run_name": run_name,
                "dataset": "wikitext2",
                "model": "lstm",
                "optimizer_mode": optimizer_mode,
                "best_validation_loss": 4.8,
                "best_validation_perplexity": best_perplexity,
                "clip_c": "" if clip_c is None else clip_c,
                "clip_c_res": "" if clip_c_res is None else clip_c_res,
                "diagnostics": "{}",
            }
        )
    header = list(rows[0].keys())
    (sweep_dir / "run_summaries.csv").write_text(
        ",".join(header)
        + "\n"
        + "\n".join(",".join(str(row[key]) for key in header) for row in rows)
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_registered_experiment.py",
            "--name",
            "wikitext2-lstm-report",
            "--",
            "--sweep-dir",
            str(sweep_dir),
            "--runs-root",
            str(runs_root),
            "--report-dir",
            str(report_dir),
            "--figure-dir",
            str(figure_dir),
        ],
        cwd=repo_root,
        check=True,
    )

    assert (report_dir / "best_runs.csv").exists()
    assert (report_dir / "wandb_context.json").exists()
    assert (figure_dir / "wikitext2_lstm_best_perplexity_vs_threshold.png").exists()
    assert (figure_dir / "wikitext2_lstm_best_trajectories.pdf").exists()
