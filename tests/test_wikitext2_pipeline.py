import json
import subprocess
import sys
from pathlib import Path

import torch

from residual_clipping.wikitext2_data import batchify, get_batch, load_corpus
from residual_clipping.wikitext2_models import make_lstm_language_model


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
