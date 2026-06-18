# Residual Clipping

Centralized experiment code for studying momentum, clipping, and residual-centered clipping.

## Scope

This repository is organized around two experiment families:

- Synthetic quadratics for lightweight optimization and threshold-sensitivity studies.
- Centralized CIFAR image-classification runs, including CIFAR-10 with `resnet20`, `resnet18`, and `vgg16`, plus CIFAR-100 with `vgg16`.
- WikiText-2 word-level language modeling with a tied-weight 2-layer LSTM.
- Storage-conscious ALBERT-base-v2 fine-tuning on GLUE/RTE with AdamW clipping variants.

The current codebase includes a resumable synthetic quadratics pipeline and a centralized CIFAR pipeline with common utilities, reproducible CLI conventions, and plotting entrypoints.

## Reproducibility Principles

- Runs are config-driven and support resume/skip behavior for interrupted work.
- W&B identity, grouping, and notes are always supplied at launch time rather than embedded in tracked code.
- Heavy outputs, checkpoints, and local planning notes stay out of git.
- The tracked config and registry files define the intended reproducible surface.

## Project Layout

```text
src/residual_clipping/   Shared library code
experiments/             Experiment-family entrypoints
scripts/                 Thin operational wrappers
configs/                 Reproducible config files
tests/                   Unit tests
docs/                    Notes and generated documentation
```

## What To Run First

For a first pass through the repository:

1. Run a small quadratics smoke command to confirm the local environment is healthy.
2. Run a named CIFAR-10 sweep from the registry with local overrides.
3. Regenerate the corresponding CIFAR-10 report artifacts and figures.

The registry-driven path is the easiest place to start because it keeps the tracked config names visible:

```bash
python scripts/list_experiments.py
python scripts/run_registered_experiment.py --name cifar10-resnet20-sweep -- --resume
python scripts/run_registered_experiment.py --name cifar10-resnet20-report
```

## CLI Conventions

All long-running commands should converge on the same launch-time interface:

- W&B: `--wandb-mode`, `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-run-name`, `--wandb-job-type`, `--wandb-tags`, `--wandb-notes`
- Resume: `--resume`, `--overwrite`, `--run-id`, `--output-dir`

The codebase intentionally keeps W&B identity and grouping details out of tracked defaults. Those settings should come from CLI flags or the shell environment.

When W&B logging is enabled and `--wandb-group` is omitted, the CIFAR sweep path derives a neutral default group from the dataset, model scope, beta, and seed range. Multi-machine runs should still pass an explicit shared `--wandb-group` so all methods land in the same comparison bucket.

## Main Workflows

### Quadratics

```bash
python -m experiments.quadratics.run --wandb-mode disabled
python -m experiments.quadratics.search --wandb-mode disabled --resume
python -m experiments.quadratics.plot
```

### CIFAR-10: Single Run

```bash
python -m experiments.cifar10.run --model resnet20 --optimizer-mode sgd_momentum --wandb-mode disabled
```

### CIFAR-10: Sweep From Tracked Config

```bash
python scripts/run_cifar10_config.py --config configs/cifar10/resnet20_sweep.yaml -- --resume
```

### CIFAR-10: Sweep By Registry Name

```bash
python scripts/list_experiments.py
python scripts/run_registered_experiment.py --name cifar10-resnet20-sweep -- --resume
```

### CIFAR-100: VGG-16 Sweep By Registry Name

```bash
python scripts/run_registered_experiment.py --name cifar100-vgg16-sweep -- --resume
```

### CIFAR-10: Report Collection And Plotting

```bash
python -m experiments.cifar10.collect
python -m experiments.cifar10.plot
python scripts/plot_cifar10_config.py --config configs/cifar10/report_resnet20.yaml
```

### CIFAR-100: VGG-16 Report Collection And Plotting

```bash
python scripts/run_registered_experiment.py --name cifar100-vgg16-report
```

### WikiText-2: 2-Layer LSTM Sweep

```bash
python scripts/run_registered_experiment.py --name wikitext2-lstm-sweep -- --resume --download
```

### WikiText-2: 2-Layer LSTM Report Collection And Plotting

```bash
python scripts/run_registered_experiment.py --name wikitext2-lstm-report
```

### WikiText-2: BPTT-70 LSTM Sweep

This variant keeps the same model and clipping grids, uses learning rates `30,40,50`,
and writes to separate run, report, figure, and W&B namespaces.

```bash
python scripts/run_registered_experiment.py --name wikitext2-lstm-bptt70-sweep -- --resume --download
python scripts/run_registered_experiment.py --name wikitext2-lstm-bptt70-report
```

### ALBERT Base v2: RTE Fine-Tuning

The ALBERT/RTE path compares exactly:

- `adamw_uncut`
- `adamw_clip`
- `adamw_resclip_euclidean`
- `adamw_resclip_metric`

It defaults to scalar-only logs, no checkpoints, no final model save, and no W&B model artifacts. Use external caches when storage is tight:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DIR="$HOME/.cache/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
```

Warm the ALBERT/RTE cache once before launching runs from several SSH sessions:

```bash
python scripts/prefetch_albert_rte.py --cache-dir "$HF_HOME"
```

Then reuse the same cache from every session. Add `--no-download` after
prefetching if you want training runs to fail fast instead of touching the
network.

Smoke-test the optimizer implementation:

```bash
python scripts/smoke_test_optimizers.py
```

Run the four stage-1 RTE comparisons:

```bash
python train.py --task rte --model_checkpoint albert-base-v2 --hf-cache-dir "$HF_HOME" --no-download --optimizer_name adamw_uncut --clip_threshold inf --lr 1e-5 --batch_size 8 --max_epochs 1 --seed 123 --save_checkpoints false --wandb_log_model false

python train.py --task rte --model_checkpoint albert-base-v2 --hf-cache-dir "$HF_HOME" --no-download --optimizer_name adamw_clip --clipping_scope local --clip_threshold 1.0 --lr 1e-5 --batch_size 8 --max_epochs 1 --seed 123 --save_checkpoints false --wandb_log_model false

python train.py --task rte --model_checkpoint albert-base-v2 --hf-cache-dir "$HF_HOME" --no-download --optimizer_name adamw_resclip_euclidean --clipping_scope local --clip_threshold 1.0 --lr 1e-5 --batch_size 8 --max_epochs 1 --seed 123 --save_checkpoints false --wandb_log_model false

python train.py --task rte --model_checkpoint albert-base-v2 --hf-cache-dir "$HF_HOME" --no-download --optimizer_name adamw_resclip_metric --clipping_scope local --clip_threshold 1.0 --lr 1e-5 --batch_size 8 --max_epochs 1 --seed 123 --save_checkpoints false --wandb_log_model false
```

Scripted runs:

```bash
bash scripts/run_albert_rte_stage1.sh
bash scripts/run_albert_rte_threshold_sensitivity.sh
```

Chezhegov-style ALBERT/RTE reproduction:

```bash
python scripts/smoke_test_optimizers.py
bash scripts/run_albert_rte_reproduction_stage1.sh
bash scripts/run_albert_rte_reproduction_seeds.sh
python scripts/aggregate_rte_reproduction.py
```

### CIFAR-10: Three-Machine Grouped Workflow

Use the helper to print one launch command per method. All three machines should share the same explicit `--wandb-group`.

```bash
python scripts/three_machine_cifar10_commands.py --wandb-group my-group
```

The generated commands route through the registry-backed sweep launcher and pin each machine to one method.

## Expected CIFAR Artifacts

Sweep runs write per-run directories containing:

- `metrics.jsonl`
- `summary.json`
- `checkpoint_latest.pt`

Sweep-level summaries live under the sweep output root in:

- `cifar10_sweeps/run_summaries.csv` for CIFAR-10
- `cifar100_sweeps/run_summaries.csv` for CIFAR-100
- `<dataset>_sweeps/threshold_summary.csv`

Report collection writes compact analysis artifacts such as:

- `run_diagnostics.csv`
- `best_runs.csv`
- `best_trajectory_records.csv`
- `best_trajectory_summary.csv`
- `wandb_context.json`

Figure generation writes:

- `cifar10_best_accuracy_vs_threshold.(png|pdf)` or `cifar100_best_accuracy_vs_threshold.(png|pdf)`
- `cifar10_best_trajectories.(png|pdf)` or `cifar100_best_trajectories.(png|pdf)`
- `wikitext2_lstm_best_perplexity_vs_threshold.(png|pdf)`
- `wikitext2_lstm_best_trajectories.(png|pdf)`

## Registry And Configs

Tracked CIFAR sweep and report entries live in [configs/cifar10/registry.yaml](configs/cifar10/registry.yaml).

Tracked model sweep configs:

- [configs/cifar10/resnet20_sweep.yaml](configs/cifar10/resnet20_sweep.yaml)
- [configs/cifar10/resnet18_sweep.yaml](configs/cifar10/resnet18_sweep.yaml)
- [configs/cifar10/vgg16_sweep.yaml](configs/cifar10/vgg16_sweep.yaml)
- [configs/cifar10/all_models_sweep.yaml](configs/cifar10/all_models_sweep.yaml)
- [configs/cifar100/vgg16_sweep.yaml](configs/cifar100/vgg16_sweep.yaml)
- [configs/wikitext2/lstm_sweep.yaml](configs/wikitext2/lstm_sweep.yaml)
- [configs/wikitext2/lstm_bptt70_sweep.yaml](configs/wikitext2/lstm_bptt70_sweep.yaml)

Tracked report configs:

- [configs/cifar10/report_resnet20.yaml](configs/cifar10/report_resnet20.yaml)
- [configs/cifar10/report_all_models.yaml](configs/cifar10/report_all_models.yaml)
- [configs/cifar100/report_vgg16.yaml](configs/cifar100/report_vgg16.yaml)
- [configs/wikitext2/report_lstm.yaml](configs/wikitext2/report_lstm.yaml)
- [configs/wikitext2/report_lstm_bptt70.yaml](configs/wikitext2/report_lstm_bptt70.yaml)

## Development

```bash
python -m compileall src experiments scripts
pytest
```

See [docs/cifar10_workflow.md](docs/cifar10_workflow.md) for a concise CIFAR-10 sweep and reporting walkthrough.
