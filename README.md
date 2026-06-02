# Residual Clipping

Centralized experiment code for studying momentum, clipping, and residual-centered clipping.

## Scope

This repository is organized around two experiment families:

- Synthetic quadratics for lightweight optimization and threshold-sensitivity studies.
- Centralized CIFAR-10 image-classification runs with `resnet20`, `resnet18`, and `vgg16`.

The current codebase includes a resumable synthetic quadratics pipeline and a centralized CIFAR-10 pipeline with common utilities, reproducible CLI conventions, and plotting entrypoints.

## Project Layout

```text
src/residual_clipping/   Shared library code
experiments/             Experiment-family entrypoints
scripts/                 Thin operational wrappers
configs/                 Reproducible config files
tests/                   Unit tests
docs/                    Notes and generated documentation
```

## CLI Conventions

All long-running commands should converge on the same launch-time interface:

- W&B: `--wandb-mode`, `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-run-name`, `--wandb-job-type`, `--wandb-tags`, `--wandb-notes`
- Resume: `--resume`, `--overwrite`, `--run-id`, `--output-dir`

The codebase intentionally keeps W&B identity and grouping details out of tracked defaults. Those settings should come from CLI flags or the shell environment.

When W&B logging is enabled and `--wandb-group` is omitted, the CIFAR-10 sweep path derives a neutral default group from the dataset, model scope, beta, and seed range. Multi-machine runs should still pass an explicit shared `--wandb-group` so all methods land in the same comparison bucket.

## Commands

The repository currently exposes the quadratics and centralized CIFAR-10 pipelines through these commands:

```bash
python -m experiments.quadratics.run --wandb-mode disabled
python -m experiments.quadratics.search --wandb-mode disabled --resume
python -m experiments.quadratics.plot
python -m experiments.cifar10.run --model resnet20 --optimizer-mode sgd_momentum --wandb-mode disabled
python -m experiments.cifar10.sweep --models resnet20,resnet18,vgg16 --wandb-mode disabled --resume
python -m experiments.cifar10.plot
python -m experiments.cifar10.collect
python scripts/three_machine_cifar10_commands.py --wandb-group my-group
python scripts/run_cifar10_config.py --config configs/cifar10/resnet20_sweep.yaml -- --resume
python scripts/plot_cifar10_config.py --config configs/cifar10/report_resnet20.yaml
python scripts/list_experiments.py
python scripts/run_registered_experiment.py --name cifar10-resnet20-sweep -- --resume
```

## Development

```bash
python -m compileall src experiments scripts
pytest
```
