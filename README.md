# Residual Clipping

Centralized experiment code for studying momentum, clipping, and residual-centered clipping.

## Scope

This repository is organized around two experiment families:

- Synthetic quadratics for lightweight optimization and threshold-sensitivity studies.
- Centralized CIFAR-10 image-classification runs with `resnet20`, `resnet18`, and `vgg16`.

The current repository foundation focuses on shared structure, common utilities, and reproducible CLI conventions. The experiment implementations will be added on top of this base.

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

- W&B: `--wandb-mode`, `--wandb-project`, `--wandb-entity`, `--wandb-group`, `--wandb-run-name`
- Resume: `--resume`, `--overwrite`, `--run-id`, `--output-dir`

The codebase intentionally keeps W&B identity and grouping details out of tracked defaults. Those settings should come from CLI flags or the shell environment.

## Planned Commands

The concrete experiment implementations will plug into these commands:

```bash
python -m experiments.quadratics.run --wandb-mode disabled
python -m experiments.cifar10.run --model resnet20 --optimizer-mode sgd_momentum --wandb-mode disabled
```

## Development

```bash
python -m compileall src experiments scripts
pytest
```
