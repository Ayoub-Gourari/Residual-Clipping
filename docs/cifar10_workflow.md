# CIFAR-10 Workflow

This note summarizes the intended centralized CIFAR-10 workflow for tracked sweeps, grouped runs, and report regeneration.

## 1. Inspect The Registry

List the tracked sweep and report entries:

```bash
python scripts/list_experiments.py
```

The registry file is:

```text
configs/cifar10/registry.yaml
```

## 2. Launch A Sweep

Run a tracked sweep by name:

```bash
python scripts/run_registered_experiment.py --name cifar10-resnet20-sweep -- --resume
```

Useful launch-time overrides include:

- `--wandb-mode online`
- `--wandb-project <project>`
- `--wandb-entity <entity>`
- `--wandb-group <shared-group>`
- `--wandb-job-type sweep`
- `--loader-workers <n>`
- `--output-dir <path>`

## 3. Launch The Three-Machine Comparison

Print one command per optimizer mode:

```bash
python scripts/three_machine_cifar10_commands.py --wandb-group my-group
```

Each printed command:

- launches the same tracked sweep through the registry
- fixes one optimizer mode per machine
- attaches a shared W&B group
- labels the W&B job type as `sweep`

For multi-machine comparisons, use the same explicit `--wandb-group` on all machines.

## 4. Regenerate Reports And Figures

Once sweep outputs are present, regenerate compact CSV summaries and figures:

```bash
python scripts/run_registered_experiment.py --name cifar10-resnet20-report
```

or directly through the report config:

```bash
python scripts/plot_cifar10_config.py --config configs/cifar10/report_resnet20.yaml
```

The report side writes:

- `threshold_summary.csv`
- `run_summaries.csv`
- `run_diagnostics.csv`
- `best_runs.csv`
- `best_trajectory_records.csv`
- `best_trajectory_summary.csv`
- `wandb_context.json`

and the figures:

- `cifar10_best_accuracy_vs_threshold.(png|pdf)`
- `cifar10_best_trajectories.(png|pdf)`

## 5. Resume Behavior

Sweep reruns with `--resume` skip completed run directories when the saved summary indicates the requested epoch count was completed. This is the default way to continue interrupted work without recomputing finished runs.

## 6. Practical Notes

- In restricted environments, set `--loader-workers 0`.
- When W&B logging is enabled and no group is provided, the sweep path derives a neutral default group from the dataset, model scope, beta, and seed range.
- For method comparisons across machines, prefer an explicit shared `--wandb-group` rather than relying on the derived default.
