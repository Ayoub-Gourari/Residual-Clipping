"""Training, checkpointing, and sweep helpers for centralized CIFAR experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .cifar10_data import dataset_num_classes, make_dataloaders
from .cifar10_models import get_model
from .clipping import clip_tensor_list, tensor_list_global_norm
from .logging_utils import append_jsonl, atomic_write_json
from .randomness import capture_rng_state, restore_rng_state, set_global_seed


OPTIMIZER_MODES = ("sgd_momentum", "clipped_momentum", "residual_clipped_momentum")


def top1_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return 100.0 * predictions.eq(target).float().mean().item()


def parse_wandb_tags(value: str | None) -> list[str] | None:
    if value is None:
        return None
    tags = [item.strip() for item in value.split(",") if item.strip()]
    return tags or None


def parse_milestones(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    return sorted(int(item.strip()) for item in value.split(",") if item.strip())


def current_lr(base_lr: float, epoch: int, milestones: list[int], gamma: float) -> float:
    drops = sum((epoch - 1) >= milestone for milestone in milestones)
    return base_lr * (gamma ** drops)


def tensor_list_dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    if not left:
        return torch.tensor(0.0)
    total = torch.zeros((), device=left[0].device)
    for lhs, rhs in zip(left, right):
        total = total + (lhs.detach() * rhs.detach()).sum()
    return total


def tensor_list_sub(left: list[torch.Tensor], right: list[torch.Tensor]) -> list[torch.Tensor]:
    return [lhs - rhs for lhs, rhs in zip(left, right)]


def tensor_list_add(left: list[torch.Tensor], right: list[torch.Tensor]) -> list[torch.Tensor]:
    return [lhs + rhs for lhs, rhs in zip(left, right)]


def clipping_error_norm(center: list[torch.Tensor], update: list[torch.Tensor], target: list[torch.Tensor]) -> float:
    error = tensor_list_sub(tensor_list_add(center, update), target)
    return tensor_list_global_norm(error).item()


def collect_gradients(params: list[torch.nn.Parameter], weight_decay: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    data_grads = []
    update_grads = []
    for param in params:
        if param.grad is None:
            grad = torch.zeros_like(param)
        else:
            grad = param.grad.detach().clone()
        data_grads.append(grad.detach().clone())
        if weight_decay != 0:
            grad = grad.add(param.detach(), alpha=weight_decay)
        update_grads.append(grad.detach().clone())
    return update_grads, data_grads


class RunningDiagnostics:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {}

    def update(self, diagnostics: dict[str, Any]) -> None:
        for key, value in diagnostics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                self.values.setdefault(key, []).append(float(value))

    def summarize(self) -> dict[str, float]:
        summary = {}
        for key, values in self.values.items():
            array = np.asarray(values, dtype=np.float64)
            summary[f"{key}_mean"] = float(array.mean())
            summary[f"{key}_median"] = float(np.median(array))
        return summary


class MomentumClipper:
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        mode: str,
        beta: float,
        clip_c: float | None,
        clip_c_res: float | None,
    ) -> None:
        if mode == "clipped_momentum" and clip_c is None:
            raise ValueError("--clip-c is required for clipped_momentum.")
        if mode == "residual_clipped_momentum" and clip_c_res is None:
            raise ValueError("--clip-c-res is required for residual_clipped_momentum.")
        self.params = list(params)
        self.mode = mode
        self.beta = beta
        self.clip_c = clip_c
        self.clip_c_res = clip_c_res
        self.momentum = [torch.zeros_like(param) for param in self.params]

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "beta": self.beta,
            "clip_c": self.clip_c,
            "clip_c_res": self.clip_c_res,
            "momentum": [value.detach().cpu() for value in self.momentum],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        values = state_dict["momentum"]
        self.momentum = [tensor.to(param.device) for tensor, param in zip(values, self.params)]

    def _base_diagnostics(self, grads: list[torch.Tensor], momentum_before: list[torch.Tensor], grad_norm: float) -> dict[str, float]:
        momentum_norm = tensor_list_global_norm(momentum_before).item()
        residuals = tensor_list_sub(grads, momentum_before)
        residual_norm = tensor_list_global_norm(residuals).item()
        cosine = tensor_list_dot(grads, momentum_before).item() / (grad_norm * momentum_norm + 1e-12)
        return {
            "grad_norm": grad_norm,
            "momentum_norm": momentum_norm,
            "residual_norm": residual_norm,
            "residual_to_grad_norm_ratio": residual_norm / (grad_norm + 1e-12),
            "cos_grad_momentum": cosine,
        }

    def step(self, grads: list[torch.Tensor], lr: float) -> dict[str, float]:
        momentum_before = [value.detach().clone() for value in self.momentum]
        grad_norm = tensor_list_global_norm(grads).item()
        diagnostics = self._base_diagnostics(grads, momentum_before, grad_norm)

        if self.mode == "sgd_momentum":
            self.momentum = [
                self.beta * momentum + (1.0 - self.beta) * grad
                for momentum, grad in zip(momentum_before, grads)
            ]
        elif self.mode == "clipped_momentum":
            clipped_grad, _, scale = clip_tensor_list(grads, self.clip_c)
            diagnostics.update(
                {
                    "clip_input_norm": grad_norm,
                    "clipped_object_norm": tensor_list_global_norm(clipped_grad).item(),
                    "clip_threshold": float(self.clip_c),
                    "clip_active": float(grad_norm > float(self.clip_c)),
                    "clip_fraction": scale,
                    "standard_clip_fraction": scale,
                    "standard_clip_input_norm": grad_norm,
                    "standard_clipped_grad_norm": tensor_list_global_norm(clipped_grad).item(),
                }
            )
            self.momentum = [
                self.beta * momentum + (1.0 - self.beta) * grad
                for momentum, grad in zip(momentum_before, clipped_grad)
            ]
        elif self.mode == "residual_clipped_momentum":
            residuals = tensor_list_sub(grads, momentum_before)
            clipped_residuals, residual_norm, scale = clip_tensor_list(residuals, self.clip_c_res)
            diagnostics.update(
                {
                    "clip_input_norm": residual_norm,
                    "clipped_object_norm": tensor_list_global_norm(clipped_residuals).item(),
                    "clip_threshold": float(self.clip_c_res),
                    "clip_active": float(residual_norm > float(self.clip_c_res)),
                    "clip_fraction": scale,
                    "residual_clip_fraction": scale,
                    "residual_clip_input_norm": residual_norm,
                    "residual_clipped_residual_norm": tensor_list_global_norm(clipped_residuals).item(),
                    "residual_clipping_error_norm": clipping_error_norm(momentum_before, clipped_residuals, grads),
                    "cos_grad_center": diagnostics["cos_grad_momentum"],
                }
            )
            self.momentum = [
                momentum + (1.0 - self.beta) * residual
                for momentum, residual in zip(momentum_before, clipped_residuals)
            ]
        else:
            raise ValueError(f"Unknown optimizer_mode={self.mode}")

        with torch.no_grad():
            for param, momentum in zip(self.params, self.momentum):
                param.add_(momentum, alpha=-lr)
        diagnostics["next_momentum_norm"] = tensor_list_global_norm(self.momentum).item()
        return diagnostics


def maybe_init_wandb(args, run_name: str):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; skipping W&B logging.")
        return None

    return wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=resolved_wandb_group(args),
        name=args.wandb_run_name or run_name,
        id=args.run_id,
        job_type=args.wandb_job_type,
        tags=parse_wandb_tags(args.wandb_tags),
        notes=args.wandb_notes,
        resume="allow" if args.resume else None,
        config={**vars(args), "num_training_workers": 1},
    )


def log_wandb(payload: dict[str, Any]) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        step = payload.get("train/global_step", payload.get("validation/global_step"))
        wandb.log(payload, step=step)


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_latest.pt"


def summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def metrics_path(run_dir: Path) -> Path:
    return run_dir / "metrics.jsonl"


def sweep_output_dir(output_dir: str | Path, dataset_name: str) -> Path:
    return Path(output_dir) / f"{dataset_name}_sweeps"


def resolved_run_name(args) -> str:
    if args.run_name is not None:
        return args.run_name
    parts = [
        args.model,
        args.optimizer_mode,
    ]
    if args.clip_c is not None:
        parts.append(f"C{str(args.clip_c).replace('.', 'p')}")
    if args.clip_c_res is not None:
        parts.append(f"Cres{str(args.clip_c_res).replace('.', 'p')}")
    parts.extend(
        [
            f"lr{str(args.lr).replace('.', 'p')}",
            f"b{str(args.beta).replace('.', 'p')}",
            f"seed{args.seed}",
        ]
    )
    return args.wandb_run_name or "-".join(parts)


def default_wandb_group(args) -> str:
    model_part = getattr(args, "model", None)
    if model_part is None and hasattr(args, "models"):
        models = list(getattr(args, "models"))
        model_part = "multi-model" if len(models) > 1 else models[0]
    dataset = getattr(args, "dataset", "dataset")
    beta = getattr(args, "beta", None)
    seed_start = getattr(args, "seed_start", getattr(args, "seed", 0))
    num_seeds = getattr(args, "num_seeds", 1)
    parts = [str(dataset), str(model_part or "model")]
    if beta is not None:
        parts.append(f"b{str(beta).replace('.', 'p')}")
    parts.append(f"seeds{seed_start}-{seed_start + num_seeds - 1}")
    return "-".join(parts)


def resolved_wandb_group(args) -> str | None:
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    return args.wandb_group or default_wandb_group(args)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: MomentumClipper,
    epoch: int,
    global_step: int,
    best_accuracy: float,
    best_epoch: int,
    tracker: RunningDiagnostics,
    clip_stats: dict[str, float],
    args_dict: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "tracker_values": tracker.values,
        "clip_stats": clip_stats,
        "rng_state": capture_rng_state(),
        "args": args_dict,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: MomentumClipper,
    tracker: RunningDiagnostics,
):
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
    data_loader,
    loss_func: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc_weighted = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data = data.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            output = model(data)
            loss = loss_func(output, target)
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_acc_weighted += top1_accuracy(output, target) * batch_size
            total += batch_size
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_acc_weighted / max(total, 1),
        "length": total,
    }


def train_one_epoch(
    *,
    args,
    model: nn.Module,
    train_loader,
    loss_func: nn.Module,
    device: torch.device,
    optimizer: MomentumClipper,
    epoch: int,
    lr: float,
    global_step: int,
    tracker: RunningDiagnostics,
    clip_stats: dict[str, float],
    run_dir: Path,
) -> int:
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        data = data.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        grads, data_grads = collect_gradients(params, args.weight_decay)
        diagnostics = optimizer.step(grads, lr)
        diagnostics["data_grad_norm"] = tensor_list_global_norm(data_grads).item()
        tracker.update(diagnostics)
        if "clip_active" in diagnostics:
            clip_stats["total"] += 1
            clip_stats["active"] += diagnostics["clip_active"]

        global_step += 1
        if batch_idx % args.log_interval == 0:
            payload = {
                "train/global_step": global_step,
                "train/epoch": epoch,
                "train/batch_idx": batch_idx,
                "train/loss": loss.item(),
                "train/accuracy": top1_accuracy(output.detach(), target),
                "train/lr": lr,
            }
            payload.update({f"train/{key}": value for key, value in diagnostics.items()})
            if clip_stats["total"] > 0:
                payload["train/clip_active_percent"] = 100.0 * clip_stats["active"] / clip_stats["total"]
            append_jsonl(metrics_path(run_dir), payload)
            log_wandb(payload)
            print(
                " ".join(
                    [
                        f"E={epoch}",
                        f"B={batch_idx}",
                        f"step={global_step}",
                        f"loss={loss.item():.4f}",
                        f"acc={payload['train/accuracy']:.2f}",
                        f"grad_norm={diagnostics['grad_norm']:.4f}",
                        f"momentum_norm={diagnostics['momentum_norm']:.4f}",
                    ]
                ),
                flush=True,
            )
    return global_step


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    atomic_write_json(summary_path(run_dir), summary)


def summary_indicates_complete(run_dir: Path, epochs: int) -> bool:
    path = summary_path(run_dir)
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return int(summary.get("epochs", -1)) >= epochs and int(summary.get("completed", 0)) == 1


def run_cifar10_experiment(args) -> dict[str, Any]:
    if args.use_cuda and not torch.cuda.is_available():
        print("Requested CUDA, but no CUDA device is available; using CPU.", flush=True)
        args.use_cuda = False
    device = torch.device("cuda" if args.use_cuda else "cpu")
    pin_memory = device.type == "cuda"

    run_name = resolved_run_name(args)
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and args.resume and summary_indicates_complete(run_dir, args.epochs):
        summary = json.loads(summary_path(run_dir).read_text(encoding="utf-8"))
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

    set_global_seed(args.seed)
    train_loader, test_loader = make_dataloaders(
        args.dataset,
        args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        download=args.download,
        loader_workers=args.loader_workers,
        pin_memory=pin_memory,
        fake_train_size=args.fake_train_size,
        fake_test_size=args.fake_test_size,
    )
    model = get_model(args.model, num_classes=dataset_num_classes(args.dataset)).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = MomentumClipper(params, args.optimizer_mode, args.beta, args.clip_c, args.clip_c_res)
    tracker = RunningDiagnostics()
    clip_stats = {"active": 0.0, "total": 0.0}
    global_step = 0
    best_accuracy = 0.0
    best_epoch = 0
    start_epoch = 1
    initial_payload = None

    if args.resume and checkpoint_path(run_dir).exists():
        checkpoint = load_checkpoint(
            checkpoint_path(run_dir),
            model=model,
            optimizer=optimizer,
            tracker=tracker,
        )
        global_step = int(checkpoint["global_step"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        clip_stats = checkpoint.get("clip_stats", clip_stats)
        print(f"Resuming {run_name} from epoch {start_epoch}", flush=True)
    else:
        initial_payload = {
            "train/global_step": 0,
            "validation/global_step": 0,
            "validation/epoch": 0,
            **{f"run/{key}": value for key, value in vars(args).items() if isinstance(value, (int, float, str, bool)) or value is None},
        }
        initial_eval = evaluate(
            model, test_loader, loss_func, device, max_batches=args.max_test_batches
        )
        initial_payload.update(
            {
                "validation/loss": initial_eval["loss"],
                "validation/accuracy": initial_eval["accuracy"],
                "validation/top1": initial_eval["accuracy"],
                "validation/length": initial_eval["length"],
            }
        )
        append_jsonl(metrics_path(run_dir), initial_payload)
        best_accuracy = initial_eval["accuracy"]

    run = maybe_init_wandb(args, run_name)
    if run is not None and start_epoch == 1 and initial_payload is not None:
        log_wandb(initial_payload)

    milestones = parse_milestones(args.lr_milestones)
    for epoch in range(start_epoch, args.epochs + 1):
        lr = current_lr(args.lr, epoch, milestones, args.lr_gamma)
        global_step = train_one_epoch(
            args=args,
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            lr=lr,
            global_step=global_step,
            tracker=tracker,
            clip_stats=clip_stats,
            run_dir=run_dir,
        )
        eval_log = evaluate(
            model, test_loader, loss_func, device, max_batches=args.max_test_batches
        )
        if eval_log["accuracy"] > best_accuracy:
            best_accuracy = eval_log["accuracy"]
            best_epoch = epoch
        payload = {
            "train/global_step": global_step,
            "validation/global_step": global_step,
            "validation/epoch": epoch,
            "validation/loss": eval_log["loss"],
            "validation/accuracy": eval_log["accuracy"],
            "validation/top1": eval_log["accuracy"],
            "validation/best_accuracy": best_accuracy,
            "validation/best_epoch": best_epoch,
            "validation/length": eval_log["length"],
            "train/lr": lr,
        }
        append_jsonl(metrics_path(run_dir), payload)
        log_wandb(payload)
        print(
            f"E={epoch} validation_acc={eval_log['accuracy']:.2f} "
            f"best={best_accuracy:.2f}@{best_epoch} lr={lr}",
            flush=True,
        )
        save_checkpoint(
            checkpoint_path(run_dir),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            tracker=tracker,
            clip_stats=clip_stats,
            args_dict={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )

    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "optimizer_mode": args.optimizer_mode,
        "best_validation_accuracy": best_accuracy,
        "best_validation_epoch": best_epoch,
        "global_step": global_step,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "clip_c": args.clip_c,
        "clip_c_res": args.clip_c_res,
        "weight_decay": args.weight_decay,
        "wandb_group": resolved_wandb_group(args),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_job_type": args.wandb_job_type,
        "completed": 1,
        "diagnostics": tracker.summarize(),
    }
    write_summary(run_dir, summary)
    if run is not None:
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


@dataclass(frozen=True)
class SweepConfig:
    model: str
    optimizer_mode: str
    lr: float
    beta: float
    clip_c: float | None
    clip_c_res: float | None
    seed: int

    @property
    def config_id(self) -> str:
        clip_tag = ""
        if self.clip_c is not None:
            clip_tag = f"-C{str(self.clip_c).replace('.', 'p')}"
        if self.clip_c_res is not None:
            clip_tag = f"-Cres{str(self.clip_c_res).replace('.', 'p')}"
        return (
            f"{self.model}-{self.optimizer_mode}{clip_tag}"
            f"-lr{str(self.lr).replace('.', 'p')}-b{str(self.beta).replace('.', 'p')}-seed{self.seed}"
        )


def build_sweep_configs(args) -> list[SweepConfig]:
    configs = []
    for model in args.models:
        for lr in args.lrs:
            for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                if args.optimizer_mode in (None, "sgd_momentum"):
                    configs.append(SweepConfig(model, "sgd_momentum", lr, args.beta, None, None, seed))
                if args.optimizer_mode in (None, "clipped_momentum"):
                    for clip_c in args.clip_values:
                        configs.append(SweepConfig(model, "clipped_momentum", lr, args.beta, float(clip_c), None, seed))
                if args.optimizer_mode in (None, "residual_clipped_momentum"):
                    for clip_c_res in args.res_clip_values:
                        configs.append(SweepConfig(model, "residual_clipped_momentum", lr, args.beta, None, float(clip_c_res), seed))
    return configs


def load_summary_if_exists(run_dir: Path) -> dict[str, Any] | None:
    path = summary_path(run_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_sweep(args) -> pd.DataFrame:
    configs = build_sweep_configs(args)
    results = []
    completed = 0
    for config in configs:
        run_args = argparse.Namespace(**vars(args))
        run_args.model = config.model
        run_args.optimizer_mode = config.optimizer_mode
        run_args.lr = config.lr
        run_args.beta = config.beta
        run_args.clip_c = config.clip_c
        run_args.clip_c_res = config.clip_c_res
        run_args.seed = config.seed
        run_args.run_name = config.config_id
        summary = run_cifar10_experiment(run_args)
        results.append(summary)
        completed += 1
        print(f"[sweep] completed {completed}/{len(configs)}: {config.config_id}", flush=True)
    return pd.DataFrame(results)


def summarize_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    results = results.copy()
    results["clip_threshold"] = results["clip_c"].fillna(results["clip_c_res"])
    group_cols = ["model", "optimizer_mode", "clip_threshold"]
    grouped = results.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        best_accuracy=("best_validation_accuracy", "max"),
        mean_best_accuracy=("best_validation_accuracy", "mean"),
        run_count=("run_name", "count"),
    ).reset_index()
    return summary.sort_values(["model", "optimizer_mode", "clip_threshold"]).reset_index(drop=True)
