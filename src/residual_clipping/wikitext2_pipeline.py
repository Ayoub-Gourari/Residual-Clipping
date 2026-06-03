"""Training, checkpointing, and sweep helpers for WikiText-2 language modeling."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from .cifar10_pipeline import (
    OPTIMIZER_MODES,
    MomentumClipper,
    RunningDiagnostics,
    collect_gradients,
    current_lr,
    log_wandb,
    maybe_init_wandb,
    metrics_path,
    parse_milestones,
    resolved_wandb_group,
)
from .clipping import tensor_list_global_norm
from .logging_utils import append_jsonl, atomic_write_json
from .randomness import capture_rng_state, restore_rng_state, set_global_seed
from .wikitext2_data import batchify, get_batch, load_corpus
from .wikitext2_models import make_lstm_language_model


DATASET_NAMES = ("wikitext2", "fake_wikitext2")


def detach_hidden(hidden: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(value.detach() for value in hidden)  # type: ignore[return-value]


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_latest.pt"


def summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def sweep_output_dir(output_dir: str | Path, dataset_name: str) -> Path:
    return Path(output_dir) / f"{dataset_name}_sweeps"


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 100.0))


def resolved_run_name(args) -> str:
    if args.run_name is not None:
        return args.run_name
    parts = [args.model, args.optimizer_mode]
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


def summary_indicates_complete(run_dir: Path, epochs: int) -> bool:
    path = summary_path(run_dir)
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return int(summary.get("epochs", -1)) >= epochs and int(summary.get("completed", 0)) == 1


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: MomentumClipper,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_epoch: int,
    tracker: RunningDiagnostics,
    clip_stats: dict[str, float],
    args_dict: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "tracker_values": tracker.values,
            "clip_stats": clip_stats,
            "rng_state": capture_rng_state(),
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: MomentumClipper,
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
    data_source: torch.Tensor,
    loss_func: nn.Module,
    *,
    bptt: int,
    max_batches: int | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden = model.init_hidden(data_source.size(1), device)
    with torch.no_grad():
        for batch_idx, index in enumerate(range(0, data_source.size(0) - 1, bptt)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data, targets = get_batch(data_source, index, bptt)
            output, hidden = model(data, hidden)
            hidden = detach_hidden(hidden)
            loss = loss_func(output, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    loss_value = total_loss / max(total_tokens, 1)
    return {
        "loss": loss_value,
        "perplexity": perplexity(loss_value),
        "tokens": float(total_tokens),
    }


def train_one_epoch(
    *,
    args,
    model: nn.Module,
    train_data: torch.Tensor,
    loss_func: nn.Module,
    optimizer: MomentumClipper,
    epoch: int,
    lr: float,
    global_step: int,
    tracker: RunningDiagnostics,
    clip_stats: dict[str, float],
    run_dir: Path,
    device: torch.device,
) -> int:
    model.train()
    hidden = model.init_hidden(train_data.size(1), device)
    params = [param for param in model.parameters() if param.requires_grad]
    for batch_idx, index in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        data, targets = get_batch(train_data, index, args.bptt)
        hidden = detach_hidden(hidden)
        model.zero_grad(set_to_none=True)
        output, hidden = model(data, hidden)
        loss = loss_func(output, targets)
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
                "train/perplexity": perplexity(loss.item()),
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
                        f"ppl={payload['train/perplexity']:.2f}",
                        f"grad_norm={diagnostics['grad_norm']:.4f}",
                    ]
                ),
                flush=True,
            )
    return global_step


def run_wikitext2_experiment(args) -> dict[str, Any]:
    if args.use_cuda and not torch.cuda.is_available():
        print("Requested CUDA, but no CUDA device is available; using CPU.", flush=True)
        args.use_cuda = False
    device = torch.device("cuda" if args.use_cuda else "cpu")
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
    corpus = load_corpus(
        args.dataset,
        args.data_dir,
        download=args.download,
        fake_token_count=args.fake_token_count,
        fake_vocab_size=args.fake_vocab_size,
    )
    train_data = batchify(corpus.train, args.batch_size, device)
    valid_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.eval_batch_size, device)
    model = make_lstm_language_model(
        vocab_size=corpus.vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    ).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = MomentumClipper(params, args.optimizer_mode, args.beta, args.clip_c, args.clip_c_res)
    tracker = RunningDiagnostics()
    clip_stats = {"active": 0.0, "total": 0.0}
    global_step = 0
    start_epoch = 1
    best_validation_loss = math.inf
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
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        clip_stats = checkpoint.get("clip_stats", clip_stats)
        print(f"Resuming {run_name} from epoch {start_epoch}", flush=True)
    else:
        initial_eval = evaluate(
            model,
            valid_data,
            loss_func,
            bptt=args.bptt,
            max_batches=args.max_eval_batches,
            device=device,
        )
        initial_payload = {
            "train/global_step": 0,
            "validation/global_step": 0,
            "validation/epoch": 0,
            "validation/loss": initial_eval["loss"],
            "validation/perplexity": initial_eval["perplexity"],
            "validation/tokens": initial_eval["tokens"],
            **{f"run/{key}": value for key, value in vars(args).items() if isinstance(value, (int, float, str, bool)) or value is None},
        }
        append_jsonl(metrics_path(run_dir), initial_payload)
        best_validation_loss = initial_eval["loss"]

    run = maybe_init_wandb(args, run_name)
    if run is not None and start_epoch == 1 and initial_payload is not None:
        log_wandb(initial_payload)

    milestones = parse_milestones(args.lr_milestones)
    for epoch in range(start_epoch, args.epochs + 1):
        lr = current_lr(args.lr, epoch, milestones, args.lr_gamma)
        global_step = train_one_epoch(
            args=args,
            model=model,
            train_data=train_data,
            loss_func=loss_func,
            optimizer=optimizer,
            epoch=epoch,
            lr=lr,
            global_step=global_step,
            tracker=tracker,
            clip_stats=clip_stats,
            run_dir=run_dir,
            device=device,
        )
        validation = evaluate(
            model,
            valid_data,
            loss_func,
            bptt=args.bptt,
            max_batches=args.max_eval_batches,
            device=device,
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
        payload = {
            "train/global_step": global_step,
            "validation/global_step": global_step,
            "validation/epoch": epoch,
            "validation/loss": validation["loss"],
            "validation/perplexity": validation["perplexity"],
            "validation/best_loss": best_validation_loss,
            "validation/best_perplexity": perplexity(best_validation_loss),
            "validation/best_epoch": best_epoch,
            "validation/tokens": validation["tokens"],
            "train/lr": lr,
        }
        append_jsonl(metrics_path(run_dir), payload)
        log_wandb(payload)
        print(
            f"E={epoch} validation_loss={validation['loss']:.4f} "
            f"validation_ppl={validation['perplexity']:.2f} "
            f"best_ppl={perplexity(best_validation_loss):.2f}@{best_epoch} lr={lr}",
            flush=True,
        )
        save_checkpoint(
            checkpoint_path(run_dir),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            tracker=tracker,
            clip_stats=clip_stats,
            args_dict={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )

    test = evaluate(
        model,
        test_data,
        loss_func,
        bptt=args.bptt,
        max_batches=args.max_eval_batches,
        device=device,
    )
    summary = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "optimizer_mode": args.optimizer_mode,
        "best_validation_loss": best_validation_loss,
        "best_validation_perplexity": perplexity(best_validation_loss),
        "best_validation_epoch": best_epoch,
        "test_loss": test["loss"],
        "test_perplexity": test["perplexity"],
        "global_step": global_step,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "clip_c": args.clip_c,
        "clip_c_res": args.clip_c_res,
        "weight_decay": args.weight_decay,
        "vocab_size": corpus.vocab_size,
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "bptt": args.bptt,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "tie_weights": args.tie_weights,
        "wandb_group": resolved_wandb_group(args),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_job_type": args.wandb_job_type,
        "completed": 1,
        "diagnostics": tracker.summarize(),
    }
    atomic_write_json(summary_path(run_dir), summary)
    if run is not None:
        run.finish()
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


@dataclass(frozen=True)
class SweepConfig:
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
            f"lstm-{self.optimizer_mode}{clip_tag}"
            f"-lr{str(self.lr).replace('.', 'p')}-b{str(self.beta).replace('.', 'p')}-seed{self.seed}"
        )


def build_sweep_configs(args) -> list[SweepConfig]:
    configs = []
    for lr in args.lrs:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            if args.optimizer_mode in (None, "sgd_momentum"):
                configs.append(SweepConfig("sgd_momentum", lr, args.beta, None, None, seed))
            if args.optimizer_mode in (None, "clipped_momentum"):
                for clip_c in args.clip_values:
                    configs.append(SweepConfig("clipped_momentum", lr, args.beta, float(clip_c), None, seed))
            if args.optimizer_mode in (None, "residual_clipped_momentum"):
                for clip_c_res in args.res_clip_values:
                    configs.append(
                        SweepConfig("residual_clipped_momentum", lr, args.beta, None, float(clip_c_res), seed)
                    )
    return configs


def run_sweep(args) -> pd.DataFrame:
    configs = build_sweep_configs(args)
    results = []
    completed = 0
    for config in configs:
        run_args = argparse.Namespace(**vars(args))
        run_args.optimizer_mode = config.optimizer_mode
        run_args.lr = config.lr
        run_args.beta = config.beta
        run_args.clip_c = config.clip_c
        run_args.clip_c_res = config.clip_c_res
        run_args.seed = config.seed
        run_args.run_name = config.config_id
        summary = run_wikitext2_experiment(run_args)
        results.append(summary)
        completed += 1
        print(f"[sweep] completed {completed}/{len(configs)}: {config.config_id}", flush=True)
    return pd.DataFrame(results)


def summarize_sweep_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    results = results.copy()
    for column in ["clip_c", "clip_c_res", "best_validation_loss", "best_validation_perplexity"]:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    results["clip_threshold"] = results["clip_c"].fillna(results["clip_c_res"])
    group_cols = ["model", "optimizer_mode", "clip_threshold"]
    grouped = results.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        best_perplexity=("best_validation_perplexity", "min"),
        mean_best_perplexity=("best_validation_perplexity", "mean"),
        best_loss=("best_validation_loss", "min"),
        run_count=("run_name", "count"),
    ).reset_index()
    return summary.sort_values(["model", "optimizer_mode", "clip_threshold"]).reset_index(drop=True)
