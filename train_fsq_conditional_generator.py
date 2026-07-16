"""Fine-tune the frozen-FSQ token generator with style and trajectory controls."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from datasets.fsq_token_dataset import build_fsq_token_store
from datasets.fsq_trajectory_dataset import (
    FSQConditionalDataset,
    TrajectoryNormalization,
    build_fsq_trajectory_store,
    fit_trajectory_normalization,
)
from models.fsq_generator import FSQConditionalTransformerGenerator
from train_fsq_generator import build_scheduler, choose_device, maybe_subset, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an FSQ Transformer with a style prefix and root-local trajectory controls."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-database", type=Path, default=None)
    parser.add_argument("--trajectory-database", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap each split for a smoke test.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for name in (
        "token_database",
        "trajectory_database",
        "base_checkpoint",
        "outdir",
        "device",
        "epochs",
        "max_samples",
        "num_workers",
        "resume",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    required = ("token_database", "trajectory_database", "base_checkpoint")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Config must define: {missing}")
    for name in ("token_database", "trajectory_database", "base_checkpoint", "outdir", "resume"):
        if config.get(name) is not None:
            config[name] = Path(config[name])
    config["outdir"] = config.get("outdir", Path("outputs/fsq_generator_conditional"))
    return config


def build_datasets(config: dict, token_store):
    trajectory_store = build_fsq_trajectory_store(config["trajectory_database"], token_store)
    normalization = fit_trajectory_normalization(trajectory_store, token_store.split_windows["train"])
    full_datasets = {
        split: FSQConditionalDataset(split, token_store, trajectory_store, normalization)
        for split in ("train", "val", "test")
    }
    maximum = config.get("max_samples")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("max_samples must be positive")
    seed = int(config.get("seed", 3407))
    datasets = {
        split: maybe_subset(full_datasets[split], maximum, seed + offset * 1000)
        for offset, split in enumerate(("train", "val", "test"))
    }
    return datasets, trajectory_store, normalization


def build_loaders(config: dict, datasets, device: torch.device):
    batch_size = int(config.get("batch_size", 512))
    num_workers = int(config.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(config.get("persistent_workers", True))
        common["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    generator = torch.Generator().manual_seed(int(config.get("seed", 3407)))
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def load_base_generator(path: Path, token_store, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing base generator checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq_generator":
        raise ValueError(
            f"Conditional training requires an unconditional fsq_generator checkpoint, got {checkpoint.get('model_family')}"
        )
    if checkpoint.get("tokenizer_checkpoint_sha256") != token_store.checkpoint_sha256:
        raise ValueError("Base generator checkpoint and token database use different FSQ tokenizers")
    return checkpoint


def build_model(
    base_checkpoint: dict,
    token_store,
    trajectory_dim: int,
    config: dict,
) -> FSQConditionalTransformerGenerator:
    base_config = dict(base_checkpoint["model_config"])
    # The conditional model inherits all generator dimensions from the trained
    # base checkpoint.  Only the new condition branches are configurable here.
    model = FSQConditionalTransformerGenerator(
        **base_config,
        num_styles=len(token_store.style_names),
        trajectory_dim=int(trajectory_dim),
        trajectory_hidden_dim=int(config.get("trajectory_hidden_dim", 128)),
    )
    if token_store.window_size - 1 > model.context_frames:
        raise ValueError("Token windows exceed the base generator context length")
    incompatible = model.load_state_dict(base_checkpoint["model"], strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("style_embedding.") or name.startswith("trajectory_encoder.")
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "Base checkpoint did not load as an unconditional generator. "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return model


def make_optimizer(model: FSQConditionalTransformerGenerator, config: dict):
    base_parameters = []
    condition_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("style_embedding.") or name.startswith("trajectory_encoder."):
            condition_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    base_lr = float(config.get("base_lr", 3e-5))
    condition_lr = float(config.get("condition_lr", 3e-4))
    if base_lr <= 0.0 or condition_lr <= 0.0:
        raise ValueError("base_lr and condition_lr must be positive")
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": base_lr, "name": "base"},
            {"params": condition_parameters, "lr": condition_lr, "name": "condition"},
        ],
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.95))),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    return optimizer, base_lr, condition_lr


def _prepare_batch(
    batch: dict,
    device: torch.device,
    history_dropout: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = batch["indices"].to(device, non_blocking=device.type == "cuda").long()
    style_ids = batch["style_id"].to(device, non_blocking=device.type == "cuda").long()
    trajectory = batch["trajectory"].to(device, non_blocking=device.type == "cuda").float()
    valid = batch["trajectory_valid"].to(device, non_blocking=device.type == "cuda").bool()
    if indices.shape[1] < 2 or trajectory.shape[1] != indices.shape[1] - 1:
        raise ValueError("Conditional trajectories must have one entry per next-token target")
    if training and history_dropout > 0.0 and trajectory.shape[1] > 1:
        # Retain the newest command.  Earlier conditions may be absent when a
        # controller is attached mid-clip, so this prevents cache prefill from
        # depending on a perfect command history.
        keep = torch.rand_like(valid[:, :-1], dtype=torch.float32) >= history_dropout
        valid = valid.clone()
        valid[:, :-1] &= keep
        trajectory = trajectory * valid[..., None]
    return indices[:, :-1], indices[:, 1:], style_ids, trajectory, valid


def run_epoch(
    model: FSQConditionalTransformerGenerator,
    loader,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    grad_clip_norm: float = 1.0,
    history_dropout: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_nll = 0.0
    total_correct = 0
    total_level_error = 0.0
    total_tokens = 0
    total_windows = 0
    total_valid_controls = 0
    total_controls = 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            inputs, targets, style_ids, trajectory, valid = _prepare_batch(
                batch,
                device=device,
                history_dropout=history_dropout,
                training=training,
            )
            logits = model(
                inputs,
                style_ids=style_ids,
                trajectory=trajectory,
                trajectory_valid=valid,
            )["logits"]
            loss = F.cross_entropy(logits.reshape(-1, model.num_levels), targets.reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            predictions = logits.argmax(dim=-1)
            token_count = targets.numel()
            total_nll += float(loss.detach()) * token_count
            total_correct += int((predictions == targets).sum())
            total_level_error += float((predictions - targets).abs().sum())
            total_tokens += token_count
            total_windows += int(inputs.shape[0])
            total_valid_controls += int(valid.sum())
            total_controls += valid.numel()
    if total_tokens == 0:
        raise ValueError("No conditional token windows were processed")
    elapsed = time.perf_counter() - started
    nll = total_nll / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "coordinate_accuracy": total_correct / total_tokens,
        "level_mae": total_level_error / total_tokens,
        "valid_control_fraction": total_valid_controls / max(total_controls, 1),
        "windows_per_second": total_windows / max(elapsed, 1e-8),
    }


def serialize_config(config: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def load_resume(
    path: Path,
    model: FSQConditionalTransformerGenerator,
    optimizer,
    scheduler,
    token_store,
    normalization: TrajectoryNormalization,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq_conditional_generator":
        raise ValueError(f"Unsupported conditional resume checkpoint: {checkpoint.get('model_family')}")
    if checkpoint["model_config"] != model.config:
        raise ValueError("Resume model_config does not match the current model")
    if checkpoint["tokenizer_checkpoint_sha256"] != token_store.checkpoint_sha256:
        raise ValueError("Resume checkpoint and token database use different tokenizers")
    saved_normalization = TrajectoryNormalization.from_checkpoint(checkpoint["trajectory_normalization"])
    if not (
        np.allclose(saved_normalization.mean, normalization.mean)
        and np.allclose(saved_normalization.std, normalization.std)
    ):
        raise ValueError("Resume checkpoint uses different trajectory normalization statistics")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def main() -> None:
    args = parse_args()
    config = load_config(args)
    epochs = int(config.get("epochs", 40))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    history_dropout = float(config.get("history_condition_dropout", 0.5))
    if not 0.0 <= history_dropout < 1.0:
        raise ValueError("history_condition_dropout must lie in [0, 1)")
    set_seed(int(config.get("seed", 3407)), bool(config.get("deterministic", False)))
    device = choose_device(str(config.get("device", "auto")))
    token_store = build_fsq_token_store(config["token_database"])
    base_checkpoint = load_base_generator(config["base_checkpoint"], token_store, device)
    datasets, trajectory_store, normalization = build_datasets(config, token_store)
    loaders = build_loaders(config, datasets, device)
    model = build_model(base_checkpoint, token_store, trajectory_store.trajectory_dim, config).to(device)
    optimizer, base_lr, condition_lr = make_optimizer(model, config)
    total_steps = max(epochs * len(loaders["train"]), 1)
    warmup_steps = int(round(float(config.get("warmup_ratio", 0.05)) * total_steps))
    min_lr_ratio = float(config.get("min_lr_ratio", 0.1))
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1]")
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio)

    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run_name") or time.strftime("fsq_conditional_%Y%m%d-%H%M%S"))
    normalization_summary = {
        "mean": normalization.mean.tolist(),
        "std": normalization.std.tolist(),
        "valid_frames": normalization.valid_frames,
    }
    run_config = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": serialize_config(config),
        "model_config": model.config,
        "base_checkpoint": str(config["base_checkpoint"]),
        "base_checkpoint_epoch": int(base_checkpoint.get("epoch", 0)),
        "token_database": str(token_store.database),
        "trajectory_database": str(trajectory_store.database),
        "trajectory_feature_order": trajectory_store.feature_order,
        "trajectory_future_frames": trajectory_store.future_frames.tolist(),
        "trajectory_normalization": normalization_summary,
        "control_alignment": "input x_t receives trajectory at target frame t+1 and predicts x_(t+1)",
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "style_names": token_store.style_names,
        "base_lr": base_lr,
        "condition_lr": condition_lr,
    }
    with (outdir / f"{run_name}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)
    writer = SummaryWriter(log_dir=(outdir / "tensorboard" / run_name).as_posix())
    writer.add_text("config", json.dumps(run_config, indent=2, default=str))

    start_epoch = 0
    global_step = 0
    best_val_nll = math.inf
    best_epoch = 0
    patience = 0
    if config.get("resume") is not None:
        checkpoint = load_resume(
            config["resume"], model, optimizer, scheduler, token_store, normalization, device
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", start_epoch * len(loaders["train"])))
        best_val_nll = float(checkpoint["best_val_nll"])
        best_epoch = int(checkpoint["best_epoch"])
        patience = int(checkpoint.get("patience", 0))

    print(f"device={device} token_database={token_store.database}")
    print(f"trajectory_database={trajectory_store.database} controls={trajectory_store.trajectory_dim}D")
    print(
        f"split_sizes={run_config['split_sizes']} parameters={sum(p.numel() for p in model.parameters())} "
        f"normalization_frames={normalization.valid_frames}"
    )
    for epoch in range(start_epoch, epochs):
        train_stats = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
            history_dropout=history_dropout,
        )
        global_step += len(loaders["train"])
        val_stats = run_epoch(model, loaders["val"], device)
        for split, stats in (("train", train_stats), ("val", val_stats)):
            for name, value in stats.items():
                writer.add_scalar(f"{split}/{name}", value, epoch + 1)
        for group in optimizer.param_groups:
            writer.add_scalar(f"optimizer/{group['name']}_lr", group["lr"], epoch + 1)

        is_best = val_stats["nll"] < best_val_nll
        if is_best:
            best_val_nll = val_stats["nll"]
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        checkpoint = {
            "model_family": "fsq_conditional_generator",
            "model_config": model.config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_val_nll": best_val_nll,
            "patience": patience,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "run_config": run_config,
            "tokenizer_checkpoint_sha256": token_store.checkpoint_sha256,
            "style_names": token_store.style_names,
            "trajectory_database": str(trajectory_store.database),
            "trajectory_feature_order": trajectory_store.feature_order,
            "trajectory_future_frames": trajectory_store.future_frames,
            "trajectory_normalization": normalization.as_checkpoint(),
            "control_alignment": run_config["control_alignment"],
            "parent_generator_checkpoint": str(config["base_checkpoint"]),
            "parent_generator_epoch": int(base_checkpoint.get("epoch", 0)),
        }
        torch.save(checkpoint, outdir / "last.pt")
        if is_best:
            torch.save(checkpoint, outdir / "best.pt")
        print(
            f"epoch={epoch + 1} train_nll={train_stats['nll']:.5f} "
            f"val_nll={val_stats['nll']:.5f} val_ppl={val_stats['perplexity']:.4f} "
            f"val_acc={val_stats['coordinate_accuracy']:.4f} "
            f"base_lr={optimizer.param_groups[0]['lr']:.8f} "
            f"condition_lr={optimizer.param_groups[1]['lr']:.8f}"
        )
        if patience >= int(config.get("early_stop_patience", 12)):
            print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch}")
            break
    writer.close()


if __name__ == "__main__":
    main()
