"""Train a style/trajectory-conditioned FSQ token generator from scratch."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.fsq_token_dataset import build_fsq_token_store
from datasets.fsq_trajectory_dataset import (
    FSQConditionalDataset,
    TrajectoryNormalization,
    build_fsq_trajectory_store,
    fit_trajectory_normalization,
)
from models.fsq_generator import (
    FSQConditionalTransformerGenerator,
    STYLE_CACHE_POLICY,
    STYLE_CONDITIONING,
)
from train_fsq_generator import build_scheduler, choose_device, maybe_subset, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an independent FSQ Transformer with causal dynamic FiLM style and trajectory controls."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-database", type=Path, default=None)
    parser.add_argument("--trajectory-database", type=Path, default=None)
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
    required = ("token_database", "trajectory_database")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Config must define: {missing}")
    for name in ("token_database", "trajectory_database", "outdir", "resume"):
        if config.get(name) is not None:
            config[name] = Path(config[name])
    config["outdir"] = config.get("outdir", Path("outputs/fsq_generator_conditional_dynamic_film"))
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


def build_model(
    token_store,
    trajectory_dim: int,
    config: dict,
) -> FSQConditionalTransformerGenerator:
    model = FSQConditionalTransformerGenerator(
        num_coordinates=token_store.num_coordinates,
        num_levels=token_store.num_levels,
        num_styles=len(token_store.style_names),
        trajectory_dim=int(trajectory_dim),
        trajectory_hidden_dim=int(config.get("trajectory_hidden_dim", 128)),
        style_embedding_dim=int(config.get("style_embedding_dim", 128)),
        style_conditioning=STYLE_CONDITIONING,
        coordinate_embedding_dim=int(config.get("coordinate_embedding_dim", 16)),
        dim=int(config.get("d_model", 256)),
        num_layers=int(config.get("num_layers", 6)),
        num_query_heads=int(config.get("num_query_heads", 8)),
        num_kv_heads=int(config.get("num_kv_heads", 4)),
        ff_dim=int(config.get("ff_dim", 768)),
        dropout=float(config.get("dropout", 0.1)),
        context_frames=int(config.get("context_frames", token_store.window_size)),
        rope_theta=float(config.get("rope_theta", 10000.0)),
        qk_norm=bool(config.get("qk_norm", True)),
        norm_eps=float(config.get("norm_eps", 1e-5)),
    )
    if token_store.window_size - 1 > model.context_frames:
        raise ValueError("Token windows exceed the conditional generator context length")
    return model


def _prepare_batch(
    batch: dict,
    device: torch.device,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = batch["indices"].to(device, non_blocking=device.type == "cuda").long()
    style_ids = batch["style_id"].to(device, non_blocking=device.type == "cuda").long()
    trajectory = batch["trajectory"].to(device, non_blocking=device.type == "cuda").float()
    valid = batch["trajectory_valid"].to(device, non_blocking=device.type == "cuda").bool()
    if indices.shape[1] < 2 or trajectory.shape[1] != indices.shape[1] - 1:
        raise ValueError("Conditional trajectories must have one entry per next-token target")
    return indices[:, :-1], indices[:, 1:], style_ids, trajectory, valid


def run_epoch(
    model: FSQConditionalTransformerGenerator,
    loader,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    grad_clip_norm: float = 1.0,
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
    if checkpoint.get("initialization") != "scratch":
        raise ValueError("Resume checkpoint was not trained as an independent scratch model")
    if checkpoint.get("style_conditioning") != STYLE_CONDITIONING:
        raise ValueError("Resume checkpoint does not use causal dynamic style FiLM")
    if checkpoint.get("style_cache_policy") != STYLE_CACHE_POLICY:
        raise ValueError("Resume checkpoint does not use the append-only style cache policy")
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
    epochs = int(config.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(int(config.get("seed", 3407)), bool(config.get("deterministic", False)))
    device = choose_device(str(config.get("device", "auto")))
    token_store = build_fsq_token_store(config["token_database"])
    datasets, trajectory_store, normalization = build_datasets(config, token_store)
    loaders = build_loaders(config, datasets, device)
    model = build_model(token_store, trajectory_store.trajectory_dim, config).to(device)
    lr = float(config.get("lr", 3e-4))
    min_lr = float(config.get("min_lr", 1e-5))
    if lr <= 0.0 or not 0.0 <= min_lr <= lr:
        raise ValueError("Require lr > 0 and 0 <= min_lr <= lr")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.95))),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    total_steps = max(epochs * len(loaders["train"]), 1)
    warmup_steps = int(round(float(config.get("warmup_ratio", 0.05)) * total_steps))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, min_lr / lr)

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
        "initialization": "scratch",
        "style_conditioning": STYLE_CONDITIONING,
        "style_cache_policy": STYLE_CACHE_POLICY,
        "token_database": str(token_store.database),
        "trajectory_database": str(trajectory_store.database),
        "trajectory_feature_order": trajectory_store.feature_order,
        "trajectory_future_frames": trajectory_store.future_frames.tolist(),
        "trajectory_normalization": normalization_summary,
        "control_alignment": "input x_t receives trajectory at target frame t+1 and predicts x_(t+1)",
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "style_names": token_store.style_names,
        "lr": lr,
        "min_lr": min_lr,
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
        )
        global_step += len(loaders["train"])
        val_stats = run_epoch(model, loaders["val"], device)
        for split, stats in (("train", train_stats), ("val", val_stats)):
            for name, value in stats.items():
                writer.add_scalar(f"{split}/{name}", value, epoch + 1)
        writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch + 1)

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
            "initialization": "scratch",
            "style_conditioning": STYLE_CONDITIONING,
            "style_cache_policy": STYLE_CACHE_POLICY,
        }
        torch.save(checkpoint, outdir / "last.pt")
        if is_best:
            torch.save(checkpoint, outdir / "best.pt")
        print(
            f"epoch={epoch + 1} train_nll={train_stats['nll']:.5f} "
            f"val_nll={val_stats['nll']:.5f} val_ppl={val_stats['perplexity']:.4f} "
            f"val_acc={val_stats['coordinate_accuracy']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )
        if patience >= int(config.get("early_stop_patience", 12)):
            print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch}")
            break
    writer.close()


if __name__ == "__main__":
    main()
