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

from datasets.fsq_token_dataset import FSQTokenDataset, build_fsq_token_store
from models.fsq_generator import FSQCausalTransformerGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal Transformer on frozen FSQ token clips.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-database", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap each split for smoke tests.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for name in ("token_database", "device", "epochs", "max_samples", "num_workers", "outdir", "resume"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if "token_database" not in config:
        raise ValueError("Config must define token_database")
    config["outdir"] = Path(config.get("outdir", "outputs/fsq_generator"))
    if config.get("resume") is not None:
        config["resume"] = Path(config["resume"])
    return config


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def maybe_subset(dataset, maximum: int | None, seed: int):
    if maximum is None or maximum >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    selection = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return Subset(dataset, selection)


def build_loaders(config: dict, store, device: torch.device):
    seed = int(config.get("seed", 3407))
    maximum = config.get("max_samples")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("max_samples must be positive")
    datasets = {
        split: maybe_subset(FSQTokenDataset(split, store), maximum, seed + index * 1000)
        for index, split in enumerate(("train", "val", "test"))
    }
    common = {
        "batch_size": int(config.get("batch_size", 512)),
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    if common["batch_size"] <= 0 or common["num_workers"] < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    if common["num_workers"] > 0:
        common["persistent_workers"] = bool(config.get("persistent_workers", True))
        common["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return datasets, loaders


def build_model(config: dict, store) -> FSQCausalTransformerGenerator:
    model = FSQCausalTransformerGenerator(
        num_coordinates=store.num_coordinates,
        num_levels=store.num_levels,
        coordinate_embedding_dim=int(config.get("coordinate_embedding_dim", 16)),
        dim=int(config.get("d_model", 256)),
        num_layers=int(config.get("num_layers", 6)),
        num_query_heads=int(config.get("num_query_heads", 8)),
        num_kv_heads=int(config.get("num_kv_heads", 4)),
        ff_dim=int(config.get("ff_dim", 768)),
        dropout=float(config.get("dropout", 0.1)),
        context_frames=int(config.get("context_frames", store.window_size)),
        rope_theta=float(config.get("rope_theta", 10000.0)),
        qk_norm=bool(config.get("qk_norm", True)),
        norm_eps=float(config.get("norm_eps", 1e-5)),
    )
    if store.window_size - 1 > model.context_frames:
        raise ValueError(
            f"Token window requires {store.window_size - 1} input frames, "
            f"but context_frames={model.context_frames}"
        )
    return model


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_lr_ratio, float(step + 1) / float(warmup_steps))
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def autoregressive_batch(model, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if indices.shape[1] < 2:
        raise ValueError("Token windows must contain at least two frames")
    inputs = indices[:, :-1]
    targets = indices[:, 1:]
    logits = model(inputs)["logits"]
    loss = F.cross_entropy(logits.reshape(-1, model.num_levels), targets.reshape(-1))
    return logits, targets, loss


def run_epoch(
    model,
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
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            indices = batch["indices"].to(device, non_blocking=device.type == "cuda").long()
            logits, targets, loss = autoregressive_batch(model, indices)
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
            total_windows += int(indices.shape[0])

    if total_tokens == 0:
        raise ValueError("No token windows were processed")
    elapsed = time.perf_counter() - started
    nll = total_nll / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "coordinate_accuracy": total_correct / total_tokens,
        "level_mae": total_level_error / total_tokens,
        "windows_per_second": total_windows / max(elapsed, 1e-8),
    }


def serialize_config(config: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def load_resume(path: Path, model, optimizer, scheduler, store, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq_generator":
        raise ValueError(f"Unsupported resume checkpoint family: {checkpoint.get('model_family')}")
    if checkpoint["model_config"] != model.config:
        raise ValueError("Resume checkpoint model_config does not match the current model")
    if checkpoint["tokenizer_checkpoint_sha256"] != store.checkpoint_sha256:
        raise ValueError("Resume checkpoint and token database use different FSQ tokenizers")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def main() -> None:
    cli_args = parse_args()
    config = load_config(cli_args)
    epochs = int(config.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    seed = int(config.get("seed", 3407))
    set_seed(seed, bool(config.get("deterministic", False)))
    device = choose_device(str(config.get("device", "auto")))
    store = build_fsq_token_store(config["token_database"])
    datasets, loaders = build_loaders(config, store, device)
    model = build_model(config, store).to(device)

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
    run_name = str(config.get("run_name") or time.strftime("fsq_generator_%Y%m%d-%H%M%S"))
    run_config = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": serialize_config(config),
        "model_config": model.config,
        "token_database": str(store.database),
        "tokenizer_checkpoint": store.checkpoint_path,
        "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "num_coordinates": store.num_coordinates,
        "num_levels": store.num_levels,
        "window_size": store.window_size,
    }
    config_path = outdir / f"{run_name}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)
    writer = SummaryWriter(log_dir=(outdir / "tensorboard" / run_name).as_posix())
    writer.add_text("config", json.dumps(run_config, indent=2))

    start_epoch = 0
    global_step = 0
    best_val_nll = math.inf
    best_epoch = 0
    patience = 0
    if config.get("resume") is not None:
        checkpoint = load_resume(Path(config["resume"]), model, optimizer, scheduler, store, device)
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", start_epoch * len(loaders["train"])))
        best_val_nll = float(checkpoint["best_val_nll"])
        best_epoch = int(checkpoint["best_epoch"])
        patience = int(checkpoint.get("patience", 0))

    print(f"device={device} token_database={store.database}")
    print(f"checkpoint_sha256={store.checkpoint_sha256}")
    print(f"split_sizes={run_config['split_sizes']} parameters={sum(p.numel() for p in model.parameters())}")
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
            "model_family": "fsq_generator",
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
            "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
        }
        torch.save(checkpoint, outdir / "last.pt")
        if is_best:
            torch.save(checkpoint, outdir / "best.pt")
        print(
            f"epoch={epoch + 1} train_nll={train_stats['nll']:.5f} "
            f"val_nll={val_stats['nll']:.5f} val_ppl={val_stats['perplexity']:.4f} "
            f"val_acc={val_stats['coordinate_accuracy']:.4f} lr={optimizer.param_groups[0]['lr']:.8f}"
        )
        if patience >= int(config.get("early_stop_patience", 15)):
            print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch}")
            break
    writer.close()


if __name__ == "__main__":
    main()
