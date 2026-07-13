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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets.fsq_token_dataset import FSQTokenDataset, build_fsq_token_store
from models.fsq_style_gate import FSQStyleGateExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dynamic style gate on frozen FSQ token clips.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap each split for smoke tests.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for name in ("device", "epochs", "max_samples", "num_workers", "outdir"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if "token_database" not in config:
        raise ValueError("Config must define token_database")
    config["outdir"] = Path(config.get("outdir", "outputs/fsq_style_gate"))
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


def split_style_labels(dataset: FSQTokenDataset) -> np.ndarray:
    return np.asarray(
        [dataset.store.style_ids[window.range_idx] for window in dataset.windows],
        dtype=np.int64,
    )


def maybe_subset(dataset: FSQTokenDataset, maximum: int | None, seed: int):
    if maximum is None or maximum >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    selection = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return Subset(dataset, selection)


def labels_for_dataset(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        labels = split_style_labels(dataset.dataset)
        return labels[np.asarray(dataset.indices, dtype=np.int64)]
    return split_style_labels(dataset)


def build_loaders(config: dict, store, device: torch.device):
    seed = int(config.get("seed", 3407))
    maximum = config.get("max_samples")
    datasets = {
        split: maybe_subset(FSQTokenDataset(split, store), maximum, seed + index * 1000)
        for index, split in enumerate(("train", "val", "test"))
    }
    train_labels = labels_for_dataset(datasets["train"])
    counts = np.bincount(train_labels, minlength=len(store.style_names)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing style classes: {np.flatnonzero(counts == 0).tolist()}")
    weights = torch.from_numpy((1.0 / counts[train_labels]).astype(np.float64))
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    common = {
        "batch_size": int(config.get("batch_size", 128)),
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    if common["num_workers"] > 0:
        common["persistent_workers"] = bool(config.get("persistent_workers", True))
        common["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    loaders = {
        "train": DataLoader(datasets["train"], sampler=sampler, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return datasets, loaders, counts.astype(np.int64)


def random_crop(indices: torch.Tensor, crop_length: int) -> torch.Tensor:
    if crop_length <= 0 or crop_length > indices.shape[1]:
        raise ValueError("crop_length must be between 1 and sequence length")
    if crop_length == indices.shape[1]:
        return indices
    starts = torch.randint(0, indices.shape[1] - crop_length + 1, (indices.shape[0],), device=indices.device)
    offsets = torch.arange(crop_length, device=indices.device)
    gather_indices = starts[:, None] + offsets[None]
    return indices.gather(1, gather_indices[:, :, None].expand(-1, -1, indices.shape[2]))


def temperature_at_epoch(config: dict, epoch: int) -> float:
    start = float(config.get("gate_temperature_start", 1.5))
    end = float(config.get("gate_temperature_end", 0.2))
    epochs = max(int(config.get("epochs", 100)) - 1, 1)
    progress = min(max(epoch / epochs, 0.0), 1.0)
    return start * ((end / start) ** progress)


def update_confusion(confusion: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> None:
    num_classes = confusion.shape[0]
    values = labels.to(torch.int64) * num_classes + predictions.to(torch.int64)
    confusion += torch.bincount(values.cpu(), minlength=num_classes * num_classes).reshape_as(confusion)


def classification_metrics(confusion: torch.Tensor) -> dict[str, float]:
    matrix = confusion.double()
    true_count = matrix.sum(dim=1)
    pred_count = matrix.sum(dim=0)
    tp = matrix.diag()
    recall = tp / true_count.clamp_min(1.0)
    precision = tp / pred_count.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    valid = true_count > 0
    return {
        "balanced_accuracy": float(recall[valid].mean()),
        "macro_f1": float(f1[valid].mean()),
        "accuracy": float(tp.sum() / matrix.sum().clamp_min(1.0)),
    }


def crop_consistency(model, indices: torch.Tensor, temperature: float, crop_length: int) -> torch.Tensor:
    if crop_length <= 0 or crop_length >= indices.shape[1]:
        return indices.new_zeros((), dtype=torch.float32)
    first = random_crop(indices, crop_length)
    second = random_crop(indices, crop_length)
    first_gate = model.gate_tokens(first, temperature, stochastic=False, hard=False)
    second_gate = model.gate_tokens(second, temperature, stochastic=False, hard=False)
    return F.l1_loss(first_gate["mask_probability"], second_gate["mask_probability"])


def batch_losses(model, batch, device: torch.device, temperature: float, config: dict, training: bool):
    indices = batch["indices"].to(device, non_blocking=device.type == "cuda")
    labels = batch["style_id"].to(device, non_blocking=device.type == "cuda").long()
    output = model(indices, temperature=temperature, stochastic=training, hard=not training)
    dynamic_ce = F.cross_entropy(output["dynamic_logits"], labels)
    full_ce = F.cross_entropy(output["full_logits"], labels)
    random_ce = F.cross_entropy(output["random_logits"], labels)
    l0 = output["expected_l0"].mean()
    probability = output["mask_probability"]
    binary = (probability * (1.0 - probability)).mean()
    consistency = crop_consistency(
        model,
        indices,
        temperature,
        int(config.get("consistency_crop_length", 48)),
    )
    total = (
        dynamic_ce
        + float(config.get("baseline_weight", 1.0)) * (full_ce + random_ce)
        + float(config.get("l0_weight", 0.001)) * l0
        + float(config.get("binary_weight", 0.001)) * binary
        + float(config.get("consistency_weight", 0.1)) * consistency
    )
    losses = {
        "loss": total,
        "dynamic_ce": dynamic_ce,
        "full_ce": full_ce,
        "random_ce": random_ce,
        "expected_l0": l0,
        "binary": binary,
        "consistency": consistency,
    }
    return output, labels, losses


def run_epoch(model, loader, device, temperature, config, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    num_styles = int(model.config["num_styles"])
    confusion = {name: torch.zeros(num_styles, num_styles, dtype=torch.int64) for name in ("dynamic", "full", "random")}
    totals = {name: 0.0 for name in ("loss", "dynamic_ce", "full_ce", "random_ce", "expected_l0", "binary", "consistency")}
    mask_totals = {name: 0.0 for name in ("active_pair_ratio", "active_coordinates", "active_levels_per_coordinate", "mask_entropy")}
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            output, labels, losses = batch_losses(model, batch, device, temperature, config, training)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
                optimizer.step()
            batch_size = labels.shape[0]
            count += batch_size
            for name, value in losses.items():
                totals[name] += float(value.detach()) * batch_size
            for name in confusion:
                update_confusion(confusion[name], labels, output[f"{name}_logits"].argmax(dim=-1))

            hard_mask = output["mask"].detach() >= 0.5
            probability = output["mask_probability"].detach().clamp(1e-7, 1.0 - 1e-7)
            mask_totals["active_pair_ratio"] += float(hard_mask.float().mean()) * batch_size
            mask_totals["active_coordinates"] += float(hard_mask.any(dim=-1).float().sum(dim=-1).mean()) * batch_size
            mask_totals["active_levels_per_coordinate"] += float(hard_mask.float().sum(dim=-1).mean()) * batch_size
            entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
            mask_totals["mask_entropy"] += float(entropy.mean()) * batch_size
    if count == 0:
        raise ValueError("No samples were processed")
    stats = {name: value / count for name, value in totals.items()}
    stats.update({name: value / count for name, value in mask_totals.items()})
    for name, matrix in confusion.items():
        for metric, value in classification_metrics(matrix).items():
            stats[f"{name}_{metric}"] = value
    return stats


def build_model(config: dict, store) -> FSQStyleGateExperiment:
    return FSQStyleGateExperiment(
        num_coordinates=store.num_coordinates,
        num_levels=store.num_levels,
        num_styles=len(store.style_names),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_heads=int(config.get("num_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        ff_dim=int(config.get("ff_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
        max_seq_len=int(config.get("max_seq_len", store.window_size)),
    )


def serialize_config(config: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}


def main() -> None:
    cli_args = parse_args()
    config = load_config(cli_args)
    if int(config.get("epochs", 100)) <= 0:
        raise ValueError("epochs must be positive")
    seed = int(config.get("seed", 3407))
    set_seed(seed, bool(config.get("deterministic", False)))
    device = choose_device(str(config.get("device", "auto")))
    store = build_fsq_token_store(config["token_database"])
    datasets, loaders, train_style_counts = build_loaders(config, store, device)
    model = build_model(config, store).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(config.get("epochs", 100)), 1),
        eta_min=float(config.get("min_lr", 1e-5)),
    )

    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = str(config.get("run_name") or time.strftime("fsq_style_gate_%Y%m%d-%H%M%S"))
    run_config = {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": serialize_config(config),
        "model_config": model.config,
        "token_database": str(store.database),
        "tokenizer_checkpoint": store.checkpoint_path,
        "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
        "style_names": store.style_names,
        "action_names": store.action_names,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "train_style_counts": train_style_counts.tolist(),
    }
    config_path = outdir / f"{run_name}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)
    writer = SummaryWriter(log_dir=(outdir / "tensorboard" / run_name).as_posix())
    writer.add_text("config", json.dumps(run_config, indent=2))

    best_accuracy = -math.inf
    best_epoch = 0
    patience = 0
    epochs = int(config.get("epochs", 100))
    print(f"device={device} token_database={store.database}")
    print(f"checkpoint_sha256={store.checkpoint_sha256}")
    print(f"split_sizes={run_config['split_sizes']} style_counts={train_style_counts.tolist()}")
    for epoch in range(epochs):
        temperature = temperature_at_epoch(config, epoch)
        train_stats = run_epoch(model, loaders["train"], device, temperature, config, optimizer)
        val_stats = run_epoch(model, loaders["val"], device, temperature, config)
        scheduler.step()
        for split, stats in (("train", train_stats), ("val", val_stats)):
            for name, value in stats.items():
                writer.add_scalar(f"{split}/{name}", value, epoch + 1)
        writer.add_scalar("gate/temperature", temperature, epoch + 1)
        writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch + 1)
        print(
            f"epoch={epoch + 1} temperature={temperature:.4f} "
            f"train_dynamic_bal_acc={train_stats['dynamic_balanced_accuracy']:.4f} "
            f"val_dynamic_bal_acc={val_stats['dynamic_balanced_accuracy']:.4f} "
            f"val_full_bal_acc={val_stats['full_balanced_accuracy']:.4f} "
            f"val_random_bal_acc={val_stats['random_balanced_accuracy']:.4f} "
            f"val_active_ratio={val_stats['active_pair_ratio']:.4f}"
        )
        is_best = val_stats["dynamic_balanced_accuracy"] > best_accuracy
        if is_best:
            best_accuracy = val_stats["dynamic_balanced_accuracy"]
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        checkpoint = {
            "model_family": "fsq_style_gate",
            "model_config": model.config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_accuracy,
            "temperature": temperature,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "run_config": run_config,
            "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
            "style_names": store.style_names,
            "action_names": store.action_names,
        }
        torch.save(checkpoint, outdir / "last.pt")
        if is_best:
            torch.save(checkpoint, outdir / "best.pt")
        if patience >= int(config.get("early_stop_patience", 15)):
            print(f"early_stop epoch={epoch + 1} best_epoch={best_epoch}")
            break
    writer.close()


if __name__ == "__main__":
    main()
