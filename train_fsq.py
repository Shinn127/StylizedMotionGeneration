import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.feature_dataset import FeatureDataset, build_feature_store
from models.fsq import FSQMotionAutoencoder
from models.losses import compute_motion_reconstruction_losses
from motion_features import serialize_motion_feature_stats


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 1:
        return 0
    return min(8, cpu_count - 1)


def load_config(config_path: Path | None) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, required=True)
    pre_args, remaining = pre_parser.parse_known_args(argv)
    config = load_config(pre_args.config)

    def cfg(name, default):
        return config.get(name, default)

    def cfg_path(name, default):
        value = cfg(name, default)
        return None if value is None else Path(value)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--split-train", default=cfg("split_train", "train"))
    parser.add_argument("--split-val", default=cfg("split_val", "val"))
    parser.add_argument("--feature-database", type=Path, default=cfg_path("feature_database", None))
    parser.add_argument("--batch-size", type=int, default=cfg("batch_size", 32))
    parser.add_argument("--epochs", type=int, default=cfg("epochs", 100))
    parser.add_argument("--lr", type=float, default=cfg("lr", 2e-4))
    parser.add_argument("--min-lr", type=float, default=cfg("min_lr", 1e-5))
    parser.add_argument("--warmup-epochs", type=int, default=cfg("warmup_epochs", 2))
    parser.add_argument("--seed", type=int, default=cfg("seed", 3407))
    parser.add_argument("--deterministic", action="store_true", default=cfg("deterministic", False))
    parser.add_argument("--grad-clip-norm", type=float, default=cfg("grad_clip_norm", 1.0))
    parser.add_argument("--resume", type=Path, default=cfg_path("resume", None))
    parser.add_argument("--save-every", type=int, default=cfg("save_every", 0))
    parser.add_argument("--num-workers", type=int, default=cfg("num_workers", default_num_workers()))
    parser.add_argument("--prefetch-factor", type=int, default=cfg("prefetch_factor", 4))
    parser.add_argument("--log-every", type=int, default=cfg("log_every", 50))
    parser.add_argument("--run-name", type=str, default=cfg("run_name", None))
    parser.add_argument("--model-type", choices=["frame_causal_cnn"], default=cfg("model_type", "frame_causal_cnn"))
    parser.add_argument("--code-dim", type=int, default=cfg("code_dim", 256))
    parser.add_argument("--width", type=int, default=cfg("width", 512))
    parser.add_argument("--depth", type=int, default=cfg("depth", 6))
    parser.add_argument("--dilation-growth-rate", type=int, default=cfg("dilation_growth_rate", 2))
    parser.add_argument("--fsq-num-latent-tokens", type=int, default=cfg("fsq_num_latent_tokens", 40))
    parser.add_argument("--fsq-num-levels", type=int, default=cfg("fsq_num_levels", 9))
    parser.add_argument("--fsq-scale", type=float, default=cfg("fsq_scale", None))
    parser.add_argument("--fsq-preserve-symmetry", action="store_true", default=cfg("fsq_preserve_symmetry", False))
    parser.add_argument("--fsq-noise-dropout", type=float, default=cfg("fsq_noise_dropout", 0.0))
    parser.add_argument("--delta-weight", type=float, default=cfg("delta_weight", 1.0))
    parser.add_argument("--root-pos-weight", type=float, default=cfg("root_pos_weight", 0.0))
    parser.add_argument("--root-rot-weight", type=float, default=cfg("root_rot_weight", 0.0))
    parser.add_argument("--root-dt", type=float, default=cfg("root_dt", 1.0 / 60.0))
    parser.add_argument("--outdir", type=Path, default=cfg_path("outdir", Path("outputs/fsq")))
    parser.add_argument("--data-parallel", action="store_true", default=cfg("data_parallel", False))
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=cfg("pin_memory", torch.cuda.is_available()))
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=cfg("persistent_workers", True))
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    return args


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataloader_kwargs(args, shuffle: bool, pin_memory: bool) -> dict:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_dataloaders(args, pin_memory: bool):
    store = build_feature_store(args.feature_database)
    train_dataset = FeatureDataset(split=args.split_train, store=store)
    val_dataset = FeatureDataset(split=args.split_val, store=store)
    train_loader = DataLoader(train_dataset, **dataloader_kwargs(args, shuffle=True, pin_memory=pin_memory))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(args, shuffle=False, pin_memory=pin_memory))
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(args, motion_dim):
    if args.model_type != "frame_causal_cnn":
        raise ValueError(f"Unsupported FSQ model_type: {args.model_type}")
    return FSQMotionAutoencoder(
        motion_dim=motion_dim,
        code_dim=args.code_dim,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        num_latent_tokens=args.fsq_num_latent_tokens,
        num_levels=args.fsq_num_levels,
        fsq_scale=args.fsq_scale,
        fsq_preserve_symmetry=args.fsq_preserve_symmetry,
        fsq_noise_dropout=args.fsq_noise_dropout,
    )


def build_lr_scheduler(optimizer, args):
    if args.lr <= 0.0:
        raise ValueError(f"lr must be positive, got {args.lr}")
    min_factor = args.min_lr / args.lr
    warmup_epochs = max(0, int(args.warmup_epochs))
    decay_epochs = max(1, int(args.epochs) - warmup_epochs)

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return max(min_factor, float(epoch_index + 1) / float(warmup_epochs))
        progress = min(max(float(epoch_index - warmup_epochs) / float(decay_epochs), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def init_metric_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "recon": 0.0,
        "delta": 0.0,
        "root_pos": 0.0,
        "root_rot": 0.0,
        "level_perplexity": 0.0,
        "level_usage": 0.0,
    }


def update_metric_totals(totals, losses, output, batch_size):
    totals["loss"] += losses.loss.item() * batch_size
    totals["recon"] += losses.recon.item() * batch_size
    totals["delta"] += losses.delta.item() * batch_size
    totals["root_pos"] += losses.root_pos.item() * batch_size
    totals["root_rot"] += losses.root_rot.item() * batch_size
    totals["level_perplexity"] += output["level_perplexity"].item() * batch_size
    totals["level_usage"] += output["level_usage"].item() * batch_size


def finalize_metric_totals(totals, count):
    if count == 0:
        raise ValueError("Cannot finalize metrics with count=0")
    return {name: value / count for name, value in totals.items()}


def move_motion_to_device(batch, device, non_blocking: bool):
    return batch["motion"].to(device, non_blocking=non_blocking)


def compute_losses(motion, output, feature_weights, feature_offset, feature_scale, args):
    return compute_motion_reconstruction_losses(
        batch_motion=motion,
        output=output,
        feature_weights=feature_weights,
        feature_offset=feature_offset,
        feature_scale=feature_scale,
        delta_weight=args.delta_weight,
        commit_weight=0.0,
        root_pos_weight=args.root_pos_weight,
        root_rot_weight=args.root_rot_weight,
        root_dt=args.root_dt,
    )


def evaluate(model, loader, feature_weights, feature_offset, feature_scale, device, args, non_blocking: bool):
    model.eval()
    totals = init_metric_totals()
    count = 0
    with torch.no_grad():
        for batch in loader:
            motion = move_motion_to_device(batch, device, non_blocking=non_blocking)
            output = model(motion)
            losses = compute_losses(motion, output, feature_weights, feature_offset, feature_scale, args)
            batch_size = motion.shape[0]
            update_metric_totals(totals, losses, output, batch_size)
            count += batch_size
    return finalize_metric_totals(totals, count)


def train_one_epoch(
    model,
    loader,
    optimizer,
    feature_weights,
    feature_offset,
    feature_scale,
    device,
    args,
    grad_clip_norm,
    writer,
    global_step,
    log_every,
    non_blocking: bool,
):
    model.train()
    totals = init_metric_totals()
    count = 0
    start_time = time.perf_counter()

    for batch in loader:
        motion = move_motion_to_device(batch, device, non_blocking=non_blocking)
        output = model(motion)
        losses = compute_losses(motion, output, feature_weights, feature_offset, feature_scale, args)
        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = motion.shape[0]
        update_metric_totals(totals, losses, output, batch_size)
        count += batch_size
        global_step += 1

        if writer is not None and (global_step == 1 or global_step % log_every == 0):
            writer.add_scalar("train_step/loss", losses.loss.item(), global_step)
            writer.add_scalar("train_step/recon", losses.recon.item(), global_step)
            writer.add_scalar("train_step/delta", losses.delta.item(), global_step)
            writer.add_scalar("train_step/root_pos", losses.root_pos.item(), global_step)
            writer.add_scalar("train_step/root_rot", losses.root_rot.item(), global_step)
            writer.add_scalar("train_step/level_perplexity", output["level_perplexity"].item(), global_step)
            writer.add_scalar("train_step/level_usage", output["level_usage"].item(), global_step)
            writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_step)

    epoch_time = time.perf_counter() - start_time
    stats = finalize_metric_totals(totals, count)
    stats["epoch_seconds"] = epoch_time
    stats["samples_per_second"] = count / max(epoch_time, 1e-8)
    return stats, global_step


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    return time.strftime("%Y%m%d-%H%M%S")


def serialize_args(args) -> dict[str, str | int | float | bool | None]:
    serialized = {}
    source = args if isinstance(args, dict) else vars(args)
    for key, value in source.items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def build_run_config(args, run_name: str, train_dataset, val_dataset) -> dict:
    return {
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "args": serialize_args(args),
        "model_family": "fsq",
        "dataset": {
            "feature_database": str(train_dataset.feature_database),
            "joint_subset": train_dataset.joint_subset,
            "motion_dim": train_dataset.motion_dim,
            "num_joints": train_dataset.num_joints,
            "train_summary": train_dataset.split_summary(),
            "val_summary": val_dataset.split_summary(),
        },
    }


def save_run_config(args, run_name: str, train_dataset, val_dataset) -> Path:
    config = build_run_config(args, run_name, train_dataset, val_dataset)
    config_path = args.outdir / f"{run_name}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
    return config_path


def load_resume_checkpoint(resume_path: Path, model, optimizer, scheduler, device):
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq":
        raise ValueError(f"Expected an FSQ checkpoint, got model_family={checkpoint.get('model_family')}")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def build_checkpoint(
    model,
    optimizer,
    scheduler,
    args,
    run_name,
    config_path,
    train_dataset,
    epoch,
    global_step,
    best_val,
    train_stats,
    val_stats,
) -> dict:
    return {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": serialize_args(args),
        "run_name": run_name,
        "config_path": str(config_path),
        "model_family": "fsq",
        "motion_dim": train_dataset.motion_dim,
        "stats": serialize_motion_feature_stats(
            train_dataset.feature_stats(),
            names=train_dataset.names,
            parents=train_dataset.parents,
            joint_subset=train_dataset.joint_subset,
        ),
        "epoch": epoch + 1,
        "global_step": global_step,
        "best_val": best_val,
        "train_stats": train_stats,
        "val_stats": val_stats,
    }


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = bool(args.pin_memory and device.type == "cuda")
    non_blocking = use_pin_memory
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args, pin_memory=use_pin_memory)
    feature_weights = torch.from_numpy(train_dataset.model_feature_weights().astype("float32")).to(device)
    feature_stats = train_dataset.feature_stats()
    feature_offset = torch.from_numpy(feature_stats.offset.astype("float32")).to(device)
    feature_scale = torch.from_numpy(feature_stats.scale.astype("float32")).to(device)

    model = build_model(args, motion_dim=train_dataset.motion_dim)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args)

    run_name = build_run_name(args)
    config_path = save_run_config(args, run_name, train_dataset, val_dataset)
    log_dir = args.outdir / "tensorboard" / run_name
    writer = SummaryWriter(log_dir=log_dir.as_posix())
    writer.add_text("config/args", json.dumps(serialize_args(args), indent=2, ensure_ascii=False))
    writer.add_text("dataset/train_summary", json.dumps(train_dataset.split_summary(), indent=2))
    writer.add_text("dataset/val_summary", json.dumps(val_dataset.split_summary(), indent=2))

    print(f"device={device}")
    print(f"train_summary={train_dataset.split_summary()}")
    print(f"val_summary={val_dataset.split_summary()}")
    print(f"config_yaml={config_path}")
    print(f"tensorboard_logdir={log_dir}")

    best_val = None
    global_step = 0
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_resume_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_val = checkpoint["best_val"]
        print(f"resumed_from={args.resume} start_epoch={start_epoch + 1} global_step={global_step} best_val={best_val}")

    for epoch in range(start_epoch, args.epochs):
        train_stats, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            feature_weights,
            feature_offset,
            feature_scale,
            device,
            args=args,
            grad_clip_norm=args.grad_clip_norm,
            writer=writer,
            global_step=global_step,
            log_every=args.log_every,
            non_blocking=non_blocking,
        )
        val_stats = evaluate(
            model,
            val_loader,
            feature_weights,
            feature_offset,
            feature_scale,
            device,
            args=args,
            non_blocking=non_blocking,
        )

        for name in ["loss", "recon", "delta", "root_pos", "root_rot", "level_perplexity", "level_usage"]:
            writer.add_scalar(f"train_epoch/{name}", train_stats[name], epoch + 1)
            writer.add_scalar(f"val/{name}", val_stats[name], epoch + 1)
        writer.add_scalar("train_epoch/samples_per_second", train_stats["samples_per_second"], epoch + 1)
        writer.add_scalar("train_epoch/epoch_seconds", train_stats["epoch_seconds"], epoch + 1)
        writer.add_scalar("optimizer/lr", optimizer.param_groups[0]["lr"], epoch + 1)
        if device.type == "cuda":
            writer.add_scalar(
                "system/max_memory_allocated_mb",
                torch.cuda.max_memory_allocated(device=device) / (1024 ** 2),
                epoch + 1,
            )

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_stats['loss']:.6f} "
            f"train_recon={train_stats['recon']:.6f} "
            f"train_delta={train_stats['delta']:.6f} "
            f"train_root_pos={train_stats['root_pos']:.6f} "
            f"train_root_rot={train_stats['root_rot']:.6f} "
            f"train_level_perplexity={train_stats['level_perplexity']:.6f} "
            f"train_level_usage={train_stats['level_usage']:.6f} "
            f"train_samples_per_second={train_stats['samples_per_second']:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_recon={val_stats['recon']:.6f} "
            f"val_delta={val_stats['delta']:.6f} "
            f"val_root_pos={val_stats['root_pos']:.6f} "
            f"val_root_rot={val_stats['root_rot']:.6f} "
            f"val_level_perplexity={val_stats['level_perplexity']:.6f} "
            f"val_level_usage={val_stats['level_usage']:.6f}"
        )

        is_best = best_val is None or val_stats["loss"] < best_val
        if is_best:
            best_val = val_stats["loss"]

        scheduler.step()
        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            run_name=run_name,
            config_path=config_path,
            train_dataset=train_dataset,
            epoch=epoch,
            global_step=global_step,
            best_val=best_val,
            train_stats=train_stats,
            val_stats=val_stats,
        )
        torch.save(checkpoint, args.outdir / "last.pt")
        if is_best:
            torch.save(checkpoint, args.outdir / "best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, args.outdir / f"epoch_{epoch + 1:04d}.pt")

    writer.close()


if __name__ == "__main__":
    main()
