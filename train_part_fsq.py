import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.feature_dataset import FeatureDataset, build_feature_store
from models.losses import compute_motion_reconstruction_losses
from models.part_fsq import HierarchicalPartFSQMotionAutoencoder
from models.part_fsq_losses import adaptive_part_fsq_reuse_loss
from models.part_layout import GROUP_NAMES
from motion_features import serialize_motion_feature_stats


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 0
    return 0 if cpu_count <= 1 else min(4, cpu_count - 1)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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

    parser = argparse.ArgumentParser(description="Train a full-FP32 Hierarchical Part-FSQ tokenizer on 64-frame windows.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--split-train", default=cfg("split_train", "train"))
    parser.add_argument("--split-val", default=cfg("split_val", "val"))
    parser.add_argument("--feature-database", type=Path, default=cfg_path("feature_database", None))
    parser.add_argument("--batch-size", type=int, default=cfg("batch_size", 256))
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
    parser.add_argument("--prefetch-factor", type=int, default=cfg("prefetch_factor", 2))
    parser.add_argument("--log-every", type=int, default=cfg("log_every", 50))
    parser.add_argument("--metrics-interval", type=int, default=cfg("metrics_interval", 100))
    parser.add_argument("--run-name", type=str, default=cfg("run_name", None))
    parser.add_argument("--stream-dim", type=int, default=cfg("stream_dim", 64))
    parser.add_argument("--fsq-num-levels", type=int, default=cfg("fsq_num_levels", 9))
    parser.add_argument("--fsq-scale", type=float, default=cfg("fsq_scale", None))
    parser.add_argument("--fsq-preserve-symmetry", action="store_true", default=cfg("fsq_preserve_symmetry", False))
    parser.add_argument("--fsq-noise-dropout", type=float, default=cfg("fsq_noise_dropout", 0.0))
    parser.add_argument("--reuse-weight", type=float, default=cfg("reuse_weight", 0.01))
    parser.add_argument("--reuse-thresholds", type=float, nargs=len(GROUP_NAMES), default=cfg("reuse_thresholds", None))
    parser.add_argument("--contact-transition-threshold", type=float, default=cfg("contact_transition_threshold", 0.25))
    parser.add_argument("--delta-weight", type=float, default=cfg("delta_weight", 3.0))
    parser.add_argument("--root-pos-weight", type=float, default=cfg("root_pos_weight", 0.1))
    parser.add_argument("--root-rot-weight", type=float, default=cfg("root_rot_weight", 0.1))
    parser.add_argument("--joint-weight", type=float, default=cfg("joint_weight", 0.5))
    parser.add_argument("--contact-weight", type=float, default=cfg("contact_weight", 0.1))
    parser.add_argument("--foot-slide-weight", type=float, default=cfg("foot_slide_weight", 0.1))
    parser.add_argument("--foot-height-weight", type=float, default=cfg("foot_height_weight", 0.1))
    parser.add_argument("--contact-temperature", type=float, default=cfg("contact_temperature", 10.0))
    parser.add_argument("--root-dt", type=float, default=cfg("root_dt", 1.0 / 60.0))
    parser.add_argument("--outdir", type=Path, default=cfg_path("outdir", Path("outputs/part_fsq")))
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
    kwargs = {"batch_size": args.batch_size, "shuffle": shuffle, "num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_dataloaders(args, pin_memory: bool):
    store = build_feature_store(args.feature_database)
    train_dataset = FeatureDataset(args.split_train, store)
    val_dataset = FeatureDataset(args.split_val, store)
    train_loader = DataLoader(train_dataset, **dataloader_kwargs(args, shuffle=True, pin_memory=pin_memory))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(args, shuffle=False, pin_memory=pin_memory))
    return train_dataset, val_dataset, train_loader, val_loader


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
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def build_loss_metadata(dataset, device: torch.device) -> dict[str, object]:
    try:
        foot_indices = (dataset.names.index("LeftToeBase"), dataset.names.index("RightToeBase"))
    except ValueError as exc:
        raise ValueError("Joint and foot losses require LeftToeBase and RightToeBase joints") from exc
    joint_weights = np.ones(len(dataset.names), dtype=np.float32)
    joint_weights[0] = 0.0
    for index, name in enumerate(dataset.names):
        if any(token in name for token in ("Head", "Hand", "Foot", "Toe")):
            joint_weights[index] = 2.0
    return {
        "ref_pos": torch.from_numpy(dataset.feature_stats().ref_pos.astype("float32")).to(device),
        "parents": tuple(int(parent) for parent in dataset.parents.tolist()),
        "joint_weights": torch.from_numpy(joint_weights).to(device),
        "foot_indices": foot_indices,
    }


def compute_losses(model, motion, output, feature_weights, feature_offset, feature_scale, loss_metadata, args):
    reconstruction = compute_motion_reconstruction_losses(
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
        joint_weight=args.joint_weight,
        contact_weight=args.contact_weight,
        foot_slide_weight=args.foot_slide_weight,
        foot_height_weight=args.foot_height_weight,
        contact_temperature=args.contact_temperature,
        ref_pos=loss_metadata["ref_pos"],
        parents=loss_metadata["parents"],
        joint_weights=loss_metadata["joint_weights"],
        foot_indices=loss_metadata["foot_indices"],
    )
    reuse = adaptive_part_fsq_reuse_loss(
        codes=output["fsq_codes"],
        target_motion=motion,
        layout=unwrap_model(model).layout,
        thresholds=args.reuse_thresholds,
        feature_weights=feature_weights,
        contact_values=reconstruction.target_contact,
        contact_transition_threshold=args.contact_transition_threshold,
        level_step=2.0 / float(args.fsq_num_levels - 1),
    )
    return reconstruction, reuse, reconstruction.loss + float(args.reuse_weight) * reuse.loss


_BASE_METRICS = ("loss", "recon", "delta", "root_pos", "root_rot", "joint", "contact", "foot_slide", "foot_height", "reuse")
_REPRESENTATION_METRICS = (
    "level_perplexity",
    "level_usage",
    "tuple_unique_ratio",
    "tuple_change_rate",
    "coordinate_change_rate",
)


def init_totals(device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((), device=device) for name in (*_BASE_METRICS, *_REPRESENTATION_METRICS)}


def update_totals(totals, reconstruction, reuse, total_loss, output, batch_size, collect_metrics: bool):
    totals["loss"].add_(total_loss.detach() * batch_size)
    for name in ("recon", "delta", "root_pos", "root_rot", "joint", "contact", "foot_slide", "foot_height"):
        totals[name].add_(getattr(reconstruction, name).detach() * batch_size)
    totals["reuse"].add_(reuse.loss.detach() * batch_size)
    if collect_metrics:
        for name in _REPRESENTATION_METRICS:
            totals[name].add_(output[name].detach() * batch_size)


def finalize_totals(totals, count: int, representation_count: int) -> dict[str, float]:
    if count == 0:
        raise ValueError("No samples were processed")
    result = {name: float(totals[name] / count) for name in _BASE_METRICS}
    if representation_count > 0:
        result.update({name: float(totals[name] / representation_count) for name in _REPRESENTATION_METRICS})
    return result


@torch.no_grad()
def evaluate(model, loader, feature_weights, feature_offset, feature_scale, loss_metadata, device, args, non_blocking):
    model.eval()
    totals = init_totals(device)
    count = 0
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=non_blocking)
        output = model(motion, collect_metrics=True)
        reconstruction, reuse, total_loss = compute_losses(
            model, motion, output, feature_weights, feature_offset, feature_scale, loss_metadata, args
        )
        batch_size = motion.shape[0]
        update_totals(totals, reconstruction, reuse, total_loss, output, batch_size, collect_metrics=True)
        count += batch_size
    return finalize_totals(totals, count, count)


def train_one_epoch(
    model, loader, optimizer, feature_weights, feature_offset, feature_scale, loss_metadata, device, args, non_blocking, writer, global_step
):
    model.train()
    totals = init_totals(device)
    count = 0
    representation_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=non_blocking)
        collect_metrics = global_step == 0 or global_step % args.metrics_interval == 0
        output = model(motion, collect_metrics=collect_metrics)
        reconstruction, reuse, total_loss = compute_losses(
            model, motion, output, feature_weights, feature_offset, feature_scale, loss_metadata, args
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        batch_size = motion.shape[0]
        update_totals(totals, reconstruction, reuse, total_loss, output, batch_size, collect_metrics)
        count += batch_size
        if collect_metrics:
            representation_count += batch_size
        global_step += 1
        if global_step == 1 or global_step % args.log_every == 0:
            writer.add_scalar("train_step/loss", total_loss.item(), global_step)
            writer.add_scalar("train_step/reuse", reuse.loss.item(), global_step)
            for group, value in zip(GROUP_NAMES, reuse.group_losses):
                writer.add_scalar(f"train_step/reuse_{group}", value.item(), global_step)
            if collect_metrics:
                for group, value in zip(GROUP_NAMES, output["group_coordinate_change_rates"]):
                    writer.add_scalar(f"train_step/change_{group}", value.item(), global_step)
            writer.add_scalar("train_step/lr", optimizer.param_groups[0]["lr"], global_step)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    stats = finalize_totals(totals, count, representation_count)
    stats["epoch_seconds"] = seconds
    stats["samples_per_second"] = count / max(seconds, 1e-8)
    return stats, global_step


def serializable_args(args) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def build_checkpoint(model, optimizer, scheduler, args, train_dataset, epoch, global_step, best_val, train_stats, val_stats):
    unwrapped = unwrap_model(model)
    return {
        "model_family": "part_fsq",
        "model_config": unwrapped.config,
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": serializable_args(args),
        "representation": {
            "tokenizer": "hierarchical_part_fsq",
            "coordinate_order": list(GROUP_NAMES),
            "coordinate_counts": {group: unwrapped.layout.group_slices[group].stop - unwrapped.layout.group_slices[group].start for group in GROUP_NAMES},
            "num_coordinates": unwrapped.num_coordinates,
            "num_levels": unwrapped.num_levels,
            "stream_dim": unwrapped.stream_dim,
            "temporal_downsample": 1,
            "frame_rate": 60,
            "receptive_field": unwrapped.receptive_field,
            "lookahead_frames": unwrapped.lookahead_frames,
            "precision": "fp32",
        },
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
    if args.feature_database is None:
        raise ValueError("--feature-database is required")
    if args.stream_dim <= 0 or args.fsq_num_levels <= 1:
        raise ValueError("stream_dim must be positive and fsq_num_levels must be > 1")
    if args.metrics_interval <= 0:
        raise ValueError("metrics_interval must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = bool(args.pin_memory and device.type == "cuda")
    non_blocking = use_pin_memory
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args, pin_memory=use_pin_memory)
    model = HierarchicalPartFSQMotionAutoencoder(
        names=train_dataset.names,
        parents=train_dataset.parents,
        motion_dim=train_dataset.motion_dim,
        stream_dim=args.stream_dim,
        num_levels=args.fsq_num_levels,
        fsq_scale=args.fsq_scale,
        fsq_preserve_symmetry=args.fsq_preserve_symmetry,
        fsq_noise_dropout=args.fsq_noise_dropout,
    )
    if train_dataset.window_size != model.receptive_field:
        raise ValueError(f"Dataset window_size={train_dataset.window_size}, model receptive_field={model.receptive_field}")
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    feature_weights = torch.from_numpy(train_dataset.model_feature_weights().astype("float32")).to(device)
    feature_stats = train_dataset.feature_stats()
    feature_offset = torch.from_numpy(feature_stats.offset.astype("float32")).to(device)
    feature_scale = torch.from_numpy(feature_stats.scale.astype("float32")).to(device)
    loss_metadata = build_loss_metadata(train_dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    config_path = args.outdir / f"{run_name}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_args(args), handle, sort_keys=False, allow_unicode=False)
    writer = SummaryWriter(log_dir=(args.outdir / "tensorboard" / run_name).as_posix())
    writer.add_text("config/args", json.dumps(serializable_args(args), indent=2), 0)

    best_val = None
    global_step = 0
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_family") != "part_fsq":
            raise ValueError(f"Unsupported Part-FSQ checkpoint family: {checkpoint.get('model_family')}")
        if checkpoint["model_config"] != unwrap_model(model).config:
            raise ValueError("Resume checkpoint model_config does not match the current Part-FSQ model")
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_val = checkpoint["best_val"]

    print(f"device={device} precision=fp32")
    print(f"train_summary={train_dataset.split_summary()}")
    print(f"val_summary={val_dataset.split_summary()}")
    print(f"part_layout={unwrap_model(model).layout.to_dict()}")
    for epoch in range(start_epoch, args.epochs):
        train_stats, global_step = train_one_epoch(
            model, train_loader, optimizer, feature_weights, feature_offset, feature_scale, loss_metadata,
            device, args, non_blocking, writer, global_step
        )
        val_stats = evaluate(
            model, val_loader, feature_weights, feature_offset, feature_scale, loss_metadata, device, args, non_blocking
        )
        scheduler.step()
        for prefix, stats in (("train", train_stats), ("val", val_stats)):
            for name, value in stats.items():
                writer.add_scalar(f"{prefix}/{name}", value, epoch + 1)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_stats['loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} val_reuse={val_stats['reuse']:.6f} "
            f"train_samples_per_second={train_stats['samples_per_second']:.2f}"
        )
        is_best = best_val is None or val_stats["loss"] < best_val
        if is_best:
            best_val = val_stats["loss"]
        if is_best or (args.save_every > 0 and (epoch + 1) % args.save_every == 0):
            checkpoint = build_checkpoint(
                model, optimizer, scheduler, args, train_dataset, epoch, global_step, best_val, train_stats, val_stats
            )
            suffix = "best" if is_best else f"epoch{epoch + 1}"
            torch.save(checkpoint, args.outdir / f"{run_name}_{suffix}.pt")
    writer.close()


if __name__ == "__main__":
    main()
