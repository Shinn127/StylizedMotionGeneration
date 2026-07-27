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
from models.part_layout import PART_NAMES
from models.residual_part_fsq import GROUP_COORDINATES, GROUP_NAMES, ResidualPartFSQMotionAutoencoder
from models.residual_part_fsq_losses import adaptive_base_reuse_loss
from motion_features import serialize_motion_feature_stats


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 0
    return 0 if cpu_count <= 1 else min(4, cpu_count - 1)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, required=True)
    pre_args, remaining = pre.parse_known_args(argv)
    config = load_config(pre_args.config)

    def cfg(name, default):
        return config.get(name, default)

    def cfg_path(name, default):
        value = cfg(name, default)
        return None if value is None else Path(value)

    parser = argparse.ArgumentParser(description="Train full-FP32 Residual Part-FSQ on fixed 64-frame windows.")
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
    parser.add_argument("--prefetch-factor", type=int, default=cfg("prefetch_factor", 4))
    parser.add_argument("--log-every", type=int, default=cfg("log_every", 50))
    parser.add_argument("--metrics-interval", type=int, default=cfg("metrics_interval", 100))
    parser.add_argument("--run-name", type=str, default=cfg("run_name", None))
    parser.add_argument("--base-code-dim", type=int, default=cfg("base_code_dim", 128))
    parser.add_argument("--base-width", type=int, default=cfg("base_width", 512))
    parser.add_argument("--part-state-dim", type=int, default=cfg("part_state_dim", 64))
    parser.add_argument("--residual-decoder-dim", type=int, default=cfg("residual_decoder_dim", 128))
    parser.add_argument("--residual-decoder-width", type=int, default=cfg("residual_decoder_width", 128))
    parser.add_argument("--residual-hidden-dim", type=int, default=cfg("residual_hidden_dim", 64))
    parser.add_argument("--residual-group-dropout", type=float, default=cfg("residual_group_dropout", 0.1))
    parser.add_argument("--fsq-num-levels", type=int, default=cfg("fsq_num_levels", 9))
    parser.add_argument("--fsq-scale", type=float, default=cfg("fsq_scale", None))
    parser.add_argument("--fsq-preserve-symmetry", action="store_true", default=cfg("fsq_preserve_symmetry", False))
    parser.add_argument("--fsq-noise-dropout", type=float, default=cfg("fsq_noise_dropout", 0.0))
    parser.add_argument("--base-loss-weight", type=float, default=cfg("base_loss_weight", 0.25))
    parser.add_argument("--base-reuse-weight", type=float, default=cfg("base_reuse_weight", 0.0025))
    parser.add_argument("--base-reuse-threshold", type=float, default=cfg("base_reuse_threshold", 1.0))
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
    parser.add_argument("--outdir", type=Path, default=cfg_path("outdir", Path("outputs/residual_part_fsq")))
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
    train_dataset = FeatureDataset(args.split_train, store)
    val_dataset = FeatureDataset(args.split_val, store)
    train_loader = DataLoader(train_dataset, **dataloader_kwargs(args, True, pin_memory))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(args, False, pin_memory))
    return train_dataset, val_dataset, train_loader, val_loader


def build_lr_scheduler(optimizer, args):
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")
    min_factor = args.min_lr / args.lr
    warmup_epochs = max(0, int(args.warmup_epochs))
    decay_epochs = max(1, int(args.epochs) - warmup_epochs)

    def lr_lambda(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return max(min_factor, float(epoch_index + 1) / warmup_epochs)
        progress = min(max(float(epoch_index - warmup_epochs) / decay_epochs, 0.0), 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def build_loss_metadata(dataset, device: torch.device) -> dict[str, object]:
    try:
        foot_indices = (dataset.names.index("LeftToeBase"), dataset.names.index("RightToeBase"))
    except ValueError as exc:
        raise ValueError("Losses require LeftToeBase and RightToeBase") from exc
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


def reconstruction_loss(motion, recon, feature_weights, feature_offset, feature_scale, metadata, args, base: bool):
    output = {"recon_state": recon, "commit_loss": recon.new_zeros(())}
    return compute_motion_reconstruction_losses(
        batch_motion=motion,
        output=output,
        feature_weights=feature_weights,
        feature_offset=feature_offset,
        feature_scale=feature_scale,
        delta_weight=1.0 if base else args.delta_weight,
        commit_weight=0.0,
        root_pos_weight=0.1 if base else args.root_pos_weight,
        root_rot_weight=0.1 if base else args.root_rot_weight,
        root_dt=args.root_dt,
        joint_weight=0.25 if base else args.joint_weight,
        contact_weight=0.1 if base else args.contact_weight,
        foot_slide_weight=0.0 if base else args.foot_slide_weight,
        foot_height_weight=0.0 if base else args.foot_height_weight,
        contact_temperature=args.contact_temperature,
        ref_pos=metadata["ref_pos"],
        parents=metadata["parents"],
        joint_weights=metadata["joint_weights"],
        foot_indices=metadata["foot_indices"],
    )


def compute_losses(model, motion, output, feature_weights, feature_offset, feature_scale, metadata, args):
    final = reconstruction_loss(
        motion, output["recon_state"], feature_weights, feature_offset, feature_scale, metadata, args, base=False
    )
    base = reconstruction_loss(
        motion, output["base_recon_state"], feature_weights, feature_offset, feature_scale, metadata, args, base=True
    )
    reuse = adaptive_base_reuse_loss(
        base_codes=output["base_codes"],
        target_motion=motion,
        layout=unwrap_model(model).layout,
        thresholds=args.base_reuse_threshold,
        feature_weights=feature_weights,
        contact_values=final.target_contact,
        contact_transition_threshold=args.contact_transition_threshold,
        level_step=2.0 / float(args.fsq_num_levels - 1),
    )
    total = final.loss + float(args.base_loss_weight) * base.loss + float(args.base_reuse_weight) * reuse.loss
    return final, base, reuse, total


LOSS_METRICS = (
    "loss", "recon", "delta", "root_pos", "root_rot", "joint", "contact", "foot_slide", "foot_height",
    "base_loss", "base_recon", "base_delta", "base_joint", "base_reuse",
)
REPRESENTATION_METRICS = (
    "level_perplexity", "level_usage", "tuple_unique_ratio", "tuple_change_rate", "coordinate_change_rate",
)
RESIDUAL_METRICS = tuple(f"residual_energy_{part}" for part in PART_NAMES)


def init_totals(device: torch.device):
    return {name: torch.zeros((), device=device) for name in (*LOSS_METRICS, *REPRESENTATION_METRICS, *RESIDUAL_METRICS)}


def update_totals(totals, final, base, reuse, total, output, batch_size: int, collect_metrics: bool):
    values = {
        "loss": total,
        "recon": final.recon,
        "delta": final.delta,
        "root_pos": final.root_pos,
        "root_rot": final.root_rot,
        "joint": final.joint,
        "contact": final.contact,
        "foot_slide": final.foot_slide,
        "foot_height": final.foot_height,
        "base_loss": base.loss,
        "base_recon": base.recon,
        "base_delta": base.delta,
        "base_joint": base.joint,
        "base_reuse": reuse.loss,
    }
    for name, value in values.items():
        totals[name].add_(value.detach() * batch_size)
    for part in PART_NAMES:
        totals[f"residual_energy_{part}"].add_(output["part_residuals"][part].detach().abs().mean() * batch_size)
    if collect_metrics:
        for name in REPRESENTATION_METRICS:
            totals[name].add_(output[name].detach() * batch_size)


def finalize_totals(totals, count: int, representation_count: int):
    if count == 0:
        raise ValueError("No samples were processed")
    result = {name: float(totals[name] / count) for name in (*LOSS_METRICS, *RESIDUAL_METRICS)}
    if representation_count > 0:
        result.update({name: float(totals[name] / representation_count) for name in REPRESENTATION_METRICS})
    return result


@torch.no_grad()
def evaluate(model, loader, tensors, metadata, device, args, non_blocking):
    model.eval()
    totals = init_totals(device)
    count = 0
    for batch in loader:
        motion = batch["motion"].to(device, non_blocking=non_blocking)
        output = model(motion, collect_metrics=True)
        final, base, reuse, total = compute_losses(model, motion, output, *tensors, metadata, args)
        batch_size = motion.shape[0]
        update_totals(totals, final, base, reuse, total, output, batch_size, True)
        count += batch_size
    return finalize_totals(totals, count, count)


def train_one_epoch(model, loader, optimizer, tensors, metadata, device, args, non_blocking, writer, global_step):
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
        final, base, reuse, total = compute_losses(model, motion, output, *tensors, metadata, args)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if args.grad_clip_norm is not None and args.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        batch_size = motion.shape[0]
        update_totals(totals, final, base, reuse, total, output, batch_size, collect_metrics)
        count += batch_size
        if collect_metrics:
            representation_count += batch_size
        global_step += 1
        if global_step == 1 or global_step % args.log_every == 0:
            writer.add_scalar("train_step/loss", total.item(), global_step)
            writer.add_scalar("train_step/base_loss", base.loss.item(), global_step)
            writer.add_scalar("train_step/base_reuse", reuse.loss.item(), global_step)
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


def serializable_args(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def build_checkpoint(model, optimizer, scheduler, args, dataset, epoch, global_step, best_val, train_stats, val_stats):
    unwrapped = unwrap_model(model)
    return {
        "model_family": "residual_part_fsq",
        "model_config": unwrapped.config,
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": serializable_args(args),
        "representation": {
            "tokenizer": "residual_part_fsq_40x9_v1",
            "coordinate_order": list(GROUP_NAMES),
            "coordinate_counts": dict(GROUP_COORDINATES),
            "num_coordinates": unwrapped.num_coordinates,
            "num_levels": unwrapped.num_levels,
            "temporal_downsample": 1,
            "frame_rate": 60,
            "receptive_field": unwrapped.receptive_field,
            "lookahead_frames": unwrapped.lookahead_frames,
            "precision": "fp32",
        },
        "stats": serialize_motion_feature_stats(
            dataset.feature_stats(), names=dataset.names, parents=dataset.parents, joint_subset=dataset.joint_subset
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
    if args.metrics_interval <= 0 or args.fsq_num_levels <= 1:
        raise ValueError("metrics_interval must be positive and fsq_num_levels must be > 1")
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    non_blocking = pin_memory
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args, pin_memory)
    model = ResidualPartFSQMotionAutoencoder(
        names=train_dataset.names,
        parents=train_dataset.parents,
        motion_dim=train_dataset.motion_dim,
        base_code_dim=args.base_code_dim,
        base_width=args.base_width,
        part_state_dim=args.part_state_dim,
        residual_decoder_dim=args.residual_decoder_dim,
        residual_decoder_width=args.residual_decoder_width,
        residual_hidden_dim=args.residual_hidden_dim,
        num_levels=args.fsq_num_levels,
        fsq_scale=args.fsq_scale,
        fsq_preserve_symmetry=args.fsq_preserve_symmetry,
        fsq_noise_dropout=args.fsq_noise_dropout,
        residual_group_dropout=args.residual_group_dropout,
    )
    if train_dataset.window_size != model.receptive_field:
        raise ValueError(f"Dataset window_size={train_dataset.window_size}, model RF={model.receptive_field}")
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    stats = train_dataset.feature_stats()
    tensors = (
        torch.from_numpy(train_dataset.model_feature_weights().astype("float32")).to(device),
        torch.from_numpy(stats.offset.astype("float32")).to(device),
        torch.from_numpy(stats.scale.astype("float32")).to(device),
    )
    metadata = build_loss_metadata(train_dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args)
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    with (args.outdir / f"{run_name}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_args(args), handle, sort_keys=False)
    writer = SummaryWriter(log_dir=(args.outdir / "tensorboard" / run_name).as_posix())
    writer.add_text("config/args", json.dumps(serializable_args(args), indent=2), 0)

    best_val = None
    global_step = 0
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_family") != "residual_part_fsq":
            raise ValueError(f"Unsupported checkpoint family: {checkpoint.get('model_family')}")
        if checkpoint["model_config"] != unwrap_model(model).config:
            raise ValueError("Resume model_config does not match")
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_val = checkpoint["best_val"]

    print(f"device={device} precision=fp32")
    print(f"train_summary={train_dataset.split_summary()}")
    print(f"val_summary={val_dataset.split_summary()}")
    print(f"coordinate_layout={GROUP_COORDINATES}")
    for epoch in range(start_epoch, args.epochs):
        train_stats, global_step = train_one_epoch(
            model, train_loader, optimizer, tensors, metadata, device, args, non_blocking, writer, global_step
        )
        val_stats = evaluate(model, val_loader, tensors, metadata, device, args, non_blocking)
        scheduler.step()
        for prefix, values in (("train", train_stats), ("val", val_stats)):
            for name, value in values.items():
                writer.add_scalar(f"{prefix}/{name}", value, epoch + 1)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_stats['loss']:.6f} "
            f"train_recon={train_stats['recon']:.6f} train_base_recon={train_stats['base_recon']:.6f} "
            f"val_loss={val_stats['loss']:.6f} val_recon={val_stats['recon']:.6f} "
            f"val_base_recon={val_stats['base_recon']:.6f} "
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

