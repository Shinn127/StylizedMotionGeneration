import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.motion_dataset import MotionDataset
from models.vqvae import CausalMotionVQVAE


def default_num_workers() -> int:
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 1:
        return 0
    return min(8, cpu_count - 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-val", default="val")
    parser.add_argument("--database-path", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--use-full-skeleton", action="store_true")
    parser.add_argument("--use-root-cond", dest="use_root_cond", action="store_true")
    parser.add_argument("--no-use-root-cond", dest="use_root_cond", action="store_false")
    parser.set_defaults(use_root_cond=True)
    parser.add_argument("--root-cond-dim", type=int, default=6)
    parser.add_argument("--code-dim", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--commit-weight", type=float, default=0.25)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/vqvae"))
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=torch.cuda.is_available())
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)
    return parser.parse_args()


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
    dataset_kwargs = {
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "database_path": args.database_path,
        "use_full_skeleton": args.use_full_skeleton,
    }
    train_dataset = MotionDataset(split=args.split_train, **dataset_kwargs)
    val_dataset = MotionDataset(split=args.split_val, **dataset_kwargs)
    train_loader = DataLoader(train_dataset, **dataloader_kwargs(args, shuffle=True, pin_memory=pin_memory))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(args, shuffle=False, pin_memory=pin_memory))
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(args, motion_dim):
    return CausalMotionVQVAE(
        motion_dim=motion_dim,
        root_cond_dim=args.root_cond_dim,
        use_root_cond=args.use_root_cond,
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
        num_heads=args.num_heads,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
    )


def compute_losses(batch_motion, output, feature_weights, delta_weight, commit_weight):
    recon = output["recon_state"]
    feature_weights = feature_weights.view(1, 1, -1).to(batch_motion.device)
    recon_loss = torch.mean(feature_weights * torch.abs(recon - batch_motion))
    delta_loss = F.l1_loss(recon[:, 1:] - recon[:, :-1], batch_motion[:, 1:] - batch_motion[:, :-1])
    commit_loss = output["commit_loss"]
    loss = recon_loss + delta_weight * delta_loss + commit_weight * commit_loss
    return loss, recon_loss, delta_loss, commit_loss


def init_metric_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "recon": 0.0,
        "delta": 0.0,
        "commit": 0.0,
        "mean_head_perplexity": 0.0,
    }


def update_metric_totals(totals, loss, recon_loss, delta_loss, commit_loss, perplexity, batch_size):
    totals["loss"] += loss.item() * batch_size
    totals["recon"] += recon_loss.item() * batch_size
    totals["delta"] += delta_loss.item() * batch_size
    totals["commit"] += commit_loss.item() * batch_size
    totals["mean_head_perplexity"] += perplexity.item() * batch_size


def finalize_metric_totals(totals, count):
    if count == 0:
        raise ValueError("Cannot finalize metrics with count=0")
    return {name: value / count for name, value in totals.items()}


def move_motion_to_device(batch, device, non_blocking: bool):
    return batch["motion"].to(device, non_blocking=non_blocking)


def evaluate(model, loader, feature_weights, device, delta_weight, commit_weight, non_blocking: bool):
    model.eval()
    totals = init_metric_totals()
    count = 0
    with torch.no_grad():
        for batch in loader:
            motion = move_motion_to_device(batch, device, non_blocking=non_blocking)
            output = model(motion)
            loss, recon_loss, delta_loss, commit_loss = compute_losses(
                motion,
                output,
                feature_weights,
                delta_weight=delta_weight,
                commit_weight=commit_weight,
            )
            batch_size = motion.shape[0]
            update_metric_totals(
                totals,
                loss,
                recon_loss,
                delta_loss,
                commit_loss,
                output["mean_head_perplexity"],
                batch_size,
            )
            count += batch_size
    return finalize_metric_totals(totals, count)


def train_one_epoch(
    model,
    loader,
    optimizer,
    feature_weights,
    device,
    delta_weight,
    commit_weight,
    writer,
    global_step,
    log_every,
    non_blocking: bool,
):
    model.train()
    totals = init_metric_totals()
    count = 0
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        motion = move_motion_to_device(batch, device, non_blocking=non_blocking)
        output = model(motion)
        loss, recon_loss, delta_loss, commit_loss = compute_losses(
            motion,
            output,
            feature_weights,
            delta_weight=delta_weight,
            commit_weight=commit_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = motion.shape[0]
        update_metric_totals(
            totals,
            loss,
            recon_loss,
            delta_loss,
            commit_loss,
            output["mean_head_perplexity"],
            batch_size,
        )
        count += batch_size
        global_step += 1

        if writer is not None and (global_step == 1 or global_step % log_every == 0):
            writer.add_scalar("train_step/loss", loss.item(), global_step)
            writer.add_scalar("train_step/recon", recon_loss.item(), global_step)
            writer.add_scalar("train_step/delta", delta_loss.item(), global_step)
            writer.add_scalar("train_step/commit", commit_loss.item(), global_step)
            writer.add_scalar("train_step/mean_head_perplexity", output["mean_head_perplexity"].item(), global_step)
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
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = bool(args.pin_memory and device.type == "cuda")
    non_blocking = use_pin_memory
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(args, pin_memory=use_pin_memory)
    feature_weights = torch.from_numpy(train_dataset.feature_stats().weights.astype("float32"))

    model = build_model(args, motion_dim=train_dataset.motion_dim)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_name = build_run_name(args)
    log_dir = args.outdir / "tensorboard" / run_name
    writer = SummaryWriter(log_dir=log_dir.as_posix())
    writer.add_text("config/args", json.dumps(serialize_args(args), indent=2, ensure_ascii=False))
    writer.add_text("dataset/train_summary", json.dumps(train_dataset.split_summary(), indent=2))
    writer.add_text("dataset/val_summary", json.dumps(val_dataset.split_summary(), indent=2))

    print(f"device={device}")
    print(f"train_summary={train_dataset.split_summary()}")
    print(f"val_summary={val_dataset.split_summary()}")
    print(f"tensorboard_logdir={log_dir}")

    best_val = None
    global_step = 0
    for epoch in range(args.epochs):
        train_stats, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            feature_weights,
            device,
            delta_weight=args.delta_weight,
            commit_weight=args.commit_weight,
            writer=writer,
            global_step=global_step,
            log_every=args.log_every,
            non_blocking=non_blocking,
        )

        val_stats = evaluate(
            model,
            val_loader,
            feature_weights,
            device,
            delta_weight=args.delta_weight,
            commit_weight=args.commit_weight,
            non_blocking=non_blocking,
        )

        writer.add_scalar("train_epoch/loss", train_stats["loss"], epoch + 1)
        writer.add_scalar("train_epoch/recon", train_stats["recon"], epoch + 1)
        writer.add_scalar("train_epoch/delta", train_stats["delta"], epoch + 1)
        writer.add_scalar("train_epoch/commit", train_stats["commit"], epoch + 1)
        writer.add_scalar("train_epoch/mean_head_perplexity", train_stats["mean_head_perplexity"], epoch + 1)
        writer.add_scalar("train_epoch/samples_per_second", train_stats["samples_per_second"], epoch + 1)
        writer.add_scalar("train_epoch/epoch_seconds", train_stats["epoch_seconds"], epoch + 1)
        writer.add_scalar("val/loss", val_stats["loss"], epoch + 1)
        writer.add_scalar("val/recon", val_stats["recon"], epoch + 1)
        writer.add_scalar("val/delta", val_stats["delta"], epoch + 1)
        writer.add_scalar("val/commit", val_stats["commit"], epoch + 1)
        writer.add_scalar("val/mean_head_perplexity", val_stats["mean_head_perplexity"], epoch + 1)
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
            f"train_commit={train_stats['commit']:.6f} "
            f"train_perplexity={train_stats['mean_head_perplexity']:.6f} "
            f"train_samples_per_second={train_stats['samples_per_second']:.2f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_recon={val_stats['recon']:.6f} "
            f"val_delta={val_stats['delta']:.6f} "
            f"val_commit={val_stats['commit']:.6f} "
            f"val_perplexity={val_stats['mean_head_perplexity']:.6f}"
        )

        checkpoint = {
            "model": unwrap_model(model).state_dict(),
            "args": serialize_args(args),
            "motion_dim": train_dataset.motion_dim,
            "epoch": epoch + 1,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        torch.save(checkpoint, args.outdir / "last.pt")

        if best_val is None or val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(checkpoint, args.outdir / "best.pt")

    writer.close()


if __name__ == "__main__":
    main()
