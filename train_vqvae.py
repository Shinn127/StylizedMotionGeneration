import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.motion_dataset import MotionDataset
from models.vqvae import CausalMotionVQVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-train", default="train")
    parser.add_argument("--split-val", default="val")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-full-skeleton", action="store_true")
    parser.add_argument("--use-root-cond", dest="use_root_cond", action="store_true")
    parser.add_argument("--no-use-root-cond", dest="use_root_cond", action="store_false")
    parser.set_defaults(use_root_cond=True)
    parser.add_argument("--root-cond-dim", type=int, default=6)
    parser.add_argument("--code-dim", type=int, default=256)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--commit-weight", type=float, default=0.25)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/vqvae"))
    return parser.parse_args()


def build_dataloaders(args):
    train_dataset = MotionDataset(
        split=args.split_train,
        window_size=args.window_size,
        use_full_skeleton=args.use_full_skeleton,
    )
    val_dataset = MotionDataset(
        split=args.split_val,
        window_size=args.window_size,
        use_full_skeleton=args.use_full_skeleton,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataset, train_loader, val_loader


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


def evaluate(model, loader, feature_weights, device, delta_weight, commit_weight):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            output = model(motion)
            loss, _, _, _ = compute_losses(motion, output, feature_weights, delta_weight, commit_weight)
            total += loss.item() * motion.shape[0]
            count += motion.shape[0]
    return total / count


def train_one_epoch(model, loader, optimizer, feature_weights, device, delta_weight, commit_weight):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_delta = 0.0
    total_commit = 0.0
    total_perplexity = 0.0
    count = 0

    for batch in loader:
        motion = batch["motion"].to(device)
        output = model(motion)
        loss, recon_loss, delta_loss, commit_loss = compute_losses(
            motion,
            output,
            feature_weights,
            delta_weight=delta_weight,
            commit_weight=commit_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = motion.shape[0]
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_delta += delta_loss.item() * batch_size
        total_commit += commit_loss.item() * batch_size
        total_perplexity += output["mean_head_perplexity"].item() * batch_size
        count += batch_size

    return {
        "loss": total_loss / count,
        "recon": total_recon / count,
        "delta": total_delta / count,
        "commit": total_commit / count,
        "mean_head_perplexity": total_perplexity / count,
    }


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader, val_loader = build_dataloaders(args)
    feature_weights = torch.from_numpy(train_dataset.feature_stats().weights.astype("float32"))

    model = build_model(args, motion_dim=train_dataset.motion_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = None
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            feature_weights,
            device,
            delta_weight=args.delta_weight,
            commit_weight=args.commit_weight,
        )

        val_loss = evaluate(
            model,
            val_loader,
            feature_weights,
            device,
            delta_weight=args.delta_weight,
            commit_weight=args.commit_weight,
        )
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_stats['loss']:.6f} "
            f"train_recon={train_stats['recon']:.6f} "
            f"train_delta={train_stats['delta']:.6f} "
            f"train_commit={train_stats['commit']:.6f} "
            f"mean_head_perplexity={train_stats['mean_head_perplexity']:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "motion_dim": train_dataset.motion_dim,
                },
                args.outdir / "best.pt",
            )


if __name__ == "__main__":
    main()
