"""Teacher-forced control ablations for a conditional FSQ generator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.fsq_token_dataset import build_fsq_token_store
from datasets.fsq_trajectory_dataset import (
    FSQConditionalDataset,
    TrajectoryNormalization,
    build_fsq_trajectory_store,
)
from models.fsq_generator import (
    FSQConditionalTransformerGenerator,
    STYLE_CACHE_POLICY,
    STYLE_CONDITIONING,
)
from train_fsq_generator import choose_device, maybe_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate style/trajectory ablations for a conditional FSQ generator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-database", type=Path, required=True)
    parser.add_argument("--trajectory-database", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--ablations",
        type=str,
        default="true,shuffled-style,zero-trajectory,shuffled-trajectory",
        help="Comma-separated choices: true, shuffled-style, zero-trajectory, shuffled-trajectory.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path, token_store, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq_conditional_generator":
        raise ValueError(f"Unsupported checkpoint family: {checkpoint.get('model_family')}")
    if checkpoint.get("style_conditioning") != STYLE_CONDITIONING:
        raise ValueError("Checkpoint is not a causal dynamic-FiLM conditional generator")
    if checkpoint.get("style_cache_policy") != STYLE_CACHE_POLICY:
        raise ValueError("Checkpoint does not use the append-only style cache policy")
    if checkpoint.get("tokenizer_checkpoint_sha256") != token_store.checkpoint_sha256:
        raise ValueError("Conditional checkpoint and token database use different FSQ tokenizers")
    model = FSQConditionalTransformerGenerator(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model


def prepare_batch(batch: dict, device: torch.device):
    indices = batch["indices"].to(device, non_blocking=device.type == "cuda").long()
    styles = batch["style_id"].to(device, non_blocking=device.type == "cuda").long()
    trajectory = batch["trajectory"].to(device, non_blocking=device.type == "cuda").float()
    valid = batch["trajectory_valid"].to(device, non_blocking=device.type == "cuda").bool()
    return indices[:, :-1], indices[:, 1:], styles, trajectory, valid


def ablate(
    name: str,
    style_ids: torch.Tensor,
    trajectory: torch.Tensor,
    valid: torch.Tensor,
    num_styles: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if name == "true":
        return style_ids, trajectory, valid
    if name == "shuffled-style":
        # Dataset windows are grouped by style, so rolling a batch often leaves
        # nearly every label unchanged. A cyclic offset guarantees a wrong label.
        return (style_ids + 1) % num_styles, trajectory, valid
    if name == "zero-trajectory":
        return style_ids, torch.zeros_like(trajectory), torch.zeros_like(valid)
    if name == "shuffled-trajectory":
        return style_ids, trajectory.roll(1, dims=0), valid.roll(1, dims=0)
    raise ValueError(f"Unsupported ablation {name!r}")


def evaluate(model: FSQConditionalTransformerGenerator, loader, device: torch.device, ablation_name: str) -> dict[str, float]:
    total_nll = 0.0
    total_correct = 0
    total_level_error = 0.0
    total_tokens = 0
    total_valid_controls = 0
    total_controls = 0
    with torch.inference_mode():
        for batch in loader:
            inputs, targets, styles, trajectory, valid = prepare_batch(batch, device)
            styles, trajectory, valid = ablate(
                ablation_name,
                styles,
                trajectory,
                valid,
                model.num_styles,
            )
            logits = model(
                inputs,
                style_ids=styles,
                trajectory=trajectory,
                trajectory_valid=valid,
            )["logits"]
            loss = F.cross_entropy(logits.reshape(-1, model.num_levels), targets.reshape(-1))
            prediction = logits.argmax(dim=-1)
            token_count = targets.numel()
            total_nll += float(loss) * token_count
            total_correct += int((prediction == targets).sum())
            total_level_error += float((prediction - targets).abs().sum())
            total_tokens += token_count
            total_valid_controls += int(valid.sum())
            total_controls += valid.numel()
    if total_tokens == 0:
        raise ValueError("No windows were evaluated")
    nll = total_nll / total_tokens
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "coordinate_accuracy": total_correct / total_tokens,
        "level_mae": total_level_error / total_tokens,
        "valid_control_fraction": total_valid_controls / max(total_controls, 1),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers must be non-negative")
    token_store = build_fsq_token_store(args.token_database)
    device = choose_device(args.device)
    checkpoint, model = load_checkpoint(args.checkpoint, token_store, device)
    trajectory_database = args.trajectory_database or Path(checkpoint["trajectory_database"])
    trajectory_store = build_fsq_trajectory_store(trajectory_database, token_store)
    normalization = TrajectoryNormalization.from_checkpoint(checkpoint["trajectory_normalization"])
    dataset = FSQConditionalDataset(args.split, token_store, trajectory_store, normalization)
    dataset = maybe_subset(dataset, args.max_samples, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    ablations = [name.strip() for name in args.ablations.split(",") if name.strip()]
    if not ablations:
        raise ValueError("At least one ablation is required")
    results = {name: evaluate(model, loader, device, name) for name in ablations}
    output = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(checkpoint.get("epoch", 0)),
        "token_database": str(token_store.database),
        "trajectory_database": str(trajectory_store.database),
        "split": args.split,
        "windows": len(dataset),
        "trajectory_feature_order": trajectory_store.feature_order,
        "future_frames": trajectory_store.future_frames.tolist(),
        "control_alignment": checkpoint.get("control_alignment"),
        "ablations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
