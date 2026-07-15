from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets.fsq_token_dataset import build_fsq_token_store
from evaluate_fsq_generator import (
    load_fsq_decoder,
    load_generator_checkpoint,
    resolve_fsq_checkpoint,
)
from train_fsq_generator import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and decode a motion continuation in FSQ token space.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="FSQ generator checkpoint.")
    parser.add_argument("--token-database", type=Path, required=True)
    parser.add_argument("--fsq-checkpoint", type=Path, default=None)
    parser.add_argument("--range-idx", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--seed-frames", type=int, default=64)
    parser.add_argument("--generate-frames", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Use categorical sampling instead of greedy generation.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_request(args: argparse.Namespace, store, model, indices: np.ndarray) -> None:
    if args.range_idx < 0 or args.range_idx >= len(store.token_files):
        raise ValueError(f"range_idx must be in [0, {len(store.token_files) - 1}], got {args.range_idx}")
    if args.start < 0:
        raise ValueError("start must be non-negative")
    if args.seed_frames <= 0 or args.generate_frames <= 0:
        raise ValueError("seed_frames and generate_frames must be positive")
    if args.seed_frames > model.context_frames:
        raise ValueError(
            f"seed_frames={args.seed_frames} exceeds generator context_frames={model.context_frames}"
        )
    if args.start + args.seed_frames > indices.shape[0]:
        raise ValueError(
            f"Seed [{args.start}, {args.start + args.seed_frames}) exceeds shard length {indices.shape[0]}"
        )
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    store = build_fsq_token_store(args.token_database)
    generator_checkpoint, model = load_generator_checkpoint(args.checkpoint, store, device)
    if args.range_idx < 0 or args.range_idx >= len(store.token_files):
        raise ValueError(f"range_idx must be in [0, {len(store.token_files) - 1}], got {args.range_idx}")
    shard_indices = np.load(store.token_files[args.range_idx], mmap_mode="r")
    validate_request(args, store, model, shard_indices)

    seed_np = np.asarray(
        shard_indices[args.start : args.start + args.seed_frames],
        dtype=np.int64,
    ).copy()
    seed_indices = torch.from_numpy(seed_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        generated_indices = model.generate(
            seed_indices,
            num_steps=args.generate_frames,
            temperature=args.temperature,
            greedy=not args.sample,
        )

    fsq_path = resolve_fsq_checkpoint(args.fsq_checkpoint, store)
    _, fsq_model = load_fsq_decoder(fsq_path, device)
    combined_indices = torch.cat((seed_indices, generated_indices), dim=1)
    with torch.inference_mode():
        decoded_features = fsq_model.decode_from_indices(combined_indices)
    generated_features = decoded_features[:, args.seed_frames :]

    target_available = args.start + args.seed_frames + args.generate_frames <= shard_indices.shape[0]
    target_indices = None
    target_features = None
    if target_available:
        target_np = np.asarray(
            shard_indices[
                args.start : args.start + args.seed_frames + args.generate_frames
            ],
            dtype=np.int64,
        ).copy()
        target_indices = torch.from_numpy(target_np).unsqueeze(0).to(device)
        with torch.inference_mode():
            target_features = fsq_model.decode_from_indices(target_indices)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "seed_indices.npy", seed_np.astype(np.uint8))
    np.save(
        args.output_dir / "generated_indices.npy",
        generated_indices[0].cpu().numpy().astype(np.uint8),
    )
    np.save(
        args.output_dir / "seed_and_generated_indices.npy",
        combined_indices[0].cpu().numpy().astype(np.uint8),
    )
    np.save(
        args.output_dir / "decoded_features.npy",
        decoded_features[0].cpu().numpy().astype(np.float32),
    )
    np.save(
        args.output_dir / "generated_features.npy",
        generated_features[0].cpu().numpy().astype(np.float32),
    )
    if target_indices is not None and target_features is not None:
        np.save(
            args.output_dir / "target_indices.npy",
            target_indices[0].cpu().numpy().astype(np.uint8),
        )
        np.save(
            args.output_dir / "target_features.npy",
            target_features[0].cpu().numpy().astype(np.float32),
        )

    metadata = {
        "generator_checkpoint": str(args.checkpoint),
        "generator_epoch": int(generator_checkpoint.get("epoch", 0)),
        "fsq_checkpoint": str(fsq_path),
        "token_database": str(store.database),
        "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
        "range_idx": args.range_idx,
        "range_name": str(store.range_names[args.range_idx]),
        "mirror": bool(store.range_mirror[args.range_idx]),
        "start": args.start,
        "seed_frames": args.seed_frames,
        "generate_frames": args.generate_frames,
        "sampling": "categorical" if args.sample else "greedy",
        "temperature": args.temperature,
        "target_available": target_available,
        "device": str(device),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
