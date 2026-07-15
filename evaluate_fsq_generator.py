from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.fsq_token_dataset import FSQTokenDataset, build_fsq_token_store
from encode_fsq_database import sha256_file
from models.fsq import FSQMotionAutoencoder
from models.fsq_generator import FSQCausalTransformerGenerator, FSQGeneratorCache
from models.losses import denormalize_motion_features, integrate_root_trajectory, reconstruct_joint_positions
from preprocess import quat
from train_fsq_generator import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a causal FSQ token generator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-database", type=Path, required=True)
    parser.add_argument("--fsq-checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rollout-samples", type=int, default=16)
    parser.add_argument("--seed-frames", type=int, default=32)
    parser.add_argument("--rollout-frames", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Use categorical sampling instead of greedy rollout.")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-steps", type=int, default=100)
    parser.add_argument("--root-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_generator_checkpoint(path: Path, store, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq_generator":
        raise ValueError(f"Unsupported checkpoint family: {checkpoint.get('model_family')}")
    if checkpoint["tokenizer_checkpoint_sha256"] != store.checkpoint_sha256:
        raise ValueError("Generator checkpoint and token database use different FSQ tokenizers")
    model = FSQCausalTransformerGenerator(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if (model.num_coordinates, model.num_levels) != (store.num_coordinates, store.num_levels):
        raise ValueError("Generator dimensions do not match the token database")
    return checkpoint, model


def resolve_fsq_checkpoint(cli_path: Path | None, store) -> Path:
    path = cli_path or Path(store.checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing FSQ checkpoint {path}; provide the matching path with --fsq-checkpoint"
        )
    digest = sha256_file(path)
    if digest != store.checkpoint_sha256:
        raise ValueError(
            f"FSQ checkpoint SHA256 {digest} does not match token database {store.checkpoint_sha256}"
        )
    return path


def load_fsq_decoder(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq":
        raise ValueError(f"Unsupported FSQ checkpoint family: {checkpoint.get('model_family')}")
    model = FSQMotionAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model


def maybe_subset(dataset, maximum: int | None, seed: int):
    if maximum is None or maximum >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    selection = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return Subset(dataset, selection)


def evaluate_teacher_forced(model, loader, device: torch.device) -> dict[str, object]:
    coordinate_nll = torch.zeros(model.num_coordinates, dtype=torch.float64)
    coordinate_correct = torch.zeros(model.num_coordinates, dtype=torch.float64)
    coordinate_level_error = torch.zeros(model.num_coordinates, dtype=torch.float64)
    coordinate_count = torch.zeros(model.num_coordinates, dtype=torch.float64)
    repeat_correct = 0
    repeat_level_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            indices = batch["indices"].to(device).long()
            inputs, targets = indices[:, :-1], indices[:, 1:]
            logits = model(inputs)["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, model.num_levels),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            predictions = logits.argmax(dim=-1)
            coordinate_nll += loss.sum(dim=(0, 1)).double().cpu()
            coordinate_correct += (predictions == targets).sum(dim=(0, 1)).double().cpu()
            coordinate_level_error += (predictions - targets).abs().sum(dim=(0, 1)).double().cpu()
            coordinate_count += torch.full(
                (model.num_coordinates,),
                targets.shape[0] * targets.shape[1],
                dtype=torch.float64,
            )
            repeat_correct += int((inputs == targets).sum())
            repeat_level_error += float((inputs - targets).abs().sum())

    if not bool((coordinate_count > 0).all()):
        raise ValueError("No teacher-forced samples were processed")
    per_coordinate = []
    for coordinate in range(model.num_coordinates):
        nll = float(coordinate_nll[coordinate] / coordinate_count[coordinate])
        per_coordinate.append(
            {
                "coordinate": coordinate,
                "nll": nll,
                "perplexity": math.exp(min(nll, 50.0)),
                "accuracy": float(coordinate_correct[coordinate] / coordinate_count[coordinate]),
                "level_mae": float(coordinate_level_error[coordinate] / coordinate_count[coordinate]),
            }
        )
    total = float(coordinate_count.sum())
    nll = float(coordinate_nll.sum() / total)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "coordinate_accuracy": float(coordinate_correct.sum() / total),
        "level_mae": float(coordinate_level_error.sum() / total),
        "normalized_level_mae": float(coordinate_level_error.sum() / total) / (model.num_levels - 1),
        "per_coordinate": per_coordinate,
        "repeat_last": {
            "coordinate_accuracy": repeat_correct / total,
            "level_mae": repeat_level_error / total,
        },
    }


def fit_frequency_baselines(store, batch_size: int, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    unigram = torch.ones(store.num_coordinates, store.num_levels, dtype=torch.float64)
    transition = torch.ones(
        store.num_coordinates,
        store.num_levels,
        store.num_levels,
        dtype=torch.float64,
    )
    loader = DataLoader(
        FSQTokenDataset("train", store),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    for batch in loader:
        indices = batch["indices"].long()
        for coordinate in range(store.num_coordinates):
            values = indices[:, :, coordinate]
            unigram[coordinate] += torch.bincount(values.reshape(-1), minlength=store.num_levels)
            pairs = values[:, :-1] * store.num_levels + values[:, 1:]
            transition[coordinate] += torch.bincount(
                pairs.reshape(-1),
                minlength=store.num_levels * store.num_levels,
            ).reshape(store.num_levels, store.num_levels)
    return unigram / unigram.sum(dim=-1, keepdim=True), transition / transition.sum(dim=-1, keepdim=True)


def evaluate_frequency_baselines(loader, unigram: torch.Tensor, transition: torch.Tensor) -> dict[str, object]:
    unigram_nll = 0.0
    unigram_correct = 0
    markov_nll = 0.0
    markov_correct = 0
    count = 0
    num_coordinates, num_levels = unigram.shape
    coordinate_ids = torch.arange(num_coordinates).view(1, 1, -1)
    unigram_predictions = unigram.argmax(dim=-1).view(1, 1, -1)
    for batch in loader:
        indices = batch["indices"].long()
        inputs, targets = indices[:, :-1], indices[:, 1:]
        expanded_coordinates = coordinate_ids.expand_as(targets)
        unigram_probability = unigram[expanded_coordinates, targets]
        markov_probability = transition[expanded_coordinates, inputs, targets]
        markov_predictions = transition[expanded_coordinates, inputs].argmax(dim=-1)
        unigram_nll -= float(unigram_probability.log().sum())
        markov_nll -= float(markov_probability.log().sum())
        unigram_correct += int((unigram_predictions == targets).sum())
        markov_correct += int((markov_predictions == targets).sum())
        count += targets.numel()
    if count == 0:
        raise ValueError("No baseline evaluation samples were processed")
    unigram_mean = unigram_nll / count
    markov_mean = markov_nll / count
    return {
        "unigram": {
            "nll": unigram_mean,
            "perplexity": math.exp(min(unigram_mean, 50.0)),
            "coordinate_accuracy": unigram_correct / count,
        },
        "first_order_markov": {
            "nll": markov_mean,
            "perplexity": math.exp(min(markov_mean, 50.0)),
            "coordinate_accuracy": markov_correct / count,
        },
    }


def _distribution_js(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first /= np.maximum(first.sum(axis=-1, keepdims=True), 1.0)
    second /= np.maximum(second.sum(axis=-1, keepdims=True), 1.0)
    middle = 0.5 * (first + second)
    first_term = np.zeros_like(first)
    second_term = np.zeros_like(second)
    first_active = first > 0.0
    second_active = second > 0.0
    first_term[first_active] = first[first_active] * np.log(
        first[first_active] / np.maximum(middle[first_active], 1e-12)
    )
    second_term[second_active] = second[second_active] * np.log(
        second[second_active] / np.maximum(middle[second_active], 1e-12)
    )
    return float(0.5 * (first_term.sum(axis=-1) + second_term.sum(axis=-1)).mean())


def sequence_statistics(indices: np.ndarray, num_levels: int) -> dict[str, float | list[float]]:
    batch_size, seq_len, num_coordinates = indices.shape
    occupancy = np.stack(
        [
            np.bincount(indices[:, :, coordinate].reshape(-1), minlength=num_levels)
            for coordinate in range(num_coordinates)
        ]
    )
    probability = occupancy / np.maximum(occupancy.sum(axis=-1, keepdims=True), 1.0)
    perplexity = np.exp(-(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=-1))
    if seq_len > 1:
        changes = indices[:, 1:] != indices[:, :-1]
        tuple_change_rate = float(changes.any(axis=-1).mean())
        coordinate_change_rate = float(changes.mean())
    else:
        tuple_change_rate = 0.0
        coordinate_change_rate = 0.0
    unique_ratios = [
        np.unique(indices[sample], axis=0).shape[0] / max(seq_len, 1)
        for sample in range(batch_size)
    ]
    return {
        "level_perplexity": float(perplexity.mean()),
        "level_perplexity_min": float(perplexity.min()),
        "level_perplexity_max": float(perplexity.max()),
        "level_usage": float((occupancy > 0).mean()),
        "tuple_unique_ratio": float(np.mean(unique_ratios)),
        "tuple_change_rate": tuple_change_rate,
        "coordinate_change_rate": coordinate_change_rate,
    }


def transition_counts(indices: np.ndarray, num_levels: int) -> np.ndarray:
    counts = np.zeros((indices.shape[-1], num_levels, num_levels), dtype=np.int64)
    for coordinate in range(indices.shape[-1]):
        pairs = indices[:, :-1, coordinate] * num_levels + indices[:, 1:, coordinate]
        counts[coordinate] = np.bincount(
            pairs.reshape(-1), minlength=num_levels * num_levels
        ).reshape(num_levels, num_levels)
    return counts.reshape(indices.shape[-1], -1)


def occupancy_counts(indices: np.ndarray, num_levels: int) -> np.ndarray:
    return np.stack(
        [
            np.bincount(indices[:, :, coordinate].reshape(-1), minlength=num_levels)
            for coordinate in range(indices.shape[-1])
        ]
    )


def collect_rollout_batch(dataset, samples: int, seed_frames: int, rollout_frames: int) -> torch.Tensor:
    take = min(samples, len(dataset))
    if take <= 0:
        raise ValueError("rollout_samples must select at least one sample")
    windows = torch.stack([dataset[index]["indices"] for index in range(take)]).long()
    required = seed_frames + rollout_frames
    if seed_frames <= 0 or rollout_frames <= 0 or required > windows.shape[1]:
        raise ValueError(
            f"Require positive seed/rollout lengths totaling at most {windows.shape[1]}, got {required}"
        )
    return windows[:, :required]


def decoded_rollout_metrics(
    fsq_checkpoint: dict,
    fsq_model: FSQMotionAutoencoder,
    generated_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    seed_frames: int,
    root_dt: float,
) -> dict[str, float]:
    generated_motion = fsq_model.decode_from_indices(generated_sequence)
    target_motion = fsq_model.decode_from_indices(target_sequence)
    generated_future = generated_motion[:, seed_frames:]
    target_future = target_motion[:, seed_frames:]
    stats = fsq_checkpoint["stats"]
    offset = torch.as_tensor(stats["offset"], dtype=torch.float32, device=generated_motion.device)
    scale = torch.as_tensor(stats["scale"], dtype=torch.float32, device=generated_motion.device)
    ref_pos = torch.as_tensor(stats["ref_pos"], dtype=torch.float32, device=generated_motion.device)
    parents = torch.as_tensor(stats["parents"], dtype=torch.long, device=generated_motion.device)
    names = [str(name) for name in np.asarray(stats["names"], dtype=object).tolist()]
    foot_indices = [names.index("LeftToeBase"), names.index("RightToeBase")]

    generated_joint = reconstruct_joint_positions(
        generated_future, offset, scale, ref_pos, parents, root_dt, world_space=False
    )
    target_joint = reconstruct_joint_positions(
        target_future, offset, scale, ref_pos, parents, root_dt, world_space=False
    )
    joint_error = generated_joint[:, :, 1:] - target_joint[:, :, 1:]
    generated_root_pos, generated_root_rot = integrate_root_trajectory(
        generated_future, offset, scale, root_dt
    )
    target_root_pos, target_root_rot = integrate_root_trajectory(target_future, offset, scale, root_dt)
    root_error = torch.linalg.vector_norm(generated_root_pos - target_root_pos, dim=-1)

    generated_raw = denormalize_motion_features(generated_future, offset, scale)
    target_raw = denormalize_motion_features(target_future, offset, scale)
    generated_contact = generated_raw[..., -2:] >= 0.5
    target_contact = target_raw[..., -2:] >= 0.5
    tp = (generated_contact & target_contact).sum().float()
    fp = (generated_contact & ~target_contact).sum().float()
    fn = (~generated_contact & target_contact).sum().float()

    generated_world = reconstruct_joint_positions(
        generated_motion, offset, scale, ref_pos, parents, root_dt, world_space=True
    )[:, seed_frames:, foot_indices]
    generated_feet_velocity = (
        generated_world[:, 1:, :, (0, 2)] - generated_world[:, :-1, :, (0, 2)]
    ) / root_dt
    contact_gate = generated_contact[:, 1:] & generated_contact[:, :-1]
    foot_slide = (
        generated_feet_velocity.abs().sum(dim=-1) * contact_gate.float()
    ).sum() / contact_gate.sum().clamp_min(1)
    if generated_future.shape[1] > 1:
        delta_l1 = F.l1_loss(
            generated_future[:, 1:] - generated_future[:, :-1],
            target_future[:, 1:] - target_future[:, :-1],
        )
    else:
        delta_l1 = generated_future.new_zeros(())

    return {
        "normalized_feature_l1": float(F.l1_loss(generated_future, target_future)),
        "delta_l1": float(delta_l1),
        "joint_mpjpe_m": float(torch.linalg.vector_norm(joint_error, dim=-1).mean()),
        "root_mean_error_m": float(root_error.mean()),
        "root_final_error_m": float(root_error[:, -1].mean()),
        "root_rotation_error_rad": float(quat.torch_quat_angle(generated_root_rot, target_root_rot).mean()),
        "contact_accuracy": float((generated_contact == target_contact).float().mean()),
        "contact_f1": float(2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)),
        "generated_foot_slide_mps": float(foot_slide),
    }


def evaluate_rollout(
    model,
    windows: torch.Tensor,
    device: torch.device,
    seed_frames: int,
    rollout_frames: int,
    temperature: float,
    sample: bool,
    fsq_checkpoint: dict | None,
    fsq_model: FSQMotionAutoencoder | None,
    root_dt: float,
) -> dict[str, object]:
    windows = windows.to(device)
    seed = windows[:, :seed_frames]
    target = windows[:, seed_frames : seed_frames + rollout_frames]
    with torch.inference_mode():
        generated = model.generate(
            seed,
            num_steps=rollout_frames,
            temperature=temperature,
            greedy=not sample,
        )
    generated_np = generated.cpu().numpy()
    target_np = target.cpu().numpy()
    result: dict[str, object] = {
        "num_samples": int(windows.shape[0]),
        "seed_frames": seed_frames,
        "rollout_frames": rollout_frames,
        "sampling": "categorical" if sample else "greedy",
        "temperature": temperature,
        "coordinate_accuracy": float((generated == target).float().mean()),
        "level_mae": float((generated - target).abs().float().mean()),
        "generated_statistics": sequence_statistics(generated_np, model.num_levels),
        "target_statistics": sequence_statistics(target_np, model.num_levels),
        "occupancy_js": _distribution_js(
            occupancy_counts(generated_np, model.num_levels),
            occupancy_counts(target_np, model.num_levels),
        ),
        "transition_js": _distribution_js(
            transition_counts(generated_np, model.num_levels),
            transition_counts(target_np, model.num_levels),
        ),
    }
    if fsq_checkpoint is not None and fsq_model is not None:
        generated_sequence = torch.cat((seed, generated), dim=1)
        target_sequence = windows[:, : seed_frames + rollout_frames]
        with torch.inference_mode():
            result["decoded_motion"] = decoded_rollout_metrics(
                fsq_checkpoint,
                fsq_model,
                generated_sequence,
                target_sequence,
                seed_frames,
                root_dt,
            )
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_latency(
    model,
    seed: torch.Tensor,
    device: torch.device,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    if warmup < 0 or steps <= 0:
        raise ValueError("latency_warmup must be non-negative and latency_steps must be positive")
    seed = seed[:1].to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            logits, cache = model.prefill(seed)
            current = logits.argmax(dim=-1)
            model.decode_step(current, cache)

        prefill_ms = []
        for _ in range(steps):
            synchronize(device)
            started = time.perf_counter()
            model.prefill(seed)
            synchronize(device)
            prefill_ms.append((time.perf_counter() - started) * 1000.0)

        logits, cache = model.prefill(seed)
        current = logits.argmax(dim=-1)
        cached_ms = []
        for _ in range(steps):
            synchronize(device)
            started = time.perf_counter()
            logits, cache = model.decode_step(current, cache)
            synchronize(device)
            cached_ms.append((time.perf_counter() - started) * 1000.0)
            current = logits.argmax(dim=-1)

        context = seed.clone()
        uncached_ms = []
        for _ in range(steps):
            synchronize(device)
            started = time.perf_counter()
            logits = model(context[:, -model.context_frames :])["logits"][:, -1]
            synchronize(device)
            uncached_ms.append((time.perf_counter() - started) * 1000.0)
            current = logits.argmax(dim=-1)[:, None]
            context = torch.cat((context, current), dim=1)

    def summarize(values: list[float], prefix: str) -> dict[str, float]:
        array = np.asarray(values)
        return {
            f"{prefix}_mean_ms": float(array.mean()),
            f"{prefix}_p50_ms": float(np.percentile(array, 50)),
            f"{prefix}_p95_ms": float(np.percentile(array, 95)),
        }

    cache_bytes = sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in cache.layers)
    return {
        **summarize(prefill_ms, "prefill"),
        **summarize(cached_ms, "cached_step"),
        **summarize(uncached_ms, "uncached_step"),
        "cache_megabytes": cache_bytes / (1024.0 * 1024.0),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    store = build_fsq_token_store(args.token_database)
    checkpoint, model = load_generator_checkpoint(args.checkpoint, store, device)
    dataset = maybe_subset(FSQTokenDataset(args.split, store), args.max_samples, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    teacher_forced = evaluate_teacher_forced(model, loader, device)
    unigram, transition = fit_frequency_baselines(store, args.batch_size, args.num_workers)
    baselines = evaluate_frequency_baselines(loader, unigram, transition)
    rollout_windows = collect_rollout_batch(
        dataset,
        args.rollout_samples,
        args.seed_frames,
        args.rollout_frames,
    )

    fsq_path = None
    fsq_checkpoint = None
    fsq_model = None
    if not args.skip_decode:
        fsq_path = resolve_fsq_checkpoint(args.fsq_checkpoint, store)
        fsq_checkpoint, fsq_model = load_fsq_decoder(fsq_path, device)
    rollout = evaluate_rollout(
        model,
        rollout_windows,
        device,
        args.seed_frames,
        args.rollout_frames,
        args.temperature,
        args.sample,
        fsq_checkpoint,
        fsq_model,
        args.root_dt,
    )
    latency = benchmark_latency(
        model,
        rollout_windows[:, : args.seed_frames],
        device,
        args.latency_warmup,
        args.latency_steps,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "token_database": str(store.database),
        "tokenizer_checkpoint_sha256": store.checkpoint_sha256,
        "fsq_checkpoint": str(fsq_path) if fsq_path is not None else None,
        "split": args.split,
        "num_samples": len(dataset),
        "device": str(device),
        "model_config": checkpoint["model_config"],
        "teacher_forced": teacher_forced,
        "baselines": baselines,
        "rollout": rollout,
        "latency": latency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
