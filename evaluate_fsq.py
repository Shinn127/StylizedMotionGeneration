from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.feature_dataset import FeatureDataset, build_feature_store
from models.fsq import FSQMotionAutoencoder
from models.losses import (
    denormalize_motion_features,
    integrate_root_trajectory,
    reconstruct_joint_positions,
)
from preprocess import quat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more FSQ checkpoints with a common physical and representation metric suite."
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="Checkpoint path. Repeat for comparisons.")
    parser.add_argument("--feature-database", type=Path, required=True, help="Feature database defining the common evaluation set.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--root-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--contact-temperature", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


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


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, FSQMotionAutoencoder]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "fsq":
        raise ValueError(f"{path} is not an FSQ checkpoint")
    if "model_config" not in checkpoint:
        raise ValueError(f"{path} is missing model_config and cannot be evaluated by this entry point")
    model = FSQMotionAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model


def checkpoint_kinematics(checkpoint: dict, dataset, device: torch.device) -> dict[str, object]:
    stats = checkpoint["stats"]
    required = {"offset", "scale", "weights", "ref_pos", "parents", "names"}
    missing = required.difference(stats)
    if missing:
        raise ValueError(f"Checkpoint stats are missing: {sorted(missing)}")

    names = [str(name) for name in np.asarray(stats["names"], dtype=object).tolist()]
    if names != dataset.names:
        raise ValueError("Checkpoint and evaluation database use different skeleton joint ordering")
    try:
        foot_indices = (names.index("LeftToeBase"), names.index("RightToeBase"))
    except ValueError as exc:
        raise ValueError("Evaluation requires LeftToeBase and RightToeBase joints") from exc

    joint_weights = torch.ones(len(names), device=device)
    joint_weights[0] = 0.0
    for index, name in enumerate(names):
        if any(token in name for token in ("Head", "Hand", "Foot", "Toe")):
            joint_weights[index] = 2.0

    return {
        "offset": torch.as_tensor(stats["offset"], dtype=torch.float32, device=device),
        "scale": torch.as_tensor(stats["scale"], dtype=torch.float32, device=device),
        "weights": torch.as_tensor(stats["weights"], dtype=torch.float32, device=device),
        "ref_pos": torch.as_tensor(stats["ref_pos"], dtype=torch.float32, device=device),
        "parents": torch.as_tensor(stats["parents"], dtype=torch.long, device=device),
        "joint_weights": joint_weights,
        "foot_indices": foot_indices,
    }


def init_totals() -> dict[str, float]:
    return {
        "weighted_feature_l1": 0.0,
        "delta_l1": 0.0,
        "joint_mpjpe_m": 0.0,
        "joint_l1_m": 0.0,
        "root_mean_error_m": 0.0,
        "root_final_error_m": 0.0,
        "root_rotation_error_rad": 0.0,
        "contact_bce": 0.0,
        "foot_slide_mps_l1": 0.0,
        "foot_height_error_m": 0.0,
        "level_perplexity": 0.0,
        "level_usage": 0.0,
        "level_perplexity_min": 0.0,
        "level_perplexity_max": 0.0,
        "level_usage_min": 0.0,
        "level_usage_max": 0.0,
        "tuple_unique_ratio": 0.0,
        "tuple_change_rate": 0.0,
        "coordinate_change_rate": 0.0,
        "contact_tp": 0.0,
        "contact_fp": 0.0,
        "contact_fn": 0.0,
        "contact_tn": 0.0,
    }


def add_weighted(totals: dict[str, float], name: str, value: torch.Tensor, batch_size: int) -> None:
    totals[name] += float(value) * batch_size


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict,
    model: FSQMotionAutoencoder,
    dataset,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    kinematics = checkpoint_kinematics(checkpoint, dataset, device)
    eval_stats = dataset.feature_stats()
    eval_offset = torch.from_numpy(eval_stats.offset.astype(np.float32)).to(device)
    eval_scale = torch.from_numpy(eval_stats.scale.astype(np.float32)).to(device)
    eval_weights = torch.from_numpy(dataset.model_feature_weights().astype(np.float32)).to(device).view(1, 1, -1)

    totals = init_totals()
    count = 0
    with torch.inference_mode():
        for batch in loader:
            dataset_motion = batch["motion"].to(device).float()
            if args.max_samples is not None:
                remaining = args.max_samples - count
                if remaining <= 0:
                    break
                dataset_motion = dataset_motion[:remaining]
            batch_size = dataset_motion.shape[0]
            if batch_size == 0:
                break

            # The evaluator's feature database is the common metric space. Re-normalize raw motion
            # with each checkpoint's own training statistics before inference.
            target_raw = denormalize_motion_features(dataset_motion, eval_offset, eval_scale)
            model_motion = (target_raw - kinematics["offset"].view(1, 1, -1)) / kinematics["scale"].view(1, 1, -1)
            output = model(model_motion)
            prediction_raw = denormalize_motion_features(
                output["recon_state"], kinematics["offset"], kinematics["scale"]
            )
            prediction_eval = (prediction_raw - eval_offset.view(1, 1, -1)) / eval_scale.view(1, 1, -1)

            add_weighted(
                totals,
                "weighted_feature_l1",
                torch.mean(eval_weights * torch.abs(prediction_eval - dataset_motion)),
                batch_size,
            )
            add_weighted(
                totals,
                "delta_l1",
                F.l1_loss(prediction_eval[:, 1:] - prediction_eval[:, :-1], dataset_motion[:, 1:] - dataset_motion[:, :-1]),
                batch_size,
            )

            pred_joint = reconstruct_joint_positions(
                output["recon_state"],
                kinematics["offset"],
                kinematics["scale"],
                kinematics["ref_pos"],
                kinematics["parents"],
                args.root_dt,
                world_space=False,
            )
            target_joint = reconstruct_joint_positions(
                model_motion,
                kinematics["offset"],
                kinematics["scale"],
                kinematics["ref_pos"],
                kinematics["parents"],
                args.root_dt,
                world_space=False,
            )
            joint_error = pred_joint[:, :, 1:] - target_joint[:, :, 1:]
            add_weighted(totals, "joint_mpjpe_m", torch.linalg.vector_norm(joint_error, dim=-1).mean(), batch_size)
            add_weighted(totals, "joint_l1_m", joint_error.abs().mean(), batch_size)

            pred_root_pos, pred_root_rot = integrate_root_trajectory(
                output["recon_state"], kinematics["offset"], kinematics["scale"], args.root_dt
            )
            target_root_pos, target_root_rot = integrate_root_trajectory(
                model_motion, kinematics["offset"], kinematics["scale"], args.root_dt
            )
            root_error = torch.linalg.vector_norm(pred_root_pos - target_root_pos, dim=-1)
            add_weighted(totals, "root_mean_error_m", root_error.mean(), batch_size)
            add_weighted(totals, "root_final_error_m", root_error[:, -1].mean(), batch_size)
            add_weighted(
                totals,
                "root_rotation_error_rad",
                quat.torch_quat_angle(pred_root_rot, target_root_rot).mean(),
                batch_size,
            )

            target_contact = target_raw[..., -2:].clamp(0.0, 1.0)
            contact_logits = args.contact_temperature * (prediction_raw[..., -2:] - args.contact_threshold)
            add_weighted(totals, "contact_bce", F.binary_cross_entropy_with_logits(contact_logits, target_contact), batch_size)
            pred_contact = prediction_raw[..., -2:] >= args.contact_threshold
            target_contact_bool = target_contact >= args.contact_threshold
            totals["contact_tp"] += float((pred_contact & target_contact_bool).sum())
            totals["contact_fp"] += float((pred_contact & ~target_contact_bool).sum())
            totals["contact_fn"] += float((~pred_contact & target_contact_bool).sum())
            totals["contact_tn"] += float((~pred_contact & ~target_contact_bool).sum())

            pred_world = reconstruct_joint_positions(
                output["recon_state"],
                kinematics["offset"],
                kinematics["scale"],
                kinematics["ref_pos"],
                kinematics["parents"],
                args.root_dt,
                world_space=True,
            )
            target_world = reconstruct_joint_positions(
                model_motion,
                kinematics["offset"],
                kinematics["scale"],
                kinematics["ref_pos"],
                kinematics["parents"],
                args.root_dt,
                world_space=True,
            )
            foot_indices = list(kinematics["foot_indices"])
            pred_feet = pred_world[:, :, foot_indices]
            target_feet = target_world[:, :, foot_indices]
            contact_gate = target_contact[:, 1:] * target_contact[:, :-1]
            horizontal_velocity = (pred_feet[:, 1:, :, (0, 2)] - pred_feet[:, :-1, :, (0, 2)]) / args.root_dt
            foot_slide = (horizontal_velocity.abs().sum(dim=-1) * contact_gate).sum() / contact_gate.sum().clamp_min(1.0)
            foot_height = ((pred_feet[..., 1] - target_feet[..., 1]).abs() * target_contact).sum() / target_contact.sum().clamp_min(1.0)
            add_weighted(totals, "foot_slide_mps_l1", foot_slide, batch_size)
            add_weighted(totals, "foot_height_error_m", foot_height, batch_size)

            for name in (
                "level_perplexity",
                "level_usage",
                "level_perplexity_min",
                "level_perplexity_max",
                "level_usage_min",
                "level_usage_max",
                "tuple_unique_ratio",
                "tuple_change_rate",
                "coordinate_change_rate",
            ):
                add_weighted(totals, name, output[name], batch_size)
            count += batch_size

    if count == 0:
        raise ValueError("No evaluation samples were processed")
    for name in list(totals):
        if not name.startswith("contact_"):
            totals[name] /= count

    tp, fp, fn, tn = (totals[name] for name in ("contact_tp", "contact_fp", "contact_fn", "contact_tn"))
    totals["contact_accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    totals["contact_precision"] = tp / max(tp + fp, 1.0)
    totals["contact_recall"] = tp / max(tp + fn, 1.0)
    totals["contact_f1"] = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    for name in ("contact_tp", "contact_fp", "contact_fn", "contact_tn"):
        del totals[name]

    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "model_config": checkpoint["model_config"],
        "representation": checkpoint.get("representation", {}),
        "feature_database": str(dataset.feature_database),
        "split": dataset.split,
        "num_samples": count,
        "metrics": totals,
    }


def print_report(report: dict[str, object]) -> None:
    print(f"checkpoint={report['checkpoint']}")
    print(f"split={report['split']} samples={report['num_samples']} epoch={report['epoch']}")
    for name, value in report["metrics"].items():
        print(f"{name}={value:.8f}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")

    device = choose_device(args.device)
    store = build_feature_store(args.feature_database)
    dataset = FeatureDataset(args.split, store)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    reports = []
    for checkpoint_path in args.checkpoint:
        checkpoint, model = load_checkpoint(checkpoint_path, device)
        report = evaluate_checkpoint(checkpoint_path, checkpoint, model, dataset, loader, device, args)
        print_report(report)
        reports.append(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump({"reports": reports}, handle, indent=2)
        print(f"report={args.output}")


if __name__ == "__main__":
    main()

