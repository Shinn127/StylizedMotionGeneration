from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.feature_dataset import FeatureDataset, build_feature_store
from models.latent_residual_part_fsq import LatentResidualPartFSQMotionAutoencoder
from models.part_fsq import HierarchicalPartFSQMotionAutoencoder
from models.part_layout import PART_NAMES
from models.residual_part_fsq import ResidualPartFSQMotionAutoencoder
from models.losses import denormalize_motion_features


TokenizerModel = (
    HierarchicalPartFSQMotionAutoencoder
    | ResidualPartFSQMotionAutoencoder
    | LatentResidualPartFSQMotionAutoencoder
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate discrete body-part index editing for Part-FSQ and Residual Part-FSQ."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Tokenizer checkpoint. Repeat to compare models on identical source/donor pairs.",
    )
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--donor-offset",
        type=int,
        default=17,
        help="Deterministic donor index offset. Source i uses donor (i + offset) mod split size.",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=None,
        help="Parts to edit. Omit or pass all parts; valid names are torso, left_leg, right_leg, left_arm, right_arm.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save normalized source/donor/reconstruction/edit features and indices for GenoView.",
    )
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/part_editing_debug"))
    parser.add_argument("--debug-count", type=int, default=2)
    return parser.parse_args(argv)


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


def resolve_parts(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None or "all" in values:
        return PART_NAMES
    parts = tuple(values)
    unknown = sorted(set(parts).difference(PART_NAMES))
    if unknown:
        raise ValueError(f"Unsupported parts: {unknown}; expected one of {PART_NAMES}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"Parts must be unique, got {parts}")
    return parts


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict, TokenizerModel, str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    family = checkpoint.get("model_family")
    if family == "part_fsq":
        model: TokenizerModel = HierarchicalPartFSQMotionAutoencoder(**checkpoint["model_config"])
    elif family == "residual_part_fsq":
        model = ResidualPartFSQMotionAutoencoder(**checkpoint["model_config"])
    elif family == "latent_residual_part_fsq":
        model = LatentResidualPartFSQMotionAutoencoder(**checkpoint["model_config"])
    else:
        raise ValueError(
            f"{path} is not a Part-FSQ editing checkpoint: {family!r}; "
            "expected 'part_fsq', 'residual_part_fsq', or 'latent_residual_part_fsq'"
        )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return checkpoint, model, family


def checkpoint_stats(checkpoint: dict, dataset: FeatureDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stats = checkpoint.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("Checkpoint is missing serialized feature stats")
    required = {"offset", "scale", "names", "parents"}
    missing = required.difference(stats)
    if missing:
        raise ValueError(f"Checkpoint stats are missing: {sorted(missing)}")
    names = [str(name) for name in np.asarray(stats["names"], dtype=object).tolist()]
    parents = [int(parent) for parent in np.asarray(stats["parents"], dtype=np.int32).tolist()]
    if names != dataset.names:
        raise ValueError("Checkpoint and evaluation database use different joint ordering")
    if parents != [int(parent) for parent in dataset.parents.tolist()]:
        raise ValueError("Checkpoint and evaluation database use different parent ordering")
    offset = torch.as_tensor(stats["offset"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(stats["scale"], dtype=torch.float32, device=device)
    if offset.ndim != 1 or scale.shape != offset.shape or offset.shape[0] != dataset.motion_dim:
        raise ValueError("Checkpoint feature stats have an invalid shape")
    return offset, scale


class IndexedPairDataset(Dataset):
    def __init__(self, dataset: FeatureDataset, source_indices: list[int], donor_indices: list[int]) -> None:
        if len(source_indices) != len(donor_indices):
            raise ValueError("source_indices and donor_indices must have the same length")
        self.dataset = dataset
        self.source_indices = source_indices
        self.donor_indices = donor_indices

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        source_index = self.source_indices[index]
        donor_index = self.donor_indices[index]
        return {
            "source_motion": self.dataset[source_index]["motion"],
            "donor_motion": self.dataset[donor_index]["motion"],
            "source_index": source_index,
            "donor_index": donor_index,
        }


def build_pairs(dataset: FeatureDataset, max_samples: int | None, donor_offset: int) -> IndexedPairDataset:
    if len(dataset) < 2:
        raise ValueError("Part editing requires at least two samples in the selected split")
    if donor_offset == 0 or abs(donor_offset) >= len(dataset):
        raise ValueError(f"donor_offset must be nonzero and smaller than split size, got {donor_offset}")
    count = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    if count <= 0:
        raise ValueError("max_samples must be positive")
    source_indices = list(range(count))
    donor_indices = [int((index + donor_offset) % len(dataset)) for index in source_indices]
    return IndexedPairDataset(dataset, source_indices, donor_indices)


def model_group_slices(model: TokenizerModel) -> dict[str, slice]:
    if isinstance(model, HierarchicalPartFSQMotionAutoencoder):
        return model.layout.group_slices
    return model.group_slices


def model_feature_indices(model: TokenizerModel) -> dict[str, torch.Tensor]:
    return model.layout.feature_indices(model.motion_dim)


def edit_indices(
    source_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    group_slice: slice,
) -> torch.Tensor:
    if source_indices.shape != donor_indices.shape:
        raise ValueError("source and donor index tensors must have the same shape")
    edited = source_indices.clone()
    edited[..., group_slice] = donor_indices[..., group_slice]
    return edited


def build_multi_part_edits(
    source_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    parts: tuple[str, ...],
    group_slices: dict[str, slice],
) -> torch.Tensor:
    edits = source_indices.unsqueeze(1).expand(-1, len(parts), -1, -1).clone()
    for part_index, part in enumerate(parts):
        edits[:, part_index, :, group_slices[part]] = donor_indices[..., group_slices[part]]
    return edits


def mean_abs_on_features(
    left: torch.Tensor,
    right: torch.Tensor,
    feature_index: torch.Tensor,
) -> torch.Tensor:
    values = (left - right).abs().index_select(-1, feature_index)
    return values.mean(dim=(-1, -2))


def complement_features(feature_index: torch.Tensor, motion_dim: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(motion_dim, dtype=torch.bool, device=device)
    mask[feature_index] = False
    return torch.arange(motion_dim, device=device, dtype=torch.long)[mask]


def _batch_metric(value: torch.Tensor) -> float:
    return float(value.detach().mean().cpu())


def _new_part_accumulator() -> dict[str, float]:
    return {
        "target_response": 0.0,
        "non_target_leakage": 0.0,
        "leakage_ratio": 0.0,
        "source_target_reconstruction_error": 0.0,
        "edited_target_donor_error": 0.0,
        "target_transfer_gain": 0.0,
        "base_change": 0.0,
    }


def _accumulate_part_metrics(
    accumulator: dict[str, float],
    source_recon: torch.Tensor,
    edited: torch.Tensor,
    source_target: torch.Tensor,
    donor_target: torch.Tensor,
    target_features: torch.Tensor,
    non_target_features: torch.Tensor,
    base_change: torch.Tensor | None,
) -> None:
    target_response = mean_abs_on_features(edited, source_recon, target_features)
    non_target_leakage = mean_abs_on_features(edited, source_recon, non_target_features)
    source_target_error = mean_abs_on_features(source_recon, source_target, target_features)
    edited_target_error = mean_abs_on_features(edited, donor_target, target_features)
    transfer_gain = (source_target_error - edited_target_error) / source_target_error.clamp_min(1e-6)
    values = {
        "target_response": target_response,
        "non_target_leakage": non_target_leakage,
        "leakage_ratio": non_target_leakage / target_response.clamp_min(1e-6),
        "source_target_reconstruction_error": source_target_error,
        "edited_target_donor_error": edited_target_error,
        "target_transfer_gain": transfer_gain,
    }
    if base_change is not None:
        values["base_change"] = base_change
    for name, value in values.items():
        accumulator[name] += _batch_metric(value)


def _finalize_part_metrics(accumulator: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(count, 1) for name, value in accumulator.items()}


def _save_debug(
    debug_dir: Path,
    family: str,
    part: str,
    source_index: int,
    donor_index: int,
    source_features: torch.Tensor,
    donor_features: torch.Tensor,
    source_recon: torch.Tensor,
    donor_recon: torch.Tensor,
    edited_features: torch.Tensor,
    source_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    edited_indices: torch.Tensor,
) -> Path:
    target_dir = debug_dir / family / part
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"pair_{source_index:06d}_{donor_index:06d}.npz"
    np.savez_compressed(
        path,
        source_features=source_features.detach().cpu().numpy().astype(np.float32),
        donor_features=donor_features.detach().cpu().numpy().astype(np.float32),
        source_recon_features=source_recon.detach().cpu().numpy().astype(np.float32),
        donor_recon_features=donor_recon.detach().cpu().numpy().astype(np.float32),
        edited_features=edited_features.detach().cpu().numpy().astype(np.float32),
        source_indices=source_indices.detach().cpu().numpy().astype(np.int64),
        donor_indices=donor_indices.detach().cpu().numpy().astype(np.int64),
        edited_indices=edited_indices.detach().cpu().numpy().astype(np.int64),
    )
    return path


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict,
    model: TokenizerModel,
    family: str,
    dataset: FeatureDataset,
    loader: DataLoader,
    parts: tuple[str, ...],
    device: torch.device,
    save_debug: bool,
    debug_dir: Path,
    debug_count: int,
) -> dict[str, object]:
    checkpoint_offset, checkpoint_scale = checkpoint_stats(checkpoint, dataset, device)
    dataset_stats = dataset.feature_stats()
    dataset_offset = torch.as_tensor(dataset_stats.offset, dtype=torch.float32, device=device)
    dataset_scale = torch.as_tensor(dataset_stats.scale, dtype=torch.float32, device=device)
    feature_indices = {
        name: index.to(device)
        for name, index in model_feature_indices(model).items()
    }
    group_slices = model_group_slices(model)
    totals = {part: _new_part_accumulator() for part in parts}
    count = 0
    debug_saved = {part: 0 for part in parts}
    debug_paths: list[str] = []

    with torch.inference_mode():
        for batch in loader:
            source_dataset = batch["source_motion"].to(device, non_blocking=device.type == "cuda").float()
            donor_dataset = batch["donor_motion"].to(device, non_blocking=device.type == "cuda").float()
            batch_size = source_dataset.shape[0]

            source_raw = denormalize_motion_features(source_dataset, dataset_offset, dataset_scale)
            donor_raw = denormalize_motion_features(donor_dataset, dataset_offset, dataset_scale)
            source_motion = (source_raw - checkpoint_offset.view(1, 1, -1)) / checkpoint_scale.view(1, 1, -1)
            donor_motion = (donor_raw - checkpoint_offset.view(1, 1, -1)) / checkpoint_scale.view(1, 1, -1)

            if family in {"residual_part_fsq", "latent_residual_part_fsq"}:
                encoded = model(torch.cat((source_motion, donor_motion), dim=0), decode_base=True)
            else:
                encoded = model(torch.cat((source_motion, donor_motion), dim=0))
            source_indices = encoded["indices"][:batch_size]
            donor_indices = encoded["indices"][batch_size:]
            source_recon = encoded["recon_state"][:batch_size]
            donor_recon = encoded["recon_state"][batch_size:]

            edited_indices = build_multi_part_edits(source_indices, donor_indices, parts, group_slices)
            flat_edited_indices = edited_indices.reshape(-1, edited_indices.shape[-2], edited_indices.shape[-1])
            flat_edited = model.decode_from_indices(flat_edited_indices)
            edited = flat_edited.reshape(batch_size, len(parts), flat_edited.shape[1], flat_edited.shape[2])

            source_base = None
            edited_base = None
            if family in {"residual_part_fsq", "latent_residual_part_fsq"}:
                source_base = encoded["base_recon_state"][:batch_size]
                edited_base = model.decode_base_from_indices(flat_edited_indices).reshape(
                    batch_size, len(parts), flat_edited.shape[1], flat_edited.shape[2]
                )

            for part_index, part in enumerate(parts):
                target_features = feature_indices[part]
                non_target_features = complement_features(target_features, model.motion_dim, device)
                base_change = None
                if source_base is not None and edited_base is not None:
                    base_change = mean_abs_on_features(
                        edited_base[:, part_index],
                        source_base,
                        torch.arange(model.motion_dim, device=device, dtype=torch.long),
                    )
                _accumulate_part_metrics(
                    totals[part],
                    source_recon,
                    edited[:, part_index],
                    source_motion,
                    donor_motion,
                    target_features,
                    non_target_features,
                    base_change,
                )

                if save_debug and debug_saved[part] < debug_count:
                    source_index = int(batch["source_index"][0])
                    donor_index = int(batch["donor_index"][0])
                    debug_path = _save_debug(
                        debug_dir,
                        family,
                        part,
                        source_index,
                        donor_index,
                        source_motion[0],
                        donor_motion[0],
                        source_recon[0],
                        donor_recon[0],
                        edited[0, part_index],
                        source_indices[0],
                        donor_indices[0],
                        edited_indices[0, part_index],
                    )
                    debug_paths.append(str(debug_path))
                    debug_saved[part] += 1

            count += batch_size

    metrics = {
        part: _finalize_part_metrics(totals[part], count)
        for part in parts
    }
    return {
        "checkpoint": str(checkpoint_path),
        "model_family": family,
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "feature_database": str(dataset.feature_database),
        "split": dataset.split,
        "num_pairs": count,
        "parts": list(parts),
        "metrics": metrics,
        "debug_files": debug_paths,
    }


def print_report(report: dict[str, object]) -> None:
    print(
        f"checkpoint={report['checkpoint']} family={report['model_family']} "
        f"split={report['split']} pairs={report['num_pairs']}"
    )
    for part, metrics in report["metrics"].items():
        print(
            f"part={part} target_response={metrics['target_response']:.8f} "
            f"non_target_leakage={metrics['non_target_leakage']:.8f} "
            f"leakage_ratio={metrics['leakage_ratio']:.8f} "
            f"target_transfer_gain={metrics['target_transfer_gain']:.8f}"
        )
        if report["model_family"] in {"residual_part_fsq", "latent_residual_part_fsq"}:
            print(f"part={part} base_change={metrics['base_change']:.8f}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.debug_count < 0:
        raise ValueError("--debug-count must be non-negative")
    parts = resolve_parts(args.parts)
    device = choose_device(args.device)
    store = build_feature_store(args.feature_database)
    dataset = FeatureDataset(args.split, store)
    pairs = build_pairs(dataset, args.max_samples, args.donor_offset)
    loader = DataLoader(
        pairs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    reports = []
    for checkpoint_path in args.checkpoint:
        checkpoint, model, family = load_checkpoint(checkpoint_path, device)
        report = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            model=model,
            family=family,
            dataset=dataset,
            loader=loader,
            parts=parts,
            device=device,
            save_debug=args.save_debug,
            debug_dir=args.debug_dir,
            debug_count=args.debug_count,
        )
        print_report(report)
        reports.append(report)

    result = {
        "protocol": {
            "operation": "discrete_part_index_swap",
            "donor_offset": args.donor_offset,
            "parts": list(parts),
            "device": str(device),
            "metrics_space": "checkpoint_normalized_features",
        },
        "reports": reports,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"report={args.output}")


if __name__ == "__main__":
    main()
