from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from stylized_motion.anim.features import (
    denormalize_motion_features,
    deserialize_motion_feature_stats,
    normalize_motion_features,
)
from stylized_motion.anim.view_motion_sequence import choose_device
from stylized_motion.data import open_feature_store
from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.representation import (
    FLAT_FSQ_FAMILY,
    LATENT_RESIDUAL_FSQ_FAMILY,
    LATENT_RESIDUAL_FSQ_V2_FAMILY,
    PART_FSQ_FAMILY,
    RESIDUAL_PART_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.util.paths import RESOURCE_DIR


DIRECT_CODE_EDIT_FAMILIES = {PART_FSQ_FAMILY, RESIDUAL_PART_FSQ_FAMILY}
LATENT_EDIT_FAMILIES = {LATENT_RESIDUAL_FSQ_FAMILY, LATENT_RESIDUAL_FSQ_V2_FAMILY}
SUPPORTED_FAMILIES = DIRECT_CODE_EDIT_FAMILIES | LATENT_EDIT_FAMILIES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a body-part edit between two local dataset segments.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--target-range-idx", type=int, required=True)
    parser.add_argument("--target-start", type=int, required=True, help="Frame offset relative to the target range.")
    parser.add_argument("--donor-range-idx", type=int, required=True)
    parser.add_argument("--donor-start", type=int, required=True, help="Frame offset relative to the donor range.")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--part", choices=PART_NAMES, required=True)
    parser.add_argument("--compare-with", choices=("target", "donor"), default="target")
    parser.add_argument("--compare-spacing", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--resources-root", type=Path, default=RESOURCE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _coordinate_slice(metadata: Mapping[str, object], part: str) -> slice:
    order = metadata.get("coordinate_order")
    counts = metadata.get("coordinate_counts")
    if not isinstance(order, (list, tuple)) or not isinstance(counts, Mapping):
        raise ValueError("Checkpoint representation metadata is missing its coordinate layout")
    start = 0
    for group in order:
        name = str(group)
        if name not in counts:
            raise ValueError(f"Coordinate count is missing for group {name!r}")
        end = start + int(counts[name])
        if name == part:
            return slice(start, end)
        start = end
    raise ValueError(f"Representation has no editable group {part!r}")


def _read_range_window(store, range_idx: int, relative_start: int, length: int, history: int):
    if range_idx < 0 or range_idx >= len(store.range_names):
        raise ValueError(f"Range index {range_idx} must be in [0, {len(store.range_names) - 1}]")
    if relative_start < 0:
        raise ValueError("Range-relative start must be non-negative")
    if length <= 0:
        raise ValueError("--length must be positive")

    range_start = int(store.range_starts[range_idx])
    range_stop = int(store.range_stops[range_idx])
    absolute_start = range_start + relative_start
    absolute_stop = absolute_start + length
    if absolute_stop > range_stop:
        available = range_stop - range_start
        raise ValueError(
            f"Requested range-relative [{relative_start}, {relative_start + length}) "
            f"exceeds range {range_idx} length {available}"
        )

    shard_idx = int(store.range_shard_indices[range_idx])
    motion = np.load(store.motion_files[shard_idx], mmap_mode="r")
    if motion.ndim != 2 or motion.shape[1] != store.motion_dim:
        raise ValueError(f"Unexpected motion shard shape {motion.shape}")

    wanted_start = absolute_start - history
    read_start = max(range_start, wanted_start)
    window = np.asarray(motion[read_start:absolute_stop], dtype=np.float32).copy()
    left_pad = read_start - wanted_start
    if left_pad:
        window = np.concatenate((np.repeat(window[:1], left_pad, axis=0), window), axis=0)
    expected = history + length
    if window.shape != (expected, store.motion_dim):
        raise RuntimeError(f"Expected inference window {(expected, store.motion_dim)}, got {window.shape}")

    source = np.asarray(motion[absolute_start:absolute_stop], dtype=np.float32).copy()
    return window, source, {
        "range_idx": range_idx,
        "range_name": str(store.range_names[range_idx]),
        "mirror": bool(store.range_mirror[range_idx]),
        "shard_idx": shard_idx,
        "relative_start": relative_start,
        "absolute_start": absolute_start,
        "left_pad": left_pad,
    }


def _edit_pair(model, pair: torch.Tensor, part: str, representation_metadata: Mapping[str, object]):
    family = model.family
    if family in DIRECT_CODE_EDIT_FAMILIES:
        output = model(pair, collect_metrics=False)
        target_indices = output["indices"][0:1]
        donor_indices = output["indices"][1:2]
        edited_indices = target_indices.clone()
        part_slice = _coordinate_slice(representation_metadata, part)
        edited_indices[..., part_slice] = donor_indices[..., part_slice]
        edited_recon = model.decode_from_indices(edited_indices)
        return {
            "target_recon_state": output["recon_state"][0:1],
            "donor_recon_state": output["recon_state"][1:2],
            "edited_recon_state": edited_recon,
            "target_indices": target_indices,
            "donor_indices": donor_indices,
            "edited_indices": edited_indices,
        }
    if family in LATENT_EDIT_FAMILIES:
        permutation = torch.tensor([1, 0], device=pair.device, dtype=torch.long)
        output = model(
            pair,
            collect_metrics=False,
            edit_part=part,
            donor_permutation=permutation,
        )
        return {
            "target_recon_state": output["recon_state"][0:1],
            "donor_recon_state": output["recon_state"][1:2],
            "edited_recon_state": output["edit_recon_state"][0:1],
            "target_indices": output["indices"][0:1],
            "donor_indices": output["indices"][1:2],
            "edited_indices": None,
        }
    if family == FLAT_FSQ_FAMILY:
        raise ValueError("Flat-FSQ does not support body-part editing")
    raise ValueError(f"Unsupported representation family for part editing: {family!r}")


def _validate_checkpoint_store(checkpoint, model, store):
    if model.motion_dim != store.motion_dim:
        raise ValueError(f"Model motion_dim={model.motion_dim} does not match feature database motion_dim={store.motion_dim}")
    checkpoint_schema = checkpoint.get("feature_schema")
    if not isinstance(checkpoint_schema, Mapping):
        raise ValueError("Checkpoint is missing feature_schema")
    store_schema = store.feature_schema()
    for key in ("name", "motion_dim", "joint_subset", "names_sha256"):
        if checkpoint_schema.get(key) != store_schema.get(key):
            raise ValueError(f"Checkpoint and feature database structural schema mismatch at {key!r}")

    feature_stats = checkpoint.get("feature_stats")
    if not isinstance(feature_stats, dict):
        raise ValueError("Checkpoint is missing feature_stats")
    model_stats, stats_metadata = deserialize_motion_feature_stats(feature_stats)
    names = stats_metadata.get("names")
    parents = stats_metadata.get("parents")
    if names != store.names or not isinstance(parents, np.ndarray) or not np.array_equal(parents, store.parents):
        raise ValueError("Checkpoint and feature database skeletons differ")
    return model_stats


def _save_debug(output_dir: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved debug outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    store = open_feature_store(args.feature_database)
    try:
        checkpoint, model = load_representation_checkpoint(args.checkpoint, torch.device("cpu"))
        if model.family not in SUPPORTED_FAMILIES:
            if model.family == FLAT_FSQ_FAMILY:
                raise ValueError("Flat-FSQ does not support body-part editing")
            raise ValueError(f"Unsupported representation family for part editing: {model.family!r}")
        model_stats = _validate_checkpoint_store(checkpoint, model, store)
        history = int(model.history_frames)
        target_window, target_source, target_meta = _read_range_window(
            store, args.target_range_idx, args.target_start, args.length, history
        )
        donor_window, donor_source, donor_meta = _read_range_window(
            store, args.donor_range_idx, args.donor_start, args.length, history
        )

        target_source = denormalize_motion_features(target_source, store.stats)
        donor_source = denormalize_motion_features(donor_source, store.stats)
        target_input = normalize_motion_features(
            denormalize_motion_features(target_window, store.stats), model_stats
        )
        donor_input = normalize_motion_features(
            denormalize_motion_features(donor_window, store.stats), model_stats
        )

        device = choose_device(args.device)
        model = model.to(device).eval()
        pair = torch.from_numpy(np.stack((target_input, donor_input))).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            result = _edit_pair(model, pair, args.part, model.representation_metadata())

        trim = slice(history, history + args.length)
        target_recon_normalized = result["target_recon_state"][0, trim].detach().cpu().numpy().astype(np.float32)
        donor_recon_normalized = result["donor_recon_state"][0, trim].detach().cpu().numpy().astype(np.float32)
        edited_normalized = result["edited_recon_state"][0, trim].detach().cpu().numpy().astype(np.float32)
        target_indices = result["target_indices"][0, trim].detach().cpu().numpy()
        donor_indices = result["donor_indices"][0, trim].detach().cpu().numpy()
        edited_indices_value = result["edited_indices"]
        edited_indices = None if edited_indices_value is None else edited_indices_value[0, trim].detach().cpu().numpy()

        feature_indices = model.module.layout.feature_indices(model.motion_dim)[args.part].cpu().numpy()
        part_mask = np.zeros(model.motion_dim, dtype=bool)
        part_mask[feature_indices] = True
        change = np.abs(edited_normalized - target_recon_normalized)
        metrics = {
            "target_part_change": float(change[:, part_mask].mean()),
            "non_target_change": float(change[:, ~part_mask].mean()),
        }

        target_recon = denormalize_motion_features(target_recon_normalized, model_stats)
        donor_recon = denormalize_motion_features(donor_recon_normalized, model_stats)
        edited = denormalize_motion_features(edited_normalized, model_stats)
        metadata = {
            "checkpoint": str(args.checkpoint),
            "feature_database": str(args.feature_database),
            "family": model.family,
            "representation_id": model.representation_id,
            "part": args.part,
            "compare_with": args.compare_with,
            "length": args.length,
            "history_frames": history,
            "device": str(device),
            "target": target_meta,
            "donor": donor_meta,
            **metrics,
        }
    finally:
        store.close()

    print(json.dumps(metadata, indent=2))
    if args.save_debug:
        output_dir = args.output_dir or (
            args.checkpoint.parent
            / "part_edit_visualization"
            / f"target_{args.target_range_idx:03d}_donor_{args.donor_range_idx:03d}_{args.part}"
        )
        arrays = {
            "target_source_features": target_source.astype(np.float32),
            "donor_source_features": donor_source.astype(np.float32),
            "target_recon_features": target_recon.astype(np.float32),
            "donor_recon_features": donor_recon.astype(np.float32),
            "edited_features": edited.astype(np.float32),
            "target_indices": target_indices,
            "donor_indices": donor_indices,
        }
        if edited_indices is not None:
            arrays["edited_indices"] = edited_indices
        _save_debug(output_dir, arrays, metadata)

    if args.dry_run:
        print("dry_run_part_edit_ready=true")
        return

    from stylized_motion.anim.genoview import GenoViewCompare, build_database_from_feature_array

    if args.compare_with == "target":
        reference, reference_label = target_recon, "Target Recon"
    else:
        reference, reference_label = donor_recon, "Donor Recon"
    reference_database = build_database_from_feature_array(
        reference, args.checkpoint, False, f"{reference_label}_{args.part}"
    )
    edited_database = build_database_from_feature_array(
        edited, args.checkpoint, False, f"Edited_{args.part}"
    )
    GenoViewCompare(
        left_database=reference_database,
        right_database=edited_database,
        resources_root=args.resources_root,
        fps=args.fps,
        left_label=reference_label,
        right_label=f"Edited {args.part}",
        compare_spacing=args.compare_spacing,
    ).run()


if __name__ == "__main__":
    main()
