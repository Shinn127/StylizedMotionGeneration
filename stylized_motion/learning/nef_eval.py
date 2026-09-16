"""NEF-FSQ evaluation: per-stream reconstruction report and node/edge token transfer.

``report`` produces the per-skeleton/per-stream reconstruction, rotation, FK and
token-utilization numbers of the design's experiment report.  ``transfer``
implements same-skeleton strict/full-part token swaps over a half-open frame
interval and reports donor transfer, local feature preservation, kinematic
leakage outside FK descendants, boundary velocity and foot metrics.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from stylized_motion.anim import quat
from stylized_motion.anim.features import denormalize_motion_features
from stylized_motion.data import open_any_feature_store
from stylized_motion.data.sampling import FixedWindowSampler
from stylized_motion.learning.losses import (
    _masked_weighted_mean,
    integrate_root_trajectory,
    reconstruct_joint_positions,
    rotation_6d_to_matrix,
)
from stylized_motion.learning.nef_layout import (
    NEF_EDIT_PARTS,
    NEF_STREAM_COORDINATES,
    NEF_STREAM_NAMES,
    NEFLayout,
    nef_edit_streams,
)
from stylized_motion.learning.nef_data import (
    read_clip_window,
    read_sampler_window,
    store_length,
)
from stylized_motion.learning.representation import NEF_FSQ_FAMILY, load_representation_checkpoint
from stylized_motion.learning.runner import choose_device


# The decoder is causal with RF=34, so an edited token at frame t can move
# features up to t + 33.
DECODER_INFLUENCE_FRAMES = 33
# CPU conv kernels leak round-off (~1e-7 relative) a few frames ahead of the
# mathematical causal boundary, so boundary checks compare against a tolerance.
CAUSAL_TOLERANCE = 1e-6


def _validate_nef_model(model) -> None:
    if model.family != NEF_FSQ_FAMILY:
        raise ValueError(f"NEF-FSQ evaluation requires a {NEF_FSQ_FAMILY!r} checkpoint, got {model.family!r}")


def _frame_mask(frames: int, start: int, stop: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, frames, dtype=torch.bool, device=device)
    mask[:, max(start, 0) : max(stop, 0)] = True
    return mask


def rotation_angle_error(pred_6d: torch.Tensor, target_6d: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two [..., 3, 2] rotation-matrix columns."""
    pred = rotation_6d_to_matrix(pred_6d)
    target = rotation_6d_to_matrix(target_6d)
    relative = pred.transpose(-1, -2) @ target
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.arccos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def _stream_errors(
    recon: torch.Tensor,
    target: torch.Tensor,
    layout: NEFLayout,
    weights: torch.Tensor,
    motion_dim: int,
) -> dict[str, dict[str, float]]:
    feature_indices = layout.feature_indices(motion_dim)
    frame_mask = torch.ones(recon.shape[:2], dtype=torch.bool, device=recon.device)
    pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
    result: dict[str, dict[str, float]] = {}
    for stream in NEF_STREAM_NAMES:
        index = feature_indices[stream].to(recon.device)
        stream_weights = weights[index]
        result[stream] = {
            "recon": float(_masked_weighted_mean((recon - target).abs()[..., index], frame_mask, stream_weights)),
            "delta": float(
                _masked_weighted_mean(
                    ((recon[:, 1:] - recon[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()[..., index],
                    pair_mask,
                    stream_weights,
                )
            ),
        }
    return result


def _accumulate_token_statistics(
    accumulator: dict[str, dict[str, torch.Tensor | int]],
    indices_by_stream: Mapping[str, torch.Tensor],
    num_levels: int,
) -> None:
    """Accumulate dataset-level level counts and transition counts per stream."""
    for stream, values in indices_by_stream.items():
        counts = torch.stack(
            [
                torch.bincount(values[..., token].reshape(-1), minlength=num_levels).float()
                for token in range(values.shape[-1])
            ]
        )
        changes = values[:, 1:] != values[:, :-1]
        if stream not in accumulator:
            accumulator[stream] = {
                "counts": torch.zeros_like(counts),
                "changes": 0,
                "transitions": 0,
            }
        stats = accumulator[stream]
        stored_counts = stats["counts"]
        assert isinstance(stored_counts, torch.Tensor)
        stats["counts"] = stored_counts + counts
        stats["changes"] = int(stats["changes"]) + int(changes.sum())
        stats["transitions"] = int(stats["transitions"]) + changes.numel()


def _token_utilization(
    accumulator: Mapping[str, Mapping[str, torch.Tensor | int]],
) -> dict[str, dict[str, float]]:
    """Summarize utilization after all evaluation windows have been accumulated."""
    result: dict[str, dict[str, float]] = {}
    for stream, stats in accumulator.items():
        counts = stats["counts"]
        assert isinstance(counts, torch.Tensor)
        probabilities = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        perplexity = torch.exp(-(probabilities * torch.log(probabilities + 1e-7)).sum(dim=-1))
        transitions = int(stats["transitions"])
        result[stream] = {
            "level_usage": float((counts > 0).float().mean()),
            "level_perplexity": float(perplexity.mean()),
            "coordinate_change_rate": int(stats["changes"]) / transitions if transitions else 0.0,
        }
    return result


def _accumulate(target: dict[str, list[float]], values: Mapping[str, float]) -> None:
    for name, value in values.items():
        target.setdefault(name, []).append(float(value))


def _mean_of(values: Mapping[str, list[float]]) -> dict[str, float]:
    return {name: float(np.mean(items)) for name, items in values.items()}


def read_window(store, range_idx: int, relative_start: int, length: int, history: int):
    """Reads ``[history + length]`` feature frames with left padding.

    ``range_idx`` is a v3 range row or a v4 clip row, and ``relative_start`` is
    relative to that row's start.  Both store generations are served by the same
    shared reader (``nef_data.read_clip_window``).
    """
    total = store_length(store)
    if range_idx < 0 or range_idx >= total:
        raise ValueError(f"Range index {range_idx} must be in [0, {total - 1}]")
    if relative_start < 0 or length <= 0:
        raise ValueError("Range-relative start must be non-negative and length positive")
    from stylized_motion.learning.nef_data import split_clip_geometry

    _shard, offset, _clip_length = split_clip_geometry(store, int(range_idx))
    window, shard_idx = read_clip_window(
        store,
        int(range_idx),
        offset + int(relative_start),
        int(length),
        history=int(history),
    )
    return window, shard_idx


def model_space(window: np.ndarray, store, feature_stats: Mapping[str, object]) -> torch.Tensor:
    """The store normalizes with its own statistics; the checkpoint may differ."""
    raw = denormalize_motion_features(window, store.stats)
    return torch.from_numpy(renormalize(raw, feature_stats))


def renormalize(raw: np.ndarray, feature_stats: Mapping[str, object]) -> np.ndarray:
    offset = np.asarray(feature_stats["offset"], dtype=np.float32)
    scale = np.asarray(feature_stats["scale"], dtype=np.float32)
    return ((raw - offset) / scale).astype(np.float32)


def swap_stream_tokens(
    target_indices: torch.Tensor,
    donor_indices: torch.Tensor,
    layout: NEFLayout,
    streams: Sequence[str],
    start: int,
    stop: int,
) -> torch.Tensor:
    """Replaces only the given stream slices inside the half-open [start, stop) interval."""
    if target_indices.ndim != 3 or target_indices.shape != donor_indices.shape:
        raise ValueError("Token swap requires matching [B, T, 40] target and donor tensors")
    if target_indices.shape[-1] != layout.num_coordinates:
        raise ValueError(f"Token swap requires {layout.num_coordinates} coordinates per frame")
    unknown = sorted(set(streams) - set(NEF_STREAM_NAMES))
    if unknown:
        raise ValueError(f"Unknown NEF streams {unknown}; expected {list(NEF_STREAM_NAMES)}")
    edited = target_indices.clone()
    start = max(int(start), 0)
    stop = min(int(stop), target_indices.shape[1])
    if stop <= start:
        return edited
    slices = layout.stream_slices
    for stream in streams:
        edited[:, start:stop, slices[stream]] = donor_indices[:, start:stop, slices[stream]]
    return edited


def _descendants(parents: Sequence[int], joints: Sequence[int]) -> set[int]:
    children: dict[int, list[int]] = {index: [] for index in range(len(parents))}
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)
    result: set[int] = set()
    stack = list(joints)
    while stack:
        for child in children[stack.pop()]:
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def _world_positions(motion, offset, scale, ref_pos, parents, dt) -> torch.Tensor:
    return reconstruct_joint_positions(motion, offset, scale, ref_pos, parents, dt, world_space=True)


def contacts_from_toe_motion(
    positions: torch.Tensor,
    toe_indices: Sequence[int],
    dt: float,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Infer contact from reconstructed 3D toe speed using the preprocessing threshold."""
    if positions.ndim != 4 or positions.shape[1] < 2:
        raise ValueError("Contact inference requires positions [B,T,J,3] with T >= 2")
    toe_speed = (
        positions[:, 1:, toe_indices] - positions[:, :-1, toe_indices]
    ).norm(dim=-1) / float(dt)
    contacts = toe_speed < float(threshold)
    return torch.cat((contacts[:, :1], contacts), dim=1)


def validate_checkpoint_store(checkpoint, model, store) -> None:
    """Validate semantic feature and skeleton fields without relying on hashes."""
    module = model.module
    layout = module.layout
    if module.motion_dim != store.motion_dim:
        raise ValueError("NEF-FSQ checkpoint and feature database motion dimensions differ")
    if layout.names != tuple(store.names) or layout.parents != tuple(int(value) for value in store.parents):
        raise ValueError("NEF-FSQ checkpoint and feature database skeletons differ")
    checkpoint_schema = checkpoint.get("feature_schema")
    if not isinstance(checkpoint_schema, Mapping):
        raise ValueError("NEF-FSQ checkpoint is missing feature schema metadata")
    store_schema = store.feature_schema()
    for key in ("name", "motion_dim", "joint_subset"):
        if checkpoint_schema.get(key) != store_schema.get(key):
            raise ValueError(f"NEF-FSQ checkpoint and feature database differ at feature schema field {key!r}")


def run_report(args: argparse.Namespace) -> dict[str, object]:
    store = open_any_feature_store(args.feature_database)
    try:
        checkpoint, model = load_representation_checkpoint(args.checkpoint, torch.device("cpu"))
        _validate_nef_model(model)
        validate_checkpoint_store(checkpoint, model, store)
        device = choose_device(args.device)
        model = model.to(device).eval()
        module = model.module
        layout = module.layout
        feature_stats = checkpoint["feature_stats"]
        offset = torch.as_tensor(feature_stats["offset"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(feature_stats["scale"], dtype=torch.float32, device=device)
        ref_pos = torch.as_tensor(feature_stats["ref_pos"], dtype=torch.float32, device=device)
        weights = torch.as_tensor(
            np.asarray(feature_stats["weights"], dtype=np.float32), dtype=torch.float32, device=device
        )
        parents = tuple(int(value) for value in np.asarray(feature_stats["parents"]).tolist())
        windows = list(FixedWindowSampler(store, args.split, target_frames=64, stride=64, include_tail=True))
        if args.windows > 0:
            windows = windows[: args.windows]
        if not windows:
            raise ValueError(f"No {args.split} windows are available for the NEF-FSQ report")
        head_index = layout.names.index("Head")
        rotation_slice = slice(9, 9 + (layout.num_joints - 1) * 6)

        totals: dict[str, list[float]] = {}
        stream_sums: dict[str, dict[str, list[float]]] = {}
        token_statistics: dict[str, dict[str, torch.Tensor | int]] = {}
        head_sums: dict[str, list[float]] = {}
        with torch.inference_mode():
            for window in windows:
                raw = np.asarray(read_sampler_window(store, window), dtype=np.float32)
                motion = torch.from_numpy(renormalize(denormalize_motion_features(raw, store.stats), feature_stats))
                motion = motion.to(device)[None]
                output = model(motion, collect_metrics=False)
                recon = output["recon_state"]
                for stream, values in _stream_errors(recon, motion, layout, weights, module.motion_dim).items():
                    _accumulate(stream_sums.setdefault(stream, {}), values)
                _accumulate_token_statistics(token_statistics, output["stream_indices"], module.num_levels)

                frame_mask = torch.ones(1, 64, dtype=torch.bool, device=device)
                pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
                _accumulate(
                    totals,
                    {
                        "recon": float(_masked_weighted_mean((recon - motion).abs(), frame_mask, weights)),
                        "delta": float(
                            _masked_weighted_mean(
                                ((recon[:, 1:] - recon[:, :-1]) - (motion[:, 1:] - motion[:, :-1])).abs(),
                                pair_mask,
                                weights,
                            )
                        ),
                    },
                )
                positions = _world_positions(
                    torch.cat((recon, motion), dim=0), offset, scale, ref_pos, parents, args.root_dt
                )
                pred_positions, target_positions = positions.split(1, dim=0)
                fk_error = (pred_positions[0] - target_positions[0]).norm(dim=-1).mean(dim=0)
                rotation_shape = (1, recon.shape[1], layout.num_joints - 1, 3, 2)
                recon_raw = recon * scale + offset
                motion_raw = motion * scale + offset
                rotation_error = rotation_angle_error(
                    recon_raw[..., rotation_slice].reshape(rotation_shape),
                    motion_raw[..., rotation_slice].reshape(rotation_shape),
                ).mean(dim=(0, 1))
                pred_root_positions, pred_root_rotations = integrate_root_trajectory(recon, offset, scale, args.root_dt)
                target_root_positions, target_root_rotations = integrate_root_trajectory(motion, offset, scale, args.root_dt)
                _accumulate(
                    totals,
                    {
                        "root_pos": float((pred_root_positions[:, 1:] - target_root_positions[:, 1:]).abs().mean()),
                        "root_rot": float(
                            quat.torch_quat_angle(pred_root_rotations[:, 1:], target_root_rotations[:, 1:]).mean()
                        ),
                        "fk_world": float(fk_error.mean()),
                    },
                )
                _accumulate(
                    head_sums,
                    {
                        "fk_world": float(fk_error[head_index]),
                        "rotation": float(rotation_error[head_index - 1]),
                    },
                )
        token_summary = _token_utilization(token_statistics)
        return {
            "family": model.family,
            "representation_id": model.representation_id,
            "skeleton": layout.skeleton,
            "num_joints": layout.num_joints,
            "motion_dim": module.motion_dim,
            "split": args.split,
            "windows": len(windows),
            "overall": _mean_of(totals),
            "streams": {
                stream: {**_mean_of(stream_sums.get(stream, {})), **token_summary.get(stream, {})}
                for stream in NEF_STREAM_NAMES
            },
            "head": {
                "joint_index": int(head_index),
                "feature_dim": int(layout.feature_indices(module.motion_dim)["head_node"].numel()),
                "coordinates": NEF_STREAM_COORDINATES["head_node"],
                **_mean_of(head_sums),
            },
        }
    finally:
        store.close()


def run_transfer(args: argparse.Namespace) -> dict[str, object]:
    store = open_any_feature_store(args.feature_database)
    try:
        checkpoint, model = load_representation_checkpoint(args.checkpoint, torch.device("cpu"))
        _validate_nef_model(model)
        validate_checkpoint_store(checkpoint, model, store)
        device = choose_device(args.device)
        model = model.to(device).eval()
        module = model.module
        layout = module.layout
        feature_stats = checkpoint["feature_stats"]
        offset = torch.as_tensor(feature_stats["offset"], dtype=torch.float32, device=device)
        scale = torch.as_tensor(feature_stats["scale"], dtype=torch.float32, device=device)
        ref_pos = torch.as_tensor(feature_stats["ref_pos"], dtype=torch.float32, device=device)
        weights = torch.as_tensor(
            np.asarray(feature_stats["weights"], dtype=np.float32), dtype=torch.float32, device=device
        )
        parents = layout.parents
        history = int(model.history_frames)
        length = int(args.length)

        target_window, _ = read_window(store, args.target_range_idx, args.target_start, length, history)
        donor_window, _ = read_window(store, args.donor_range_idx, args.donor_start, length, history)
        pair = torch.stack(
            (model_space(target_window, store, feature_stats), model_space(donor_window, store, feature_stats))
        ).to(device)

        streams = nef_edit_streams(args.part, full_part=args.edit == "full")
        edit_start = history + int(args.edit_start)
        edit_stop = history + (length if args.edit_stop is None else int(args.edit_stop))
        feature_indices = layout.feature_indices(module.motion_dim)
        edited_features = sorted(index for stream in streams for index in feature_indices[stream].tolist())
        edited_joints = sorted({joint for stream in streams for joint in layout.stream_joints(stream)})
        influenced_joints = set(edited_joints) | _descendants(parents, edited_joints)
        descendant_joints = sorted(influenced_joints - set(edited_joints))
        non_target_joints = sorted(set(range(layout.num_joints)) - influenced_joints)

        stream_mask = torch.zeros(module.motion_dim, dtype=torch.bool, device=device)
        for stream in streams:
            stream_mask[feature_indices[stream].to(device)] = True

        with torch.inference_mode():
            indices = model.encode_to_indices(pair)
            target_indices, donor_indices = indices[0:1], indices[1:2]
            edited_indices = swap_stream_tokens(
                target_indices, donor_indices, layout, streams, edit_start, edit_stop
            )
            recon = model.decode_from_indices(torch.cat((target_indices, donor_indices, edited_indices), dim=0))
            target_recon, donor_recon, edited_recon = recon[0:1], recon[1:2], recon[2:3]

            frames = target_indices.shape[1]
            influence_stop = min(frames, edit_stop + DECODER_INFLUENCE_FRAMES)
            edit_frames = _frame_mask(frames, edit_start, edit_stop, device)

            local_change = (edited_recon - target_recon).abs()
            changed_streams = [
                stream
                for stream in NEF_STREAM_NAMES
                if float(local_change[..., feature_indices[stream].to(device)].max()) > 0.0
            ]
            non_target_features = sorted(set(range(module.motion_dim)) - set(edited_features))

            positions = _world_positions(
                torch.cat((target_recon, edited_recon, donor_recon), dim=0),
                offset, scale, ref_pos, parents, args.root_dt,
            )
            target_positions, edited_positions, donor_positions = positions.split(1, dim=0)
            world_change = (edited_positions[0] - target_positions[0]).norm(dim=-1)  # [T, J]

            velocity = edited_recon[:, 1:] - edited_recon[:, :-1]
            target_velocity = target_recon[:, 1:] - target_recon[:, :-1]
            velocity_step = (velocity - target_velocity)[0][:, stream_mask].abs().mean(dim=-1)  # [T-1]

            def step_at(index: int) -> float:
                return float(velocity_step[index]) if 0 <= index < velocity_step.numel() else 0.0

            toe_indices = [layout.names.index("LeftToeBase"), layout.names.index("RightToeBase")]
            edited_contact_feature = ((edited_recon[..., -2:] * scale[-2:] + offset[-2:]) > 0.5).to(torch.float32)
            target_contact = ((target_recon[..., -2:] * scale[-2:] + offset[-2:]) > 0.5).to(torch.float32)
            contact_gate = torch.clamp(target_contact[:, 1:] * target_contact[:, :-1], 0.0, 1.0)
            # Same horizontal-speed convention as the runner's foot-slide term.
            foot_speed = (edited_positions[:, 1:, toe_indices] - edited_positions[:, :-1, toe_indices])[
                ..., (0, 2)
            ].abs().mean(dim=-1) / float(args.root_dt)
            target_foot_speed = (
                target_positions[:, 1:, toe_indices] - target_positions[:, :-1, toe_indices]
            )[..., (0, 2)].abs().mean(dim=-1) / float(args.root_dt)
            # Preprocessing labels contact below 0.15 m/s.  The first frame has
            # no backward difference, so reuse the first available transition.
            inferred_contact = contacts_from_toe_motion(
                edited_positions, toe_indices, args.root_dt
            )
            nonzero = target_indices[0, edit_start:edit_stop, layout.stream_slices["global"]]
            edited_global = edited_indices[0, edit_start:edit_stop, layout.stream_slices["global"]]

        transfer = float(
            _masked_weighted_mean((edited_recon - donor_recon).abs()[..., stream_mask], edit_frames, weights[stream_mask])
        )
        report = {
            "family": model.family,
            "representation_id": model.representation_id,
            "skeleton": layout.skeleton,
            "part": args.part,
            "edit": args.edit,
            "streams": list(streams),
            "edit_interval": [int(edit_start), int(edit_stop)],
            "influence_interval": [int(edit_start), int(influence_stop)],
            "donor_transfer": transfer,
            "target_recon_deviation": float(
                _masked_weighted_mean(
                    (edited_recon - target_recon).abs()[..., stream_mask], edit_frames, weights[stream_mask]
                )
            ),
            "local_preservation": {
                "exact_stream_isolation": set(changed_streams) == set(streams),
                "changed_streams": changed_streams,
                "max_abs_change": float(local_change[..., non_target_features].max()) if non_target_features else 0.0,
                "max_abs_change_before_edit": float(local_change[:, :edit_start].max()) if edit_start > 0 else 0.0,
                "pre_edit_unchanged_within_tolerance": bool(
                    float(local_change[:, :edit_start].max()) <= CAUSAL_TOLERANCE if edit_start > 0 else True
                ),
                "frames_after_influence_unchanged": bool(
                    float(local_change[:, influence_stop:].abs().max()) <= CAUSAL_TOLERANCE
                )
                if influence_stop < frames
                else True,
            },
            "kinematic_influence": {
                "edited_joints": [layout.names[joint] for joint in edited_joints],
                "edited_joints_world_change": float(world_change[:, edited_joints].mean()),
                "descendant_joints": [layout.names[joint] for joint in descendant_joints],
                "descendant_world_change": float(world_change[:, descendant_joints].mean())
                if descendant_joints
                else 0.0,
                "non_target_joints": [layout.names[joint] for joint in non_target_joints],
                "non_target_world_change": float(world_change[:, non_target_joints].mean())
                if non_target_joints
                else 0.0,
                "non_target_world_change_max": float(world_change[:, non_target_joints].max())
                if non_target_joints
                else 0.0,
                "influence_window_world_change": float(world_change[influence_stop - 1, non_target_joints].mean())
                if non_target_joints
                else 0.0,
            },
            "boundary": {
                "velocity_step_at_start": step_at(edit_start - 1),
                "velocity_step_at_stop": step_at(edit_stop - 1),
                "max_velocity_step_in_influence": float(
                    velocity_step[max(edit_start - 1, 0) : max(influence_stop - 1, 1)].max()
                ),
            },
            "foot": {
                "toe_joints": [layout.names[index] for index in toe_indices],
                "contact_feature_preservation_mismatch_rate": float(
                    (edited_contact_feature != target_contact).any(dim=-1).float().mean()
                ),
                "kinematic_contact_mismatch_rate": float(
                    (inferred_contact != target_contact.bool()).any(dim=-1).float().mean()
                ),
                "kinematic_contact_mismatch_rate_in_edit": float(
                    (inferred_contact[:, edit_start:edit_stop] != target_contact[:, edit_start:edit_stop].bool())
                    .any(dim=-1)
                    .float()
                    .mean()
                )
                if edit_stop > edit_start
                else 0.0,
                "foot_slide_edited": float(_masked_weighted_mean(foot_speed, contact_gate)),
                "foot_slide_target": float(_masked_weighted_mean(target_foot_speed, contact_gate)),
                "contact_frames": int(contact_gate.sum()),
            },
            "token_change": {
                "global_stream_changed": bool(not torch.equal(nonzero, edited_global)),
            },
        }
    finally:
        store.close()
    return report


# Public names are the shared evaluation entry points; the underscored aliases
# are kept because tests and older callers import them.
_rotation_angle_error = rotation_angle_error
_contacts_from_toe_motion = contacts_from_toe_motion
_validate_checkpoint_store = validate_checkpoint_store
_read_window = read_window
_model_space = model_space
_renormalize = renormalize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a NEF-FSQ checkpoint: per-stream report or token transfer.")
    parser.add_argument("--metric", choices=["report", "transfer"], default="transfer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--windows", type=int, default=16, help="Report mode: number of 64-frame windows (0 = all).")
    parser.add_argument("--target-range-idx", type=int, default=None)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--donor-range-idx", type=int, default=None)
    parser.add_argument("--donor-start", type=int, default=0)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--part", choices=sorted(NEF_EDIT_PARTS), default="left_arm")
    parser.add_argument("--edit", choices=["strict", "full"], default="strict")
    parser.add_argument("--edit-start", type=int, default=0)
    parser.add_argument("--edit-stop", type=int, default=None)
    parser.add_argument("--root-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.metric == "transfer" and (args.target_range_idx is None or args.donor_range_idx is None):
        raise ValueError("transfer mode requires --target-range-idx and --donor-range-idx")
    report = run_report(args) if args.metric == "report" else run_transfer(args)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
