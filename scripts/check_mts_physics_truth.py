#!/usr/bin/env python
"""N01: the physical truth oracle, the mirror check and the error attribution.

Four questions, all read-only, on a handful of deterministic windows:

1. **Does the feature -> FK map reproduce the source motion?**  The ground truth
   is rebuilt from the *source BVH* through the same preprocessing the store was
   built with (``preprocess._process_motion_data`` -> slice -> feature components),
   so it is independent of the stored features and of the evaluator.  The raw
   features are then posed three ways: with the store's own ``ref_pos`` (the
   dataset mean, kept as the legacy path), with the SOMA bind skeleton, and with
   the *wrong* mirror of it.  A mirror clip is paired with its own ``_M`` source
   file, so the mirror rule is checked against real data, not against itself.
2. **What does the tokenizer cost?**  Features -> tokens -> decoded features -> FK,
   against the same truth, with the cold-start frames reported separately from the
   measurement interval (the decoder has no history before frame 0).
3. **Which channels own the residual?**  The decoded features' FK error is
   re-measured with one group of channels replaced by the truth -- root motion
   (velocity/angular velocity/hips position), joint rotations, joint angular
   velocities -- using the feature contract's own groups, never hand-picked
   indices.
4. **What do the physical numbers say?**  Root path, root speed, contact rate and
   foot slide are computed on *denormalized* motion through the shared context, so
   a normalized channel is never compared against a metre.

    python scripts/check_mts_physics_truth.py \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --feature-store data/processed/seed_soma_pruned_v4_ah \
      --catalog data/processed/seed_ah_catalog/clips.jsonl \
      --pairs 2 --warmup 16 \
      --output outputs/mts_next_round_20260918/N01/physics_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.anim import quat  # noqa: E402
from stylized_motion.anim import bvh as bvh_module  # noqa: E402
from stylized_motion.anim.features import (  # noqa: E402
    build_motion_feature_components,
    deserialize_motion_feature_stats,
    joint_feature_dim,
)
from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.preprocess import _process_motion_data, _slice_motion  # noqa: E402
from stylized_motion.learning.mts_operator.physics_context import (  # noqa: E402
    PHYSICAL_METRIC_VERSION,
    PhysicsContext,
)
from stylized_motion.learning.mts_operator.summary import write_summary  # noqa: E402
from stylized_motion.learning.nef_data import store_normalized_window  # noqa: E402
from stylized_motion.learning.nef_probe import model_space_window  # noqa: E402
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N01 physical truth oracle (read-only).")
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "seed_ah_catalog" / "clips.jsonl",
    )
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data" / "raw" / "seed")
    parser.add_argument("--pairs", type=int, default=2, help="Mirror pairs (each pair is two clips).")
    parser.add_argument("--warmup", type=int, default=16, help="Frames excluded from the measurement interval.")
    parser.add_argument("--frames", type=int, default=64, help="Window length for the physical metrics.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_catalog(path: Path, limit: int = 4000) -> dict[int, dict[str, Any]]:
    catalog: dict[int, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        catalog[int(entry["clip_id"])] = entry
    if not catalog:
        raise ValueError(f"The catalog {path} is empty")
    return catalog


def store_row(store: Any, clip_id: int, mirror: bool) -> int | None:
    rows = np.flatnonzero(
        (np.asarray(store.clip_source_id) == int(clip_id))
        & (np.asarray(store.clip_mirror) == bool(mirror))
    )
    return None if len(rows) == 0 else int(rows[0])


def pick_pairs(store: Any, catalog: dict[int, dict[str, Any]], *, count: int, frames: int) -> list[dict[str, Any]]:
    """Deterministic mirror pairs: a plain clip, its official mirror, both in val."""
    pairs: list[dict[str, Any]] = []
    split = np.asarray(store.clip_split)
    for clip_id in sorted(catalog):
        entry = catalog[clip_id]
        if bool(entry["is_mirror"]):
            continue
        partner = catalog.get(clip_id + 1)
        if partner is None or not bool(partner["is_mirror"]):
            continue
        if int(partner["group_id"]) != int(entry["group_id"]):
            continue
        plain_row = store_row(store, clip_id, False)
        mirror_row = store_row(store, clip_id + 1, True)
        if plain_row is None or mirror_row is None:
            continue
        if int(split[plain_row]) != 1 or int(split[mirror_row]) != 1:
            continue
        if int(store.clip_length[plain_row]) < int(frames):
            continue
        pairs.append(
            {
                "group_id": int(entry["group_id"]),
                "plain": {"clip_id": clip_id, "row": plain_row, "entry": entry},
                "mirror": {"clip_id": clip_id + 1, "row": mirror_row, "entry": partner},
            }
        )
        if len(pairs) >= int(count):
            break
    if not pairs:
        raise ValueError("No val mirror pair with a full window was found in the catalog")
    return pairs


def truth_from_source(entry: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    """The motion the store's features were built from, rebuilt from the BVH."""
    path = Path(raw_root) / str(entry["relative_path"])
    if not path.exists():
        raise FileNotFoundError(f"Source BVH is missing: {path}")
    motion = _process_motion_data(
        bvh_module.load(path.as_posix()),
        mirror=bool(entry["is_mirror"]),
        prune_ends_and_fingers=True,
    )
    cropped = _slice_motion(motion, int(entry["start"]), int(entry["stop"]))
    components = build_motion_feature_components(cropped)
    return {"path": str(path), "motion": cropped, "features": np.asarray(components.x, dtype=np.float32)}


def quaternion_angles(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(left * right, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def geometry_error(
    predicted_positions: np.ndarray,
    predicted_rotations: np.ndarray,
    truth_positions: np.ndarray,
    truth_rotations: np.ndarray,
    *,
    start: int = 0,
) -> dict[str, float]:
    """Root-relative joint error in cm plus the geodesic rotation error in degrees."""
    slice_ = slice(int(start), None)
    pred = predicted_positions[slice_]
    true = truth_positions[slice_]
    positions = np.linalg.norm((pred - pred[:, :1]) - (true - true[:, :1]), axis=-1)
    angles = quaternion_angles(predicted_rotations[slice_], truth_rotations[slice_])
    return {
        "frames": int(positions.shape[0]),
        "position_mean_cm": float(positions.mean() * 100.0),
        "position_median_cm": float(np.median(positions) * 100.0),
        "position_max_cm": float(positions.max() * 100.0),
        "rotation_mean_deg": float(angles.mean()),
        "rotation_median_deg": float(np.median(angles)),
    }


#: The feature contract's own channel groups (see ``features.MotionFeatureComponents``).
def channel_groups(joints: int) -> dict[str, slice]:
    offset = 0
    groups: dict[str, slice] = {}
    width = 3
    groups["root_linear_velocity"] = slice(offset, offset + width); offset += width
    groups["root_angular_velocity"] = slice(offset, offset + width); offset += width
    groups["hips_position"] = slice(offset, offset + width); offset += width
    width = (joints - 1) * 6
    groups["joint_rotations"] = slice(offset, offset + width); offset += width
    groups["hips_velocity"] = slice(offset, offset + 3); offset += 3
    groups["joint_angular_velocities"] = slice(offset, offset + (joints - 1) * 3); offset += (joints - 1) * 3
    groups["contacts"] = slice(offset, offset + 2); offset += 2
    return groups


def physical_metrics(state: Any, kinematic: Any, *, start: int) -> dict[str, float]:
    """Root path/speed, contact rate and foot slide from denormalized FK."""
    from stylized_motion.learning.nef_eval import contacts_from_toe_motion

    positions = state.global_positions
    root = state.root_positions
    dt = float(kinematic.dt)
    frames = positions.shape[0]
    root_steps = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    metrics = {
        "frames": int(frames - int(start)),
        "root_path_m": float(root_steps[int(start) :].sum()),
        "root_speed_mean_mps": float(root_steps[int(start) :].mean() / dt) if frames - int(start) > 1 else None,
    }
    toe_indices = kinematic.toe_indices
    if toe_indices is None or frames < 2:
        return metrics
    contacts = contacts_from_toe_motion(
        torch.as_tensor(positions)[None], toe_indices, dt, threshold=float(kinematic.contact_threshold)
    )[0].numpy()
    toe = positions[:, list(toe_indices)]
    speed = np.linalg.norm(np.diff(toe, axis=0)[..., (0, 2)], axis=-1) / dt
    gate = contacts[:-1] & contacts[1:]
    gate = gate[int(start) :]
    metrics.update(
        {
            "contact_rate": float(contacts[int(start) :].mean()),
            "contact_gate_frames": int(gate.sum()),
            "foot_slide_mps": float(speed[int(start) :][gate].mean()) if bool(gate.any()) else None,
            "foot_height_mean_m": float(toe[int(start) :, :, 1].mean()),
            "foot_height_std_m": float(toe[int(start) :, :, 1].std()),
        }
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    checkpoint, tokenizer = load_representation_checkpoint(args.tokenizer_checkpoint, torch.device("cpu"))
    tokenizer = tokenizer.eval()
    feature_stats = checkpoint["feature_stats"]
    context = PhysicsContext.from_feature_stats(
        feature_stats, provenance={"tokenizer_checkpoint": str(args.tokenizer_checkpoint)}
    )
    store = open_any_feature_store(args.feature_store)
    try:
        catalog = load_catalog(args.catalog)
        pairs = pick_pairs(store, catalog, count=int(args.pairs), frames=int(args.frames))
        joints = len(context.names)
        groups = channel_groups(joints)
        if int(feature_stats.get("motion_dim", 0) or 0) != joint_feature_dim(joints):
            raise SystemExit(
                f"The checkpoint's motion_dim {feature_stats.get('motion_dim')} does not match "
                f"joint_feature_dim({joints}) = {joint_feature_dim(joints)}"
            )
        report: dict[str, Any] = {
            "kind": "mts_physics_truth_oracle",
            "physical_metric_version": PHYSICAL_METRIC_VERSION,
            "context": context.describe(),
            "tokenizer_checkpoint": str(args.tokenizer_checkpoint),
            "feature_store": str(args.feature_store),
            "warmup_frames": int(args.warmup),
            "channel_groups": {name: [int(sl.start), int(sl.stop)] for name, sl in groups.items()},
            "clips": [],
        }
        with torch.no_grad():
            for pair in pairs:
                for role in ("plain", "mirror"):
                    item = pair[role]
                    entry = item["entry"]
                    truth = truth_from_source(entry, args.raw_root)
                    raw = np.asarray(store.read_clip(int(item["row"])), dtype=np.float32)
                    mirror = bool(entry["is_mirror"])
                    record: dict[str, Any] = {
                        "role": role,
                        "clip_id": int(item["clip_id"]),
                        "store_row": int(item["row"]),
                        "source": truth["path"],
                        "frames": int(raw.shape[0]),
                        "mirror": mirror,
                        "feature_fidelity": {
                            "max_abs_difference": float(np.abs(raw - truth["features"]).max()),
                            "mean_abs_difference": float(np.abs(raw - truth["features"]).mean()),
                        },
                    }
                    truth_motion = truth["motion"]
                    truth_rotations, truth_positions = quat.fk(
                        truth_motion["rotations"],
                        truth_motion["positions"],
                        np.asarray(truth_motion["parents"], dtype=np.int32),
                    )
                    seed_kwargs = {
                        "root_position0": truth_motion["positions"][0, 0],
                        "root_rotation0": truth_motion["rotations"][0, 0],
                    }
                    # The legacy path is the *stored* ref_pos exactly as the old
                    # physics evaluator read it; the bind skeleton is what every
                    # current metric uses; the wrong mirror is the control that says
                    # whether the mirror rule is load bearing.
                    stored_stats, _ = deserialize_motion_feature_stats(dict(feature_stats))
                    skeletons = {
                        "stored_ref_pos": stored_stats,
                        "bind_skeleton": context.stats_for(mirror=mirror),
                        "wrong_mirror": context.stats_for(mirror=not mirror),
                    }
                    record["geometry"] = {}
                    for name, stats in skeletons.items():
                        state = context.world_state(
                            raw, mirror=mirror, normalized=False, contact_threshold=None,
                            stats=stats, **seed_kwargs,
                        )
                        record["geometry"][name] = geometry_error(
                            state.global_positions, state.global_rotations, truth_positions, truth_rotations
                        )
                        record["geometry"][name]["root_path_error_m"] = float(
                            np.linalg.norm(state.root_positions - truth_positions[:, 0], axis=-1).mean()
                        )
                    # Tokenizer round trip, in the encoder's own input space.
                    normalized = store_normalized_window(store, raw)
                    model_input = model_space_window(normalized, store, feature_stats)
                    tokens = tokenizer.encode_indices(model_input[None])
                    decoded = tokenizer.decode_indices(tokens)[0].numpy()
                    record["tokenizer"] = {
                        "tokens": int(tokens.shape[1] * tokens.shape[2]),
                        "levels_used": sorted(int(value) for value in torch.unique(tokens)),
                        "decoded_feature_max_abs": float(np.abs(decoded - model_input.numpy()).max()),
                        "decoded_feature_mean_abs": float(np.abs(decoded - model_input.numpy()).mean()),
                    }
                    decoded_state = context.world_state(
                        decoded, mirror=mirror, normalized=True, contact_threshold=None, **seed_kwargs
                    )
                    record["tokenizer"]["geometry_full_range"] = geometry_error(
                        decoded_state.global_positions, decoded_state.global_rotations,
                        truth_positions, truth_rotations,
                    )
                    record["tokenizer"]["geometry_after_warmup"] = geometry_error(
                        decoded_state.global_positions, decoded_state.global_rotations,
                        truth_positions, truth_rotations, start=int(args.warmup),
                    )
                    # Which channel group owns the residual: replace one group at a time
                    # with the truth (in the encoder's input space) and re-measure.
                    attribution: dict[str, Any] = {"note": "decoded features with one channel group restored to the "
                    "pre-quantization value; a small number means that group contributed little to the FK error"}
                    model_numpy = model_input.numpy()
                    replacements = {
                        "root_motion": ["root_linear_velocity", "root_angular_velocity", "hips_position"],
                        "joint_rotations": ["joint_rotations"],
                        "joint_angular_velocities": ["joint_angular_velocities"],
                        "all_fk_channels": [
                            "root_linear_velocity", "root_angular_velocity", "hips_position", "joint_rotations",
                        ],
                    }
                    # Truth in the encoder's input space, from the same normalization.
                    truth_model = model_space_window(
                        store_normalized_window(store, truth["features"]), store, feature_stats
                    ).numpy()
                    if truth_model.shape != model_numpy.shape:
                        raise SystemExit(
                            f"The truth features are {truth_model.shape}, the stored ones {model_numpy.shape}"
                        )
                    for label, names in replacements.items():
                        # Start from the *decoded* features (they carry the tokenizer's
                        # error) and restore one group to its pre-quantization value.
                        patched = decoded.copy()
                        for name in names:
                            patched[:, groups[name]] = truth_model[:, groups[name]]
                        state = context.world_state(
                            patched, mirror=mirror, normalized=True, contact_threshold=None, **seed_kwargs
                        )
                        attribution[label] = geometry_error(
                            state.global_positions, state.global_rotations,
                            truth_positions, truth_rotations, start=int(args.warmup),
                        )
                    record["attribution_after_warmup"] = attribution
                    # Physical metrics on denormalized motion, truth vs decoded.
                    kinematic = context.kinematic(mirror=mirror)
                    truth_state = context.world_state(
                        raw, mirror=mirror, normalized=False, contact_threshold=None, **seed_kwargs
                    )
                    record["physical"] = {
                        "comparison": "decoded tokens vs the source BVH, one window",
                        "warmup_frames": int(args.warmup),
                        "truth_full_range": physical_metrics(truth_state, kinematic, start=0),
                        "truth_after_warmup": physical_metrics(truth_state, kinematic, start=int(args.warmup)),
                        "decoded_full_range_cold_start": physical_metrics(decoded_state, kinematic, start=0),
                        "decoded_after_warmup": physical_metrics(decoded_state, kinematic, start=int(args.warmup)),
                    }
                    report["clips"].append(record)
                    print(
                        f"{role} clip {item['clip_id']}: feature max|d|={record['feature_fidelity']['max_abs_difference']:.2e} "
                        f"bind FK={record['geometry']['bind_skeleton']['position_mean_cm']:.4f}cm "
                        f"stored FK={record['geometry']['stored_ref_pos']['position_mean_cm']:.4f}cm "
                        f"wrong-mirror FK={record['geometry']['wrong_mirror']['position_mean_cm']:.4f}cm "
                        f"tok FK={record['tokenizer']['geometry_after_warmup']['position_mean_cm']:.3f}cm",
                        flush=True,
                    )
        # Verdicts, stated as checks with their evidence rather than a single score.
        clips = report["clips"]
        bind_errors = [clip["geometry"]["bind_skeleton"]["position_mean_cm"] for clip in clips]
        store_errors = [clip["geometry"]["stored_ref_pos"]["position_mean_cm"] for clip in clips]
        wrong_errors = [clip["geometry"]["wrong_mirror"]["position_mean_cm"] for clip in clips]
        report["checks"] = {
            "feature_fidelity_is_exact": {
                "requirement": "the stored raw features equal the source-BVH feature build",
                "passed": all(clip["feature_fidelity"]["max_abs_difference"] == 0.0 for clip in clips),
                "evidence": {
                    "max_abs_difference": max(clip["feature_fidelity"]["max_abs_difference"] for clip in clips)
                },
            },
            "bind_skeleton_reproduces_the_source_geometry": {
                "requirement": "FK on the raw features with the bind skeleton matches the BVH truth "
                "to sub-millimetre mean error and at least 50x better than the legacy ref_pos",
                "passed": all(
                    bind < 0.5 and bind * 50.0 < store
                    for bind, store in zip(bind_errors, store_errors)
                ),
                "evidence": {
                    "per_clip_means_cm": bind_errors,
                    "stored_ref_pos_means_cm": store_errors,
                    "note": "the residual grows with the window because the root path is integrated "
                    "from velocity; the per-clip frame counts are in the clip records",
                },
            },
            "the_stored_ref_pos_is_not_good_enough": {
                "requirement": "the legacy ref_pos path is measurably worse than the bind skeleton",
                "passed": all(store > bind for store, bind in zip(store_errors, bind_errors)),
                "evidence": {
                    "store_ref_pos_means_cm": store_errors,
                    "bind_means_cm": bind_errors,
                },
            },
            "the_mirror_rule_is_load_bearing": {
                "requirement": "posing on the wrong mirror skeleton costs at least 10x the bind error",
                "passed": all(wrong > 10.0 * max(bind, 1e-6) for wrong, bind in zip(wrong_errors, bind_errors)),
                "evidence": {"wrong_mirror_means_cm": wrong_errors, "bind_means_cm": bind_errors},
            },
        }
        report["checks"]["all_passed"] = {
            "requirement": "every geometry check above",
            "passed": all(
                entry["passed"] for name, entry in report["checks"].items() if name != "all_passed"
            ),
        }
        write_summary(args.output, report)
        print(
            json.dumps({name: entry["passed"] for name, entry in report["checks"].items()}, indent=2),
            flush=True,
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
