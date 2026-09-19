#!/usr/bin/env python
"""Decoder-side physical quality of one or more NEF-FSQ checkpoints.

    python scripts/evaluate_nef_physics.py \
      --checkpoint outputs/nef_fsq_soma_packed_40x9_1h/best.pt \
      --checkpoint outputs/nef_fsq_soma_packed_40x9_physical_ft/best.pt \
      --feature-database data/processed/seed_soma_pruned_v4 \
      --split test --windows 128 --output outputs/nef_physics

The training loss cannot answer plan §4.2's question: the v1 baseline sets every
physical *weight* to zero, so its joint/contact/foot losses are never computed and
`val_full` reports 0 for them.  These metrics are measured on the decoded motion
instead, which makes them weight-independent and comparable across checkpoints:

    fk_joint_error_*   world-space joint position error (cm), mean/median/worst
    root_drift_cm      integrated root translation error over the window
    root_rotation_deg  geodesic root rotation error
    foot_slide_mps     horizontal toe speed over frames the *target* marks contact
    contact_match_rate fraction of frames whose kinematic contact agrees with the
                       contact features (the label the tokenizer must reproduce)
    foot_height_cm     toe height error over contact frames
"""

from __future__ import annotations

import argparse
import csv
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
from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.sampling import FixedWindowSampler  # noqa: E402
from stylized_motion.learning.losses import (  # noqa: E402
    integrate_root_trajectory,
    reconstruct_joint_positions,
)
from stylized_motion.learning.mts_operator.physics_context import (  # noqa: E402
    PHYSICAL_METRIC_VERSION,
    PhysicsContext,
)
from stylized_motion.learning.nef_data import read_clip_window  # noqa: E402
from stylized_motion.learning.nef_eval import contacts_from_toe_motion  # noqa: E402
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402
from stylized_motion.learning.runner import choose_device  # noqa: E402

CSV_FIELDS = (
    "checkpoint", "family", "physical_metric_version", "skeleton_source", "windows", "frames",
    "fk_joint_error_mean_cm", "fk_joint_error_median_cm", "fk_joint_error_max_cm",
    "fk_worst_joint", "fk_worst_joint_error_cm",
    "root_drift_cm", "root_rotation_deg",
    "foot_slide_mps", "contact_match_rate", "contact_precision", "contact_recall",
    "foot_height_cm",
    "target_contact_rate", "inferred_contact_rate", "target_contact_feature_mean",
)


def evaluate(args: argparse.Namespace, checkpoint_path: Path, store: Any, windows: list[Any]) -> dict[str, Any]:
    device = choose_device(args.device)
    checkpoint, model = load_representation_checkpoint(checkpoint_path, device)
    model = model.to(device).eval()
    module = model.module
    stats = checkpoint["feature_stats"]
    # N01: the FK context comes from one place (bind asset, units, joint order and
    # the mirror rule), not from the store's own ``ref_pos`` -- that value is the
    # dataset mean of the local positions, and mirroring cancels every constant
    # axis-aligned offset, so it collapses the spine.  Historical numbers measured
    # against it are not comparable with these.
    context = PhysicsContext.from_feature_stats(stats, dt=float(args.root_dt))
    offset = torch.as_tensor(np.asarray(context.stats.offset, dtype=np.float32), device=device)
    scale = torch.as_tensor(np.asarray(context.stats.scale, dtype=np.float32), device=device)
    parents = context.parents
    names = list(context.names)
    toe_indices = context.toe_indices
    dt = float(context.dt)
    frames = int(args.frames)

    joint_errors, root_errors, slides, height_errors = [], [], [], []
    contact_raw_means = []
    inferred_contacts = 0
    contact_true_positive = contact_false_positive = contact_false_negative = contact_frames = 0
    total_frames = 0
    with torch.no_grad():
        for window in windows:
            raw, _ = read_clip_window(
                store, int(window.variant_idx), int(window.target_start), frames, history=0
            )
            motion = torch.from_numpy(
                ((raw - np.asarray(stats["offset"], np.float32)) / np.asarray(stats["scale"], np.float32)).astype(np.float32)
            )[None].to(device)
            recon = model(motion, collect_metrics=False)["recon_state"]
            pair = torch.cat((recon, motion), dim=0)
            # A mirrored clip needs the mirrored skeleton; the store's own table
            # says which clips those are.
            mirrored = bool(np.asarray(store.clip_mirror)[int(window.variant_idx)])
            ref_pos = torch.as_tensor(
                np.asarray(context.stats_for(mirror=mirrored).ref_pos, dtype=np.float32), device=device
            )
            positions = reconstruct_joint_positions(pair, offset, scale, ref_pos, parents, dt, world_space=True)
            pred_positions, target_positions = positions[0], positions[1]
            joint_errors.append((pred_positions - target_positions).norm(dim=-1).cpu().numpy())

            pred_root, pred_rot = integrate_root_trajectory(recon, offset, scale, dt)
            target_root, target_rot = integrate_root_trajectory(motion, offset, scale, dt)
            root_errors.append(
                (
                    float((pred_root[0, 1:] - target_root[0, 1:]).norm(dim=-1).mean() * 100),
                    float(quat.torch_quat_angle(pred_rot[:, 1:], target_rot[:, 1:]).mean().cpu()) * 180.0 / np.pi,
                )
            )
            # Contact labels come from the target's own features, which are
            # *normalized* in the store: threshold the denormalized values, the
            # same way the training loss does (it clamps raw contacts to [0, 1]).
            contact_raw = motion[0, :, -2:] * scale[-2:] + offset[-2:]
            contact_features = contact_raw.clamp(0.0, 1.0) > 0.5
            inferred = contacts_from_toe_motion(pred_positions[None], toe_indices, dt, threshold=0.15)[0]
            matches = inferred == contact_features
            contact_raw_means.append(float(contact_raw.clamp(0.0, 1.0).mean().cpu()))
            inferred_contacts += int(inferred.sum())
            contact_true_positive += int((inferred & contact_features).sum())
            contact_false_positive += int((inferred & ~contact_features).sum())
            contact_false_negative += int((~inferred & contact_features).sum())
            contact_frames += int(contact_features.sum())
            total_frames += frames

            gate = (contact_features[1:] & contact_features[:-1]).any(dim=-1)
            if bool(gate.any()):
                pred_toe = pred_positions[1:, list(toe_indices)]
                target_toe = target_positions[1:, list(toe_indices)]
                speed = (pred_toe[1:] - pred_toe[:-1])[..., (0, 2)].abs().mean(dim=-1) / dt
                gate_pairs = gate[1:]
                if bool(gate_pairs.any()):
                    slides.append(float(speed[gate_pairs].mean().cpu()))
                height_errors.append(
                    float((pred_toe[..., 1] - target_toe[..., 1]).abs()[gate].mean().cpu() * 100)
                )
    error = np.concatenate(joint_errors)  # [N*T, J]
    per_joint = error.mean(axis=0)
    worst = int(np.argmax(per_joint))
    root_drift = float(np.mean([value[0] for value in root_errors]))
    root_rotation = float(np.mean([value[1] for value in root_errors]))
    precision = contact_true_positive / max(contact_true_positive + contact_false_positive, 1)
    recall = contact_true_positive / max(contact_true_positive + contact_false_negative, 1)
    return {
        "checkpoint": str(checkpoint_path),
        "family": model.family,
        "physical_metric_version": PHYSICAL_METRIC_VERSION,
        "skeleton_source": context.skeleton_source,
        "bind_asset_sha256": context.bind_asset_sha256,
        "windows": len(windows),
        "frames": total_frames,
        "fk_joint_error_mean_cm": float(per_joint.mean() * 100),
        "fk_joint_error_median_cm": float(np.median(per_joint) * 100),
        "fk_joint_error_max_cm": float(per_joint.max() * 100),
        "fk_worst_joint": names[worst],
        "fk_worst_joint_error_cm": float(per_joint[worst] * 100),
        "root_drift_cm": root_drift,
        "root_rotation_deg": root_rotation,
        "foot_slide_mps": float(np.mean(slides)) if slides else 0.0,
        # Accuracy over every toe/frame cell: true positives plus true negatives.
        "contact_match_rate": float(
            (
                contact_true_positive
                + (
                    total_frames * 2
                    - contact_true_positive
                    - contact_false_positive
                    - contact_false_negative
                )
            )
            / max(total_frames * 2, 1)
        ),
        "contact_precision": float(precision),
        "contact_recall": float(recall),
        "foot_height_cm": float(np.mean(height_errors)) if height_errors else 0.0,
        "target_contact_rate": float(contact_frames / max(total_frames * 2, 1)),
        "inferred_contact_rate": float(inferred_contacts / max(total_frames * 2, 1)),
        "target_contact_feature_mean": float(np.mean(contact_raw_means)) if contact_raw_means else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decoder-side physical metrics for NEF-FSQ checkpoints.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--root-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = open_any_feature_store(args.feature_database)
    try:
        windows = list(
            FixedWindowSampler(store, args.split, target_frames=args.frames, stride=args.frames, include_tail=True)
        )
        if args.windows > 0:
            windows = windows[: args.windows]
        if not windows:
            raise ValueError(f"No {args.split} windows available")
        rows = []
        for path in args.checkpoint:
            print(f"evaluating {path}", flush=True)
            rows.append(evaluate(args, Path(path), store, windows))
    finally:
        store.close()
    payload = {
        "kind": "nef_physics",
        "feature_database": str(args.feature_database),
        "split": args.split,
        "windows": len(windows),
        "frames_per_window": args.frames,
        "results": rows,
        "note": (
            "Metrics are computed on decoded motion, so they are independent of the "
            "training weights: the v1 baseline never computes joint/contact/foot losses "
            "and cannot report them from its own val_full block.  N01: the FK context now "
            "comes from the SOMA bind asset (see physical_metric_version and "
            "skeleton_source in every row); results measured against the store's own "
            "ref_pos are not comparable with these."
        ),
    }
    output = args.output or Path("outputs/nef_physics/run")
    output.mkdir(parents=True, exist_ok=True)
    (output / "physics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    with (output / "physics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
