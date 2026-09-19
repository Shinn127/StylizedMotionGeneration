#!/usr/bin/env python
"""Split a ``visualize_nef_fsq.py`` run's joint error into pose and placement.

``summary.json`` reports one joint error in metres and a root/body-local split.
Both are translations only: the root/local split subtracts the root joint's own
displacement, so a clip whose *whole body* is reconstructed at the wrong
orientation still shows up as "body-local" error, which reads as "the 40x9
alphabet could not represent the pose".  That reading is wrong (see
``postmortem_convulsions_side_loop``: the torso is off by 27 degrees while the
pose itself is fine).

This script re-measures the same saved window with a per-frame optimal rigid
alignment (Kabsch), which separates the two:

* ``pose_aligned_m``  - residual after translation+rotation alignment, i.e. the
  joint angles / body shape the codebook actually had to represent;
* ``placement_m``     - the rest, i.e. where and how the body sits in the world.

It reads the ``source_raw.npy`` / ``recon_raw.npy`` / ``summary.json`` that the
visualization already wrote, so it costs a CPU second per run and needs no render.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/nef_fsq_error_split.py \
      --checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --root outputs/nef_fsq_viz --output outputs/nef_fsq_viz/error_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.anim.features import (  # noqa: E402
    deserialize_motion_feature_stats,
    reconstruct_motion_state_from_features,
    stats_with_reference_skeleton,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402


def aligned_pose_error(global_positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-frame distance from ``reference`` to ``global_positions`` after Kabsch.

    Both are ``[T, J, 3]``.  Removing the optimal rotation as well as the
    translation leaves only the difference in body shape and joint angles.
    """
    target_centre = reference - reference.mean(axis=1, keepdims=True)
    moving_centre = global_positions - global_positions.mean(axis=1, keepdims=True)
    covariance = np.einsum("tji,tjk->tik", moving_centre, target_centre)
    u, _, vt = np.linalg.svd(covariance)
    flip = np.sign(np.linalg.det(np.einsum("tij,tkj->tik", vt, u)))
    diagonal = np.zeros_like(covariance)
    diagonal[:, 0, 0] = 1.0
    diagonal[:, 1, 1] = 1.0
    diagonal[:, 2, 2] = flip
    rotation = np.einsum(
        "tij,tjk->tik", np.einsum("tij,tjk->tik", vt.transpose(0, 2, 1), diagonal),
        u.transpose(0, 2, 1),
    )
    aligned = np.einsum("tij,tkj->tik", moving_centre, rotation) + reference.mean(axis=1, keepdims=True)
    return np.linalg.norm(reference - aligned, axis=-1)


def torso_tilt(global_positions: np.ndarray, hip: int, head: int) -> np.ndarray:
    """Angle of the pelvis->head axis away from world up, in degrees."""
    axis = global_positions[:, head] - global_positions[:, hip]
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(axis[:, 1], -1.0, 1.0)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pose vs placement split of a visualization run.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("outputs/nef_fsq_viz"))
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON with the same rows.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint, model = load_representation_checkpoint(args.checkpoint, torch.device("cpu"))
    stats, metadata = deserialize_motion_feature_stats(dict(checkpoint["feature_stats"]))
    names = [str(name) for name in metadata["names"]]
    parents = np.asarray(metadata["parents"])
    hip, head = names.index("Hips"), names.index("Head")

    root: Path = args.root
    runs = args.runs if args.runs is not None else sorted(p.name for p in root.iterdir() if p.is_dir())

    rows = []
    header = (
        f"{'run':<20} {'raw':>7} {'pose':>7} {'place':>7} {'place%':>7}"
        f" | {'pelvisY src':>11} {'rec':>6} | {'tilt src':>8} {'rec':>6}"
    )
    print(header)
    for run in runs:
        directory = root / run
        summary_path = directory / "summary.json"
        source_path = directory / "source_raw.npy"
        recon_path = directory / "recon_raw.npy"
        if not (summary_path.exists() and source_path.exists() and recon_path.exists()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # The window may come from the unmirrored sibling, which is what the summary records.
        run_stats = stats_with_reference_skeleton(
            stats, names, mirror=bool(summary.get("clip_mirror", False))
        )
        states = [
            reconstruct_motion_state_from_features(
                x=np.load(path), stats=run_stats, parents=parents, normalized=False
            )
            for path in (source_path, recon_path)
        ]
        source_global = np.asarray(states[0].global_positions)
        recon_global = np.asarray(states[1].global_positions)
        if source_global.shape[1] != len(names):  # [T, 3, J] -> [T, J, 3]
            source_global = source_global.transpose(0, 2, 1)
            recon_global = recon_global.transpose(0, 2, 1)

        raw = np.linalg.norm(source_global - recon_global, axis=-1)
        pose = aligned_pose_error(recon_global, source_global)
        source_tilt = torso_tilt(source_global, hip, head)
        recon_tilt = torso_tilt(recon_global, hip, head)
        row = {
            "run": run,
            "clip": summary.get("clip"),
            "clip_name": summary.get("clip_name"),
            "action": summary.get("action"),
            "style": summary.get("style"),
            "frames": int(raw.shape[0]),
            "raw_mean_m": float(raw.mean()),
            "raw_p95_m": float(np.percentile(raw, 95)),
            "raw_max_m": float(raw.max()),
            "pose_aligned_mean_m": float(pose.mean()),
            "placement_mean_m": float((raw - pose).mean()),
            "placement_share": float((raw - pose).mean() / raw.mean()) if raw.mean() else 0.0,
            "pelvis_height_src_m": float(source_global[:, hip, 1].mean()),
            "pelvis_height_recon_m": float(recon_global[:, hip, 1].mean()),
            "torso_tilt_src_deg": float(source_tilt.mean()),
            "torso_tilt_recon_deg": float(recon_tilt.mean()),
        }
        rows.append(row)
        print(
            f"{run:<20} {row['raw_mean_m']:7.4f} {row['pose_aligned_mean_m']:7.4f} "
            f"{row['placement_mean_m']:7.4f} {row['placement_share']:7.1%}"
            f" | {row['pelvis_height_src_m']:11.3f} {row['pelvis_height_recon_m']:6.3f}"
            f" | {row['torso_tilt_src_deg']:8.1f} {row['torso_tilt_recon_deg']:6.1f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
