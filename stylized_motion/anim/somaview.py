"""SomaView: the GenoView renderer driven by the BONES-SEED SOMA rig.

SOMA and Geno share enough conventions (a ``Hips``/``Spine2`` simulation-root
pair, centimeter BVHs, ZYX euler channels) that the viewer runs unchanged once
the assets are swapped. The assets themselves are produced by
``stylized_motion.anim.soma_assets`` from the ``soma_shapes`` directory of the
BONES-SEED dataset; see the README for the download and conversion commands.

SOMA motions are captured at 120 fps, so ``--fps`` defaults to 120 here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stylized_motion.anim.genoview import (
    RigSpec,
    GenoView,
    build_database_from_bvh,
    build_database_from_features,
    load_database_dict,
)
from stylized_motion.util.paths import SOMA_RESOURCE_DIR


SOMA_RIG = RigSpec(
    model_filename="SOMA.bin",
    bind_bvh_filename="SOMA_bind.bvh",
    sim_position_joint="Spine2",
    sim_rotation_joint="Hips",
    unit_scale=0.01,
    window_title=b"SomaView",
)

SOMA_FPS = 120


def main():
    parser = argparse.ArgumentParser(description="High-quality SOMA viewer driven by database.npz or a SOMA BVH clip.")
    parser.add_argument("--database", type=Path, default=None, help="Path to database.npz in the SOMA skeleton contract")
    parser.add_argument("--bvh", type=Path, default=None, help="Path to a soma_uniform BVH clip")
    parser.add_argument("--features", type=Path, default=None, help="Path to .npy or .npz feature arrays (requires a SOMA feature store; reserved).")
    parser.add_argument("--feature-key", type=str, default="motion", help="Array key for .npz feature input.")
    parser.add_argument("--stats-source", type=Path, default=None, help="Checkpoint .pt or schema-v3 feature store directory containing SOMA feature stats.")
    parser.add_argument("--normalized", action="store_true", help="Treat --features as normalized feature values.")
    parser.add_argument("--range-name", type=str, default="features", help="Range name used in feature visualization mode.")
    parser.add_argument("--root-position0", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--root-rotation0", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--trajectory", type=Path, default=None, help="Optional path to trajectory.npz")
    parser.add_argument(
        "--resources-root",
        type=Path,
        default=SOMA_RESOURCE_DIR,
        help="Directory containing SOMA.bin, SOMA_bind.bvh and shader files",
    )
    parser.add_argument("--fps", type=int, default=None, help="Playback FPS (defaults to the input frame time; SOMA is 120 Hz)")
    args = parser.parse_args()

    selected_inputs = [args.database is not None, args.bvh is not None, args.features is not None]
    if sum(selected_inputs) != 1:
        raise ValueError("Exactly one of --database, --bvh, or --features is required")
    if args.features is not None and args.stats_source is None:
        raise ValueError("--stats-source is required when using --features")

    database = (
        load_database_dict(args.database)
        if args.database is not None
        else build_database_from_bvh(args.bvh, range_name=args.range_name, rig=SOMA_RIG)
        if args.bvh is not None
        else build_database_from_features(
            features_path=args.features,
            stats_source=args.stats_source,
            feature_key=args.feature_key,
            normalized=args.normalized,
            range_name=args.range_name,
            root_position0=args.root_position0,
            root_rotation0=args.root_rotation0,
        )
    )

    viewer = GenoView(
        database=database,
        trajectory_path=args.trajectory,
        resources_root=args.resources_root,
        fps=args.fps,
        rig=SOMA_RIG,
    )
    viewer.run()


if __name__ == "__main__":
    main()
