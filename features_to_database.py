import argparse
from pathlib import Path

import numpy as np

from motion_features import (
    MotionFeatureStats,
    build_motion_features,
    load_database,
    normalize_motion_features,
    reconstruct_motion_state_from_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct a Genoview-compatible database.npz from 230D motion features.")
    parser.add_argument("--features", type=Path, default=None, help="Path to .npy or .npz containing motion features [T, D].")
    parser.add_argument("--database", type=Path, required=True, help="Reference pruned database.npz used for stats/metadata.")
    parser.add_argument("--output", type=Path, required=True, help="Output database.npz path.")
    parser.add_argument("--key", type=str, default="motion", help="Array key for .npz feature input.")
    parser.add_argument("--normalized", action="store_true", help="Treat input features as normalized.")
    parser.add_argument("--start-frame", type=int, default=0, help="Insert reconstructed frames starting at this database frame.")
    parser.add_argument("--num-frames", type=int, default=None, help="Optionally limit to the first N frames from the feature input.")
    parser.add_argument("--root-position0", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--root-rotation0", type=float, nargs=4, default=None, metavar=("W", "X", "Y", "Z"))
    parser.add_argument(
        "--use-database-motion",
        action="store_true",
        help="Ignore --features and reconstruct from the reference database's own 230D representation.",
    )
    return parser.parse_args()


def load_feature_array(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    data = np.load(path, allow_pickle=True)
    if key in data.files:
        return data[key].astype(np.float32)
    if len(data.files) == 1:
        return data[data.files[0]].astype(np.float32)
    raise KeyError(f"Could not find key {key!r} in {path}. Available keys: {list(data.files)}")


def make_stats_from_database(database: dict[str, np.ndarray]) -> MotionFeatureStats:
    _features, stats = build_motion_features(database)
    return stats


def export_database(args):
    database = load_database(args.database)
    stats = make_stats_from_database(database)

    if args.use_database_motion:
        features, _ = build_motion_features(database)
        if args.normalized:
            features = normalize_motion_features(features, stats)
    else:
        if args.features is None:
            raise ValueError("--features is required unless --use-database-motion is set.")
        features = load_feature_array(args.features, args.key)

    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(f"Expected [T, D] or [1, T, D], got {features.shape}")
        features = features[0]

    if args.num_frames is not None:
        features = features[: args.num_frames]

    root_position0 = (
        np.asarray(args.root_position0, dtype=np.float32)
        if args.root_position0 is not None
        else database["positions"].astype(np.float32)[int(args.start_frame), 0].copy()
    )
    root_rotation0 = (
        np.asarray(args.root_rotation0, dtype=np.float32)
        if args.root_rotation0 is not None
        else database["rotations"].astype(np.float32)[int(args.start_frame), 0].copy()
    )

    state = reconstruct_motion_state_from_features(
        x=features,
        stats=stats,
        parents=database["parents"].astype(np.int32),
        normalized=bool(args.normalized),
        root_position0=root_position0,
        root_rotation0=root_rotation0,
    )

    positions = database["positions"].astype(np.float32).copy()
    rotations = database["rotations"].astype(np.float32).copy()
    velocities = database["velocities"].astype(np.float32).copy()
    angular_velocities = database["angular_velocities"].astype(np.float32).copy()
    contacts = database["contacts"].astype(np.uint8).copy()

    start = int(args.start_frame)
    stop = start + len(state.local_positions)
    if start < 0 or stop > len(positions):
        raise ValueError(f"Requested frame range [{start}, {stop}) exceeds database length {len(positions)}")

    positions[start:stop] = state.local_positions
    rotations[start:stop] = state.local_rotations
    velocities[start:stop] = state.local_velocities
    angular_velocities[start:stop] = state.local_angular_velocities
    contacts[start:stop] = np.asarray(state.contacts > 0.5, dtype=np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        positions=positions,
        velocities=velocities,
        rotations=rotations,
        angular_velocities=angular_velocities,
        parents=database["parents"].astype(np.int32),
        names=database["names"],
        range_starts=database["range_starts"].astype(np.int32),
        range_stops=database["range_stops"].astype(np.int32),
        range_mirror=database["range_mirror"].astype(bool),
        range_names=database["range_names"],
        contacts=contacts,
        tag_range_starts=database["tag_range_starts"].astype(np.int32),
        tag_range_stops=database["tag_range_stops"].astype(np.int32),
        tag_range_names=database["tag_range_names"],
        tag_tags=database["tag_tags"],
        tag_mirror=database["tag_mirror"].astype(bool),
        joint_subset=np.array(str(database["joint_subset"].item()), dtype=object),
    )

    print(f"Exported reconstructed database to {args.output}")
    print(f"Inserted {len(state.local_positions)} frames into [{start}, {stop})")


def main():
    args = parse_args()
    export_database(args)


if __name__ == "__main__":
    main()
