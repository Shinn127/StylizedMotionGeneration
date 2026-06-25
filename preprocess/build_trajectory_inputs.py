import argparse
from pathlib import Path

import numpy as np

import quat


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def intersect_tagged_ranges(tag_range_starts, tag_range_stops, tag_tags, tags):
    if not tags:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    if len(tags) == 1:
        mask = tag_tags == tags[0]
        return tag_range_starts[mask].copy(), tag_range_stops[mask].copy()

    result_starts, result_stops = [], []
    tag_ranges = {}
    for tag in tags:
        mask = tag_tags == tag
        tag_ranges[tag] = list(zip(tag_range_starts[mask], tag_range_stops[mask]))

    base_tag = min(tags, key=lambda t: len(tag_ranges[t]))
    other_tags = [t for t in tags if t != base_tag]

    for base_start, base_stop in tag_ranges[base_tag]:
        candidates = [(int(base_start), int(base_stop))]
        for other_tag in other_tags:
            new_candidates = []
            for cand_start, cand_stop in candidates:
                for other_start, other_stop in tag_ranges[other_tag]:
                    overlap_start = max(cand_start, int(other_start))
                    overlap_stop = min(cand_stop, int(other_stop))
                    if overlap_start < overlap_stop:
                        new_candidates.append((overlap_start, overlap_stop))
            candidates = new_candidates
            if not candidates:
                break

        for start, stop in candidates:
            result_starts.append(start)
            result_stops.append(stop)

    return np.array(result_starts, dtype=np.int32), np.array(result_stops, dtype=np.int32)


def parse_tags(tags_arg, available_tags):
    if tags_arg:
        return [tag.strip() for tag in tags_arg.split(",") if tag.strip()]
    if "locomotion" in available_tags:
        return ["locomotion"]
    return ["all"]


def resolve_database_path(dataset, database_path):
    if database_path is not None:
        return database_path
    return PROCESSED_DIR / dataset / "database.npz"


def build_trajectory_inputs(database_path, output_path, tags=None, future_frames=None):
    data = np.load(database_path, allow_pickle=True)

    tag_range_starts = data["tag_range_starts"]
    tag_range_stops = data["tag_range_stops"]
    tag_range_names = data["tag_range_names"]
    tag_tags = data["tag_tags"]
    tag_mirror = data["tag_mirror"]

    xroot_pos = data["positions"].astype(np.float32)[:, 0]
    xroot_rot = data["rotations"].astype(np.float32)[:, 0]
    xroot_dir = quat.mul_vec(xroot_rot, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    available_tags = set(tag_tags.tolist())
    selected_tags = parse_tags(tags, available_tags)

    if not all(tag in available_tags for tag in selected_tags):
        missing = [tag for tag in selected_tags if tag not in available_tags]
        raise ValueError(f"Missing tags in database: {missing}. Available tags include: {sorted(available_tags)[:20]}")

    future_frames = np.asarray(future_frames or [20, 40, 60], dtype=np.int32)
    max_future = int(np.max(future_frames))

    selected_starts, selected_stops = intersect_tagged_ranges(
        tag_range_starts,
        tag_range_stops,
        tag_tags,
        selected_tags,
    )

    indices = []
    future_positions = []
    future_directions = []
    sample_range_names = []
    sample_mirror = []

    for rs, re in zip(selected_starts, selected_stops):
        pose_indices = np.arange(rs + 1, re - max_future, dtype=np.int32)
        if len(pose_indices) == 0:
            continue

        cpos = quat.inv_mul_vec(
            xroot_rot[pose_indices][:, None],
            xroot_pos[pose_indices[:, None] + future_frames] - xroot_pos[pose_indices][:, None],
        ).astype(np.float32)
        cdir = quat.inv_mul_vec(
            xroot_rot[pose_indices][:, None],
            xroot_dir[pose_indices[:, None] + future_frames],
        ).astype(np.float32)

        indices.append(pose_indices)
        future_positions.append(cpos)
        future_directions.append(cdir)

        mask = (tag_range_starts == rs) & (tag_range_stops == re) & np.isin(tag_tags, selected_tags)
        if np.any(mask):
            range_name = tag_range_names[mask][0]
            mirror_flag = bool(tag_mirror[mask][0])
        else:
            range_name = "unknown"
            mirror_flag = False
        sample_range_names.append(np.full(len(pose_indices), range_name, dtype=object))
        sample_mirror.append(np.full(len(pose_indices), mirror_flag, dtype=bool))

    if not indices:
        raise ValueError(
            f"No valid trajectory samples found for tags {selected_tags} and future frames {future_frames.tolist()} "
            f"in {database_path}"
        )

    indices = np.concatenate(indices, axis=0)
    future_positions = np.concatenate(future_positions, axis=0)
    future_directions = np.concatenate(future_directions, axis=0)
    sample_range_names = np.concatenate(sample_range_names, axis=0)
    sample_mirror = np.concatenate(sample_mirror, axis=0)
    trajectory = np.concatenate([future_positions, future_directions], axis=-1).reshape(len(indices), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        indices=indices.astype(np.int32),
        T=trajectory.astype(np.float32),
        Tpos=future_positions.astype(np.float32),
        Tdir=future_directions.astype(np.float32),
        future_frames=future_frames.astype(np.int32),
        selected_tags=np.array(selected_tags, dtype=object),
        sample_range_names=sample_range_names,
        sample_mirror=sample_mirror.astype(bool),
        database_path=np.array(str(database_path), dtype=object),
    )


def main():
    parser = argparse.ArgumentParser(description="Build ControlOperators-style trajectory inputs from database.npz.")
    parser.add_argument("--dataset", choices=["lafan", "100style"], required=True)
    parser.add_argument("--database", type=Path, default=None, help="Optional path to database.npz.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for trajectory npz.")
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tag intersection to use. Defaults to `locomotion` when available, otherwise `all`.",
    )
    parser.add_argument(
        "--future-frames",
        type=str,
        default="20,40,60",
        help="Comma-separated future frame offsets, e.g. 20,40,60",
    )
    args = parser.parse_args()

    future_frames = [int(v.strip()) for v in args.future_frames.split(",") if v.strip()]
    database_path = resolve_database_path(args.dataset, args.database)
    output_path = args.output or (database_path.parent / "trajectory.npz")

    build_trajectory_inputs(
        database_path=database_path,
        output_path=output_path,
        tags=args.tags,
        future_frames=future_frames,
    )
    print(f"Saved trajectory inputs to {output_path}")


if __name__ == "__main__":
    main()
