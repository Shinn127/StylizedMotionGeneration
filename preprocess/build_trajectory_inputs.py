import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

import quat


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

_WORKER_XROOT_POS = None
_WORKER_XROOT_ROT = None
_WORKER_XROOT_DIR = None
_WORKER_TAG_RANGE_STARTS = None
_WORKER_TAG_RANGE_STOPS = None
_WORKER_TAG_RANGE_NAMES = None
_WORKER_TAG_TAGS = None
_WORKER_TAG_MIRROR = None
_WORKER_SELECTED_TAGS = None
_WORKER_FUTURE_FRAMES = None
_WORKER_MAX_FUTURE = None


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



def _init_worker(
    xroot_pos,
    xroot_rot,
    xroot_dir,
    tag_range_starts,
    tag_range_stops,
    tag_range_names,
    tag_tags,
    tag_mirror,
    selected_tags,
    future_frames,
    max_future,
):
    global _WORKER_XROOT_POS
    global _WORKER_XROOT_ROT
    global _WORKER_XROOT_DIR
    global _WORKER_TAG_RANGE_STARTS
    global _WORKER_TAG_RANGE_STOPS
    global _WORKER_TAG_RANGE_NAMES
    global _WORKER_TAG_TAGS
    global _WORKER_TAG_MIRROR
    global _WORKER_SELECTED_TAGS
    global _WORKER_FUTURE_FRAMES
    global _WORKER_MAX_FUTURE

    _WORKER_XROOT_POS = xroot_pos
    _WORKER_XROOT_ROT = xroot_rot
    _WORKER_XROOT_DIR = xroot_dir
    _WORKER_TAG_RANGE_STARTS = tag_range_starts
    _WORKER_TAG_RANGE_STOPS = tag_range_stops
    _WORKER_TAG_RANGE_NAMES = tag_range_names
    _WORKER_TAG_TAGS = tag_tags
    _WORKER_TAG_MIRROR = tag_mirror
    _WORKER_SELECTED_TAGS = selected_tags
    _WORKER_FUTURE_FRAMES = future_frames
    _WORKER_MAX_FUTURE = max_future


def _build_range_samples(range_bounds):
    rs, re = range_bounds
    pose_indices = np.arange(rs + 1, re - _WORKER_MAX_FUTURE, dtype=np.int32)
    if len(pose_indices) == 0:
        return None

    cpos = quat.inv_mul_vec(
        _WORKER_XROOT_ROT[pose_indices][:, None],
        _WORKER_XROOT_POS[pose_indices[:, None] + _WORKER_FUTURE_FRAMES] - _WORKER_XROOT_POS[pose_indices][:, None],
    ).astype(np.float32)
    cdir = quat.inv_mul_vec(
        _WORKER_XROOT_ROT[pose_indices][:, None],
        _WORKER_XROOT_DIR[pose_indices[:, None] + _WORKER_FUTURE_FRAMES],
    ).astype(np.float32)

    mask = (
        (_WORKER_TAG_RANGE_STARTS == rs)
        & (_WORKER_TAG_RANGE_STOPS == re)
        & np.isin(_WORKER_TAG_TAGS, _WORKER_SELECTED_TAGS)
    )
    if np.any(mask):
        range_name = _WORKER_TAG_RANGE_NAMES[mask][0]
        mirror_flag = bool(_WORKER_TAG_MIRROR[mask][0])
    else:
        range_name = "unknown"
        mirror_flag = False

    sample_range_names = np.full(len(pose_indices), range_name, dtype=object)
    sample_mirror = np.full(len(pose_indices), mirror_flag, dtype=bool)
    return pose_indices, cpos, cdir, sample_range_names, sample_mirror


def _build_all_range_samples(tasks, workers):
    workers = max(1, int(workers))
    if workers == 1:
        results = (_build_range_samples(task) for task in tqdm(tasks, desc="Building trajectory"))
        return [result for result in results if result is not None]

    context = mp.get_context("fork")
    chunksize = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = executor.map(_build_range_samples, tasks, chunksize=chunksize)
        return [result for result in tqdm(results, total=len(tasks), desc="Building trajectory") if result is not None]


def build_trajectory_inputs(database_path, output_path, tags=None, future_frames=None, workers=1):
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

    tasks = list(zip(selected_starts.tolist(), selected_stops.tolist()))
    _init_worker(
        xroot_pos,
        xroot_rot,
        xroot_dir,
        tag_range_starts,
        tag_range_stops,
        tag_range_names,
        tag_tags,
        tag_mirror,
        np.array(selected_tags, dtype=object),
        future_frames,
        max_future,
    )
    range_results = _build_all_range_samples(tasks, workers=workers)

    if not range_results:
        raise ValueError(
            f"No valid trajectory samples found for tags {selected_tags} and future frames {future_frames.tolist()} "
            f"in {database_path}"
        )

    indices, future_positions, future_directions, sample_range_names, sample_mirror = zip(*range_results)

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-range trajectory generation. Use 1 for serial processing.",
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
        workers=args.workers,
    )
    print(f"Saved trajectory inputs to {output_path}")


if __name__ == "__main__":
    main()
