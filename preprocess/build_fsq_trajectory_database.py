"""Align root-local trajectory controls with frozen FSQ token shards.

``build_trajectory_inputs.py`` operates on the concatenated raw-motion database,
whereas the generator trains from one FSQ token file per motion range.  This
script materializes a lightweight, mmap-friendly control database with exactly
the same shard layout as the token database.  It intentionally stores raw
trajectory values; normalization is fitted from *training windows only* by the
conditional-generator training script.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

# Scripts in preprocess/ are invoked directly from the repository root.  Add
# that root explicitly so this standalone CLI can use the shared token loader.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.fsq_token_dataset import build_fsq_token_store


def _resolve_database_path(value: str, trajectory_path: Path) -> Path:
    candidate = Path(value)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(trajectory_path.parent / candidate)
    for path in candidates:
        if path.exists():
            return path
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve source database for {trajectory_path}; tried: {attempted}")


def _source_range_metadata(data: np.lib.npyio.NpzFile, trajectory_path: Path) -> tuple[np.ndarray, ...]:
    required = (
        "database_range_names",
        "database_range_mirror",
        "database_range_starts",
        "database_range_stops",
    )
    if all(name in data.files for name in required):
        return tuple(np.asarray(data[name]) for name in required)

    if "database_path" not in data.files:
        raise ValueError(
            "Trajectory input lacks range-local metadata. Rebuild it with "
            "preprocess/build_trajectory_inputs.py from the current repository."
        )
    database_path = _resolve_database_path(str(np.asarray(data["database_path"]).item()), trajectory_path)
    source = np.load(database_path, allow_pickle=True)
    return (
        np.asarray(source["range_names"], dtype=object),
        np.asarray(source["range_mirror"], dtype=bool),
        np.asarray(source["range_starts"], dtype=np.int32),
        np.asarray(source["range_stops"], dtype=np.int32),
    )


def _sample_range_offsets(
    data: np.lib.npyio.NpzFile,
    starts: np.ndarray,
    stops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if "sample_range_indices" in data.files and "sample_local_indices" in data.files:
        return (
            np.asarray(data["sample_range_indices"], dtype=np.int32),
            np.asarray(data["sample_local_indices"], dtype=np.int32),
        )
    if "indices" not in data.files:
        raise ValueError("Trajectory input must contain either range-local offsets or global indices")
    indices = np.asarray(data["indices"], dtype=np.int32)
    range_indices = np.searchsorted(starts, indices, side="right") - 1
    if np.any(range_indices < 0) or np.any(indices >= stops[range_indices]):
        raise ValueError("Trajectory global indices do not fit their source database ranges")
    return range_indices.astype(np.int32), (indices - starts[range_indices]).astype(np.int32)


def _match_source_ranges_to_token_shards(
    source_names: np.ndarray,
    source_mirror: np.ndarray,
    source_starts: np.ndarray,
    source_stops: np.ndarray,
    token_store,
) -> np.ndarray:
    """Map each raw-database range index to a token shard index, or ``-1``.

    Name/mirror pairs are normally unique.  Matching on length too keeps the
    mapping deterministic if a source database happens to contain duplicate
    names (for example concatenated subsets).
    """
    source_to_shard = np.full(len(source_names), -1, dtype=np.int32)
    source_groups: dict[tuple[str, bool, int], list[int]] = defaultdict(list)
    target_groups: dict[tuple[str, bool, int], list[int]] = defaultdict(list)
    for index, (name, mirror, start, stop) in enumerate(
        zip(source_names.tolist(), source_mirror.tolist(), source_starts.tolist(), source_stops.tolist())
    ):
        source_groups[(str(name), bool(mirror), int(stop) - int(start))].append(index)
    for shard_idx, (name, mirror, length) in enumerate(
        zip(
            token_store.range_names.tolist(),
            token_store.range_mirror.tolist(),
            token_store.num_frames.tolist(),
        )
    ):
        target_groups[(str(name), bool(mirror), int(length))].append(shard_idx)

    missing = []
    for key, shard_indices in target_groups.items():
        source_indices = source_groups.get(key, [])
        if len(source_indices) != len(shard_indices):
            missing.append((key, len(source_indices), len(shard_indices)))
            continue
        for source_idx, shard_idx in zip(source_indices, shard_indices):
            source_to_shard[source_idx] = shard_idx
    if missing:
        preview = "; ".join(
            f"{key}: raw={raw_count}, token={token_count}" for key, raw_count, token_count in missing[:5]
        )
        raise ValueError(
            "Trajectory source ranges do not match the FSQ token database. "
            f"Examples: {preview}"
        )
    return source_to_shard


def build_fsq_trajectory_database(
    token_database: Path,
    trajectory_input: Path,
    output: Path,
    overwrite: bool = False,
) -> None:
    token_store = build_fsq_token_store(token_database)
    token_database = token_store.database.resolve()
    trajectory_input = trajectory_input.resolve()
    output = output.resolve()
    if output == token_database:
        raise ValueError("Trajectory output must be a separate directory from the token database")
    metadata_path = output / "metadata.npz"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite to rebuild it")

    data = np.load(trajectory_input, allow_pickle=True)
    if "T" not in data.files:
        raise ValueError(f"Trajectory input {trajectory_input} does not contain T")
    trajectory = np.asarray(data["T"], dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        raise ValueError(f"Trajectory T must have shape [N,D] with N>0, got {trajectory.shape}")
    source_names, source_mirror, source_starts, source_stops = _source_range_metadata(data, trajectory_input)
    source_names = np.asarray(source_names, dtype=object)
    source_mirror = np.asarray(source_mirror, dtype=bool)
    source_starts = np.asarray(source_starts, dtype=np.int32)
    source_stops = np.asarray(source_stops, dtype=np.int32)
    if not (len(source_names) == len(source_mirror) == len(source_starts) == len(source_stops)):
        raise ValueError("Source trajectory range metadata arrays have inconsistent lengths")
    sample_ranges, sample_local = _sample_range_offsets(data, source_starts, source_stops)
    if len(sample_ranges) != len(trajectory) or len(sample_local) != len(trajectory):
        raise ValueError("Trajectory values and range-local indices have inconsistent lengths")
    if np.any(sample_ranges < 0) or np.any(sample_ranges >= len(source_names)):
        raise ValueError("Trajectory samples refer to invalid source range indices")

    source_to_shard = _match_source_ranges_to_token_shards(
        source_names,
        source_mirror,
        source_starts,
        source_stops,
        token_store,
    )
    sample_shards = source_to_shard[sample_ranges]
    keep = sample_shards >= 0
    if not np.any(keep):
        raise ValueError("No trajectory samples mapped to the supplied token database")
    output.mkdir(parents=True, exist_ok=True)
    trajectory_files: list[str] = []
    valid_files: list[str] = []
    for shard_idx, num_frames in enumerate(token_store.num_frames.tolist()):
        trajectory_rel = Path("trajectory") / f"trajectory_{shard_idx:05d}.npy"
        valid_rel = Path("valid") / f"valid_{shard_idx:05d}.npy"
        trajectory_path = output / trajectory_rel
        valid_path = output / valid_rel
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        values = np.lib.format.open_memmap(
            trajectory_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(num_frames), trajectory.shape[1]),
        )
        valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=bool, shape=(int(num_frames),))
        values[:] = 0.0
        valid[:] = False
        selected = np.nonzero(sample_shards == shard_idx)[0]
        if len(selected):
            local = sample_local[selected]
            if np.any(local < 0) or np.any(local >= int(num_frames)):
                raise ValueError(f"Trajectory local indices exceed token shard {shard_idx}")
            values[local] = trajectory[selected]
            valid[local] = True
        values.flush()
        valid.flush()
        trajectory_files.append(trajectory_rel.as_posix())
        valid_files.append(valid_rel.as_posix())

    feature_order = (
        str(np.asarray(data["trajectory_feature_order"]).item())
        if "trajectory_feature_order" in data.files
        else "legacy-unspecified"
    )
    np.savez(
        metadata_path,
        schema_version=np.asarray(1, dtype=np.int32),
        token_database=np.asarray(str(token_database), dtype=object),
        tokenizer_checkpoint_sha256=np.asarray(token_store.checkpoint_sha256, dtype=object),
        trajectory_input=np.asarray(str(trajectory_input), dtype=object),
        trajectory_dim=np.asarray(trajectory.shape[1], dtype=np.int32),
        future_frames=np.asarray(data["future_frames"], dtype=np.int32),
        trajectory_feature_order=np.asarray(feature_order, dtype=object),
        trajectory_files=np.asarray(trajectory_files, dtype=object),
        valid_files=np.asarray(valid_files, dtype=object),
        num_frames=token_store.num_frames.astype(np.int32),
        range_names=token_store.range_names,
        range_mirror=token_store.range_mirror.astype(bool),
    )
    print(f"saved={output}")
    print(f"shards={len(trajectory_files)} trajectory_dim={trajectory.shape[1]} samples={int(keep.sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align trajectory.npz to an FSQ token database.")
    parser.add_argument("--token-database", type=Path, required=True)
    parser.add_argument("--trajectory-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_fsq_trajectory_database(
        token_database=args.token_database,
        trajectory_input=args.trajectory_input,
        output=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
