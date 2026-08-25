"""Schema-v3 feature store and tensor-only representation dataset."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from stylized_motion.anim.features import MotionFeatureStats
from .sampling import SampleRequest


DATA_SCHEMA_VERSION = 3
FRAME_RATE = 60
MOTION_DIM = 230
_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MMapShardCache:
    """A process-local LRU cache of read-only numpy memmaps."""

    def __init__(self, paths: Iterable[Path], max_open_shards: int = 32) -> None:
        if int(max_open_shards) <= 0:
            raise ValueError("max_open_shards must be positive")
        self.paths = [Path(path) for path in paths]
        self.max_open_shards = int(max_open_shards)
        self._arrays: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, shard_idx: int) -> np.ndarray:
        shard_idx = int(shard_idx)
        if shard_idx < 0 or shard_idx >= len(self.paths):
            raise IndexError(f"Invalid shard index {shard_idx}")
        array = self._arrays.pop(shard_idx, None)
        if array is None:
            array = np.load(self.paths[shard_idx], mmap_mode="r", allow_pickle=False)
        self._arrays[shard_idx] = array
        while len(self._arrays) > self.max_open_shards:
            self._arrays.popitem(last=False)
        return array

    def close(self) -> None:
        self._arrays.clear()

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_arrays"] = OrderedDict()
        return state


@dataclass
class FeatureCache:
    """Raw frame-level feature cache; it intentionally has no split metadata."""

    database: Path
    manifest: dict[str, object]
    motion_files: list[Path]
    shard_num_frames: np.ndarray
    range_names: tuple[str, ...]
    range_mirror: np.ndarray
    names: list[str]
    parents: np.ndarray
    joint_subset: str
    feature_schema_hash: str
    ref_pos: np.ndarray
    max_open_shards: int = 32
    _cache: MMapShardCache | None = None

    @property
    def motion_dim(self) -> int:
        return MOTION_DIM

    def read_motion(self, shard_idx: int, start: int = 0, frames: int | None = None) -> np.ndarray:
        if self._cache is None:
            self._cache = MMapShardCache(self.motion_files, self.max_open_shards)
        values = self._cache.get(int(shard_idx))
        start = int(start)
        stop = len(values) if frames is None else start + int(frames)
        if start < 0 or stop > len(values) or stop <= start:
            raise IndexError("FeatureCache frame range is invalid")
        return np.ascontiguousarray(values[start:stop], dtype=np.float32)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
        self._cache = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_cache"] = None
        return state


def open_feature_cache(database: str | Path, *, max_open_shards: int = 32) -> FeatureCache:
    database = Path(database)
    manifest = _read_manifest(database, "Feature cache")
    if manifest.get("store_type") != "feature_cache":
        raise ValueError("Expected a feature_cache manifest")
    shard_files = _manifest_files(database, manifest, label="Feature cache")
    if not shard_files:
        raise ValueError("Feature cache must contain at least one shard")
    index = _load_index(database)
    required = {"shard_num_frames", "range_mirror", "ref_pos"}
    _require_index(index, required, "Feature cache")
    schema = manifest.get("feature_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("Feature cache manifest must contain feature_schema")
    names = [str(value) for value in schema.get("names", [])]
    parents = np.asarray(schema.get("parents", []), dtype=np.int32)
    if not names or parents.shape != (len(names),):
        raise ValueError("Feature cache skeleton schema is invalid")
    frames = np.asarray(index["shard_num_frames"], dtype=np.int64)
    if len(frames) != len(shard_files) or np.any(frames <= 0):
        raise ValueError("Feature cache shard_num_frames is invalid")
    range_names = tuple(str(value) for value in manifest.get("range_names", []))
    range_mirror = np.asarray(index["range_mirror"], dtype=bool)
    source_records = manifest.get("source_clips", [])
    if len(range_names) != len(shard_files) or len(range_mirror) != len(shard_files):
        raise ValueError("Feature cache range metadata is invalid")
    if not isinstance(source_records, list) or len(source_records) * 2 != len(shard_files):
        raise ValueError("Feature cache source clip metadata is invalid")
    for source_idx, record in enumerate(source_records):
        if not isinstance(record, Mapping):
            raise ValueError("Feature cache source clip records must be objects")
        pair = slice(source_idx * 2, source_idx * 2 + 2)
        if range_names[pair.start] != range_names[pair.start + 1] or tuple(range_mirror[pair]) != (False, True):
            raise ValueError("Feature cache shards must be ordered as original/mirror pairs")
    _validate_shard_arrays(
        shard_files,
        frames,
        dtype=np.float32,
        tail_shape=(MOTION_DIM,),
        label="Feature cache",
    )
    return FeatureCache(
        database=database,
        manifest=manifest,
        motion_files=shard_files,
        shard_num_frames=frames,
        range_names=range_names,
        range_mirror=range_mirror,
        names=names,
        parents=parents,
        joint_subset=str(schema.get("joint_subset", "unknown")),
        feature_schema_hash=str(manifest.get("feature_schema_hash", "")),
        ref_pos=np.asarray(index["ref_pos"], dtype=np.float32),
        max_open_shards=int(max_open_shards),
    )


def _require_manifest(manifest: Mapping[str, object], store_type: str) -> None:
    required = {
        "data_schema_version",
        "store_type",
        "frame_rate",
        "num_shards",
        "shard_files",
        "shard_sha256",
        "split_manifest_hash",
        "feature_schema_hash",
        "created_by",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"{store_type} manifest is missing required fields: {missing}")
    if int(manifest["data_schema_version"]) != DATA_SCHEMA_VERSION:
        raise ValueError(f"Only data schema v{DATA_SCHEMA_VERSION} is supported")
    if str(manifest["store_type"]) != store_type:
        raise ValueError(f"Expected store_type={store_type!r}")
    if int(manifest["frame_rate"]) != FRAME_RATE:
        raise ValueError("Canonical data frame_rate must be 60")
    shard_files = manifest["shard_files"]
    shard_hashes = manifest["shard_sha256"]
    if not isinstance(shard_files, list) or not isinstance(shard_hashes, list):
        raise ValueError("shard_files and shard_sha256 must be JSON lists")
    if len(shard_files) != int(manifest["num_shards"]) or len(shard_hashes) != len(shard_files):
        raise ValueError("Manifest shard metadata has inconsistent lengths")
    for key in ("split_manifest_hash", "feature_schema_hash", "created_by"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"Manifest field {key!r} must be a non-empty string")
    if any(not isinstance(value, str) or not value for value in shard_hashes):
        raise ValueError("Manifest shard_sha256 entries must be non-empty strings")


def _require_relative_path(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"Store manifest {label} paths must be non-empty strings")
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Store manifest {label} paths must be relative to the store root")


def _manifest_files(
    database: Path,
    manifest: Mapping[str, object],
    key: str = "shard_files",
    label: str = "Store",
) -> list[Path]:
    values = manifest.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{label} manifest field {key!r} must be a JSON list")
    for value in values:
        _require_relative_path(value, key.removesuffix("_files"))
    paths = [database / value for value in values]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{label} manifest references missing files: {missing[:3]}")
    return paths


def _validate_shard_arrays(
    paths: Iterable[Path],
    frame_counts: Iterable[int],
    *,
    dtype: np.dtype[Any],
    tail_shape: tuple[int, ...] = (),
    label: str,
) -> None:
    paths = list(paths)
    frame_counts = [int(value) for value in frame_counts]
    if len(paths) != len(frame_counts):
        raise ValueError(f"{label} shard metadata has inconsistent lengths")
    expected_dtype = np.dtype(dtype)
    for path, frames in zip(paths, frame_counts):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (frames, *tail_shape)
        if values.dtype != expected_dtype or values.shape != expected_shape:
            raise ValueError(
                f"{label} shard {path} has dtype/shape {values.dtype}/{values.shape}, "
                f"expected {expected_dtype}/{expected_shape}"
            )


def _validate_shard_hashes(paths: Iterable[Path], expected: Iterable[object], label: str) -> None:
    paths = list(paths)
    expected = list(expected)
    if len(paths) != len(expected):
        raise ValueError(f"{label} checksum metadata has inconsistent lengths")
    for path, digest in zip(paths, expected):
        if sha256_file(path) != str(digest):
            raise ValueError(f"{label} checksum mismatch: {path}")


def _load_index(database: Path) -> dict[str, np.ndarray]:
    path = database / "index.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing data index: {path}")
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _read_manifest(database: Path, label: str) -> dict[str, object]:
    path = database / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} manifest must be a JSON object")
    return value


def _require_index(index: Mapping[str, np.ndarray], keys: set[str], label: str = "Store") -> None:
    missing = sorted(keys - set(index))
    if missing:
        raise ValueError(f"{label} index is missing required fields: {missing}")


def _check_dtype(index: Mapping[str, np.ndarray], key: str, dtype: np.dtype[Any]) -> None:
    if np.asarray(index[key]).dtype != np.dtype(dtype):
        raise ValueError(f"Index field {key!r} must have dtype {np.dtype(dtype)}, got {index[key].dtype}")


def _check_common_index(
    index: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
    shard_count: int,
    label: str,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_index(
        index,
        {
            "shard_num_frames",
            "clip_ids",
            "source_clip_ids",
            "range_shard_indices",
            "range_starts",
            "range_stops",
            "range_mirror",
            "split_ids",
        },
        label,
    )
    for key, dtype in (
        ("shard_num_frames", np.int64),
        ("clip_ids", np.int32),
        ("source_clip_ids", np.int32),
        ("range_shard_indices", np.int32),
        ("range_starts", np.int64),
        ("range_stops", np.int64),
        ("range_mirror", np.bool_),
        ("split_ids", np.uint8),
    ):
        _check_dtype(index, key, dtype)
    shard_num_frames = np.asarray(index["shard_num_frames"], dtype=np.int64)
    if len(shard_num_frames) != shard_count or np.any(shard_num_frames <= 0):
        raise ValueError(f"{label} shard_num_frames is invalid")
    arrays = tuple(np.asarray(index[key]) for key in (
        "clip_ids", "source_clip_ids", "range_shard_indices", "range_starts",
        "range_stops", "range_mirror", "split_ids",
    ))
    range_count = len(arrays[0])
    if any(len(array) != range_count for array in arrays[1:]):
        raise ValueError(f"{label} range index arrays have inconsistent lengths")
    range_shards, range_starts, range_stops = arrays[2], arrays[3], arrays[4]
    if np.any(range_shards < 0) or np.any(range_shards >= shard_count):
        raise ValueError(f"{label} range_shard_indices contains an invalid shard")
    if np.any(range_starts < 0) or np.any(range_stops <= range_starts):
        raise ValueError(f"{label} ranges must be non-empty half-open intervals")
    for shard_idx, start, stop in zip(range_shards.tolist(), range_starts.tolist(), range_stops.tolist()):
        if int(stop) > int(shard_num_frames[int(shard_idx)]):
            raise ValueError(f"{label} range exceeds its shard frame count")
    split_ids = arrays[6]
    if np.any(split_ids > 2):
        raise ValueError(f"{label} split_ids must use train=0, val=1, test=2")
    for key in ("clip_ids", "source_clip_ids"):
        if np.any(np.asarray(index[key]) < 0):
            raise ValueError(f"{label} {key} must be non-negative")
    if int(manifest["num_shards"]) != shard_count:
        raise ValueError(f"{label} manifest num_shards does not match index")
    return (range_count, *arrays)


def _stats_sha256(stats: MotionFeatureStats) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("offset", stats.offset),
        ("scale", stats.scale),
        ("weights", stats.weights),
        ("ref_pos", stats.ref_pos),
    ):
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()


@dataclass
class FeatureStore:
    database: Path
    manifest: dict[str, object]
    motion_files: list[Path]
    shard_num_frames: np.ndarray
    clip_ids: np.ndarray
    source_clip_ids: np.ndarray
    range_shard_indices: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    range_mirror: np.ndarray
    split_ids: np.ndarray
    range_names: tuple[str, ...]
    source_clip_names: tuple[str, ...]
    style_names: tuple[str, ...]
    action_names: tuple[str, ...]
    style_ids: np.ndarray
    action_ids: np.ndarray
    frame_rate: int
    motion_dim: int
    stats: MotionFeatureStats
    names: list[str]
    parents: np.ndarray
    joint_subset: str
    names_sha256: str
    stats_sha256: str
    feature_schema_hash: str
    max_open_shards: int = 32
    _cache: MMapShardCache | None = None

    @property
    def split_manifest_hash(self) -> str:
        return str(self.manifest["split_manifest_hash"])

    @property
    def num_joints(self) -> int:
        return len(self.names)

    def feature_schema(self) -> dict[str, object]:
        return {
            "name": "motion_feature_v2",
            "motion_dim": self.motion_dim,
            "joint_subset": self.joint_subset,
            "names_sha256": self.names_sha256,
            "stats_sha256": self.stats_sha256,
            "feature_schema_hash": self.feature_schema_hash,
        }

    def _read_motion_with_cache(self, request: SampleRequest, cache: MMapShardCache) -> np.ndarray:
        if int(request.target_frames) <= 0:
            raise ValueError("target_frames must be positive")
        variant_idx = int(request.variant_idx)
        if variant_idx < 0 or variant_idx >= len(self.range_names):
            raise IndexError(f"Invalid feature range index {variant_idx}")
        shard_idx = int(request.shard_idx)
        if int(self.range_shard_indices[variant_idx]) != shard_idx:
            raise ValueError("SampleRequest shard_idx does not match variant_idx")
        range_start = int(self.range_starts[variant_idx])
        range_stop = int(self.range_stops[variant_idx])
        target_start = int(request.target_start)
        target_stop = target_start + int(request.target_frames)
        if target_start < range_start or target_stop > range_stop:
            raise IndexError("SampleRequest target exceeds its source range")
        array = cache.get(shard_idx)
        return np.ascontiguousarray(array[target_start:target_stop], dtype=np.float32)

    def read_motion(self, request: SampleRequest) -> np.ndarray:
        if not isinstance(request, SampleRequest):
            raise TypeError("FeatureStore.read_motion requires a validated SampleRequest")
        if self._cache is None:
            self._cache = MMapShardCache(self.motion_files, self.max_open_shards)
        return self._read_motion_with_cache(request, self._cache)

    def split_intervals(self, split: str) -> np.ndarray:
        split_id = _SPLIT_IDS.get(split)
        if split_id is None:
            raise ValueError(f"Unsupported split {split!r}")
        rows = np.flatnonzero(self.split_ids == split_id)
        return np.column_stack((
            self.range_shard_indices[rows], self.range_starts[rows], self.range_stops[rows], rows,
        )).astype(np.int64, copy=False)

    def model_feature_weights(self) -> np.ndarray:
        return self.stats.weights.astype(np.float32, copy=True)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
        self._cache = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_cache"] = None
        return state


def open_feature_store(database: str | Path, *, max_open_shards: int = 32) -> FeatureStore:
    database = Path(database)
    manifest = _read_manifest(database, "Feature")
    _require_manifest(manifest, "feature")
    index = _load_index(database)
    shard_files = _manifest_files(database, manifest, label="Feature")
    if len(shard_files) != int(manifest["num_shards"]):
        raise ValueError("Feature manifest shard count is invalid")
    common = _check_common_index(index, manifest, len(shard_files), "Feature")
    _, clip_ids, source_clip_ids, range_shards, range_starts, range_stops, range_mirror, split_ids = common
    schema = manifest.get("feature_schema")
    if not isinstance(schema, dict):
        raise ValueError("Feature manifest must contain feature_schema")
    names = [str(value) for value in schema.get("names", [])]
    parents = np.asarray(schema.get("parents", []), dtype=np.int32)
    if not names or parents.shape != (len(names),):
        raise ValueError("Feature schema names/parents are invalid")
    motion_dim = int(manifest.get("motion_dim", schema.get("motion_dim", 0)))
    if motion_dim != MOTION_DIM:
        raise ValueError(f"FeatureStore motion_dim must be {MOTION_DIM}, got {motion_dim}")
    _validate_shard_arrays(
        shard_files,
        index["shard_num_frames"],
        dtype=np.float32,
        tail_shape=(motion_dim,),
        label="Feature",
    )
    _require_index(index, {"offset", "scale", "weights", "ref_pos"}, "Feature")
    stats = MotionFeatureStats(
        offset=np.asarray(index["offset"], dtype=np.float32),
        scale=np.asarray(index["scale"], dtype=np.float32),
        dist=np.asarray(index.get("dist", np.ones(motion_dim, dtype=np.float32)), dtype=np.float32),
        weights=np.asarray(index["weights"], dtype=np.float32),
        ref_pos=np.asarray(index["ref_pos"], dtype=np.float32),
    )
    if (
        stats.offset.shape != (motion_dim,)
        or stats.scale.shape != (motion_dim,)
        or stats.dist.shape != (motion_dim,)
        or stats.weights.shape != (motion_dim,)
        or stats.ref_pos.ndim != 2
        or stats.ref_pos.shape != (len(names), 3)
    ):
        raise ValueError("Feature normalization arrays do not match motion_dim")
    names_sha256 = sha256_bytes(canonical_json_bytes(names))
    stats_sha256 = _stats_sha256(stats)
    schema_payload = {
        "name": "motion_feature_v2",
        "motion_dim": motion_dim,
        "joint_subset": str(schema.get("joint_subset", "unknown")),
        "names_sha256": names_sha256,
        "stats_sha256": stats_sha256,
    }
    feature_schema_hash = sha256_bytes(canonical_json_bytes(schema_payload))
    if str(schema.get("names_sha256", names_sha256)) != names_sha256:
        raise ValueError("Feature schema names_sha256 does not match names")
    if str(schema.get("stats_sha256", stats_sha256)) != stats_sha256:
        raise ValueError("Feature schema stats_sha256 does not match statistics")
    if str(manifest["feature_schema_hash"]) != feature_schema_hash:
        raise ValueError("Feature manifest feature_schema_hash does not match feature schema")
    range_names = tuple(str(value) for value in manifest.get("range_names", []))
    source_clip_names = tuple(str(value) for value in manifest.get("source_clip_names", []))
    style_names = tuple(str(value) for value in manifest.get("style_names", []))
    action_names = tuple(str(value) for value in manifest.get("action_names", []))
    range_count = len(range_starts)
    if len(range_names) != range_count or len(source_clip_names) == 0:
        raise ValueError("Feature manifest range/source clip names are invalid")
    style_ids = np.asarray(index.get("style_ids", np.zeros(range_count, dtype=np.int32)), dtype=np.int32)
    action_ids = np.asarray(index.get("action_ids", np.zeros(range_count, dtype=np.int32)), dtype=np.int32)
    if len(style_ids) != range_count or len(action_ids) != range_count:
        raise ValueError("Feature style/action index arrays have invalid lengths")
    return FeatureStore(
        database=database,
        manifest=manifest,
        motion_files=shard_files,
        shard_num_frames=np.asarray(index["shard_num_frames"], dtype=np.int64),
        clip_ids=np.asarray(clip_ids, dtype=np.int32),
        source_clip_ids=np.asarray(source_clip_ids, dtype=np.int32),
        range_shard_indices=np.asarray(range_shards, dtype=np.int32),
        range_starts=np.asarray(range_starts, dtype=np.int64),
        range_stops=np.asarray(range_stops, dtype=np.int64),
        range_mirror=np.asarray(range_mirror, dtype=bool),
        split_ids=np.asarray(split_ids, dtype=np.uint8),
        range_names=range_names,
        source_clip_names=source_clip_names,
        style_names=style_names,
        action_names=action_names,
        style_ids=style_ids,
        action_ids=action_ids,
        frame_rate=FRAME_RATE,
        motion_dim=motion_dim,
        stats=stats,
        names=names,
        parents=parents,
        joint_subset=str(schema.get("joint_subset", "unknown")),
        names_sha256=names_sha256,
        stats_sha256=stats_sha256,
        feature_schema_hash=feature_schema_hash,
        max_open_shards=int(max_open_shards),
    )


class FeatureDataset(Dataset):
    """Consume sampler requests and return [B,64,230] CPU tensors."""

    def __init__(
        self,
        split: str,
        store: FeatureStore,
        *,
        requests: Iterable[SampleRequest] | None = None,
        max_open_shards: int = 32,
        return_metadata: bool = False,
    ) -> None:
        if split not in _SPLIT_IDS:
            raise ValueError(f"Unsupported split {split!r}")
        self.split = split
        self.store = store
        self.requests = None if requests is None else list(requests)
        self.return_metadata = bool(return_metadata)
        store.max_open_shards = int(max_open_shards)

    def __len__(self) -> int:
        return 0 if self.requests is None else len(self.requests)

    def _request(self, value: int | SampleRequest) -> SampleRequest:
        request = value
        if not isinstance(request, SampleRequest):
            if self.requests is None:
                raise IndexError("FeatureDataset without requests requires a sampler request")
            request = self.requests[int(value)]
        variant_idx = int(request.variant_idx)
        if variant_idx < 0 or variant_idx >= len(self.store.split_ids):
            raise IndexError(f"Invalid feature range index {variant_idx}")
        if int(self.store.split_ids[variant_idx]) != _SPLIT_IDS[self.split]:
            raise ValueError(f"SampleRequest range {variant_idx} does not belong to split {self.split!r}")
        if int(request.target_frames) != 64:
            raise ValueError("Representation FeatureDataset requires target_frames=64")
        return request

    def _read_numpy(self, request: SampleRequest) -> np.ndarray:
        return self.store.read_motion(request)

    def _batch(self, values: list[int | SampleRequest]) -> dict[str, Any]:
        requests = [self._request(value) for value in values]
        if not requests:
            raise ValueError("Cannot collate an empty feature batch")
        frames = int(requests[0].target_frames)
        if any(int(request.target_frames) != frames for request in requests):
            raise ValueError("A feature batch must use one window length")
        motion = np.empty((len(requests), frames, self.store.motion_dim), dtype=np.float32)
        loss_mask = np.ones((len(requests), frames), dtype=bool)
        grouped: dict[int, list[tuple[int, SampleRequest]]] = {}
        for batch_idx, request in enumerate(requests):
            grouped.setdefault(int(request.shard_idx), []).append((batch_idx, request))
        for shard_idx in sorted(grouped):
            for batch_idx, request in grouped[shard_idx]:
                motion[batch_idx] = self._read_numpy(request)
        batch: dict[str, Any] = {
            "motion": torch.from_numpy(motion),
            "loss_mask": torch.from_numpy(loss_mask),
        }
        if self.return_metadata:
            batch["metadata"] = [
                {
                    "shard_idx": int(request.shard_idx),
                    "target_start": int(request.target_start),
                    "target_frames": int(request.target_frames),
                    "variant_idx": int(request.variant_idx),
                    "range_name": self.store.range_names[int(request.variant_idx)],
                }
                for request in requests
            ]
        return batch

    def __getitem__(self, index: int | SampleRequest) -> dict[str, Any]:
        return self._item_request(self._request(index))

    def _item_request(self, request: SampleRequest) -> dict[str, Any]:
        # MMapShardCache opens shards read-only. PyTorch tensors must not share
        # that storage because writes through the tensor would be undefined.
        motion = np.array(self._read_numpy(request), dtype=np.float32, copy=True, order="C")
        mask = np.ones((motion.shape[0],), dtype=bool)
        item: dict[str, Any] = {"motion": torch.from_numpy(motion), "loss_mask": torch.from_numpy(mask)}
        if self.return_metadata:
            item["metadata"] = {
                "shard_idx": int(request.shard_idx),
                "target_start": int(request.target_start),
                "target_frames": int(request.target_frames),
                "variant_idx": int(request.variant_idx),
                "range_name": self.store.range_names[int(request.variant_idx)],
            }
        return item

    def __getitems__(self, values: list[int | SampleRequest]) -> dict[str, Any]:
        return self._batch(values)

    def close(self) -> None:
        self.store.close()

    def __getstate__(self) -> dict[str, object]:
        return dict(self.__dict__)


__all__ = ["FeatureCache", "FeatureDataset", "FeatureStore", "open_feature_cache", "open_feature_store"]
