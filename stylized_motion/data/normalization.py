"""Versioned, streaming feature-normalization statistics.

Plan §4.2/§4.3.6: features are persisted *un-normalized* exactly once, while
normalization lives in its own versioned artifact. Changing the split or the
statistics must never rewrite the feature bytes, so this module owns its own
``normalization.npz``/``normalization.json`` pair and its own hash.

The accumulators are float64 Welford and mergeable: shard-ordered scans can be
split across workers and merged without the cancellation error of a naive
``sum(x^2) - mean^2`` pass. The *rule* that turns per-dimension standard
deviations into the model's ``scale``/``weights`` is shared with the legacy v3
implementation so both paths stay numerically interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stylized_motion.anim.features import MotionFeatureStats, default_joint_weights, joint_feature_dim

NORMALIZATION_VERSION = "feature_normalization_v1"
_STATS_ARRAY_KEYS = ("offset", "scale", "dist", "weights", "ref_pos")
_SCALE_FLOOR = 1e-8


def block_bounds(nbones: int) -> tuple[int, int, int]:
    """Return the (rotation_stop, hip_velocity_stop, angular_stop) feature split."""
    rotation_stop = 9 + (int(nbones) - 1) * 6
    hip_velocity_stop = rotation_stop + 3
    angular_stop = hip_velocity_stop + (int(nbones) - 1) * 3
    return rotation_stop, hip_velocity_stop, angular_stop


def compute_scale(names: Sequence[str], std: np.ndarray) -> np.ndarray:
    """Per-dimension scale from per-dimension standard deviations.

    Each feature block collapses to the mean of its own standard deviations, so
    the loss sees comparable magnitudes across root translation, rotation
    (6D), hips velocity, angular velocity and contacts.
    """
    std = np.asarray(std, dtype=np.float64)
    nbones = len(names)
    rotation_stop, hip_velocity_stop, angular_stop = block_bounds(nbones)
    if std.shape != (joint_feature_dim(nbones),):
        raise ValueError(f"std has shape {std.shape}, expected ({joint_feature_dim(nbones)},)")
    scale = np.concatenate(
        (
            np.full(3, std[0:3].mean(), dtype=np.float64),
            np.full(3, std[3:6].mean(), dtype=np.float64),
            np.full(3, std[6:9].mean(), dtype=np.float64),
            np.full(rotation_stop - 9, std[9:rotation_stop].mean(), dtype=np.float64),
            np.full(3, std[rotation_stop:hip_velocity_stop].mean(), dtype=np.float64),
            np.full(angular_stop - hip_velocity_stop, std[hip_velocity_stop:angular_stop].mean(), dtype=np.float64),
            np.full(2, std[angular_stop:].mean(), dtype=np.float64),
        )
    )
    return np.maximum(scale, _SCALE_FLOOR)


def compute_weights(names: Sequence[str]) -> np.ndarray:
    """Per-dimension loss weights implied by the skeleton's mesh weights."""
    nbones = len(names)
    rotation_stop, hip_velocity_stop, angular_stop = block_bounds(nbones)
    joint_weights = default_joint_weights([str(name) for name in names])
    del rotation_stop, hip_velocity_stop, angular_stop
    return np.concatenate(
        (
            np.ones(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            joint_weights[1:].repeat(6).astype(np.float32) * (nbones - 1),
            np.ones(3, dtype=np.float32),
            joint_weights[1:].repeat(3).astype(np.float32) * (nbones - 1),
            np.ones(2, dtype=np.float32),
        )
    ).astype(np.float32)


class FeatureStatsAccumulator:
    """Mergeable float64 Welford accumulator over feature frames."""

    def __init__(self, motion_dim: int, num_joints: int) -> None:
        if int(motion_dim) <= 0 or int(num_joints) <= 0:
            raise ValueError("motion_dim and num_joints must be positive")
        self.motion_dim = int(motion_dim)
        self.num_joints = int(num_joints)
        self.count = 0
        self.mean = np.zeros(self.motion_dim, dtype=np.float64)
        self.m2 = np.zeros(self.motion_dim, dtype=np.float64)
        self.ref_pos_sum = np.zeros((self.num_joints, 3), dtype=np.float64)
        self.blocks = 0

    def update(self, values: np.ndarray, mask: np.ndarray | None = None) -> None:
        """Accumulate a contiguous float32 [N, D] block, optionally masked."""
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != self.motion_dim:
            raise ValueError(f"Expected a [N, {self.motion_dim}] feature block, got {array.shape}")
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (array.shape[0],):
                raise ValueError("Statistics mask does not match the block's frame count")
            if not np.any(mask):
                return
            array = array[mask]
        values64 = np.asarray(array, dtype=np.float64)
        self._merge_block(values64.shape[0], values64.sum(axis=0), np.square(values64).sum(axis=0))

    def add_ref_pos_sum(self, position_sum: np.ndarray, frames: int) -> None:
        """Accumulate a pre-summed joint-position block for the reference skeleton.

        Packed stores persist one ``[J, 3]`` position sum per clip instead of
        every frame's joint positions, so the train-only reference skeleton is
        assembled from clip sums rather than re-reading motion.
        """
        if int(frames) <= 0:
            raise ValueError("frames must be positive")
        values = np.asarray(position_sum, dtype=np.float64)
        if values.shape != (self.num_joints, 3):
            raise ValueError(f"Position sums must be [{self.num_joints}, 3], got {values.shape}")
        self.ref_pos_sum += values

    def _merge_block(self, count: int, total: np.ndarray, total_squared: np.ndarray) -> None:
        if count <= 0:
            return
        if self.count == 0:
            self.mean = total / count
            self.m2 = np.maximum(total_squared / count - np.square(self.mean), 0.0) * count
            self.count = count
            self.blocks = 1
            return
        delta = (total / count) - self.mean
        total_count = self.count + count
        self.mean = self.mean + delta * (count / total_count)
        self.m2 = (
            self.m2
            + np.maximum(total_squared / count - np.square(total / count), 0.0) * count
            + np.square(delta) * (self.count * count / total_count)
        )
        self.count = total_count
        self.blocks += 1

    def merge(self, other: "FeatureStatsAccumulator") -> None:
        if other.motion_dim != self.motion_dim or other.num_joints != self.num_joints:
            raise ValueError("Cannot merge feature accumulators with different layouts")
        if other.count <= 0:
            return
        block_mean = other.mean
        block_m2 = other.m2
        self._merge_block(
            other.count,
            block_mean * other.count,
            (block_m2 + np.square(block_mean) * other.count),
        )
        self.ref_pos_sum = self.ref_pos_sum + other.ref_pos_sum
        self.blocks += max(0, other.blocks - 1)

    @property
    def std(self) -> np.ndarray:
        if self.count <= 0:
            raise ValueError("No frames have been accumulated")
        variance = np.maximum(self.m2 / self.count, 0.0)
        return np.sqrt(variance)

    def finalize(self, names: Sequence[str]) -> MotionFeatureStats:
        if self.count <= 0:
            raise ValueError("No training frames available for feature statistics")
        names = [str(name) for name in names]
        if len(names) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint names, got {len(names)}")
        scale = compute_scale(names, self.std)
        dist = (self.std / scale).astype(np.float32)
        return MotionFeatureStats(
            offset=self.mean.astype(np.float32),
            scale=scale.astype(np.float32),
            dist=dist,
            weights=compute_weights(names),
            ref_pos=(self.ref_pos_sum / self.count).astype(np.float32),
        )


def _sha256_float32_arrays(pairs: Iterable[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in pairs:
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tobytes())
    return digest.hexdigest()


@dataclass
class FeatureNormalization:
    """Versioned statistics artifact bound to a skeleton and a feature schema."""

    stats: MotionFeatureStats
    names: tuple[str, ...]
    names_sha256: str
    train_frames: int
    feature_schema_hash: str
    split_manifest_hash: str
    version: str = NORMALIZATION_VERSION
    source: str = ""

    @property
    def motion_dim(self) -> int:
        return int(self.stats.offset.shape[0])

    def stats_hash(self) -> str:
        return _sha256_float32_arrays(
            (key, getattr(self.stats, key)) for key in ("offset", "scale", "weights", "ref_pos")
        )

    def normalization_hash(self) -> str:
        from stylized_motion.data.feature_data import canonical_json_bytes

        payload = {
            "version": self.version,
            "stats_sha256": self.stats_hash(),
            "names_sha256": self.names_sha256,
            "motion_dim": self.motion_dim,
            "train_frames": int(self.train_frames),
            "feature_schema_hash": self.feature_schema_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "source": self.source,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "motion_dim": self.motion_dim,
            "num_joints": len(self.names),
            "train_frames": int(self.train_frames),
            "names_sha256": self.names_sha256,
            "stats_sha256": self.stats_hash(),
            "normalization_hash": self.normalization_hash(),
            "feature_schema_hash": self.feature_schema_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "source": self.source,
        }

    def normalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return ((array - self.stats.offset) / self.stats.scale).astype(np.float32, copy=False)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return (array * self.stats.scale + self.stats.offset).astype(np.float32, copy=False)

    def save(self, directory: str | Path) -> Path:
        import json

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez(
            target / "normalization.npz",
            offset=self.stats.offset.astype(np.float32),
            scale=self.stats.scale.astype(np.float32),
            dist=self.stats.dist.astype(np.float32),
            weights=self.stats.weights.astype(np.float32),
            ref_pos=self.stats.ref_pos.astype(np.float32),
        )
        (target / "normalization.json").write_text(
            json.dumps(self.as_manifest(), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path, *, expected: Mapping[str, Any] | None = None) -> "FeatureNormalization":
        import json

        location = Path(path)
        if location.is_dir():
            npz_path = location / "normalization.npz"
            manifest_path = location / "normalization.json"
        else:
            npz_path = location
            manifest_path = location.parent / "normalization.json"
        if not npz_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Missing normalization artifacts near {location}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as npz:
            missing = sorted(set(_STATS_ARRAY_KEYS) - set(npz.files))
            if missing:
                raise ValueError(f"normalization.npz is missing arrays: {missing}")
            stats = MotionFeatureStats(
                offset=np.asarray(npz["offset"], dtype=np.float32),
                scale=np.asarray(npz["scale"], dtype=np.float32),
                dist=np.asarray(npz["dist"], dtype=np.float32),
                weights=np.asarray(npz["weights"], dtype=np.float32),
                ref_pos=np.asarray(npz["ref_pos"], dtype=np.float32),
            )
        value = cls(
            stats=stats,
            names=(),
            names_sha256=str(manifest["names_sha256"]),
            train_frames=int(manifest["train_frames"]),
            feature_schema_hash=str(manifest["feature_schema_hash"]),
            split_manifest_hash=str(manifest["split_manifest_hash"]),
            version=str(manifest["version"]),
            source=str(manifest.get("source", "")),
        )
        if value.motion_dim != int(manifest["motion_dim"]):
            raise ValueError("Normalization motion_dim does not match its arrays")
        if value.stats_hash() != str(manifest["stats_sha256"]):
            raise ValueError("Normalization statistics do not match their recorded hash")
        if value.normalization_hash() != str(manifest["normalization_hash"]):
            raise ValueError("normalization_hash does not match canonical normalization content")
        if expected:
            for key, expected_value in expected.items():
                actual = value.as_manifest().get(key)
                if expected_value is not None and actual is not None and str(actual) != str(expected_value):
                    raise ValueError(
                        f"Normalization {key} mismatch: expected {expected_value!r}, found {actual!r}"
                    )
        return value


def names_sha256(names: Sequence[str]) -> str:
    from stylized_motion.data.feature_data import canonical_json_bytes

    return hashlib.sha256(canonical_json_bytes([str(name) for name in names])).hexdigest()


def accumulate_blocks(
    accumulator: FeatureStatsAccumulator,
    blocks: Iterable[np.ndarray],
) -> FeatureStatsAccumulator:
    """Feed contiguous ``[N, D]`` feature blocks into an accumulator."""
    for values in blocks:
        accumulator.update(values)
    return accumulator


def compute_normalization(
    store: Any,
    *,
    split: str = "train",
    chunk_frames: int = 65536,
    source: str = "",
) -> FeatureNormalization:
    """Scan the store's train ranges and build a versioned normalization.

    Only the requested split contributes, so statistics cannot leak validation
    or test frames. Reads walk the physical shards in clip order, which keeps
    the scan close to sequential I/O instead of random access. The reference
    skeleton comes from per-clip position sums, so no store needs to keep full
    per-frame joint positions around just for statistics.
    """
    if str(split) not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split {split!r}")
    if int(chunk_frames) <= 0:
        raise ValueError("chunk_frames must be positive")
    if len(store.split_clip_indices(split)) == 0:
        raise ValueError(f"Split {split!r} has no clips to accumulate")
    accumulator = FeatureStatsAccumulator(store.motion_dim, store.num_joints)
    accumulate_blocks(accumulator, store.iter_split_feature_blocks(split, chunk_frames=chunk_frames))
    position_sum, frames = store.split_position_sum(split)
    accumulator.add_ref_pos_sum(position_sum, frames)
    if int(frames) != int(accumulator.count):
        raise ValueError(
            f"Position sums cover {frames} frames but the feature scan covered {accumulator.count}"
        )
    stats = accumulator.finalize(store.names)
    return FeatureNormalization(
        stats=stats,
        names=tuple(store.names),
        names_sha256=names_sha256(store.names),
        train_frames=int(accumulator.count),
        feature_schema_hash=getattr(store, "feature_schema_hash", ""),
        split_manifest_hash=getattr(store, "split_manifest_hash", ""),
        source=str(source or f"{split}:{getattr(store, 'name', 'store')}"),
    )


__all__ = [
    "NORMALIZATION_VERSION",
    "FeatureNormalization",
    "FeatureStatsAccumulator",
    "accumulate_blocks",
    "block_bounds",
    "compute_normalization",
    "compute_scale",
    "compute_weights",
    "names_sha256",
]
