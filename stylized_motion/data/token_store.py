"""Representation-independent mmap token database and dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TokenWindow:
    shard_idx: int
    start_idx: int
    end_idx: int
    range_idx: int


@dataclass
class TokenStore:
    database: Path
    token_files: list[Path]
    code_files: list[Path | None]
    range_names: np.ndarray
    range_mirror: np.ndarray
    style_names: list[str]
    action_names: list[str]
    style_ids: np.ndarray
    action_ids: np.ndarray
    num_frames: np.ndarray
    motion_dim: int
    window_size: int
    frame_rate: int
    temporal_downsample: int
    receptive_field: int
    context_left: int
    lookahead_frames: int
    decoder_passes_inference: int
    num_coordinates: int
    num_levels: int
    split_windows: dict[str, list[TokenWindow]]
    checkpoint_path: str
    checkpoint_sha256: str
    feature_database: str
    feature_schema_hash: str
    joint_subset: str
    names_sha256: str
    stats_sha256: str
    representation_family: str
    representation_variant: str
    representation_id: str
    model_family_legacy: str
    coordinate_order: tuple[str, ...]
    coordinate_counts: dict[str, int]
    schema_version: int = 2

    @property
    def representation_metadata(self) -> dict[str, object]:
        return {
            "family": self.representation_family,
            "variant": self.representation_variant,
            "representation_id": self.representation_id,
            "coordinate_order": list(self.coordinate_order),
            "coordinate_counts": dict(self.coordinate_counts),
            "num_coordinates": self.num_coordinates,
            "num_levels": self.num_levels,
            "temporal_downsample": self.temporal_downsample,
            "frame_rate": self.frame_rate,
            "receptive_field": self.receptive_field,
            "context_left": self.context_left,
            "lookahead_frames": self.lookahead_frames,
            "decoder_passes_inference": self.decoder_passes_inference,
            "feature_schema": self.feature_schema,
        }

    @property
    def feature_schema(self) -> dict[str, object]:
        return {
            "name": "motion_feature_v2",
            "motion_dim": self.motion_dim,
            "joint_subset": self.joint_subset,
            "names_sha256": self.names_sha256,
            "stats_sha256": self.stats_sha256,
            "feature_schema_hash": self.feature_schema_hash,
        }

    def validate_contract(
        self,
        *,
        checkpoint_sha256: str | None = None,
        representation: dict[str, object] | None = None,
        feature_schema: dict[str, object] | None = None,
        window_size: int | None = None,
        frame_rate: int | None = None,
    ) -> None:
        expected_legacy = {
            "flat_fsq": "fsq",
            "part_fsq": "part_fsq",
            "residual_part_fsq": "residual_part_fsq",
            "latent_residual_fsq": "latent_residual_part_fsq",
        }.get(self.representation_family)
        if expected_legacy is None:
            raise ValueError(f"Unsupported TokenStore representation family: {self.representation_family!r}")
        if self.model_family_legacy != expected_legacy:
            raise ValueError("TokenStore model_family_legacy does not match its canonical family")
        expected_variant = {
            "flat_fsq": "flat",
            "part_fsq": "hierarchical",
            "residual_part_fsq": "default",
            "latent_residual_fsq": "v2",
        }[self.representation_family]
        if self.representation_variant != expected_variant:
            raise ValueError("TokenStore representation_variant does not match its canonical family")
        if self.representation_id != f"{self.representation_family}_{self.num_coordinates}x{self.num_levels}":
            raise ValueError("TokenStore representation_id does not match its dimensions")
        expected_layouts = {
            "flat_fsq": (("flat",), {"flat": 40}),
            "part_fsq": (("global", "sync", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {
                "global": 6, "sync": 4, "torso": 6, "left_leg": 7, "right_leg": 7, "left_arm": 5, "right_arm": 5,
            }),
            "residual_part_fsq": (("base", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {
                "base": 20, "torso": 6, "left_leg": 4, "right_leg": 4, "left_arm": 3, "right_arm": 3,
            }),
            "latent_residual_fsq": (("base", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {
                "base": 20, "torso": 6, "left_leg": 4, "right_leg": 4, "left_arm": 3, "right_arm": 3,
            }),
        }[self.representation_family]
        if tuple(self.coordinate_order) != expected_layouts[0] or self.coordinate_counts != expected_layouts[1]:
            raise ValueError("TokenStore coordinate layout does not match the canonical representation family")
        if (self.num_coordinates, self.num_levels, self.motion_dim) != (40, 9, 230):
            raise ValueError("TokenStore must use the Draft 0.3 230D / 40x9 contract")
        expected_decoder_passes = 2 if self.representation_family == "residual_part_fsq" else 1
        if self.decoder_passes_inference != expected_decoder_passes:
            raise ValueError("TokenStore decoder_passes_inference does not match its representation family")
        if self.frame_rate != 60:
            raise ValueError("TokenStore frame_rate must be 60")
        if self.window_size != 64:
            raise ValueError("TokenStore window_size must be 64")
        if (self.receptive_field, self.context_left, self.lookahead_frames) != (64, 63, 0):
            raise ValueError("TokenStore causal metadata must be RF=64, context_left=63, lookahead=0")
        if checkpoint_sha256 is not None and self.checkpoint_sha256 != str(checkpoint_sha256):
            raise ValueError("TokenStore checkpoint_sha256 does not match the requested checkpoint")
        if representation is not None:
            for key in (
                "family", "variant", "representation_id", "num_coordinates", "num_levels",
                "coordinate_order", "coordinate_counts", "temporal_downsample", "receptive_field",
                "context_left", "lookahead_frames", "decoder_passes_inference",
            ):
                expected = representation.get(key)
                actual = self.representation_metadata.get(key)
                if expected is not None and actual != expected:
                    raise ValueError(f"TokenStore representation mismatch at {key!r}")
            expected_schema = representation.get("feature_schema")
            if expected_schema is not None and dict(expected_schema) != self.feature_schema:
                raise ValueError("TokenStore feature_schema does not match the representation metadata")
        if feature_schema is not None:
            expected_hash = feature_schema.get("feature_schema_hash")
            if expected_hash is not None and str(expected_hash) != self.feature_schema_hash:
                raise ValueError("TokenStore feature_schema_hash does not match the feature database")
            for key in ("names_sha256", "stats_sha256"):
                expected = feature_schema.get(key)
                if expected is not None and str(expected) != getattr(self, key):
                    raise ValueError(f"TokenStore feature schema mismatch at {key!r}")
        if window_size is not None and int(window_size) != self.window_size:
            raise ValueError("TokenStore window_size does not match the requested window")
        if frame_rate is not None and int(frame_rate) != self.frame_rate:
            raise ValueError("TokenStore frame_rate does not match the requested frame rate")
        if self.lookahead_frames != 0:
            raise ValueError("TokenStore lookahead_frames must be zero")
        if self.temporal_downsample != 1:
            raise ValueError("TokenStore temporal_downsample must be one")

    def close(self) -> None:
        self._indices = None
        self._codes = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_indices"] = None
        state["_codes"] = None
        return state


def _scalar(data: dict[str, np.ndarray], key: str, dtype: Any = object) -> Any:
    if key not in data:
        raise ValueError(f"TokenStore metadata is missing required field {key!r}")
    return np.asarray(data[key], dtype=dtype).item()


def _load_windows(values: np.ndarray) -> list[TokenWindow]:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Token windows must have shape [N,4], got {array.shape}")
    return [TokenWindow(*[int(value) for value in row]) for row in array.tolist()]


def build_token_store(
    database: str | Path,
    *,
    checkpoint_sha256: str | None = None,
    representation: dict[str, object] | None = None,
    feature_schema: dict[str, object] | None = None,
) -> TokenStore:
    database = Path(database)
    metadata_path = database / "metadata.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing TokenStore metadata: {metadata_path}")
    npz = np.load(metadata_path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    required = {
        "token_files", "code_files", "range_names", "range_mirror", "style_names", "action_names",
        "style_ids", "action_ids", "num_frames", "train_windows", "val_windows", "test_windows",
        "schema_version", "motion_dim", "window_size", "frame_rate", "temporal_downsample",
        "receptive_field", "context_left", "lookahead_frames", "decoder_passes_inference",
        "num_coordinates", "num_levels", "checkpoint_path", "checkpoint_sha256", "feature_database",
        "feature_schema_hash", "names_sha256", "stats_sha256", "joint_subset", "representation_family",
        "representation_variant", "representation_id", "model_family_legacy", "coordinate_order",
        "coordinate_counts",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"TokenStore metadata is missing required fields: {missing}")
    if int(_scalar(data, "schema_version", np.int32)) != 2:
        raise ValueError("Only TokenStore schema_version=2 is supported")

    token_files = [database / str(path) for path in np.asarray(data["token_files"], dtype=object).tolist()]
    raw_code_files = np.asarray(data["code_files"], dtype=object).tolist()
    code_files = [database / str(path) if str(path) else None for path in raw_code_files]
    range_names = np.asarray(data["range_names"], dtype=object)
    style_ids = np.asarray(data["style_ids"], dtype=np.int32)
    action_ids = np.asarray(data["action_ids"], dtype=np.int32)
    if not (len(token_files) == len(range_names) == len(style_ids) == len(action_ids)):
        raise ValueError("TokenStore shard metadata arrays must have matching lengths")
    counts = json.loads(str(_scalar(data, "coordinate_counts", object)))
    if not isinstance(counts, dict):
        raise ValueError("coordinate_counts metadata must decode to a mapping")

    store = TokenStore(
        database=database,
        token_files=token_files,
        code_files=code_files,
        range_names=range_names,
        range_mirror=np.asarray(data["range_mirror"], dtype=bool),
        style_names=[str(name) for name in np.asarray(data["style_names"], dtype=object).tolist()],
        action_names=[str(name) for name in np.asarray(data["action_names"], dtype=object).tolist()],
        style_ids=style_ids,
        action_ids=action_ids,
        num_frames=np.asarray(data["num_frames"], dtype=np.int32),
        motion_dim=int(_scalar(data, "motion_dim", np.int32)),
        window_size=int(_scalar(data, "window_size", np.int32)),
        frame_rate=int(_scalar(data, "frame_rate", np.int32)),
        temporal_downsample=int(_scalar(data, "temporal_downsample", np.int32)),
        receptive_field=int(_scalar(data, "receptive_field", np.int32)),
        context_left=int(_scalar(data, "context_left", np.int32)),
        lookahead_frames=int(_scalar(data, "lookahead_frames", np.int32)),
        decoder_passes_inference=int(_scalar(data, "decoder_passes_inference", np.int32)),
        num_coordinates=int(_scalar(data, "num_coordinates", np.int32)),
        num_levels=int(_scalar(data, "num_levels", np.int32)),
        split_windows={split: _load_windows(data[f"{split}_windows"]) for split in ("train", "val", "test")},
        checkpoint_path=str(_scalar(data, "checkpoint_path", object)),
        checkpoint_sha256=str(_scalar(data, "checkpoint_sha256", object)),
        feature_database=str(_scalar(data, "feature_database", object)),
        feature_schema_hash=str(_scalar(data, "feature_schema_hash", object)),
        names_sha256=str(_scalar(data, "names_sha256", object)),
        stats_sha256=str(_scalar(data, "stats_sha256", object)),
        joint_subset=str(_scalar(data, "joint_subset", object)),
        representation_family=str(_scalar(data, "representation_family", object)),
        representation_variant=str(_scalar(data, "representation_variant", object)),
        representation_id=str(_scalar(data, "representation_id", object)),
        model_family_legacy=str(_scalar(data, "model_family_legacy", object)),
        coordinate_order=tuple(str(value) for value in np.asarray(data["coordinate_order"], dtype=object).tolist()),
        coordinate_counts={str(key): int(value) for key, value in counts.items()},
    )
    store.validate_contract(
        checkpoint_sha256=checkpoint_sha256,
        representation=representation,
        feature_schema=feature_schema,
    )
    return store


def build_full_clip_windows(store: TokenStore, window_size: int, stride: int) -> list[TokenWindow]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    windows: list[TokenWindow] = []
    for shard_idx, length in enumerate(store.num_frames.tolist()):
        length = int(length)
        if length < window_size:
            continue
        starts = list(range(0, length - window_size + 1, stride))
        tail_start = length - window_size
        if starts[-1] != tail_start:
            starts.append(tail_start)
        windows.extend(TokenWindow(shard_idx, start, start + window_size, shard_idx) for start in starts)
    return windows


class TokenDataset(Dataset):
    def __init__(
        self,
        split: str,
        store: TokenStore,
        include_codes: bool = False,
        windows: list[TokenWindow] | None = None,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported token split: {split}")
        self.store = store
        self.split = split
        self.include_codes = include_codes
        self.windows = windows if windows is not None else (
            [window for name in ("train", "val", "test") for window in store.split_windows[name]]
            if split == "all" else store.split_windows[split]
        )
        if include_codes and any(path is None for path in store.code_files):
            raise ValueError("TokenStore does not contain quantized float codes")
        self._indices: list[np.ndarray] | None = None
        self._codes: list[np.ndarray | None] | None = None
        self._validate_windows()
        self._ensure_open()

    def _validate_windows(self) -> None:
        for window in self.windows:
            if window.shard_idx < 0 or window.shard_idx >= len(self.store.token_files):
                raise ValueError(f"Token window references invalid shard {window.shard_idx}")
            if window.range_idx < 0 or window.range_idx >= len(self.store.range_names):
                raise ValueError(f"Token window references invalid range {window.range_idx}")
            if window.start_idx < 0 or window.end_idx <= window.start_idx:
                raise ValueError(f"Token window has invalid bounds: {window}")
            if window.end_idx > int(self.store.num_frames[window.shard_idx]):
                raise ValueError(f"Token window exceeds shard {window.shard_idx}: {window}")

    def _ensure_open(self) -> None:
        if self._indices is not None:
            return
        self._indices = [np.load(path, mmap_mode="r") for path in self.store.token_files]
        self._codes = [np.load(path, mmap_mode="r") if path is not None else None for path in self.store.code_files]
        for shard_idx, indices in enumerate(self._indices):
            expected = (int(self.store.num_frames[shard_idx]), self.store.num_coordinates)
            if indices.shape != expected:
                raise ValueError(f"Token shard {shard_idx} has shape {indices.shape}, expected {expected}")
            if np.any(indices < 0) or np.any(indices >= self.store.num_levels):
                raise ValueError(f"Token shard {shard_idx} contains an index outside [0,{self.store.num_levels})")
        if self.include_codes:
            for shard_idx, codes in enumerate(self._codes):
                assert codes is not None
                expected = (int(self.store.num_frames[shard_idx]), self.store.num_coordinates)
                if codes.shape != expected:
                    raise ValueError(f"Code shard {shard_idx} has shape {codes.shape}, expected {expected}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        self._ensure_open()
        assert self._indices is not None and self._codes is not None
        window = self.windows[index]
        indices = np.asarray(self._indices[window.shard_idx][window.start_idx:window.end_idx], dtype=np.int64).copy()
        style_id = int(self.store.style_ids[window.range_idx])
        action_id = int(self.store.action_ids[window.range_idx])
        item: dict[str, torch.Tensor | int | str | bool] = {
            "indices": torch.from_numpy(indices),
            "style_id": style_id,
            "action_id": action_id,
            "style_name": self.store.style_names[style_id],
            "action_name": self.store.action_names[action_id],
            "range_name": str(self.store.range_names[window.range_idx]),
            "mirror": bool(self.store.range_mirror[window.range_idx]),
            "shard_idx": window.shard_idx,
            "start_idx": window.start_idx,
            "end_idx": window.end_idx,
        }
        if self.include_codes:
            codes = np.asarray(self._codes[window.shard_idx][window.start_idx:window.end_idx], dtype=np.float32).copy()
            item["codes"] = torch.from_numpy(codes)
        return item


__all__ = ["TokenDataset", "TokenStore", "TokenWindow", "build_full_clip_windows", "build_token_store"]
