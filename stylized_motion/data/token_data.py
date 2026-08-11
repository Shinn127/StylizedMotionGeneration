"""Schema-v3 token store and next-token dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_data import (
    DATA_SCHEMA_VERSION,
    MMapShardCache,
    _check_common_index,
    _load_index,
    _require_manifest,
    _require_relative_path,
    sha256_file,
)
from .sampling import SampleRequest


_SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass
class TokenStore:
    database: Path
    manifest: dict[str, object]
    token_files: list[Path]
    code_files: list[Path | None]
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
    num_coordinates: int
    num_levels: int
    temporal_downsample: int
    receptive_field: int
    lookahead_frames: int
    decoder_passes_inference: int
    checkpoint_sha256: str
    feature_schema_hash: str
    representation_family: str
    representation_variant: str
    representation_id: str
    model_family_legacy: str
    coordinate_order: tuple[str, ...]
    coordinate_counts: dict[str, int]
    max_open_shards: int = 32
    _token_cache: MMapShardCache | None = None
    _code_cache: MMapShardCache | None = None

    @property
    def split_manifest_hash(self) -> str:
        return str(self.manifest["split_manifest_hash"])

    @property
    def feature_schema(self) -> dict[str, object]:
        value = self.manifest.get("feature_schema")
        if isinstance(value, dict):
            return dict(value)
        return {"feature_schema_hash": self.feature_schema_hash, "motion_dim": self.motion_dim}

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
            "lookahead_frames": self.lookahead_frames,
            "decoder_passes_inference": self.decoder_passes_inference,
            "feature_schema": self.feature_schema,
        }

    def validate_contract(
        self,
        *,
        checkpoint_sha256: str | None = None,
        representation: Mapping[str, object] | None = None,
        feature_schema: Mapping[str, object] | None = None,
    ) -> None:
        if (self.motion_dim, self.num_coordinates, self.num_levels) != (230, 40, 9):
            raise ValueError("TokenStore must use the canonical 230D / 40x9 contract")
        if self.frame_rate != 60 or self.temporal_downsample != 1:
            raise ValueError("TokenStore frame-rate and temporal-downsample metadata are invalid")
        if (self.receptive_field, self.lookahead_frames) != (64, 0):
            raise ValueError("TokenStore causal metadata must be RF=64 and lookahead=0")
        expected_legacy = {
            "flat_fsq": "fsq",
            "part_fsq": "part_fsq",
            "residual_part_fsq": "residual_part_fsq",
            "latent_residual_fsq": "latent_residual_part_fsq",
            "latent_residual_fsq_v2": "latent_residual_part_fsq_v2",
        }
        expected_variant = {
            "flat_fsq": "flat",
            "part_fsq": "hierarchical",
            "residual_part_fsq": "default",
            "latent_residual_fsq": "v2",
            "latent_residual_fsq_v2": "v2",
        }
        expected_layout = {
            "flat_fsq": (("flat",), {"flat": 40}),
            "part_fsq": (("global", "sync", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {"global": 6, "sync": 4, "torso": 6, "left_leg": 7, "right_leg": 7, "left_arm": 5, "right_arm": 5}),
            "residual_part_fsq": (("base", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {"base": 20, "torso": 6, "left_leg": 4, "right_leg": 4, "left_arm": 3, "right_arm": 3}),
            "latent_residual_fsq": (("base", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {"base": 20, "torso": 6, "left_leg": 4, "right_leg": 4, "left_arm": 3, "right_arm": 3}),
            "latent_residual_fsq_v2": (("base", "torso", "left_leg", "right_leg", "left_arm", "right_arm"), {"base": 20, "torso": 6, "left_leg": 4, "right_leg": 4, "left_arm": 3, "right_arm": 3}),
        }
        if self.representation_family not in expected_legacy:
            raise ValueError(f"Unsupported TokenStore representation family: {self.representation_family!r}")
        if self.model_family_legacy != expected_legacy[self.representation_family]:
            raise ValueError("TokenStore model_family_legacy does not match its canonical family")
        if self.representation_variant != expected_variant[self.representation_family]:
            raise ValueError("TokenStore representation_variant does not match its canonical family")
        order, counts = expected_layout[self.representation_family]
        if tuple(self.coordinate_order) != order or dict(self.coordinate_counts) != counts:
            raise ValueError("TokenStore coordinate layout does not match its canonical representation family")
        if self.representation_id != f"{self.representation_family}_40x9":
            raise ValueError("TokenStore representation_id does not match the canonical dimensions")
        expected_decoder_passes = 2 if self.representation_family == "residual_part_fsq" else 1
        if self.decoder_passes_inference != expected_decoder_passes:
            raise ValueError("TokenStore decoder_passes_inference does not match its representation family")
        if checkpoint_sha256 is not None and str(checkpoint_sha256) != self.checkpoint_sha256:
            raise ValueError("TokenStore checkpoint_sha256 does not match the requested checkpoint")
        if representation is not None:
            for key in (
                "family", "variant", "representation_id", "num_coordinates", "num_levels",
                "coordinate_order", "coordinate_counts", "temporal_downsample", "receptive_field",
                "lookahead_frames", "decoder_passes_inference",
            ):
                expected = representation.get(key)
                actual = self.representation_metadata.get(key)
                if expected is not None and actual != expected:
                    raise ValueError(f"TokenStore representation mismatch at {key!r}")
        if feature_schema is not None:
            expected_hash = feature_schema.get("feature_schema_hash")
            if expected_hash is not None and str(expected_hash) != self.feature_schema_hash:
                raise ValueError("TokenStore feature_schema_hash does not match the feature database")

    def _read_with_cache(self, request: SampleRequest, sequence_frames: int, cache: MMapShardCache) -> np.ndarray:
        if not isinstance(request, SampleRequest):
            raise TypeError("TokenStore.read_indices requires a validated SampleRequest")
        if sequence_frames <= 0:
            raise ValueError("sequence_frames must be positive")
        variant_idx = int(request.variant_idx)
        if variant_idx < 0 or variant_idx >= len(self.range_names):
            raise IndexError(f"Invalid token range index {variant_idx}")
        shard_idx = int(request.shard_idx)
        if int(self.range_shard_indices[variant_idx]) != shard_idx:
            raise ValueError("SampleRequest shard_idx does not match variant_idx")
        start = int(request.target_start)
        stop = start + int(sequence_frames)
        if start < int(self.range_starts[variant_idx]) or stop > int(self.range_stops[variant_idx]):
            raise IndexError("Token sequence exceeds its source range")
        array = cache.get(shard_idx)
        expected = (int(self.shard_num_frames[shard_idx]), self.num_coordinates)
        if array.dtype != np.uint8 or array.ndim != 2 or array.shape != expected:
            raise ValueError(f"Token shard has dtype/shape {array.dtype}/{array.shape}, expected uint8/{expected}")
        return np.ascontiguousarray(array[start:stop], dtype=np.uint8)

    def read_indices(self, request: SampleRequest, sequence_frames: int) -> np.ndarray:
        if self._token_cache is None:
            self._token_cache = MMapShardCache(self.token_files, self.max_open_shards)
        return self._read_with_cache(request, int(sequence_frames), self._token_cache)

    def read_codes(self, request: SampleRequest, sequence_frames: int) -> np.ndarray:
        if not isinstance(request, SampleRequest):
            raise TypeError("TokenStore.read_codes requires a validated SampleRequest")
        if sequence_frames <= 0:
            raise ValueError("sequence_frames must be positive")
        if any(path is None for path in self.code_files):
            raise ValueError("TokenStore does not contain code shards")
        paths = [path for path in self.code_files if path is not None]
        if self._code_cache is None:
            self._code_cache = MMapShardCache(paths, self.max_open_shards)
        variant_idx = int(request.variant_idx)
        if variant_idx < 0 or variant_idx >= len(self.range_names):
            raise IndexError(f"Invalid token range index {variant_idx}")
        shard_idx = int(request.shard_idx)
        start = int(request.target_start)
        stop = start + int(sequence_frames)
        if int(self.range_shard_indices[variant_idx]) != shard_idx:
            raise ValueError("SampleRequest shard_idx does not match variant_idx")
        if start < int(self.range_starts[variant_idx]) or stop > int(self.range_stops[variant_idx]):
            raise IndexError("Token code sequence exceeds its source range")
        array = self._code_cache.get(shard_idx)
        expected = (int(self.shard_num_frames[shard_idx]), self.num_coordinates)
        if array.dtype != np.float16 or array.shape != expected:
            raise ValueError(f"Code shard has dtype/shape {array.dtype}/{array.shape}, expected float16/{expected}")
        return np.ascontiguousarray(array[start:stop], dtype=np.float32)

    def split_intervals(self, split: str) -> np.ndarray:
        split_id = {"train": 0, "val": 1, "test": 2}.get(split)
        if split_id is None:
            raise ValueError(f"Unsupported split {split!r}")
        rows = np.flatnonzero(self.split_ids == split_id)
        return np.column_stack((
            self.range_shard_indices[rows], self.range_starts[rows], self.range_stops[rows], rows,
        )).astype(np.int64, copy=False)

    def close(self) -> None:
        if self._token_cache is not None:
            self._token_cache.close()
        if self._code_cache is not None:
            self._code_cache.close()
        self._token_cache = None
        self._code_cache = None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_token_cache"] = None
        state["_code_cache"] = None
        return state


def _metadata_value(manifest: Mapping[str, object], key: str, default: Any = None) -> Any:
    if key in manifest:
        return manifest[key]
    representation = manifest.get("representation")
    if isinstance(representation, Mapping) and key in representation:
        return representation[key]
    return default


def open_token_store(database: str | Path, *, max_open_shards: int = 32) -> TokenStore:
    database = Path(database)
    manifest_path = database / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing token manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Token manifest must be a JSON object")
    if "context_left" in manifest or (
        isinstance(manifest.get("representation"), Mapping)
        and "context_left" in manifest["representation"]
    ):
        raise ValueError("TokenStore manifests must not contain context_left metadata")
    _require_manifest(manifest, "token")
    shard_files = [database / str(value) for value in manifest["shard_files"]]
    if any(not path.exists() for path in shard_files):
        raise FileNotFoundError("Token manifest references a missing shard")
    code_values = manifest.get("code_shard_files")
    code_hashes = manifest.get("code_shard_sha256")
    if code_values is not None and not isinstance(code_values, list):
        raise ValueError("code_shard_files must be a JSON list")
    code_files = None if code_values is None else [database / str(value) if value else None for value in code_values]
    if code_files is not None and len(code_files) != len(shard_files):
        raise ValueError("code_shard_files must match token shard count")
    if code_files is not None and any(path is None for path in code_files) and any(path is not None for path in code_files):
        raise ValueError("code_shard_files must contain either one code shard for every token shard or none")
    if code_values is not None:
        if not isinstance(code_values, list) or not isinstance(code_hashes, list) or len(code_hashes) != len(code_values):
            raise ValueError("code_shard_files and code_shard_sha256 must be matching JSON lists")
        for value in code_values:
            _require_relative_path(value, "code shard")
        if any(not isinstance(value, str) or not value for value in code_hashes):
            raise ValueError("code_shard_sha256 entries must be non-empty strings")
    if code_files is not None and any(path is not None and not path.exists() for path in code_files):
        raise FileNotFoundError("Token manifest references a missing code shard")
    index = _load_index(database)
    common = _check_common_index(index, manifest, len(shard_files), "Token")
    _, clip_ids, source_clip_ids, range_shards, range_starts, range_stops, range_mirror, split_ids = common
    range_names = tuple(str(value) for value in manifest.get("range_names", []))
    source_clip_names = tuple(str(value) for value in manifest.get("source_clip_names", []))
    if len(range_names) != len(range_starts) or not source_clip_names:
        raise ValueError("Token range/source clip metadata is invalid")
    style_names = tuple(str(value) for value in manifest.get("style_names", []))
    action_names = tuple(str(value) for value in manifest.get("action_names", []))
    style_ids = np.asarray(index.get("style_ids", np.zeros(len(range_names), dtype=np.int32)), dtype=np.int32)
    action_ids = np.asarray(index.get("action_ids", np.zeros(len(range_names), dtype=np.int32)), dtype=np.int32)
    if len(style_ids) != len(range_names) or len(action_ids) != len(range_names):
        raise ValueError("Token style/action index arrays have invalid lengths")
    for path, frames in zip(shard_files, index["shard_num_frames"].tolist()):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (int(frames), int(manifest.get("num_coordinates", 40)))
        if array.dtype != np.uint8 or array.ndim != 2 or array.shape != expected:
            raise ValueError(f"Token shard {path} has dtype/shape {array.dtype}/{array.shape}, expected uint8/{expected}")
        del array
    if code_files is not None:
        for path, frames in zip(code_files, index["shard_num_frames"].tolist()):
            if path is None:
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected = (int(frames), int(manifest.get("num_coordinates", 40)))
            if array.dtype != np.float16 or array.ndim != 2 or array.shape != expected:
                raise ValueError(f"Code shard {path} has dtype/shape {array.dtype}/{array.shape}")
            del array
    coordinate_counts = _metadata_value(manifest, "coordinate_counts", {})
    if not isinstance(coordinate_counts, Mapping):
        raise ValueError("coordinate_counts must be a JSON object")
    feature_schema = manifest.get("feature_schema")
    if isinstance(feature_schema, Mapping):
        if str(feature_schema.get("feature_schema_hash", manifest["feature_schema_hash"])) != str(manifest["feature_schema_hash"]):
            raise ValueError("Token manifest feature_schema hash does not match feature_schema metadata")
    store = TokenStore(
        database=database,
        manifest=manifest,
        token_files=shard_files,
        code_files=code_files if code_files is not None else [None] * len(shard_files),
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
        frame_rate=int(manifest["frame_rate"]),
        motion_dim=int(_metadata_value(manifest, "motion_dim", 230)),
        num_coordinates=int(_metadata_value(manifest, "num_coordinates", 40)),
        num_levels=int(_metadata_value(manifest, "num_levels", 9)),
        temporal_downsample=int(_metadata_value(manifest, "temporal_downsample", 1)),
        receptive_field=int(_metadata_value(manifest, "receptive_field", 64)),
        lookahead_frames=int(_metadata_value(manifest, "lookahead_frames", 0)),
        decoder_passes_inference=int(_metadata_value(manifest, "decoder_passes_inference", 1)),
        checkpoint_sha256=str(manifest.get("checkpoint_sha256", "")),
        feature_schema_hash=str(manifest["feature_schema_hash"]),
        representation_family=str(_metadata_value(manifest, "representation_family", _metadata_value(manifest, "family", ""))),
        representation_variant=str(_metadata_value(manifest, "representation_variant", _metadata_value(manifest, "variant", ""))),
        representation_id=str(_metadata_value(manifest, "representation_id", "")),
        model_family_legacy=str(manifest.get("model_family_legacy", "")),
        coordinate_order=tuple(str(value) for value in _metadata_value(manifest, "coordinate_order", [])),
        coordinate_counts={str(key): int(value) for key, value in coordinate_counts.items()},
        max_open_shards=int(max_open_shards),
    )
    if not store.checkpoint_sha256:
        raise ValueError("Token manifest is missing checkpoint_sha256")
    store.validate_contract()
    return store


class TokenDataset(Dataset):
    """Return uint8 [sequence_frames,40] token tensors; default sequence is 65."""

    def __init__(
        self,
        split: str,
        store: TokenStore,
        *,
        requests: Iterable[SampleRequest] | None = None,
        sequence_frames: int = 65,
        include_codes: bool = False,
        max_open_shards: int = 32,
        return_metadata: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if sequence_frames <= 1:
            raise ValueError("sequence_frames must be at least two")
        self.split = split
        self.store = store
        self.requests = None if requests is None else list(requests)
        self.sequence_frames = int(sequence_frames)
        self.include_codes = bool(include_codes)
        self.return_metadata = bool(return_metadata)
        store.max_open_shards = int(max_open_shards)
        if self.include_codes and any(path is None for path in store.code_files):
            raise ValueError("TokenStore does not contain code shards")

    def __len__(self) -> int:
        return 0 if self.requests is None else len(self.requests)

    def _request(self, value: int | SampleRequest) -> SampleRequest:
        request = value
        if not isinstance(request, SampleRequest):
            if self.requests is None:
                raise IndexError("TokenDataset without requests requires a sampler request")
            request = self.requests[int(value)]
        variant_idx = int(request.variant_idx)
        if variant_idx < 0 or variant_idx >= len(self.store.split_ids):
            raise IndexError(f"Invalid token range index {variant_idx}")
        if int(self.store.split_ids[variant_idx]) != _SPLIT_IDS[self.split]:
            raise ValueError(f"SampleRequest range {variant_idx} does not belong to split {self.split!r}")
        return request

    def _item(self, request: SampleRequest) -> dict[str, Any]:
        indices = np.array(
            self.store.read_indices(request, self.sequence_frames),
            dtype=np.uint8,
            copy=True,
            order="C",
        )
        item: dict[str, Any] = {"indices": torch.from_numpy(indices)}
        if self.include_codes:
            codes = np.array(
                self.store.read_codes(request, self.sequence_frames),
                dtype=np.float32,
                copy=True,
                order="C",
            )
            item["codes"] = torch.from_numpy(codes)
        if self.return_metadata:
            item["metadata"] = {
                "shard_idx": int(request.shard_idx),
                "target_start": int(request.target_start),
                "target_frames": int(request.target_frames),
                "variant_idx": int(request.variant_idx),
                "range_name": self.store.range_names[int(request.variant_idx)],
            }
        return item

    def __getitem__(self, index: int | SampleRequest) -> dict[str, Any]:
        return self._item(self._request(index))

    def __getitems__(self, values: list[int | SampleRequest]) -> dict[str, Any]:
        requests = [self._request(value) for value in values]
        if not requests:
            raise ValueError("Cannot collate an empty token batch")
        indices = np.empty((len(requests), self.sequence_frames, self.store.num_coordinates), dtype=np.uint8)
        grouped: dict[int, list[tuple[int, SampleRequest]]] = {}
        for batch_idx, request in enumerate(requests):
            grouped.setdefault(int(request.shard_idx), []).append((batch_idx, request))
        for shard_idx in sorted(grouped):
            for batch_idx, request in grouped[shard_idx]:
                indices[batch_idx] = self.store.read_indices(request, self.sequence_frames)
        batch: dict[str, Any] = {"indices": torch.from_numpy(indices)}
        if self.include_codes:
            codes = np.empty_like(indices, dtype=np.float32)
            code_groups: dict[int, list[tuple[int, SampleRequest]]] = {}
            for batch_idx, request in enumerate(requests):
                code_groups.setdefault(int(request.shard_idx), []).append((batch_idx, request))
            for shard_idx in sorted(code_groups):
                for batch_idx, request in code_groups[shard_idx]:
                    codes[batch_idx] = self.store.read_codes(request, self.sequence_frames)
            batch["codes"] = torch.from_numpy(codes)
        if self.return_metadata:
            batch["metadata"] = [self._item(request)["metadata"] for request in requests]
        return batch

    def close(self) -> None:
        self.store.close()

    def __getstate__(self) -> dict[str, object]:
        return dict(self.__dict__)


__all__ = ["TokenDataset", "TokenStore", "open_token_store"]
