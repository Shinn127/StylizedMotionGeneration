"""Schema-v4 packed token store: one encoding context per logical clip.

Plan §3.4/§4.6. The v3 token builder encoded a whole *physical* shard as one
sequence. That was harmless while a shard held exactly one clip, but a packed
store deliberately mixes many clips into one shard, so encoding a shard as one
sequence would let a clip's tokens be influenced by its neighbours. This module
encodes each logical clip on its own — chunking *inside* the clip only, with
the encoder's own history window carried across chunk boundaries — and then
packs the resulting indices with the same clip table as the feature store.

The manifest binds the artifact to the checkpoint, the feature schema, the
normalization and the split, so a token store cannot silently outlive any of
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from stylized_motion.data.feature_data import canonical_json_bytes, sha256_file
from stylized_motion.data.packed_store import (
    PACKED_SCHEMA_VERSION,
    _load_clip_table,
    _validate_clip_table,
    open_packed_feature_store,
    publish_packed_store,
    write_clip_table,
    ClipTableEntry,
)
from stylized_motion.data.resume import PREPROCESS_VERSION, UnitResult, WorkJournal
from stylized_motion.data.sampling import SampleRequest

TOKEN_PACKED_STORE_TYPE = "token_packed"
DEFAULT_TOKEN_SHARD_BYTES = 128 * 1024 * 1024


def _encode_clip(
    encoder: Any,
    motion: np.ndarray,
    *,
    chunk_size: int,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode one clip into indices (and codes), never reading past its end.

    Chunking replays ``receptive_field - 1 - lookahead_frames`` history frames
    from *inside the same clip*, which is what makes chunked encoding equal
    whole-clip encoding while keeping memory bounded.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    motion = np.asarray(motion)
    if motion.ndim != 2 or motion.dtype != np.float32:
        raise ValueError("Feature clips must be float32 [N, D]")
    num_frames = int(motion.shape[0])
    num_coordinates = int(encoder.num_coordinates)
    indices_out = np.empty((num_frames, num_coordinates), dtype=np.uint8)
    codes_out = np.empty((num_frames, num_coordinates), dtype=np.float16)
    receptive_field = int(getattr(encoder, "receptive_field"))
    lookahead = int(getattr(encoder, "lookahead_frames"))
    if receptive_field <= 0 or lookahead < 0 or lookahead >= receptive_field:
        raise ValueError("Token encoder has invalid receptive_field/lookahead_frames metadata")
    history = receptive_field - 1 - lookahead
    with torch.inference_mode():
        for start in range(0, num_frames, int(chunk_size)):
            stop = min(num_frames, start + int(chunk_size))
            read_start = max(0, start - history)
            values = (
                torch.from_numpy(np.asarray(motion[read_start:stop], dtype=np.float32).copy())
                .unsqueeze(0)
                .to(device)
            )
            values = input_adapter(values) if input_adapter is not None else values
            codes, indices = encoder.encode_to_codes(values)
            offset = start - read_start
            length = stop - start
            indices_out[start:stop] = indices[0, offset : offset + length].detach().cpu().numpy().astype(np.uint8)
            codes_out[start:stop] = codes[0, offset : offset + length].detach().cpu().numpy().astype(np.float16)
    return indices_out, codes_out


@dataclass
class PackedTokenStore:
    """mmap reader over a schema-v4 packed token store.

    Attribute names mirror the packed *feature* store for the shared clip-table
    fields, so the same samplers and split logic drive both.
    """

    database: Path
    manifest: dict[str, Any]
    shard_files: list[Path]
    shard_num_frames: np.ndarray
    clip_shard: np.ndarray
    clip_offset: np.ndarray
    clip_length: np.ndarray
    clip_source_group: np.ndarray
    clip_variant: np.ndarray
    clip_split: np.ndarray
    clip_mirror: np.ndarray
    clip_source_id: np.ndarray
    clip_style_id: np.ndarray
    clip_action_id: np.ndarray
    clip_package_id: np.ndarray
    clip_position_sum: np.ndarray
    names: list[str]
    parents: np.ndarray
    joint_subset: str
    motion_dim: int
    num_coordinates: int
    num_levels: int
    temporal_downsample: int
    receptive_field: int
    lookahead_frames: int
    decoder_passes_inference: int
    checkpoint_sha256: str
    feature_schema_hash: str
    normalization_hash: str
    representation_family: str
    representation_variant: str
    representation_id: str
    model_family_legacy: str
    coordinate_order: tuple[str, ...]
    coordinate_counts: dict[str, int]
    source_style_names: tuple[str, ...] = ()
    source_action_names: tuple[str, ...] = ()
    source_package_names: tuple[str, ...] = ()
    code_shard_files: list[Path] = field(default_factory=list)
    max_open_shards: int = 16
    name: str = "packed_token_store"
    _cache: dict[int, np.ndarray] = field(default_factory=dict)
    _cache_order: list[int] = field(default_factory=list)
    _code_cache: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num_clips(self) -> int:
        return int(len(self.clip_shard))

    @property
    def num_joints(self) -> int:
        return len(self.names)

    @property
    def frame_rate(self) -> int:
        return 60

    @property
    def total_frames(self) -> int:
        return int(self.shard_num_frames.sum())

    @property
    def split_manifest_hash(self) -> str:
        return str(self.manifest["split_manifest_hash"])

    @property
    def feature_schema(self) -> dict[str, Any]:
        value = self.manifest.get("feature_schema")
        return dict(value) if isinstance(value, Mapping) else {"motion_dim": self.motion_dim}

    @property
    def representation_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "family": self.representation_family,
            "variant": self.representation_variant,
            "representation_id": self.representation_id,
            "coordinate_order": list(self.coordinate_order),
            "coordinate_counts": dict(self.coordinate_counts),
            "num_coordinates": int(self.num_coordinates),
            "num_levels": int(self.num_levels),
            "temporal_downsample": int(self.temporal_downsample),
            "frame_rate": 60,
            "receptive_field": int(self.receptive_field),
            "lookahead_frames": int(self.lookahead_frames),
            "decoder_passes_inference": int(self.decoder_passes_inference),
            "feature_schema": self.feature_schema,
        }
        representation = self.manifest.get("representation")
        if isinstance(representation, Mapping):
            result["architecture_version"] = representation.get("architecture_version")
            result["nef_layout"] = representation.get("nef_layout")
        return result

    def _get(self, shard_idx: int) -> np.ndarray:
        shard_idx = int(shard_idx)
        if shard_idx < 0 or shard_idx >= len(self.shard_files):
            raise IndexError(f"Invalid packed token shard index {shard_idx}")
        cached = self._cache.get(shard_idx)
        if cached is None:
            cached = np.load(self.shard_files[shard_idx], mmap_mode="r", allow_pickle=False)
            self._cache[shard_idx] = cached
            self._cache_order.append(shard_idx)
            while len(self._cache_order) > max(1, int(self.max_open_shards)):
                victim = self._cache_order.pop(0)
                self._cache.pop(victim, None)
        else:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
        return cached

    def read_frames(self, shard_idx: int, start: int, frames: int) -> np.ndarray:
        values = self._get(shard_idx)
        start, frames = int(start), int(frames)
        if frames <= 0 or start < 0 or start + frames > len(values):
            raise IndexError(
                f"Packed token shard {shard_idx} cannot serve [{start}, {start + frames}) of {len(values)} frames"
            )
        return np.ascontiguousarray(values[start : start + frames])

    def read_clip(self, clip_idx: int) -> np.ndarray:
        clip_idx = int(clip_idx)
        return self.read_frames(
            int(self.clip_shard[clip_idx]), int(self.clip_offset[clip_idx]), int(self.clip_length[clip_idx])
        )

    def window_local_start(self, clip_idx: int, target_start: int) -> int:
        """Convert an absolute frame offset into a clip-local one.

        Row-aligned stores (features, tokens, trajectory) share clip rows but
        not physical shard offsets, so a request built against one store is
        re-based through the clip row before it can address another.
        """
        clip_idx = int(clip_idx)
        local = int(target_start) - int(self.clip_offset[clip_idx])
        if local < 0:
            raise IndexError(f"Target start {target_start} precedes clip {clip_idx}")
        return local

    def read_window(self, clip_idx: int, start: int, frames: int) -> np.ndarray:
        """Read a token window strictly inside one clip's valid interval."""
        clip_idx = int(clip_idx)
        if clip_idx < 0 or clip_idx >= self.num_clips:
            raise IndexError(f"Invalid clip index {clip_idx}")
        offset = int(self.clip_offset[clip_idx])
        length = int(self.clip_length[clip_idx])
        start, frames = int(start), int(frames)
        if start < offset or start + frames > offset + length:
            raise IndexError(
                f"Window [{start}, {start + frames}) leaves clip {clip_idx} interval "
                f"[{offset}, {offset + length})"
            )
        return self.read_frames(int(self.clip_shard[clip_idx]), start, frames)

    def split_clip_indices(self, split: str) -> np.ndarray:
        split_id = {"train": 0, "val": 1, "test": 2}.get(str(split))
        if split_id is None:
            raise ValueError(f"Unsupported split {split!r}")
        return np.flatnonzero(self.clip_split == split_id)

    def clip_label(self, clip_idx: int) -> dict[str, Any]:
        clip_idx = int(clip_idx)
        return {
            "clip_id": clip_idx,
            "source_group": int(self.clip_source_group[clip_idx]),
            "variant": int(self.clip_variant[clip_idx]),
            "split": int(self.clip_split[clip_idx]),
            "mirror": bool(self.clip_mirror[clip_idx]),
            "source_id": int(self.clip_source_id[clip_idx]),
            "style": self.source_style_names[int(self.clip_style_id[clip_idx])] if self.source_style_names else "",
            "action": self.source_action_names[int(self.clip_action_id[clip_idx])] if self.source_action_names else "",
            "package": self.source_package_names[int(self.clip_package_id[clip_idx])]
            if self.source_package_names
            else "",
        }

    def close(self) -> None:
        self._cache.clear()
        self._cache_order.clear()
        self._code_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = {}
        state["_cache_order"] = []
        state["_code_cache"] = {}
        return state


def open_any_token_store(database: str | Path, *, max_open_shards: int = 16) -> Any:
    """Open a v4 packed token store or fall back to the v3 reader.

    Mirrors :func:`stylized_motion.data.packed_store.open_any_feature_store`, so a
    caller can point at either generation without knowing which one it has.
    """
    path = Path(database)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing token store manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(manifest.get("data_schema_version", 0))
    if version == PACKED_SCHEMA_VERSION:
        return open_packed_token_store(path, max_open_shards=max_open_shards)
    if version == 3:
        from stylized_motion.data.token_data import open_token_store

        return open_token_store(path)
    raise ValueError(f"Unsupported token store schema version {version!r} at {path}")


def open_packed_token_store(database: str | Path, *, max_open_shards: int = 16) -> PackedTokenStore:
    database = Path(database)
    manifest_path = database / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing packed token store manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("data_schema_version", 0)) != PACKED_SCHEMA_VERSION:
        raise ValueError("Packed token store must use data schema v4")
    if str(manifest.get("store_type")) != TOKEN_PACKED_STORE_TYPE:
        raise ValueError(f"Expected store_type={TOKEN_PACKED_STORE_TYPE!r}")
    build = manifest.get("build", {})
    if str(build.get("status", "complete")) != "complete":
        raise ValueError(f"Packed token store was not published cleanly: {build.get('status')!r}")
    if (int(manifest["num_coordinates"]), int(manifest["num_levels"])) != (40, 9):
        raise ValueError("Packed token store must use the canonical 40x9 contract")
    shard_files = [database / str(value) for value in manifest["shard_files"]]
    missing = [str(path) for path in shard_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Packed token store references missing shards: {missing[:3]}")
    schema = manifest["feature_schema"]
    names = [str(value) for value in schema["names"]]
    parents = np.asarray(schema["parents"], dtype=np.int32)
    motion_dim = int(manifest["motion_dim"])
    if motion_dim != 9 * len(names) + 5:
        raise ValueError(f"Packed token store motion_dim {motion_dim} does not match its skeleton")
    table = _load_clip_table(database)
    shard_num_frames = np.zeros(len(shard_files), dtype=np.int64)
    for index, path in enumerate(shard_files):
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint8 or values.ndim != 2 or values.shape[1] != int(manifest["num_coordinates"]):
            raise ValueError(f"Packed token shard {path} has an unexpected dtype/shape {values.dtype}/{values.shape}")
        shard_num_frames[index] = int(values.shape[0])
        if int(values.max(initial=0)) >= int(manifest["num_levels"]):
            raise ValueError(f"Packed token shard {path} contains an out-of-range index")
    _validate_clip_table(table, shard_num_frames, "Packed token store")
    code_files = [database / str(value) for value in manifest.get("code_shard_files", [])]
    return PackedTokenStore(
        database=database,
        manifest=manifest,
        shard_files=shard_files,
        shard_num_frames=shard_num_frames,
        clip_shard=np.asarray(table["clip_shard"], dtype=np.int32),
        clip_offset=np.asarray(table["clip_offset"], dtype=np.int64),
        clip_length=np.asarray(table["clip_length"], dtype=np.int64),
        clip_source_group=np.asarray(table["clip_source_group"], dtype=np.int32),
        clip_variant=np.asarray(table["clip_variant"], dtype=np.uint8),
        clip_split=np.asarray(table["clip_split"], dtype=np.uint8),
        clip_mirror=np.asarray(table["clip_mirror"], dtype=bool),
        clip_source_id=np.asarray(table["clip_source_id"], dtype=np.int32),
        clip_style_id=np.asarray(table["clip_style_id"], dtype=np.int32),
        clip_action_id=np.asarray(table["clip_action_id"], dtype=np.int32),
        clip_package_id=np.asarray(table["clip_package_id"], dtype=np.int32),
        clip_position_sum=np.asarray(table["clip_position_sum"], dtype=np.float64)
        if table["clip_position_sum"].ndim == 3
        else np.zeros((len(table["clip_shard"]), 0, 3), dtype=np.float64),
        names=names,
        parents=parents,
        joint_subset=str(schema.get("joint_subset", "unknown")),
        motion_dim=motion_dim,
        num_coordinates=int(manifest["num_coordinates"]),
        num_levels=int(manifest["num_levels"]),
        temporal_downsample=int(manifest.get("temporal_downsample", 1)),
        receptive_field=int(manifest.get("receptive_field", 64)),
        lookahead_frames=int(manifest.get("lookahead_frames", 0)),
        decoder_passes_inference=int(manifest.get("decoder_passes_inference", 1)),
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
        feature_schema_hash=str(manifest["feature_schema_hash"]),
        normalization_hash=str(manifest.get("normalization_hash", "")),
        representation_family=str(manifest.get("representation_family", "")),
        representation_variant=str(manifest.get("representation_variant", "")),
        representation_id=str(manifest.get("representation_id", "")),
        model_family_legacy=str(manifest.get("model_family_legacy", "")),
        coordinate_order=tuple(str(value) for value in manifest.get("coordinate_order", [])),
        coordinate_counts={str(k): int(v) for k, v in dict(manifest.get("coordinate_counts", {})).items()},
        source_style_names=tuple(str(value) for value in manifest.get("style_names", [])),
        source_action_names=tuple(str(value) for value in manifest.get("action_names", [])),
        source_package_names=tuple(str(value) for value in manifest.get("package_names", [])),
        code_shard_files=code_files,
        max_open_shards=int(max_open_shards),
    )


class PackedTokenDataset(torch.utils.data.Dataset):
    """Next-token windows read from a packed token store, one clip at a time."""

    def __init__(
        self,
        split: str,
        store: PackedTokenStore,
        *,
        sequence_frames: int = 65,
        max_open_shards: int = 16,
        return_metadata: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {split!r}")
        if int(sequence_frames) < 2:
            raise ValueError("sequence_frames must be at least 2")
        self.split = split
        self.store = store
        self.sequence_frames = int(sequence_frames)
        self.return_metadata = bool(return_metadata)
        self.max_open_shards = int(max_open_shards)
        self.store.max_open_shards = int(max_open_shards)
        self._split_id = {"train": 0, "val": 1, "test": 2}[split]

    def __len__(self) -> int:
        return int((self.store.clip_split == self._split_id).sum())

    def _read(self, request: SampleRequest) -> np.ndarray:
        clip_idx = int(request.variant_idx)
        if clip_idx < 0 or clip_idx >= self.store.num_clips:
            raise IndexError(f"Invalid packed token clip index {clip_idx}")
        if int(self.store.clip_split[clip_idx]) != self._split_id:
            raise ValueError(f"SampleRequest clip {clip_idx} does not belong to split {self.split!r}")
        return self.store.read_window(clip_idx, int(request.target_start), int(request.target_frames))

    def _batch(self, values: Sequence[int | SampleRequest]) -> dict[str, Any]:
        if not values:
            raise ValueError("Cannot collate an empty packed token batch")
        typed = [value for value in values if isinstance(value, SampleRequest)]
        if len(typed) != len(values):
            raise TypeError("PackedTokenDataset batches require SampleRequest entries")
        frames = int(typed[0].target_frames)
        if any(int(request.target_frames) != frames for request in typed):
            raise ValueError("A packed token batch must use one window length")
        tokens = np.empty((len(typed), frames, self.store.num_coordinates), dtype=np.uint8)
        grouped: dict[int, list[tuple[int, SampleRequest]]] = {}
        for batch_idx, request in enumerate(typed):
            grouped.setdefault(int(self.store.clip_shard[int(request.variant_idx)]), []).append((batch_idx, request))
        for shard_idx in sorted(grouped):
            for batch_idx, request in grouped[shard_idx]:
                tokens[batch_idx] = self._read(request)
        batch: dict[str, Any] = {
            "tokens": torch.from_numpy(tokens),
            "loss_mask": torch.ones((len(typed), frames), dtype=torch.bool),
        }
        if self.return_metadata:
            batch["metadata"] = [
                {
                    "clip_id": int(request.variant_idx),
                    "shard_idx": int(self.store.clip_shard[int(request.variant_idx)]),
                    "target_start": int(request.target_start),
                    "target_frames": int(request.target_frames),
                    **self.store.clip_label(int(request.variant_idx)),
                }
                for request in typed
            ]
        return batch

    def __getitem__(self, index: int | SampleRequest) -> dict[str, Any]:
        if not isinstance(index, SampleRequest):
            raise TypeError("PackedTokenDataset requires a SampleRequest")
        item: dict[str, Any] = {
            "tokens": torch.from_numpy(np.array(self._read(index), dtype=np.uint8, copy=True)),
            "loss_mask": torch.ones((int(index.target_frames),), dtype=torch.bool),
        }
        if self.return_metadata:
            item["metadata"] = {
                "clip_id": int(index.variant_idx),
                "shard_idx": int(self.store.clip_shard[int(index.variant_idx)]),
                "target_start": int(index.target_start),
                "target_frames": int(index.target_frames),
                **self.store.clip_label(int(index.variant_idx)),
            }
        return item

    def __getitems__(self, values: Sequence[int | SampleRequest]) -> dict[str, Any]:
        return self._batch(list(values))

    def close(self) -> None:
        self.store.close()

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_packed_token_store(
    feature_store_path: str | Path,
    output: str | Path,
    *,
    encoder: Any,
    checkpoint_sha256: str,
    device: torch.device | str = "cpu",
    chunk_size: int = 1024,
    unit_dir: str | Path | None = None,
    save_codes: bool = False,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    shard_bytes: int = DEFAULT_TOKEN_SHARD_BYTES,
    overwrite: bool = False,
    resume: bool = True,
    model_family_legacy: str | None = None,
    splits: Sequence[str] = ("train", "val", "test"),
    limit_clips: int | None = None,
) -> dict[str, Any]:
    """Encode every logical clip independently and publish a packed token store."""
    if not checkpoint_sha256:
        raise ValueError("checkpoint_sha256 is required for a TokenStore")
    feature_store = open_packed_feature_store(feature_store_path)
    device = torch.device(device)
    metadata = encoder.representation_metadata()
    if not isinstance(metadata, Mapping):
        raise ValueError("Token encoder representation_metadata() must return a mapping")
    if (
        int(encoder.num_coordinates) != 40
        or int(encoder.num_levels) != 9
        or int(encoder.receptive_field) != 64
        or int(encoder.lookahead_frames) != 0
    ):
        raise ValueError("Token encoder does not satisfy the canonical 40x9 causal contract")
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Packed token store already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        import shutil

        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "tokens").mkdir(parents=True, exist_ok=True)
    if save_codes:
        (staging / "codes").mkdir(parents=True, exist_ok=True)
    units = Path(unit_dir) if unit_dir is not None else staging / "units"
    units.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        canonical_json_bytes(
            {
                "checkpoint_sha256": str(checkpoint_sha256),
                "feature_schema_hash": feature_store.feature_schema_hash,
                "normalization_hash": feature_store.normalization_hash,
                "split_manifest_hash": feature_store.split_manifest_hash,
                "chunk_size": int(chunk_size),
                "num_coordinates": int(encoder.num_coordinates),
                "receptive_field": int(encoder.receptive_field),
                "lookahead_frames": int(encoder.lookahead_frames),
            }
        )
    ).hexdigest()
    journal = WorkJournal(units / "journal.jsonl")
    keep = np.zeros(feature_store.num_clips, dtype=bool)
    for name in splits:
        keep[feature_store.split_clip_indices(name)] = True
    clip_indices = np.flatnonzero(keep)
    if limit_clips is not None:
        clip_indices = clip_indices[: int(limit_clips)]
    completed: dict[str, UnitResult] = {}
    reused = 0
    entries: list[ClipTableEntry] = []
    pending_clips: list[int] = []
    for clip_idx in clip_indices.tolist():
        unit_id = hashlib.sha256(f"{signature}:{clip_idx}".encode("utf-8")).hexdigest()[:32]
        record = journal.committed.get(unit_id) if resume else None
        if record is not None and record.preprocess_version == PREPROCESS_VERSION:
            path = Path(record.output)
            if path.exists() and sha256_file(path) == record.sha256:
                completed[unit_id] = record
                reused += 1
                continue
        pending_clips.append(int(clip_idx))
    report: dict[str, Any] = {
        "clips": int(len(clip_indices)),
        "reused": int(reused),
        "encoded": 0,
        "failed": 0,
        "failures": [],
    }
    if not feature_store.normalization and pending_clips:
        feature_store.close()
        raise ValueError(
            "Packed feature store has no normalization artifact; the token encoder adapter needs it"
        )
    try:
        for clip_idx in tqdm(pending_clips, desc="Encoding clips"):
            unit_id = hashlib.sha256(f"{signature}:{clip_idx}".encode("utf-8")).hexdigest()[:32]
            try:
                motion = feature_store.normalization.normalize(feature_store.read_clip(clip_idx))
                indices, codes = _encode_clip(
                    encoder,
                    motion,
                    chunk_size=chunk_size,
                    device=device,
                    input_adapter=input_adapter,
                )
                token_path = units / f"{unit_id}.npy"
                np.save(token_path, indices)
                if save_codes:
                    np.save(units / f"{unit_id}.codes.npy", codes)
                result = UnitResult(
                    unit_id=unit_id,
                    clip_id=int(clip_idx),
                    variant_id=int(feature_store.clip_variant[clip_idx]),
                    output=token_path.as_posix(),
                    sha256=sha256_file(token_path),
                    frames=int(indices.shape[0]),
                    motion_dim=int(feature_store.motion_dim),
                    skeleton_hash=feature_store.skeleton_hash,
                )
            except Exception as error:  # noqa: BLE001 - reported and retryable
                journal.record_failure(unit_id, signature=signature, error=str(error))
                report["failed"] = int(report["failed"]) + 1
                report["failures"].append({"clip": int(clip_idx), "error": str(error)[:200]})
                continue
            journal.commit(result, signature=signature)
            completed[unit_id] = result
            report["encoded"] = int(report["encoded"]) + 1
        if report["failed"]:
            raise RuntimeError(
                f"{report['failed']} clips failed to encode; rerun to retry. "
                f"First failures: {report['failures'][:3]}"
            )
        if len(completed) != int(len(clip_indices)):
            raise RuntimeError(
                f"{int(len(clip_indices)) - len(completed)} clips are missing after the run; rerun to complete"
            )
        # Pack the per-clip token arrays in exactly the order the feature store
        # used, so the two clip tables stay row-aligned for conditioning.
        from stylized_motion.data.seed_build import packed_order_key

        order_seed = int(feature_store.manifest.get("split_seed", 3407))
        keys = [
            packed_order_key(
                split=int(feature_store.clip_split[clip_idx]),
                source_group=int(feature_store.clip_source_group[clip_idx]),
                source_clip_id=int(feature_store.clip_source_id[clip_idx]),
                variant=int(feature_store.clip_variant[clip_idx]),
                seed=order_seed,
            )
            for clip_idx in clip_indices.tolist()
        ]
        order = np.asarray(sorted(range(len(keys)), key=lambda row: keys[row]), dtype=np.int64)
        buffer: list[np.ndarray] = []
        buffer_frames = 0
        frames_per_shard = max(1, int(shard_bytes) // max(1, int(encoder.num_coordinates)))
        shard_files: list[str] = []
        shard_hashes: list[str] = []
        shard_frames: list[int] = []
        code_files: list[str] = []
        code_hashes: list[str] = []

        def flush() -> None:
            nonlocal buffer, buffer_frames
            if not buffer:
                return
            values = np.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
            relative = Path("tokens") / f"shard_{len(shard_files):05d}.npy"
            np.save(staging / relative, values)
            shard_files.append(relative.as_posix())
            shard_hashes.append(sha256_file(staging / relative))
            shard_frames.append(int(values.shape[0]))
            buffer = []
            buffer_frames = 0

        for row in order.tolist():
            clip_idx = int(clip_indices[row])
            unit_id = hashlib.sha256(f"{signature}:{clip_idx}".encode("utf-8")).hexdigest()[:32]
            values = np.load(units / f"{unit_id}.npy", mmap_mode="r", allow_pickle=False)
            if buffer and buffer_frames + len(values) > frames_per_shard:
                flush()
            entries.append(
                ClipTableEntry(
                    clip_id=len(entries),
                    shard_idx=len(shard_files),
                    offset=buffer_frames,
                    length=int(len(values)),
                    source_group=int(feature_store.clip_source_group[clip_idx]),
                    variant=int(feature_store.clip_variant[clip_idx]),
                    split=int(feature_store.clip_split[clip_idx]),
                    mirror=bool(feature_store.clip_mirror[clip_idx]),
                    source_id=int(feature_store.clip_source_id[clip_idx]),
                    style_id=int(feature_store.clip_style_id[clip_idx]),
                    action_id=int(feature_store.clip_action_id[clip_idx]),
                    package_id=int(feature_store.clip_package_id[clip_idx]),
                    move_name=str(feature_store.manifest.get("clip_names", [""])[clip_idx])
                    if clip_idx < len(feature_store.manifest.get("clip_names", []))
                    else "",
                )
            )
            buffer.append(np.asarray(values))
            buffer_frames += int(len(values))
            if save_codes:
                code_path = units / f"{unit_id}.codes.npy"
                relative = Path("codes") / f"{len(code_files):05d}.npy"
                np.save(staging / relative, np.load(code_path, mmap_mode="r", allow_pickle=False))
                code_files.append(relative.as_posix())
                code_hashes.append(sha256_file(staging / relative))
        flush()
        write_clip_table(staging, entries, num_joints=len(feature_store.names))
        representation = dict(metadata)
        legacy = model_family_legacy or {
            "flat_fsq": "fsq",
            "part_fsq": "part_fsq",
            "residual_part_fsq": "residual_part_fsq",
            "latent_residual_fsq": "latent_residual_part_fsq",
            "latent_residual_fsq_v2": "latent_residual_part_fsq_v2",
            "nef_fsq": "nef_fsq",
        }.get(str(representation.get("family", "")), "")
        manifest: dict[str, Any] = {
            "data_schema_version": PACKED_SCHEMA_VERSION,
            "store_type": TOKEN_PACKED_STORE_TYPE,
            "layout": "packed",
            "frame_rate": 60,
            "created_by": "stylized_motion.data.packed_token",
            "preprocess_version": PREPROCESS_VERSION,
            "num_shards": len(shard_files),
            "shard_files": shard_files,
            "shard_sha256": shard_hashes,
            "shard_num_frames": shard_frames,
            "shard_target_bytes": int(shard_bytes),
            "num_clips": len(entries),
            "total_frames": int(sum(entry.length for entry in entries)),
            "motion_dim": int(feature_store.motion_dim),
            "feature_schema": {
                "name": "motion_feature_v2",
                "motion_dim": int(feature_store.motion_dim),
                "joint_subset": feature_store.joint_subset,
                "names": list(feature_store.names),
                "parents": [int(value) for value in feature_store.parents.tolist()],
            },
            "feature_schema_hash": feature_store.feature_schema_hash,
            "skeleton_hash": feature_store.skeleton_hash,
            "normalization_hash": feature_store.normalization_hash,
            "split_manifest_hash": feature_store.split_manifest_hash,
            "split_policy": feature_store.manifest.get("split_policy", ""),
            "split_seed": int(feature_store.manifest.get("split_seed", 0)),
            "clip_names": [
                (
                    str(feature_store.manifest.get("clip_names", [])[int(clip_indices[row])])
                    if int(clip_indices[row]) < len(feature_store.manifest.get("clip_names", []))
                    else ""
                )
                for row in order.tolist()
            ],
            "style_names": list(feature_store.source_style_names),
            "action_names": list(feature_store.source_action_names),
            "package_names": list(feature_store.source_package_names),
            "representation": representation,
            "representation_family": str(representation.get("family", "")),
            "representation_variant": str(representation.get("variant", "")),
            "representation_id": str(representation.get("representation_id", "")),
            "model_family_legacy": str(legacy),
            "checkpoint_sha256": str(checkpoint_sha256),
            "num_coordinates": int(encoder.num_coordinates),
            "num_levels": int(encoder.num_levels),
            "coordinate_order": list(representation.get("coordinate_order", [])),
            "coordinate_counts": dict(representation.get("coordinate_counts", {})),
            "temporal_downsample": int(representation.get("temporal_downsample", 1)),
            "receptive_field": int(representation.get("receptive_field", 64)),
            "lookahead_frames": int(representation.get("lookahead_frames", 0)),
            "decoder_passes_inference": int(representation.get("decoder_passes_inference", 1)),
            "unit_report": {"clips": report["clips"], "reused": reused, "encoded": report["encoded"]},
            "build": {"status": "complete", "preprocess_version": PREPROCESS_VERSION, "staging": False},
        }
        if save_codes:
            manifest["code_shard_files"] = code_files
            manifest["code_shard_sha256"] = code_hashes
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        verify_token_store(staging, feature_store=feature_store, full=False)
        publish_packed_store(staging, output, overwrite=overwrite)
        report.update(
            {
                "output": str(output),
                "shards": len(shard_files),
                "checkpoint_sha256": str(checkpoint_sha256),
                "normalization_hash": feature_store.normalization_hash,
                "feature_schema_hash": feature_store.feature_schema_hash,
            }
        )
        return report
    finally:
        feature_store.close()


def verify_token_store(
    database: Path,
    *,
    feature_store: Any | None = None,
    full: bool = True,
) -> dict[str, Any]:
    """Re-open a packed token store and check alignment with its feature store."""
    store = open_packed_token_store(database)
    try:
        report = {
            "clips": store.num_clips,
            "shards": len(store.shard_files),
            "frames": store.total_frames,
            "num_coordinates": int(store.num_coordinates),
            "num_levels": int(store.num_levels),
            "checkpoint_sha256": store.checkpoint_sha256,
            "normalization_hash": store.normalization_hash,
            "feature_schema_hash": store.feature_schema_hash,
        }
        if feature_store is not None:
            if len(store.clip_shard) > len(feature_store.clip_shard):
                raise ValueError("Token store covers more clips than the feature store")
            # Row alignment is a real contract: conditioning reads the feature row
            # and the token row for the same clip.  A `--limit-clips` build covers
            # a *subset* of the feature rows, so the contract is checked against
            # the rows the token store actually claims (`clip_source_id`) instead
            # of a plain count comparison; the physical shard offsets are not part
            # of it because token and feature arrays pack at different widths.
            # Each (catalogue source clip, variant) pair identifies exactly one
            # feature row, which is what the token table can be matched against:
            # a limited build holds a *subset* of rows in packed order, so
            # positional equality only applies to a full build.
            feature_rows = {
                (int(source), int(variant)): row
                for row, (source, variant) in enumerate(
                    zip(feature_store.clip_source_id, feature_store.clip_variant)
                )
            }
            rows: list[int] = []
            for token_row in range(len(store.clip_shard)):
                key = (int(store.clip_source_id[token_row]), int(store.clip_variant[token_row]))
                matched = feature_rows.get(key)
                if matched is None:
                    raise ValueError(
                        f"Token store clip {key} does not exist in the feature store"
                    )
                rows.append(matched)
            source_rows = np.asarray(rows, dtype=np.int64)
            is_full = len(store.clip_shard) == len(feature_store.clip_shard)
            if is_full and not np.array_equal(source_rows, np.arange(len(source_rows))):
                raise ValueError("A full token store must follow the feature store's row order")
            for key in (
                "clip_source_group",
                "clip_variant",
                "clip_split",
                "clip_mirror",
                "clip_source_id",
                "clip_style_id",
                "clip_action_id",
                "clip_package_id",
                "clip_length",
            ):
                expected = np.asarray(getattr(feature_store, key))[source_rows]
                if not np.array_equal(np.asarray(getattr(store, key)), expected):
                    raise ValueError(f"Token store clip table disagrees with the feature store at {key}")
            if store.normalization_hash != feature_store.normalization_hash:
                raise ValueError("Token store normalization hash does not match the feature store")
            if store.split_manifest_hash != feature_store.split_manifest_hash:
                raise ValueError("Token store split hash does not match the feature store")
            if store.feature_schema_hash != feature_store.feature_schema_hash:
                raise ValueError("Token store feature schema hash does not match the feature store")
            if is_full:
                if store.clip_length.sum() != feature_store.clip_length.sum():
                    raise ValueError("Token store covers a different number of frames than the feature store")
            else:
                report["limited_build"] = True
                report["feature_store_clips"] = int(len(feature_store.clip_shard))
        if full:
            for relative, digest in zip(store.manifest["shard_files"], store.manifest["shard_sha256"]):
                if sha256_file(Path(database) / str(relative)) != digest:
                    raise ValueError(f"Packed token shard checksum mismatch: {relative}")
        return report
    finally:
        store.close()


__all__ = [
    "DEFAULT_TOKEN_SHARD_BYTES",
    "open_any_token_store",
    "TOKEN_PACKED_STORE_TYPE",
    "PackedTokenDataset",
    "PackedTokenStore",
    "build_packed_token_store",
    "open_packed_token_store",
    "verify_token_store",
]
