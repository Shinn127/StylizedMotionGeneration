"""Store access shared by the NEF evaluation, probes and MTS windows.

Both store generations are read through this one module:

* v3 row stores (``FeatureStore``) index ranges with absolute shard offsets and
  expose ``range_starts`` / ``range_shard_indices`` / ``motion_files``;
* v4 packed stores (``PackedFeatureStore``) index logical clips with
  ``clip_offset`` / ``clip_shard`` / ``read_frames``.

Keeping the adaptation here means the training loader, the Phase-0 probes and
the NEF report all read the *same* frames: a discrepancy cannot hide in two
slightly different window readers, and a new store generation only has to be
taught to this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from stylized_motion.anim.features import denormalize_motion_features


def is_packed_store(store: Any) -> bool:
    """True for the schema-v4 packed reader, False for the v3 row store."""
    return hasattr(store, "clip_offset")


def store_length(store: Any) -> int:
    """Number of logical clips (v4) or ranges (v3)."""
    if is_packed_store(store):
        return int(store.num_clips)
    return len(store.range_names)


def split_clip_geometry(store: Any, clip_idx: int) -> tuple[int, int, int]:
    """``(shard_idx, offset, length)`` of one logical clip/range row."""
    clip_idx = int(clip_idx)
    if is_packed_store(store):
        if not 0 <= clip_idx < int(store.num_clips):
            raise IndexError(f"Invalid clip index {clip_idx}")
        return (
            int(store.clip_shard[clip_idx]),
            int(store.clip_offset[clip_idx]),
            int(store.clip_length[clip_idx]),
        )
    if not 0 <= clip_idx < len(store.range_names):
        raise IndexError(f"Invalid clip index {clip_idx}")
    start = int(store.range_starts[clip_idx])
    return (
        int(store.range_shard_indices[clip_idx]),
        start,
        int(store.range_stops[clip_idx]) - start,
    )


def read_features(store: Any, shard_idx: int, start: int, frames: int) -> np.ndarray:
    """Reads ``frames`` feature rows starting at absolute shard offset ``start``."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    if is_packed_store(store):
        return np.asarray(store.read_frames(int(shard_idx), int(start), int(frames)), dtype=np.float32)
    motion = np.load(store.motion_files[int(shard_idx)], mmap_mode="r", allow_pickle=False)
    window = np.asarray(motion[int(start) : int(start) + int(frames)], dtype=np.float32)
    if window.shape[0] != int(frames):
        raise IndexError(
            f"Shard {shard_idx} cannot serve [{start}, {start + frames}) of {len(motion)} frames"
        )
    return np.ascontiguousarray(window)


def read_clip_window(
    store: Any,
    clip_idx: int,
    start: int,
    frames: int,
    *,
    history: int = 0,
    shards: dict[int, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Reads ``history + frames`` rows ending at ``start + frames``.

    The window never leaves its logical clip; frames before the clip start are
    left-padded by repeating the first stored frame, so a probe or a training
    window measures the decoder rather than fabricated motion.  Returns the
    window and the shard index it came from.
    """
    shard_idx, offset, length = split_clip_geometry(store, int(clip_idx))
    start, frames, history = int(start), int(frames), int(history)
    if start < offset or start + frames > offset + length:
        raise IndexError(
            f"Window [{start}, {start + frames}) leaves clip {clip_idx} interval "
            f"[{offset}, {offset + length})"
        )
    read_start = max(offset, start - history)
    read_frames = start + frames - read_start
    if shards is not None and shard_idx in shards:
        array = shards[shard_idx]
        window = np.asarray(array[read_start : read_start + read_frames], dtype=np.float32)
        if is_packed_store(store):
            window = np.ascontiguousarray(window)
    else:
        window = read_features(store, shard_idx, read_start, read_frames)
        if shards is not None:
            files = getattr(store, "shard_files", None) or getattr(store, "motion_files")
            shards[shard_idx] = np.load(files[shard_idx], mmap_mode="r", allow_pickle=False)
    left_pad = read_start - (start - history)
    if left_pad > 0:
        window = np.concatenate((np.repeat(window[:1], left_pad, axis=0), window), axis=0)
    planned = history + frames
    if window.shape != (planned, int(store.motion_dim)):
        raise RuntimeError(f"Expected window {(planned, store.motion_dim)}, got {window.shape}")
    return window, shard_idx


def read_sampler_window(
    store: Any,
    request: Any,
    *,
    history: int = 0,
    shards: dict[int, Any] | None = None,
) -> np.ndarray:
    """Reads one sampler request (``SampleRequest``) with optional history."""
    return read_clip_window(
        store,
        int(request.variant_idx),
        int(request.target_start),
        int(request.target_frames),
        history=int(history),
        shards=shards,
    )[0]


def renormalize(raw: np.ndarray, feature_stats: Mapping[str, object]) -> np.ndarray:
    """Re-normalizes raw features into a checkpoint's statistics."""
    offset = np.asarray(feature_stats["offset"], dtype=np.float32)
    scale = np.asarray(feature_stats["scale"], dtype=np.float32)
    return ((raw - offset) / scale).astype(np.float32)


def model_space_window(
    window: np.ndarray,
    store: Any,
    feature_stats: Mapping[str, object],
) -> torch.Tensor:
    """Denormalizes with the store's statistics, then normalizes with the model's."""
    raw = denormalize_motion_features(window, store.stats)
    return torch.from_numpy(renormalize(raw, feature_stats))


def validate_checkpoint_against_store(checkpoint: Any, model: Any, store: Any) -> None:
    """Checks that a checkpoint and a store describe the same motion data.

    ``nef_eval.validate_checkpoint_store`` is NEF-specific: it compares stream
    ownership through ``module.layout``.  Representation comparisons (R1's
    flat / part / NEF table) need the skeleton and feature-schema half of that
    check without assuming a layout, so this is the shared version.
    """
    module = getattr(model, "module", model)
    if int(getattr(module, "motion_dim")) != int(store.motion_dim):
        raise ValueError("Checkpoint and feature database motion dimensions differ")
    stats = (checkpoint or {}).get("feature_stats") if isinstance(checkpoint, Mapping) else None
    if isinstance(stats, Mapping):
        names = stats.get("names")
        parents = stats.get("parents")
        if names is not None and [str(name) for name in names] != [str(name) for name in store.names]:
            raise ValueError("Checkpoint and feature database skeletons differ")
        if parents is not None and [int(value) for value in np.asarray(parents).tolist()] != [
            int(value) for value in np.asarray(store.parents).tolist()
        ]:
            raise ValueError("Checkpoint and feature database topology differs")
    checkpoint_schema = (checkpoint or {}).get("feature_schema") if isinstance(checkpoint, Mapping) else None
    if isinstance(checkpoint_schema, Mapping):
        store_schema = store.feature_schema()
        for key in ("name", "motion_dim", "joint_subset"):
            if checkpoint_schema.get(key) != store_schema.get(key):
                raise ValueError(
                    f"Checkpoint and feature database differ at feature schema field {key!r}"
                )


def module_device(model: torch.nn.Module) -> torch.device:
    """Device of a module that may hold no parameters (buffers only)."""
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


__all__ = [
    "is_packed_store",
    "model_space_window",
    "module_device",
    "read_clip_window",
    "read_features",
    "read_sampler_window",
    "renormalize",
    "split_clip_geometry",
    "validate_checkpoint_against_store",
    "store_length",
]
