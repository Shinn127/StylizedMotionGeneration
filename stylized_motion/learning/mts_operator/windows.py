"""Window and token reading shared by the MTS training and evaluation stages.

Two backends, one interface:

* a token store (v3 ``TokenStore.read_indices``) where windows are already
  encoded;
* a feature store plus the frozen tokenizer, where windows are read through the
  same clip-local, left-padded helper the probes use and encoded on the spot.

Keeping this in one place means "what the operator trains on" and "what the
probes measured" are read by identical code, so a discrepancy cannot hide in
two slightly different window readers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from stylized_motion.learning.nef_data import model_space_window, read_sampler_window

from .layout_adapter import LayoutAdapter


def has_token_indices(store: Any) -> bool:
    """True when the store can serve encoded tokens directly."""
    return hasattr(store, "read_indices")


def read_window_tokens(
    store: Any,
    request: Any,
    *,
    frames: int,
    history: int,
    tokenizer: Any | None = None,
    feature_stats: Mapping[str, object] | None = None,
    shards: dict[int, Any] | None = None,
) -> torch.Tensor:
    """Returns one ``[frames, 40]`` long token window for a sampler request."""
    frames = int(frames)
    if has_token_indices(store):
        values = np.asarray(store.read_indices(request, frames), dtype=np.int64)
        if values.shape != (frames, 40):
            raise ValueError(
                f"Token store returned {values.shape}, expected {(frames, 40)}"
            )
        return torch.from_numpy(np.ascontiguousarray(values))
    if tokenizer is None or feature_stats is None:
        raise ValueError(
            "Reading tokens from a feature store requires the frozen tokenizer and its feature stats"
        )
    if int(request.target_frames) != frames:
        raise ValueError(
            f"Feature-store windows must be requested with target_frames={frames}, "
            f"got {request.target_frames}"
        )
    window = read_sampler_window(store, request, history=int(history), shards=shards)
    motion = model_space_window(window, store, feature_stats)
    device = next(tokenizer.parameters()).device
    with torch.no_grad():
        tokens = tokenizer.encode_indices(motion[None].to(device))
    return tokens[0, int(history) : int(history) + frames].detach().cpu()


def windows_by_clip(store: Any, split: str, *, frames: int = 64, stride: int | None = None) -> dict[int, list[Any]]:
    """Groups one split's sampler requests by logical clip row."""
    from stylized_motion.data.sampling import FixedWindowSampler

    grouped: dict[int, list[Any]] = defaultdict(list)
    sampler = FixedWindowSampler(
        store, split, target_frames=int(frames), stride=int(stride or frames), include_tail=True
    )
    for request in sampler:
        grouped[int(request.variant_idx)].append(request)
    return dict(grouped)


def clip_rows_for_split(store: Any, split: str) -> Sequence[int]:
    """Clip/range rows that belong to ``split`` (v3 and v4 stores)."""
    if hasattr(store, "split_clip_indices"):
        return [int(value) for value in store.split_clip_indices(split)]
    from stylized_motion.data.sampling import SPLIT_IDS

    split_ids = np.asarray(store.split_ids)
    return [int(value) for value in np.flatnonzero(split_ids == SPLIT_IDS[split])]


def adapter_from_tokenizer(tokenizer: Any, *, num_levels: int | None = None) -> LayoutAdapter:
    layout = tokenizer.token_layout()
    if layout is None:
        raise ValueError("The tokenizer did not expose a NEF token layout")
    return LayoutAdapter(layout, num_levels=int(num_levels or tokenizer.num_levels))


__all__ = [
    "adapter_from_tokenizer",
    "clip_rows_for_split",
    "has_token_indices",
    "read_window_tokens",
    "windows_by_clip",
]
