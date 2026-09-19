"""Window and token reading shared by the MTS training and evaluation stages.

Two backends, one interface:

* a token store (v3 ``TokenStore.read_indices``) where windows are already
  encoded;
* a feature store plus the frozen tokenizer, where windows are read through the
  same clip-local, left-padded helper the probes use and encoded on the spot.

Keeping this in one place means "what the operator trains on" and "what the
probes measured" are read by identical code, so a discrepancy cannot hide in
two slightly different window readers.

The batch sources live here too (:class:`TokenSource`,
:class:`PairedBatchSource`), because the training, evaluation and generation
scripts all need "one clip -> one window" and "one pair -> one aligned example".
They used to be re-implemented per script, which is how the evaluation script
ended up conditioning on ``index % 4`` instead of a real action id, and how a
skipped window left the style ids shifted by one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from stylized_motion.learning.nef_data import (
    model_space_window,
    read_sampler_window,
    split_clip_geometry,
    store_normalized_window,
)

from .layout_adapter import LayoutAdapter

CONTENT_KINDS = ("none", "action_id")


def build_content_vocabulary(
    records: Sequence[Any], *, kind: str = "action_id"
) -> dict[str, Any]:
    """The frozen action-id map, built from the *training* records only.

    Sorted, so the same action string gets the same id in every process and in a
    later run; an unknown action is an error at use time, never a silent 0, a row
    index or a style id.
    """
    if kind not in CONTENT_KINDS:
        raise ValueError(f"Unknown content kind {kind!r}; expected {list(CONTENT_KINDS)}")
    classes = (
        tuple(sorted({str(record.content) for record in records if str(record.content)}))
        if kind == "action_id"
        else ()
    )
    return {"kind": str(kind), "classes": list(classes)}


@dataclass(frozen=True)
class ContentVocabulary:
    """A ``content.kind`` plus a frozen action-id map."""

    kind: str = "none"
    classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in CONTENT_KINDS:
            raise ValueError(f"Unknown content kind {self.kind!r}; expected {list(CONTENT_KINDS)}")
        if self.kind == "none" and self.classes:
            raise ValueError("content.kind='none' must not carry an action vocabulary")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("The action vocabulary must not contain duplicates")

    @classmethod
    def build(cls, records: Sequence[Any], *, kind: str = "action_id") -> "ContentVocabulary":
        payload = build_content_vocabulary(records, kind=kind)
        return cls(kind=payload["kind"], classes=tuple(payload["classes"]))

    @property
    def index(self) -> dict[str, int]:
        return {name: position for position, name in enumerate(self.classes)}

    @property
    def unconditional(self) -> bool:
        """``none`` means the model is not conditioned on any content label."""
        return self.kind == "none"

    def id_for(self, action: Any) -> int:
        """The frozen id of ``action``; an unknown or missing label is an error."""
        if self.unconditional:
            raise ValueError("content.kind='none' has no action ids to look up")
        key = str(action)
        position = self.index.get(key)
        if position is None:
            raise ValueError(
                f"Unknown action {key!r}: it was not in the training vocabulary "
                f"({len(self.classes)} classes, e.g. {list(self.classes[:5])}). An unseen "
                "action needs its own condition representation; it must not borrow id 0."
            )
        return position

    def vector(self, actions: Sequence[Any]) -> torch.Tensor:
        return torch.tensor([self.id_for(action) for action in actions], dtype=torch.long)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "classes": list(self.classes)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ContentVocabulary | None":
        if not payload:
            return None
        return cls(kind=str(payload.get("kind", "action_id")), classes=tuple(payload.get("classes", ())))


@dataclass(frozen=True)
class WindowSample:
    """One window plus the labels that produced it."""

    tokens: torch.Tensor  # [frames, 40] int64
    valid_mask: torch.Tensor  # [frames] bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self, *, content_condition: torch.Tensor | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tokens": self.tokens,
            "valid_mask": self.valid_mask,
            "content_condition": content_condition,
            "sample_metadata": dict(self.metadata),
        }
        return payload


class TokenSource:
    """One split's windows behind "clip row -> window" (v3, v4 or feature store)."""

    def __init__(
        self,
        *,
        store: Any,
        windows_by_clip: Mapping[int, list[Any]],
        adapter: LayoutAdapter,
        frames: int,
        history: int,
        tokenizer: Any | None = None,
        feature_stats: Mapping[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.store = store
        self.windows_by_clip = windows_by_clip
        self.adapter = adapter
        self.frames = int(frames)
        self.history = int(history)
        self.tokenizer = tokenizer
        self.feature_stats = feature_stats
        self.rng = rng or np.random.default_rng(0)
        self._shards: dict[int, Any] = {}

    def clip_ids(self) -> list[int]:
        return sorted(int(clip) for clip in self.windows_by_clip)

    def request(self, clip_id: int, start: int) -> Any:
        """A ``SampleRequest`` for one explicit window of one clip."""
        from stylized_motion.data.sampling import SampleRequest

        shard, _offset, _length = split_clip_geometry(self.store, int(clip_id))
        return SampleRequest(
            shard_idx=int(shard),
            target_start=int(start),
            target_frames=int(self.frames),
            variant_idx=int(clip_id),
        )

    def read(self, request: Any) -> torch.Tensor:
        tokens = read_window_tokens(
            self.store,
            request,
            frames=self.frames,
            history=self.history,
            tokenizer=self.tokenizer,
            feature_stats=self.feature_stats,
            shards=self._shards,
        )
        # How much context this window really used, for the sample metadata: a
        # window at the clip start uses less than the nominal history.
        self.last_context_frames = min(
            int(self.history),
            max(0, int(getattr(request, "target_start", 0)) - self._clip_offset(int(getattr(request, "variant_idx", 0)))),
        )
        return tokens

    def _clip_offset(self, clip_idx: int) -> int:
        try:
            return int(split_clip_geometry(self.store, int(clip_idx))[1])
        except Exception:  # pragma: no cover - stores without clip geometry
            return 0

    def window_at(self, clip_id: int, start: int) -> WindowSample | None:
        """One explicit window (not a random one), with the mask the reader produced."""
        candidates = self.windows_by_clip.get(int(clip_id))
        if not candidates:
            return None
        if not any(int(getattr(request, "target_start", -1)) == int(start) for request in candidates):
            return None
        request = self.request(int(clip_id), int(start))
        tokens = self.read(request)
        metadata = {
            "clip_id": int(clip_id),
            "target_start": int(start),
            "variant_idx": int(clip_id),
            "frames": int(tokens.shape[0]),
            "history_frames": int(getattr(self, "last_context_frames", 0)),
        }
        device = getattr(self.store, "device", None)
        shape = (int(tokens.shape[0]),)
        valid = self.valid_mask_for(int(clip_id), int(start))
        return WindowSample(
            tokens=tokens,
            valid_mask=torch.ones(*shape, dtype=torch.bool) if valid is None else valid,
            metadata=metadata,
        )

    def valid_mask_for(self, clip_id: int, start: int) -> torch.Tensor | None:
        """Frames of that window that really exist; ``None`` means data until the end."""
        length = None
        try:
            _shard, offset, clip_length = split_clip_geometry(self.store, int(clip_id))
            length = int(offset) + int(clip_length)
        except Exception:  # pragma: no cover - stores without clip geometry
            return None
        frames = int(self.frames)
        first = int(start)
        valid = torch.zeros(frames, dtype=torch.bool)
        stop = min(first + frames, length)
        if stop > first:
            valid[: stop - first] = True
        return valid

    def window(self, clip_id: int) -> WindowSample | None:
        """A random window of ``clip_id``, or ``None`` when the clip has none."""
        candidates = self.windows_by_clip.get(int(clip_id))
        if not candidates:
            return None
        request = candidates[int(self.rng.integers(len(candidates)))]
        tokens = self.read(request)
        metadata = {
            "clip_id": int(clip_id),
            "target_start": int(getattr(request, "target_start", -1)),
            "variant_idx": int(getattr(request, "variant_idx", -1)),
            "frames": int(tokens.shape[0]),
            "history_frames": int(getattr(self, "last_context_frames", 0)),
        }
        valid = torch.ones(int(tokens.shape[0]), dtype=torch.bool)
        return WindowSample(tokens=tokens, valid_mask=valid, metadata=metadata)


class PairedBatchSource:
    """Builds aligned reference-conditioned batches from audited style pairs.

    Field alignment is the whole point: every list is appended inside the same
    ``if`` that accepts a pair, so skipping a pair whose window is unavailable
    cannot shift the style ids or the action ids relative to the tokens.
    """

    def __init__(
        self,
        *,
        token_source: TokenSource,
        sampler: Any,
        mask_generator: Any,
        device: torch.device | str,
        batch_size: int = 32,
        mode: str = "same_style",
        stage: str = "train",
        strength: tuple[float, float] | None = None,
        seed: int = 3407,
        style_index: Mapping[str, int] | None = None,
        content_vocabulary: ContentVocabulary | None = None,
        target_sampling: str | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tokens = token_source
        self.sampler = sampler
        self.mask_generator = mask_generator
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.frames = int(token_source.frames)
        self.mode = str(mode)
        self.stage = str(stage)
        self.strength = strength
        # The style-id sandbox conditions on a style *id*: the pairs already know
        # their style, so the batch carries the id the encoder expects.
        self.style_index = dict(style_index) if style_index else None
        self.content = content_vocabulary or ContentVocabulary()
        self.target_sampling = target_sampling
        self.rng = np.random.default_rng(seed)
        self.generator = generator or torch.Generator(device="cpu").manual_seed(seed)
        self.skipped_pairs = 0
        self.dropped_pairs = 0

    def sample_pairs(self, pairs: Sequence[Any]) -> dict[str, Any]:
        """Aligned tensors and metadata for the pairs whose windows exist."""
        targets: list[torch.Tensor] = []
        references: list[torch.Tensor] = []
        target_valid: list[torch.Tensor] = []
        reference_valid: list[torch.Tensor] = []
        actions: list[str] = []
        metadata: list[dict[str, Any]] = []
        kept: list[Any] = []
        for pair in pairs:
            target = self.tokens.window(pair.target.clip_id)
            reference = self.tokens.window(pair.reference.clip_id)
            if target is None or reference is None:
                self.skipped_pairs += 1
                continue
            targets.append(target.tokens)
            references.append(reference.tokens)
            target_valid.append(target.valid_mask)
            reference_valid.append(reference.valid_mask)
            # The condition describes the *motion being generated* (the target /
            # source clip), never the style reference.
            actions.append(str(pair.target.content))
            metadata.append(
                {
                    **target.metadata,
                    "style": str(pair.target.style),
                    "action": str(pair.target.content),
                    "actor": str(pair.target.performer or ""),
                    "split": str(pair.target.split),
                    "mode": str(pair.mode),
                    "reference_clip_id": int(pair.reference.clip_id),
                    "reference_style": str(pair.reference.style),
                    "reference_action": str(pair.reference.content),
                }
            )
            kept.append(pair)
        content_condition = None
        if not self.content.unconditional and actions:
            content_condition = self.content.vector(actions).to(self.device)
        return {
            "pairs": kept,
            "tokens": targets,
            "reference_tokens": references,
            "valid_mask": target_valid,
            "reference_valid_mask": reference_valid,
            "actions": actions,
            "content_condition": content_condition,
            "sample_metadata": metadata,
        }

    def batch(self, *, size: int | None = None) -> Any | None:
        from .model import OperatorBatch

        size = self.batch_size if size is None else int(size)
        pairs = self.sampler.sample(
            count=size,
            mode=self.mode,
            stage=self.stage,
            generator=self.rng,
            target_sampling=self.target_sampling,
        )
        if not pairs:
            return None
        payload = self.sample_pairs(pairs)
        if not payload["tokens"]:
            return None
        target_tokens = torch.stack(payload["tokens"])
        reference_tokens = torch.stack(payload["reference_tokens"])
        target_valid = torch.stack(payload["valid_mask"])
        reference_valid = torch.stack(payload["reference_valid_mask"])
        count, frames = target_tokens.shape[:2]
        if not bool(target_valid.all()):
            raise ValueError(
                "A paired batch contains padding; the window reader must only return real frames"
            )
        mask = self.mask_generator.sample(
            count,
            frames,
            adapter=self.tokens.adapter,
            generator=self.generator,
            device=self.device,
        )
        strength = None
        if self.strength is not None:
            low, high = self.strength
            strength = torch.from_numpy(
                self.rng.uniform(low, high, size=count).astype(np.float32)
            )
        style_ids = None
        if self.style_index is not None:
            unknown = sorted(
                {str(pair.target.style) for pair in payload["pairs"]} - set(self.style_index)
            )
            if unknown:
                raise ValueError(f"Style id map is missing {unknown}")
            style_ids = torch.tensor(
                [self.style_index[str(pair.target.style)] for pair in payload["pairs"]],
                dtype=torch.long,
            )
        # N05b: the auxiliary seen-style CE reads the *reference's* style id, so the
        # paired source carries it beside the target's own.  ``reference_style_ids``
        # is filled whenever a style index is available, whatever the encoder kind.
        reference_style_ids = None
        if self.style_index is not None:
            reference_style_ids = torch.tensor(
                [self.style_index[str(pair.reference.style)] for pair in payload["pairs"]],
                dtype=torch.long,
            )
        return OperatorBatch(
            target_tokens=target_tokens.to(self.device),
            reference_tokens=reference_tokens.to(self.device),
            style_ids=None if style_ids is None else style_ids.to(self.device),
            reference_style_ids=None
            if reference_style_ids is None
            else reference_style_ids.to(self.device),
            visible_mask=mask.visible_mask,
            target_valid_mask=target_valid.to(self.device),
            reference_valid_mask=reference_valid.to(self.device),
            content_condition=payload["content_condition"],
            strength=1.0 if strength is None else strength.to(self.device),
            sample_metadata=payload["sample_metadata"],
            kind=mask.kind,
        )

    def batches(self, count: int):
        for _ in range(int(count)):
            batch = self.batch()
            if batch is not None:
                yield batch


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
    """Returns one ``[frames, 40]`` long token window for a sampler request.

    Three backends: a v3 token store (``read_indices``), a v4 packed token store
    (``read_window`` over the clip table) and a feature store plus the frozen
    tokenizer.  All three present the same window convention, so the sampler is
    the only thing that has to know which one it is reading.

    The feature-store path reads ``min(history, frames before the window)`` rows of
    *clip-local* context, encodes them together with the window and keeps the last
    ``frames`` tokens, which is exactly what the packed token builder does; the
    history length actually used is recorded in the sample metadata.
    """
    frames = int(frames)
    if has_token_indices(store):
        values = np.asarray(store.read_indices(request, frames), dtype=np.int64)
        if values.shape != (frames, 40):
            raise ValueError(
                f"Token store returned {values.shape}, expected {(frames, 40)}"
            )
        return torch.from_numpy(np.ascontiguousarray(values))
    if hasattr(store, "clip_split") and hasattr(store, "read_window") and tokenizer is None:
        values = np.asarray(
            store.read_window(int(request.variant_idx), int(request.target_start), frames),
            dtype=np.int64,
        )
        if values.shape != (frames, 40):
            raise ValueError(
                f"Packed token store returned {values.shape}, expected {(frames, 40)}"
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
    # Context comes from *inside* the clip only.  The old version padded the
    # missing history by repeating the clip's first frame, which the token-store
    # builder never does (it encodes each clip from its own first frame and lets
    # the encoder's start handling cover the frames before it), so a window near a
    # clip start produced different tokens online than the stored ones.
    _shard, offset, _length = split_clip_geometry(store, int(request.variant_idx))
    available = max(0, int(request.target_start) - int(offset))
    context = min(int(history), available)
    window = read_sampler_window(store, request, history=context, shards=shards)
    # Packs store raw frames, v3 stores are already normalized: the encoder input
    # space is the store's normalized space either way.
    motion = model_space_window(store_normalized_window(store, window), store, feature_stats)
    device = next(tokenizer.parameters()).device
    with torch.no_grad():
        tokens = tokenizer.encode_indices(motion[None].to(device))
    if tokens.shape[1] < context + frames:
        raise RuntimeError(
            f"Tokenizer returned {tokens.shape[1]} frames for a {context + frames}-frame window"
        )
    return tokens[0, context : context + frames].detach().cpu()


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
    "CONTENT_KINDS",
    "ContentVocabulary",
    "PairedBatchSource",
    "TokenSource",
    "WindowSample",
    "adapter_from_tokenizer",
    "build_content_vocabulary",
    "clip_rows_for_split",
    "has_token_indices",
    "read_window_tokens",
    "windows_by_clip",
]
