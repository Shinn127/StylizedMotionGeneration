"""The frozen evaluation protocol: which windows, masks and pairs are scored.

Revision 2 replaces "draw one random validation batch per epoch" with a list of
fixed items.  Every validation item states its split, clip, window start, mask
kind, seed, strength and content condition, so two epochs -- or two runs on the
same checkpoint -- score exactly the same thing, and a reader can recompute the
number without the training RNG.

The per-kind objective keeps fixed weights and reports the counts it was built
from: a kind with no supervised token contributes nothing and is named in
``missing_kinds`` instead of quietly re-weighting the others.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import MASK_KINDS
from .sampling import uniform_seed

EVAL_PROTOCOL_VERSION = 2
#: How many windows one mask kind contributes by default.  Smoke runs use 1, the
#: revision-2 recipes use at least 4.
DEFAULT_BATCHES_PER_KIND = 4
#: The store's own split ids, in the order ``split_ids`` / ``clip_split`` use.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def store_split_of_clip(store: Any, clip_id: int) -> str:
    """The store's *own* split label for one clip row.

    A frozen protocol must not take the caller's word for a row being held out.
    Both store generations expose a per-clip split table (v4 ``clip_split``, v3
    ``split_ids``); a store that cannot report one is refused rather than
    trusted, because "the caller said val" is not evidence.
    """
    clip_id = int(clip_id)
    if hasattr(store, "clip_split"):
        values = np.asarray(store.clip_split)
    elif hasattr(store, "split_ids"):
        values = np.asarray(store.split_ids)
    else:
        raise ValueError(
            "The store exposes no per-clip split table, so the split of clip "
            f"{clip_id} cannot be verified; the protocol is not built on trust"
        )
    if not 0 <= clip_id < len(values):
        raise IndexError(f"Invalid clip index {clip_id}")
    split_id = int(values[clip_id])
    if not 0 <= split_id < len(SPLIT_NAMES):
        raise ValueError(f"Clip {clip_id} carries split id {split_id}, which is not a known split")
    return SPLIT_NAMES[split_id]


@dataclass(frozen=True)
class ValidationSample:
    """One fixed validation item: window, mask kind, seed and conditions."""

    sample_id: int
    kind: str
    split: str
    target_clip: int
    target_start: int
    reference_clip: int
    reference_start: int
    seed: int
    content: str = ""
    style: str = ""
    actor: str = ""
    #: The complete mask configuration, not just its kind: a manifest row that says
    #: "stream" without the ratios/blocks does not describe the same experiment.
    mask_config: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": int(self.sample_id),
            "kind": self.kind,
            "split": self.split,
            "target_clip": int(self.target_clip),
            "target_start": int(self.target_start),
            "reference_clip": int(self.reference_clip),
            "reference_start": int(self.reference_start),
            "seed": int(self.seed),
            "content": self.content,
            "style": self.style,
            "actor": self.actor,
            "mask_config": dict(self.mask_config),
        }


def build_target_only_samples(
    token_source: Any,
    *,
    kinds: Sequence[str] = ("full_generation",),
    rows_per_kind: int = 8,
    frames: int,
    seed: int,
    split: str = "val",
    mask_generator: Any | None = None,
    labels: Mapping[int, Mapping[str, str]] | None = None,
    content_vocabulary: Any | None = None,
) -> tuple[list[ValidationSample], dict[str, Any]]:
    """A frozen validation set for a *base* model: held-out windows only, no pairing.

    The transport is not a style operator, so running it through the pair sampler
    would shrink its validation set to whatever can be paired -- a different
    experiment than the one the transport is trained for.  It reads windows
    directly instead, and every row is checked against the store's own split
    table (:func:`store_split_of_clip`): a pool that mixes splits is an error,
    and an empty split is an error, never a fallback to the training data.

    ``rows_per_kind`` is an explicit row count, independent of
    ``loader.batch_size``; asking for more rows than the split can supply fails
    instead of silently shrinking the protocol, because two "64-row" runs that
    scored different numbers of rows are not comparable.

    ``labels`` maps a clip row to its content/style/actor labels (the store's own
    table).  They are carried so a conditioned transport can build the same action
    condition at validation time as it saw in training; clips whose action is not
    in the frozen ``content_vocabulary`` are excluded and counted rather than
    crashing at read time.

    Returns the samples plus the selection report (pool sizes and exclusions) that
    the caller stores in the protocol artifact.
    """
    kinds = tuple(str(kind) for kind in kinds)
    if not kinds:
        raise ValueError("The protocol needs at least one mask kind")
    for kind in kinds:
        if kind not in MASK_KINDS:
            raise ValueError(f"Unknown mask kind {kind!r}; expected {list(MASK_KINDS)}")
    if str(split) not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected {list(SPLIT_NAMES)}")
    rows_per_kind = int(rows_per_kind)
    if rows_per_kind <= 0:
        raise ValueError("rows_per_kind must be a positive row count")
    frames = int(frames)
    store = getattr(token_source, "store", None)
    if store is None:
        raise ValueError(
            "The token source does not expose its store, so the split of every candidate "
            "window cannot be verified; the protocol is not built on trust"
        )
    clips = sorted(eligible_window_clips(token_source, frames))
    in_split: list[int] = []
    for clip in clips:
        actual = store_split_of_clip(store, clip)
        if actual != str(split):
            raise ValueError(
                f"The window pool for the {split!r} protocol contains clip {clip}, which the "
                f"store assigns to {actual!r}; the protocol must be built from one split's windows"
            )
        in_split.append(clip)
    if not in_split:
        raise ValueError(
            f"The {split!r} split has no clip with a full {frames}-frame window, so the frozen "
            "validation protocol cannot be built; it is never taken from another split"
        )
    lookup = dict(labels or {})
    usable = list(in_split)
    excluded_by_action: dict[str, int] = {}
    if content_vocabulary is not None and not getattr(content_vocabulary, "unconditional", True):
        known = set(content_vocabulary.index)
        kept: list[int] = []
        for clip in usable:
            action = str(lookup.get(clip, {}).get("content", ""))
            if action in known:
                kept.append(clip)
            else:
                excluded_by_action[action] = excluded_by_action.get(action, 0) + 1
        usable = kept
        if not usable:
            raise ValueError(
                f"Every {split!r} clip with a full window carries an action the frozen "
                f"vocabulary does not know ({sorted(excluded_by_action)}); the transport cannot "
                "be validated on it without a condition it was never trained with"
            )
    if len(usable) < rows_per_kind:
        raise ValueError(
            f"The {split!r} split has {len(usable)} usable clips for a {frames}-frame window, "
            f"fewer than the {rows_per_kind} rows per kind this protocol asks for; refusing to "
            "shrink a frozen protocol silently"
        )
    rng = np.random.default_rng(int(seed))
    mask_config = _mask_config_dict(mask_generator)
    samples: list[ValidationSample] = []
    for kind_index, kind in enumerate(kinds):
        chosen = rng.choice(len(usable), size=rows_per_kind, replace=False)
        for row_index, index in enumerate(chosen):
            clip_id = int(usable[int(index)])
            label = dict(lookup.get(clip_id, {}))
            samples.append(
                ValidationSample(
                    sample_id=len(samples),
                    kind=kind,
                    # Written from the verified store label, never from the argument.
                    split=store_split_of_clip(store, clip_id),
                    target_clip=clip_id,
                    target_start=_first_window_start(token_source, clip_id, frames),
                    reference_clip=clip_id,
                    reference_start=_first_window_start(token_source, clip_id, frames),
                    seed=uniform_seed(
                        int(seed), kind_index, row_index, (rows_per_kind, frames)
                    ),
                    content=str(label.get("content", "")),
                    style=str(label.get("style", "")),
                    actor=str(label.get("actor", "")),
                    mask_config=mask_config,
                )
            )
    selection = {
        "split": str(split),
        "rows_per_kind": rows_per_kind,
        "kinds": list(kinds),
        "frames": frames,
        "clips_in_split": len(in_split),
        "usable_clips": len(usable),
        "excluded_by_action": dict(sorted(excluded_by_action.items())),
        "store_clips_verified": len(clips),
    }
    return samples, selection


def content_vocabulary_filter(
    samples: Sequence[ValidationSample], vocabulary: Any | None
) -> tuple[list[ValidationSample], dict[str, Any]]:
    """Drops rows whose action the frozen vocabulary cannot condition on.

    The sampler draws from whatever the split holds; the *model* can only be
    conditioned on the train split's actions.  A row with another label is an error
    at read time (``ContentVocabulary.vector`` refuses to borrow an id), so a
    protocol that keeps such rows cannot score anything.  They are excluded and
    counted here instead, exactly as the transport's own protocol does, and a kind
    that loses every row is an error rather than a silently smaller protocol.
    """
    if vocabulary is None or getattr(vocabulary, "unconditional", True):
        return list(samples), {"excluded": 0, "by_action": {}, "kept": len(samples)}
    known = set(vocabulary.classes)
    kept: list[ValidationSample] = []
    excluded: dict[str, int] = {}
    excluded_per_kind: dict[str, int] = {}
    per_kind: dict[str, int] = {}
    for sample in samples:
        if str(sample.content) in known:
            kept.append(sample)
            per_kind[sample.kind] = per_kind.get(sample.kind, 0) + 1
        else:
            excluded[str(sample.content)] = excluded.get(str(sample.content), 0) + 1
            excluded_per_kind[sample.kind] = excluded_per_kind.get(sample.kind, 0) + 1
    requested_kinds = {sample.kind for sample in samples}
    empty_kinds = sorted(kind for kind in requested_kinds if per_kind.get(kind, 0) == 0)
    if empty_kinds:
        raise ValueError(
            f"excluded rows left no action the frozen vocabulary can condition on for mask kinds "
            f"{empty_kinds}; the protocol cannot be built from this split (excluded: {excluded})"
        )
    return kept, {
        "excluded": sum(excluded.values()),
        "by_action": dict(sorted(excluded.items())),
        "kept": len(kept),
        "kept_per_kind": dict(sorted(per_kind.items())),
        "excluded_per_kind": dict(sorted(excluded_per_kind.items())),
        "vocabulary_size": len(known),
        "note": (
            "rows whose action is outside the frozen training vocabulary are excluded and counted; "
            "an id is never borrowed for them"
        ),
    }


def build_validation_samples(
    sampler: Any,
    *,
    token_source: Any,
    mask_generator: Any | None = None,
    kinds: Sequence[str] = MASK_KINDS,
    batches_per_kind: int = DEFAULT_BATCHES_PER_KIND,
    batch_size: int,
    frames: int,
    seed: int,
    split: str = "val",
    stage: str = "val",
    target_sampling: str | None = None,
) -> list[ValidationSample]:
    """Freezes the validation items once, deterministically.

    Pairs come from the sampler under a *validation* RNG that is created here and
    never shared with training; windows are the first available window of the clip
    (not a random one) so the protocol does not depend on any RNG at read time.
    """
    kinds = tuple(str(kind) for kind in kinds)
    for kind in kinds:
        if kind not in MASK_KINDS:
            raise ValueError(f"Unknown mask kind {kind!r}; expected {list(MASK_KINDS)}")
    batches_per_kind = int(batches_per_kind)
    batch_size = int(batch_size)
    if batches_per_kind <= 0 or batch_size <= 0:
        raise ValueError("batches_per_kind and batch_size must be positive")
    rng = np.random.default_rng(int(seed))
    mask_config = _mask_config_dict(mask_generator)
    store = getattr(token_source, "store", None)
    samples: list[ValidationSample] = []
    for kind_index, kind in enumerate(kinds):
        for batch_index in range(batches_per_kind):
            pairs = sampler.sample(
                count=batch_size,
                mode="same_style",
                stage=stage,
                generator=rng,
                target_sampling=target_sampling,
            )
            if not pairs:
                raise ValueError(
                    f"The validation split produced no {kind!r} batches; the frozen protocol "
                    "cannot be built (check windows, held-out styles and split sizes)"
                )
            for pair_index, pair in enumerate(pairs):
                # The store's own table decides whether the row is really held out;
                # the sampler's stage filter is an intention, not evidence.
                actual_split = (
                    store_split_of_clip(store, pair.target.clip_id)
                    if store is not None
                    else str(split)
                )
                if store is not None and actual_split != str(split):
                    raise ValueError(
                        f"Validation pair target {pair.target.clip_id} belongs to the store's "
                        f"{actual_split!r} split, not {str(split)!r}; the protocol must score "
                        "held-out rows only"
                    )
                samples.append(
                    ValidationSample(
                        sample_id=len(samples),
                        kind=kind,
                        split=actual_split,
                        target_clip=int(pair.target.clip_id),
                        target_start=_first_window_start(token_source, pair.target.clip_id, frames),
                        reference_clip=int(pair.reference.clip_id),
                        reference_start=_first_window_start(
                            token_source, pair.reference.clip_id, frames
                        ),
                        # One seed per item, derived from the run seed and the item,
                        # so the mask is reproducible without the training stream.
                        seed=uniform_seed(int(seed), kind_index, batch_index * 1000 + pair_index, (batch_size, frames)),
                        content=str(pair.target.content),
                        style=str(pair.target.style),
                        actor=str(pair.target.performer or ""),
                        mask_config=dict(mask_config),
                    )
                )
    if not samples:
        raise ValueError("The validation protocol is empty")
    return samples


def _first_window_start(token_source: Any, clip_id: int, frames: int) -> int:
    candidates = token_source.windows_by_clip.get(int(clip_id))
    if not candidates:
        raise ValueError(f"Clip {clip_id} has no {frames}-frame window for validation")
    starts = sorted(int(getattr(request, "target_start", -1)) for request in candidates)
    return int(starts[0])


@dataclass
class ValidationProtocol:
    """A fixed list of validation items plus the objective built from them."""

    samples: tuple[ValidationSample, ...]
    kinds: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)
    version: int = EVAL_PROTOCOL_VERSION
    #: The recipe's own name: a profile protocol and a full protocol must not be
    #: confused, and two runs are only comparable when their ids match.
    protocol_id: str = ""
    #: How the rows were selected (split, row count, pool sizes, exclusions).
    selection: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("A validation protocol needs at least one sample")
        if not self.weights:
            present = sorted({sample.kind for sample in self.samples})
            self.weights = {kind: 1.0 / len(present) for kind in present}
        values = {kind: float(weight) for kind, weight in self.weights.items()}
        for kind, weight in values.items():
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Validation weight for {kind!r} must be finite and non-negative")
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("Validation weights must contain at least one positive value")
        # Normalized once, here, and fixed: a run cannot quietly use different
        # weights than the ones it recorded.
        self.weights = {kind: weight / total for kind, weight in values.items()}
        unknown = sorted(set(self.weights) - set(self.kinds))
        if unknown:
            raise ValueError(f"Configured weights for undeclared kinds: {unknown}")
        missing = sorted(set(self.kinds) - {sample.kind for sample in self.samples})
        if missing:
            raise ValueError(f"Declared kinds without samples: {missing}")
        if any(float(weight) < 0.0 for weight in self.weights.values()):
            raise ValueError("Validation weights must be non-negative")

    @classmethod
    def build(
        cls,
        sampler: Any,
        *,
        protocol_id: str = "",
        identity: Mapping[str, Any] | None = None,
        content_vocabulary: Any | None = None,
        **kwargs: Any,
    ) -> "ValidationProtocol":
        samples = build_validation_samples(sampler, **kwargs)
        samples, exclusions = content_vocabulary_filter(samples, content_vocabulary)
        kinds = tuple(sorted({sample.kind for sample in samples}))
        selection: dict[str, Any] = {}
        if identity is not None:
            selection["store_identity"] = dict(identity)
        if content_vocabulary is not None:
            selection["content_vocabulary_filter"] = exclusions
        return cls(
            samples=tuple(samples),
            kinds=kinds,
            protocol_id=str(protocol_id),
            selection=selection,
        )

    @classmethod
    def from_target_windows(
        cls,
        token_source: Any,
        *,
        protocol_id: str,
        store_identity: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ValidationProtocol":
        """The base transport's protocol: held-out windows, verified per row."""
        samples, selection = build_target_only_samples(token_source, **kwargs)
        kinds = tuple(sorted({sample.kind for sample in samples}))
        if store_identity is not None:
            selection = {**selection, "store_identity": dict(store_identity)}
        return cls(
            samples=tuple(samples),
            kinds=kinds,
            protocol_id=str(protocol_id),
            selection=dict(selection),
        )

    def counts(self) -> dict[str, int]:
        counts = {kind: 0 for kind in self.kinds}
        for sample in self.samples:
            counts[sample.kind] = counts.get(sample.kind, 0) + 1
        return counts

    def describe(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "protocol_id": self.protocol_id or None,
            "kinds": list(self.kinds),
            "weights": {kind: float(self.weights[kind]) for kind in self.kinds},
            "samples_per_kind": self.counts(),
            "samples": int(len(self.samples)),
            # The split the rows claim *and* the verified selection report: a
            # protocol that only counted rows once lied about where they came from.
            "splits": sorted({sample.split for sample in self.samples}),
            "selection": dict(self.selection),
            "seeds": sorted({int(sample.seed) for sample in self.samples}),
            "sample_ids": [int(sample.sample_id) for sample in self.samples],
        }

    def fingerprint(self) -> str:
        """The protocol's *content* identity as one digest.

        Covers every row (kind, split, clip, start, seed, labels, mask config), the
        per-kind weights, the selection report (which includes the data identity
        when the caller passed one), so two protocols that share a shape but score
        different windows cannot look like the same experiment.  A digest over the
        description alone was exactly that mistake.
        """
        payload = {
            "version": int(self.version),
            "protocol_id": self.protocol_id,
            "kinds": list(self.kinds),
            "weights": {kind: float(self.weights[kind]) for kind in self.kinds},
            "rows": [sample.as_dict() for sample in self.samples],
            "selection": dict(self.selection),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.describe(),
            "items": [sample.as_dict() for sample in self.samples],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def batches_for_kind(self, kind: str) -> list[ValidationSample]:
        return [sample for sample in self.samples if sample.kind == kind]

    def objective(self, per_kind: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Fixed-weight objective over the kinds that produced a number.

        ``per_kind[kind]`` carries ``nll_sum`` / ``supervised_tokens`` (and
        optionally ``loss``).  Kinds with no supervised token are reported in
        ``missing_kinds`` and contribute nothing; their weight is *not* moved to
        the others, so the objective stays comparable across epochs.
        """
        value = 0.0
        total_tokens = 0
        missing: list[str] = []
        invalid: list[str] = []
        counts: dict[str, int] = {}
        nll_sums: dict[str, float] = {}
        for kind in self.kinds:
            entry = per_kind.get(kind)
            supervised = int((entry or {}).get("supervised_tokens", 0) or 0)
            counts[kind] = supervised
            if entry is None or supervised <= 0:
                missing.append(kind)
                continue
            nll_sum = float(entry.get("nll_sum", 0.0))
            if not np.isfinite(nll_sum):
                invalid.append(kind)
                continue
            nll_sums[kind] = nll_sum
            value += float(self.weights[kind]) * (nll_sum / supervised)
            total_tokens += supervised
        # Every positively-weighted kind must have produced a finite number: a
        # missing kind would otherwise make the objective *smaller* and easier to
        # beat, and a NaN would silently poison the comparison.
        weighted_kinds = [kind for kind in self.kinds if self.weights.get(kind, 0.0) > 0.0]
        usable = (
            not missing
            and not invalid
            and total_tokens > 0
            and all(kind in weighted_kinds for kind in self.kinds)
        )
        return {
            "objective": value if usable else None,
            "objectives_per_kind": {
                kind: nll_sums[kind] / counts[kind] for kind in nll_sums
            },
            "supervised_tokens": total_tokens,
            "counts": counts,
            "missing_kinds": missing,
            "invalid_kinds": invalid,
            "weights": {kind: float(self.weights[kind]) for kind in self.kinds},
            "usable": usable,
            "version": int(self.version),
        }


class ValidationBatchBuilder:
    """Turns the frozen items into deterministic ``OperatorBatch`` objects.

    Windows are read by start frame (never at random), masks are drawn from each
    item's own seed, and the batch carries the item's metadata, so an epoch cannot
    change *what* is scored -- only the weights change.
    """

    def __init__(
        self,
        *,
        token_source: Any,
        mask_generator: Any,
        adapter: Any,
        device: torch.device | str,
        content_vocabulary: Any | None = None,
        strength: float = 1.0,
        style_index: Mapping[str, int] | None = None,
        encoder_kind: str = "reference",
    ) -> None:
        self.tokens = token_source
        self.mask_generator = mask_generator
        self.adapter = adapter
        self.device = torch.device(device)
        self.content = content_vocabulary
        self.strength = float(strength)
        self.style_index = None if style_index is None else {str(k): int(v) for k, v in style_index.items()}
        self.encoder_kind = str(encoder_kind)
        if self.encoder_kind == "style_id" and self.style_index is None:
            raise ValueError("A style-ID validation builder needs the checkpoint's style index")
        self.mask_config = _mask_config_dict(mask_generator)

    def _window(self, clip_id: int, start: int) -> Any:
        """The window exactly as the reader built it, valid mask included.

        The token source already knows which frames exist (padding, short clips);
        re-deriving "all frames are valid" here would quietly score padding as real
        motion.
        """
        from .windows import WindowSample

        provider = getattr(self.tokens, "window_at", None)
        if callable(provider):
            sample = provider(int(clip_id), int(start))
            if sample is None:
                raise ValueError(f"Clip {clip_id} has no window at {start}")
            return sample
        tokens = self.tokens.read(self.tokens.request(int(clip_id), int(start)))
        return WindowSample(
            tokens=tokens,
            valid_mask=torch.ones(int(tokens.shape[0]), dtype=torch.bool),
            metadata={"clip_id": int(clip_id), "target_start": int(start)},
        )

    def build_batch(self, samples: Sequence[ValidationSample]) -> Any:
        from .model import OperatorBatch

        if not samples:
            raise ValueError("Cannot build an empty validation batch")
        target_windows = [self._window(s.target_clip, s.target_start) for s in samples]
        reference_windows = [self._window(s.reference_clip, s.reference_start) for s in samples]
        targets = torch.stack([window.tokens for window in target_windows])
        references = torch.stack([window.tokens for window in reference_windows])
        kinds = {sample.kind for sample in samples}
        if len(kinds) != 1:
            raise ValueError(f"A validation batch mixes mask kinds: {sorted(kinds)}")
        kind = next(iter(kinds))
        frames = int(targets.shape[1])
        for sample in samples:
            if sample.mask_config and self.mask_config and dict(sample.mask_config) != self.mask_config:
                raise ValueError(
                    "The manifest row's mask config does not match the mask generator this run "
                    f"is configured with: row={dict(sample.mask_config)} run={self.mask_config}"
                )
        # One mask per row, each from its own seed, then stacked: a row means the
        # same thing at B=1 and at B=8, and a new batch size cannot re-derive it.
        rows = []
        for sample in samples:
            generator = torch.Generator(device="cpu").manual_seed(int(sample.seed))
            rows.append(
                self.mask_generator.sample_kind(
                    kind, 1, frames, adapter=self.adapter, generator=generator,
                    device=torch.device("cpu"),
                ).visible_mask
            )
        visible = torch.cat(rows, dim=0)
        content_condition = None
        if self.content is not None and not getattr(self.content, "unconditional", True):
            content_condition = self.content.vector([sample.content for sample in samples]).to(
                self.device
            )
        style_ids = None
        if self.encoder_kind == "style_id":
            try:
                style_ids = torch.tensor(
                    [self.style_index[str(sample.style)] for sample in samples], dtype=torch.long
                )
            except KeyError as exc:
                raise KeyError(
                    f"Style {exc.args[0]!r} is not in the checkpoint's style index "
                    f"{sorted(self.style_index)}"
                ) from exc
        return OperatorBatch(
            target_tokens=targets.to(self.device),
            reference_tokens=references.to(self.device),
            style_ids=None if style_ids is None else style_ids.to(self.device),
            visible_mask=visible.to(self.device),
            target_valid_mask=torch.stack([w.valid_mask for w in target_windows]).to(self.device),
            reference_valid_mask=torch.stack([w.valid_mask for w in reference_windows]).to(self.device),
            strength=self.strength,
            content_condition=content_condition,
            sample_metadata=[sample.as_dict() for sample in samples],
            kind=kind,
        )

    def batches(self, protocol: ValidationProtocol, *, batch_size: int) -> list[tuple[str, Any]]:
        """The whole protocol as ``(kind, batch)`` pairs, in a fixed order."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        built: list[tuple[str, Any]] = []
        for kind in protocol.kinds:
            items = protocol.batches_for_kind(kind)
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                built.append((kind, self.build_batch(chunk)))
        return built

    def transport_items(self, protocol: ValidationProtocol, *, batch_size: int) -> list[dict[str, Any]]:
        """The frozen batches in the shape the base transport's trainer expects.

        The transport is scored on the target window only, but with the *stated*
        mask, condition and valid frames, so a validation pass and the training
        step score the same thing.
        """
        return [
            {
                "tokens": batch.target_tokens,
                "valid_mask": batch.target_valid_mask,
                "content_condition": batch.content_condition,
                "visible_mask": batch.visible_mask,
                "kind": kind,
                # The rows the batch was built from: a decoded case has to be named
                # by its clip/start/action, not by its position in a list.
                "sample_metadata": list(batch.sample_metadata),
            }
            for kind, batch in self.batches(protocol, batch_size=batch_size)
        ]

    def evaluate(
        self, trainer: Any, protocol: ValidationProtocol, *, batch_size: int, transport: bool = False
    ) -> dict[str, Any]:
        """Per-kind metrics plus the fixed-weight objective.

        ``transport=True`` scores the frozen base transport (its trainer takes
        tokens plus the stated mask) instead of the style operator.
        """
        if transport:
            per_kind: dict[str, dict[str, Any]] = {}
            for item in self.transport_items(protocol, batch_size=batch_size):
                report = trainer.evaluate([item])
                kind = str(item["kind"])
                entry = per_kind.setdefault(
                    kind, {"nll_sum": 0.0, "supervised_tokens": 0, "batches": 0}
                )
                entry["nll_sum"] += float(report.get("nll_sum", 0.0) or 0.0)
                entry["supervised_tokens"] += int(report.get("supervised_tokens", 0) or 0)
                entry["batches"] += 1
            return protocol.objective(per_kind)
        per_kind = {}
        for kind, batch in self.batches(protocol, batch_size=batch_size):
            report = trainer.evaluate([batch])
            entry = per_kind.setdefault(kind, {"nll_sum": 0.0, "supervised_tokens": 0, "batches": 0})
            entry["nll_sum"] += float(report.get("nll_sum", 0.0) or 0.0)
            entry["supervised_tokens"] += int(report.get("supervised_tokens", 0) or 0)
            entry["batches"] += 1
        return protocol.objective(per_kind)


def validation_evidence(
    protocol: "ValidationProtocol",
    report: Mapping[str, Any],
    *,
    seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The validation block both checkpoint kinds record, from one evaluation.

    The same numbers feed the log, the history, the payload and the best decision;
    a checkpoint that stores a different number than the one the run selected on
    cannot be compared with anything, which was true of the operator payload
    (it stored a plain ``val_nll`` while ``best`` followed the protocol objective).
    """
    evidence: dict[str, Any] = {
        "val_objective": report.get("objective"),
        "val_per_kind": dict(report.get("objectives_per_kind") or {}),
        "val_counts": dict(report.get("counts") or {}),
        "val_supervised_tokens": int(report.get("supervised_tokens", 0) or 0),
        "val_missing_kinds": list(report.get("missing_kinds") or []),
        "val_invalid_kinds": list(report.get("invalid_kinds") or []),
        "val_usable": bool(report.get("usable", False)),
        "protocol_id": protocol.protocol_id or None,
        "protocol_hash": protocol.fingerprint(),
    }
    for key, value in (seconds or {}).items():
        evidence[str(key)] = float(value)
    return evidence


def _mask_config_dict(mask_generator: Any | None) -> dict[str, Any]:
    """The mask generator's full configuration, as a plain dict.

    The manifest must record the ratios and block sizes, not only the kind: a row
    that says "stream" without them does not describe the same experiment.
    """
    if mask_generator is None:
        return {}
    config = getattr(mask_generator, "config", None)
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    as_dict = getattr(config, "as_dict", None)
    return dict(as_dict()) if callable(as_dict) else {}


EVAL_MANIFEST_VERSION = 2
#: Roles an evaluation row can fill.  A role that has no legal clip is recorded as
#: unavailable with a reason -- never filled with a rolled or random-token tensor.
EVAL_REFERENCE_ROLES = ("correct", "wrong", "random")


@dataclass(frozen=True)
class ClipRef:
    """One evaluated clip window and the labels it must be compared under."""

    clip_id: int
    start: int
    frames: int
    style: str = ""
    action: str = ""
    actor: str = ""
    take: int = -1
    variant: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": int(self.clip_id),
            "start": int(self.start),
            "frames": int(self.frames),
            "style": self.style,
            "action": self.action,
            "actor": self.actor,
            "take": int(self.take),
            "variant": int(self.variant),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClipRef":
        return cls(
            clip_id=int(payload["clip_id"]),
            start=int(payload["start"]),
            frames=int(payload["frames"]),
            style=str(payload.get("style", "")),
            action=str(payload.get("action", "")),
            actor=str(payload.get("actor", "")),
            take=int(payload.get("take", -1)),
            variant=int(payload.get("variant", 0)),
        )


@dataclass(frozen=True)
class EvalRow:
    """One evaluated target with every reference it is scored against."""

    sample_id: int
    split: str
    target: ClipRef
    candidates: tuple[ClipRef, ...]
    positive_indices: tuple[int, ...]
    mask_kind: str
    mask_seed: int
    sample_seed: int
    correct: ClipRef | None = None
    wrong: ClipRef | None = None
    random: ClipRef | None = None
    region: str = ""
    graph_radius: int = 0
    frame_range: tuple[int, int] | None = None
    condition: str = ""
    reasons: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": int(self.sample_id),
            "split": self.split,
            "target": self.target.as_dict(),
            "correct_reference": None if self.correct is None else self.correct.as_dict(),
            "wrong_reference": None if self.wrong is None else self.wrong.as_dict(),
            "random_real_reference": None if self.random is None else self.random.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "positive_indices": [int(index) for index in self.positive_indices],
            "mask_kind": self.mask_kind,
            "mask_seed": int(self.mask_seed),
            "sample_seed": int(self.sample_seed),
            "region": self.region,
            "graph_radius": int(self.graph_radius),
            "frame_range": None if self.frame_range is None else list(self.frame_range),
            "condition": self.condition,
            "reasons": dict(self.reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalRow":
        frame_range = payload.get("frame_range")
        return cls(
            sample_id=int(payload["sample_id"]),
            split=str(payload["split"]),
            target=ClipRef.from_dict(payload["target"]),
            candidates=tuple(ClipRef.from_dict(item) for item in payload.get("candidates", ())),
            positive_indices=tuple(int(index) for index in payload.get("positive_indices", ())),
            mask_kind=str(payload["mask_kind"]),
            mask_seed=int(payload["mask_seed"]),
            sample_seed=int(payload["sample_seed"]),
            correct=None if payload.get("correct_reference") is None else ClipRef.from_dict(payload["correct_reference"]),
            wrong=None if payload.get("wrong_reference") is None else ClipRef.from_dict(payload["wrong_reference"]),
            random=None if payload.get("random_real_reference") is None else ClipRef.from_dict(payload["random_real_reference"]),
            region=str(payload.get("region", "")),
            graph_radius=int(payload.get("graph_radius", 0)),
            frame_range=None if frame_range is None else (int(frame_range[0]), int(frame_range[1])),
            condition=str(payload.get("condition", "")),
            reasons=dict(payload.get("reasons", {})),
        )


def _windows(token_source: Any, clip_id: int, frames: int) -> list[int]:
    candidates = token_source.windows_by_clip.get(int(clip_id))
    if not candidates:
        return []
    return sorted(int(getattr(request, "target_start", -1)) for request in candidates)


def eligible_window_clips(token_source: Any, frames: int) -> dict[int, list[int]]:
    """``clip_id -> starts`` for every clip that has at least one full window."""
    return {
        int(clip): starts
        for clip in sorted(token_source.windows_by_clip)
        if (starts := _windows(token_source, int(clip), frames))
    }


def build_eval_rows(
    records: Sequence[Any],
    token_source: Any,
    *,
    split: str,
    samples: int,
    batch_size: int,
    frames: int,
    seed: int = 3407,
    mask_kinds: Sequence[str] = ("full_generation",),
    region: str = "",
    graph_radius: int = 0,
    frame_range: tuple[int, int] | None = None,
    condition: str = "none",
) -> list[EvalRow]:
    """Builds the evaluation manifest without needing a trained model.

    Every row states its windows, labels, candidate set and seeds.  A role with no
    legal clip is left empty with a reason: rolling a reference or filling it with
    uniform tokens would invent evidence that does not exist.
    """
    clips = eligible_window_clips(token_source, frames)
    if not clips:
        raise ValueError("No clip has a full window; the evaluation manifest would be empty")
    by_clip = {int(record.clip_id): record for record in records}
    pool = [clip for clip in clips if clip in by_clip and str(by_clip[clip].split) == str(split)]
    if len(pool) < 2:
        raise ValueError(
            f"The {split!r} split has {len(pool)} usable clips; an evaluation needs at least two"
        )
    rng = np.random.default_rng(int(seed))
    rows: list[EvalRow] = []
    mask_kinds = tuple(str(kind) for kind in mask_kinds) or ("full_generation",)
    for index in range(int(samples)):
        mask_kind = mask_kinds[index % len(mask_kinds)]
        target_id = int(pool[int(rng.integers(len(pool)))])
        target = by_clip[target_id]
        target_ref = ClipRef(
            clip_id=target_id,
            start=int(_windows(token_source, target_id, frames)[0]),
            frames=int(frames),
            style=str(target.style),
            action=str(target.content),
            actor=str(target.performer or ""),
            take=int(target.source_group),
            variant=int(target.variant),
        )
        reasons: dict[str, str] = {}
        # Candidate set: every eligible same-style clip that is not a leak.
        candidates: list[Any] = []
        for clip in pool:
            record = by_clip[clip]
            if str(record.style) != str(target.style):
                continue
            if _leaks(record, target):
                continue
            candidates.append(record)
        if not candidates:
            reasons["candidates"] = "no_legal_same_style_clip"
        candidate_refs = tuple(
            ClipRef(
                clip_id=int(record.clip_id),
                start=int(_windows(token_source, int(record.clip_id), frames)[0]),
                frames=int(frames),
                style=str(record.style),
                action=str(record.content),
                actor=str(record.performer or ""),
                take=int(record.source_group),
                variant=int(record.variant),
            )
            for record in candidates
        )
        positive_indices = tuple(
            position
            for position, record in enumerate(candidates)
            if str(record.style) == str(target.style)
        )
        correct = candidate_refs[0] if candidate_refs else None
        wrong_record = _pick_wrong(candidates, target, by_clip, pool)
        if wrong_record is None:
            reasons["wrong_reference"] = "no_legal_different_style_clip"
        random_record = _pick_random(pool, by_clip, target, rng)
        if random_record is None:
            reasons["random_real_reference"] = "no_other_eligible_clip"
        rows.append(
            EvalRow(
                sample_id=int(index),
                split=str(split),
                target=target_ref,
                candidates=candidate_refs,
                positive_indices=positive_indices,
                mask_kind=mask_kind,
                # One seed per row, derived from the run seed and the row, so the
                # mask is reproducible without any training stream.
                mask_seed=uniform_seed(int(seed), 0, index, (int(batch_size), int(frames))),
                sample_seed=uniform_seed(int(seed), 1, index, (int(batch_size), int(frames))),
                correct=correct,
                wrong=None
                if wrong_record is None
                else ClipRef(
                    clip_id=int(wrong_record.clip_id),
                    start=int(_windows(token_source, int(wrong_record.clip_id), frames)[0]),
                    frames=int(frames),
                    style=str(wrong_record.style),
                    action=str(wrong_record.content),
                    actor=str(wrong_record.performer or ""),
                    take=int(wrong_record.source_group),
                    variant=int(wrong_record.variant),
                ),
                random=None
                if random_record is None
                else ClipRef(
                    clip_id=int(random_record.clip_id),
                    start=int(_windows(token_source, int(random_record.clip_id), frames)[0]),
                    frames=int(frames),
                    style=str(random_record.style),
                    action=str(random_record.content),
                    actor=str(random_record.performer or ""),
                    take=int(random_record.source_group),
                    variant=int(random_record.variant),
                ),
                region=str(region),
                graph_radius=int(graph_radius),
                frame_range=frame_range,
                condition=str(condition),
                reasons=reasons,
            )
        )
    return rows


def _leaks(record: Any, target: Any) -> bool:
    """Same clip, or same take (which covers mirror and sibling-crop variants).

    The catalogue gives every variant of one take the same ``source_group``, so the
    take rule already excludes the mirror pair and the neighbouring crop; the
    clip-id check is the degenerate case of it.  This is R04's data identity, not
    just "the clip ids differ".
    """
    if int(record.clip_id) == int(target.clip_id):
        return True
    if int(record.source_group) >= 0 and int(record.source_group) == int(target.source_group):
        return True
    return False


def _pick_wrong(candidates: Sequence[Any], target: Any, by_clip: Mapping[int, Any], pool: Sequence[int]):
    """A different-style clip, preferring the same action and then the same actor."""
    options = [
        by_clip[clip]
        for clip in pool
        if str(by_clip[clip].style) != str(target.style) and not _leaks(by_clip[clip], target)
    ]
    if not options:
        return None
    for predicate in (
        lambda record: str(record.content) == str(target.content)
        and str(record.performer or "") == str(target.performer or ""),
        lambda record: str(record.content) == str(target.content),
        lambda record: str(record.performer or "") == str(target.performer or ""),
        lambda record: True,
    ):
        matching = sorted(
            (record for record in options if predicate(record)), key=lambda record: int(record.clip_id)
        )
        if matching:
            return matching[0]
    return None


def _pick_random(pool: Sequence[int], by_clip: Mapping[int, Any], target: Any, rng) -> Any:
    """A real eligible clip, drawn at random; its labels are recorded as they are."""
    options = [clip for clip in pool if not _leaks(by_clip[clip], target)]
    if not options:
        return None
    return by_clip[int(options[int(rng.integers(len(options)))])]


def write_eval_manifest(
    path: str | Path,
    rows: Sequence[EvalRow],
    *,
    identity: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Writes the manifest (JSON metadata + JSONL rows) with finite values only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": EVAL_MANIFEST_VERSION,
        "identity": dict(identity or {}),
        "protocol": dict(protocol or {}),
        "rows": len(rows),
    }
    _require_json_safe(payload)
    rows_path = path.with_suffix(".jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = row.as_dict()
            _require_json_safe(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    payload["rows_file"] = rows_path.name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _require_json_safe(value: Any) -> None:
    """Rejects NaN/inf: a JSON manifest must not carry a number that is not one."""
    if isinstance(value, Mapping):
        for item in value.values():
            _require_json_safe(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_json_safe(item)
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"The manifest contains a non-finite number: {value!r}")


def read_eval_manifest(path: str | Path) -> tuple[dict[str, Any], list[EvalRow]]:
    """Reads a manifest written by :func:`write_eval_manifest`."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("protocol_version", 0)) != EVAL_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported evaluation manifest version {payload.get('protocol_version')}; "
            f"expected {EVAL_MANIFEST_VERSION}"
        )
    rows_path = path.with_name(str(payload.get("rows_file") or path.with_suffix(".jsonl").name))
    if not rows_path.exists():
        raise FileNotFoundError(f"Evaluation manifest rows are missing: {rows_path}")
    rows = [
        EvalRow.from_dict(json.loads(line))
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(payload.get("rows", len(rows))):
        raise ValueError(
            f"The manifest declares {payload.get('rows')} rows but the file has {len(rows)}"
        )
    return payload, rows


def retrieval_report(
    scores: Sequence[Sequence[float]], positive_indices: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Multi-positive top-1 retrieval, accumulated as hits/count.

    Every candidate that really shares the target's style counts as a positive, so
    two same-style references are both correct.  Ties are broken by the fixed
    candidate order (the first minimum wins), which is stated here rather than left
    to ``argmin``'s implementation.  Aggregation is a sum of hits and counts, so a
    different batch partition gives the same number.
    """
    if len(scores) != len(positive_indices):
        raise ValueError("scores and positive_indices must describe the same targets")
    hits = 0
    counted = 0
    unavailable = 0
    candidate_counts: list[int] = []
    for row_scores, positives in zip(scores, positive_indices):
        if not row_scores or not positives:
            unavailable += 1
            continue
        counted += 1
        candidate_counts.append(len(row_scores))
        best = 0
        for index in range(1, len(row_scores)):
            if float(row_scores[index]) < float(row_scores[best]):
                best = index
        hits += int(best in set(int(index) for index in positives))
    chance = (
        float(
            sum(
                len(set(int(index) for index in positives)) / max(len(row_scores), 1)
                for row_scores, positives in zip(scores, positive_indices)
                if row_scores and positives
            )
        )
        / counted
        if counted
        else None
    )
    return {
        "hits": hits,
        "count": counted,
        "unavailable": unavailable,
        "top1_accuracy": None if counted == 0 else hits / counted,
        "chance": chance,
        "candidates_min": min(candidate_counts) if candidate_counts else 0,
        "candidates_max": max(candidate_counts) if candidate_counts else 0,
    }


__all__ = [
    "content_vocabulary_filter",
    "DEFAULT_BATCHES_PER_KIND",
    "EVAL_PROTOCOL_VERSION",
    "MASK_KINDS",
    "SPLIT_NAMES",
    "EVAL_MANIFEST_VERSION",
    "ClipRef",
    "EvalRow",
    "ValidationBatchBuilder",
    "ValidationProtocol",
    "ValidationSample",
    "build_eval_rows",
    "build_target_only_samples",
    "build_validation_samples",
    "eligible_window_clips",
    "read_eval_manifest",
    "retrieval_report",
    "store_split_of_clip",
    "write_eval_manifest",
]
