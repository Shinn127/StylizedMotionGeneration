"""Tensor, masking and checkpoint contracts for the MTS style operator.

Everything downstream of the tokenizer (transport, operator, sampling, metrics)
agrees on one set of shapes and one set of rules about padding.  Keeping them in
one module is what makes the operator experiments comparable: the flat / part /
NEF runs differ in the layout adapter they are given, never in their tensor
semantics.

Shapes (plan §5.1)::

    tokens          LongTensor  [B, T, 40]       values 0..8
    visible_mask    BoolTensor  [B, T, 40]
    edit_mask       BoolTensor  [B, T, 40]       hard operator support
    valid_mask      BoolTensor  [B, T]           frames that exist at all
    stream_hidden   FloatTensor [B, T, 13, D]
    base_logits     FloatTensor [B, T, 40, 9]
    style_embedding FloatTensor [B, Ds]
    styled_probs    FloatTensor [B, T, 40, 9]

Padding is never disguised as a legal FSQ level: a batch carries an explicit
``valid_mask`` and every cross-entropy, pooling and metric call must apply it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

MTS_CONTRACT_VERSION = 1
#: Order of the mask kinds used by the base transport.  A mixture is a
#: distribution over these names, so the order is part of the contract.
MASK_KINDS: tuple[str, ...] = (
    "random_coordinate",
    "stream",
    "temporal_span",
    "spatiotemporal_block",
    "full_generation",
)


@dataclass(frozen=True)
class TokenSpec:
    """Shape and identity contract of one discrete motion alphabet."""

    num_coordinates: int = 40
    num_levels: int = 9
    num_streams: int = 13
    family: str = "nef_fsq"
    representation_id: str = ""
    layout_hash: str = ""

    def __post_init__(self) -> None:
        if self.num_coordinates <= 0 or self.num_levels <= 1 or self.num_streams <= 0:
            raise ValueError(
                "TokenSpec requires positive coordinates/streams and at least two levels"
            )

    @classmethod
    def from_layout(
        cls,
        layout: Any,
        *,
        family: str = "nef_fsq",
        representation_id: str = "",
    ) -> TokenSpec:
        if not hasattr(layout, "layout_hash"):
            raise TypeError("TokenSpec.from_layout requires a layout with a layout_hash()")
        return cls(
            num_coordinates=int(layout.num_coordinates),
            num_levels=int(getattr(layout, "num_levels", 9)),
            num_streams=len(layout.coordinate_order),
            family=str(family),
            representation_id=str(representation_id),
            layout_hash=str(layout.layout_hash()),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "num_coordinates": self.num_coordinates,
            "num_levels": self.num_levels,
            "num_streams": self.num_streams,
            "family": self.family,
            "representation_id": self.representation_id,
            "layout_hash": self.layout_hash,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- validators --------------------------------------------------------
    def validate_tokens(self, tokens: torch.Tensor, *, name: str = "tokens") -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tokens.ndim != 3 or tokens.shape[-1] != self.num_coordinates:
            raise ValueError(
                f"{name} must be [B, T, {self.num_coordinates}], got {tuple(tokens.shape)}"
            )
        if tokens.dtype not in (torch.long, torch.int64, torch.int32, torch.uint8):
            raise ValueError(f"{name} must be an integer tensor, got {tokens.dtype}")
        if tokens.numel():
            low = int(tokens.min())
            high = int(tokens.max())
            if low < 0 or high >= self.num_levels:
                raise ValueError(
                    f"{name} holds levels outside [0, {self.num_levels - 1}] "
                    f"(min={low}, max={high}); padding must not be encoded as a level"
                )
        return tokens.long()

    def validate_mask(
        self, mask: torch.Tensor, *, name: str, batch: int | None = None, frames: int | None = None
    ) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor")
        if mask.ndim != 3 or mask.shape[-1] != self.num_coordinates:
            raise ValueError(
                f"{name} must be [B, T, {self.num_coordinates}], got {tuple(mask.shape)}"
            )
        if batch is not None and mask.shape[0] != batch:
            raise ValueError(f"{name} has batch {mask.shape[0]}, expected {batch}")
        if frames is not None and mask.shape[1] != frames:
            raise ValueError(f"{name} has {mask.shape[1]} frames, expected {frames}")
        return mask

    def validate_frame_mask(
        self, valid_mask: torch.Tensor, *, batch: int, frames: int
    ) -> torch.Tensor:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean tensor")
        if valid_mask.shape != (int(batch), int(frames)):
            raise ValueError(
                f"valid_mask must be [B, T] = {(int(batch), int(frames))}, "
                f"got {tuple(valid_mask.shape)}"
            )
        return valid_mask

    def validate_logits(
        self, logits: torch.Tensor, *, name: str = "base_logits"
    ) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if logits.ndim != 4 or logits.shape[2:] != (self.num_coordinates, self.num_levels):
            raise ValueError(
                f"{name} must be [B, T, {self.num_coordinates}, {self.num_levels}], "
                f"got {tuple(logits.shape)}"
            )
        if not logits.is_floating_point():
            raise ValueError(f"{name} must be a float tensor, got {logits.dtype}")
        return logits

    def validate_probs(self, probs: torch.Tensor, *, name: str = "styled_probs") -> torch.Tensor:
        self.validate_logits(probs, name=name)
        if probs.numel():
            sums = probs.sum(dim=-1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
                raise ValueError(f"{name} rows must sum to one")
            if bool((probs < 0).any()):
                raise ValueError(f"{name} must be non-negative")
        return probs

    def validate_hidden(self, hidden: torch.Tensor, *, name: str = "stream_hidden") -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if hidden.ndim != 4 or hidden.shape[2] != self.num_streams:
            raise ValueError(
                f"{name} must be [B, T, {self.num_streams}, D], got {tuple(hidden.shape)}"
            )
        if not hidden.is_floating_point():
            raise ValueError(f"{name} must be a float tensor, got {hidden.dtype}")
        return hidden


@dataclass
class TransportOutput:
    """What the style-free base transport returns (plan §5.2)."""

    logits: torch.Tensor  # [B, T, 40, 9]
    stream_hidden: torch.Tensor | None = None  # [B, T, 13, D]
    valid_mask: torch.Tensor | None = None  # [B, T] bool
    spec: TokenSpec | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = self.spec or TokenSpec()
        self.spec = spec
        spec.validate_logits(self.logits, name="TransportOutput.logits")
        if self.stream_hidden is not None:
            spec.validate_hidden(self.stream_hidden)
            if self.stream_hidden.shape[:2] != self.logits.shape[:2]:
                raise ValueError("stream_hidden and logits must share batch and frame axes")
        if self.valid_mask is not None:
            spec.validate_frame_mask(
                self.valid_mask, batch=self.logits.shape[0], frames=self.logits.shape[1]
            )

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.softmax(dim=-1)

    def frame_mask(self, *, device: torch.device | None = None) -> torch.Tensor:
        """The frame mask as a ``[B, T]`` tensor, defaulting to all-valid."""
        if self.valid_mask is not None:
            return self.valid_mask
        return torch.ones(self.logits.shape[:2], dtype=torch.bool, device=device or self.logits.device)


def draw_device(generator: torch.Generator | None, fallback: torch.device) -> torch.device:
    """Device a random draw must happen on.

    ``torch.rand``/``randperm``/``multinomial`` all require the generator and the
    tensors to share a device, while callers reasonably keep a CPU generator for
    reproducibility and put their data on the GPU.  Drawing on the generator's
    own device and moving the *result* keeps both possible.
    """
    return generator.device if generator is not None else fallback


def require_frame_mask(
    valid_mask: torch.Tensor | None, *, batch: int, frames: int, device: torch.device
) -> torch.Tensor:
    """Returns a validated ``[B, T]`` frame mask, defaulting to all-valid."""
    if valid_mask is None:
        return torch.ones((int(batch), int(frames)), dtype=torch.bool, device=device)
    return TokenSpec().validate_frame_mask(valid_mask.to(device).bool(), batch=batch, frames=frames)


def _supervision_mask(
    shape: tuple[int, ...],
    *,
    valid_mask: torch.Tensor | None,
    coordinate_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """The single place the supervised positions are computed.

    ``valid_mask`` [B, T] excludes padding, ``coordinate_mask`` [B, T, K] excludes
    positions the caller is not predicting.  Counts are element counts of this
    mask, never an average fraction, so accumulation across batches stays exact.
    """
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    if valid_mask is not None:
        if valid_mask.shape != shape[:2]:
            raise ValueError(f"valid_mask must be {shape[:2]}, got {tuple(valid_mask.shape)}")
        mask &= valid_mask.to(device).bool().unsqueeze(-1)
    if coordinate_mask is not None:
        if coordinate_mask.shape != shape:
            raise ValueError(f"coordinate_mask must be {shape}, got {tuple(coordinate_mask.shape)}")
        mask &= coordinate_mask.to(device).bool()
    return mask


def _compute_dtype(reference: torch.Tensor) -> torch.dtype:
    """Accumulate in float32, but never lose a float64 input's precision."""
    return torch.float64 if reference.dtype == torch.float64 else torch.float32


def _reduce_selected(nll: torch.Tensor, *, reduction: str, reference: torch.Tensor) -> torch.Tensor:
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        # nll is already restricted to supervised positions.
        return nll.mean() if nll.numel() else reference.new_zeros(())
    raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'sum'")


def _selected_targets(
    targets: torch.Tensor, mask: torch.Tensor, *, num_levels: int, device: torch.device
) -> torch.Tensor:
    """Targets at supervised positions only, range-checked where they count.

    Filtering before the gather is what keeps an invalid target at a padded or
    locked position from raising: only positions the loss actually scores are
    validated.
    """
    chosen = targets.to(device).long()[mask]
    if chosen.numel():
        if int(chosen.min()) < 0 or int(chosen.max()) >= num_levels:
            raise ValueError(
                f"targets at supervised positions must be in [0, {num_levels - 1}], "
                f"got [{int(chosen.min())}, {int(chosen.max())}]"
            )
    return chosen


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    coordinate_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Token cross-entropy over the supervised positions, from **logits**.

    An empty supervision set returns a zero that keeps the autograd graph, so a
    batch without a single masked position cannot corrupt a training step; the
    caller decides whether to skip the step (see the trainers' token counts).
    """
    if logits.ndim != 4:
        raise ValueError(f"logits must be [B, T, K, levels], got {tuple(logits.shape)}")
    if targets.shape != logits.shape[:3]:
        raise ValueError(
            f"targets must be {tuple(logits.shape[:3])}, got {tuple(targets.shape)}"
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'sum'")
    device = logits.device
    mask = _supervision_mask(
        logits.shape[:3], valid_mask=valid_mask, coordinate_mask=coordinate_mask, device=device
    )
    if not bool(mask.any()):
        # A zero that still reaches the graph: a batch with nothing to predict
        # must not break backward for the caller.
        return logits.sum() * 0.0
    chosen = _selected_targets(targets, mask, num_levels=logits.shape[-1], device=device)
    log_probs = F.log_softmax(logits[mask].to(_compute_dtype(logits)), dim=-1)
    nll = -log_probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    return _reduce_selected(nll, reduction=reduction, reference=logits)


def masked_nll_from_probs(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    coordinate_mask: torch.Tensor | None = None,
    reduction: str = "mean",
    clamp_min: float = 1e-12,
) -> torch.Tensor:
    """Negative log-likelihood of the targets **from probabilities**.

    This is the operator-side objective: the CTMC and kernel families emit
    probabilities rather than logits, and feeding those into
    :func:`masked_cross_entropy` silently computed a different quantity (audit:
    1.3804 instead of 0.010050 for p_target = 0.99).  Finite/negative/mass checks
    belong at the operator boundary, not in every call (they would force a
    device sync per batch).
    """
    if probabilities.ndim != 4:
        raise ValueError(
            f"probabilities must be [B, T, K, levels], got {tuple(probabilities.shape)}"
        )
    if targets.shape != probabilities.shape[:3]:
        raise ValueError(
            f"targets must be {tuple(probabilities.shape[:3])}, got {tuple(targets.shape)}"
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction {reduction!r}; expected 'mean' or 'sum'")
    device = probabilities.device
    mask = _supervision_mask(
        probabilities.shape[:3],
        valid_mask=valid_mask,
        coordinate_mask=coordinate_mask,
        device=device,
    )
    if not bool(mask.any()):
        # A zero that still reaches the graph (see masked_cross_entropy).
        return probabilities.sum() * 0.0
    chosen = _selected_targets(
        targets, mask, num_levels=probabilities.shape[-1], device=device
    )
    selected = probabilities[mask].to(_compute_dtype(probabilities))
    picked = selected.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    nll = -picked.clamp_min(float(clamp_min)).log()
    return _reduce_selected(nll, reduction=reduction, reference=probabilities)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over ``mask`` with a zero-safe denominator."""
    weights = mask.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def tokenizer_fingerprint(
    representation_metadata: Mapping[str, object] | None,
    *,
    feature_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The identity fields an MTS model must match to use a tokenizer.

    Only fields that actually change token semantics are included: family,
    variant, representation id, coordinate layout, level count, causal window
    and (for NEF) the layout hash.  Weights are deliberately absent — the
    fingerprint answers "same alphabet", not "same file".
    """
    metadata = dict(representation_metadata or {})
    fingerprint: dict[str, object] = {
        "family": metadata.get("family"),
        "variant": metadata.get("variant"),
        "representation_id": metadata.get("representation_id"),
        "num_coordinates": metadata.get("num_coordinates"),
        "num_levels": metadata.get("num_levels"),
        "coordinate_order": metadata.get("coordinate_order"),
        "coordinate_counts": metadata.get("coordinate_counts"),
        "receptive_field": metadata.get("receptive_field"),
        "lookahead_frames": metadata.get("lookahead_frames"),
        "architecture_version": metadata.get("architecture_version"),
        "nef_layout_hash": metadata.get("nef_layout_hash"),
    }
    schema = dict(feature_schema or {})
    fingerprint["feature_schema"] = {
        key: schema.get(key)
        for key in (
            "name",
            "motion_dim",
            "joint_subset",
            "names_sha256",
            "skeleton_hash",
            "feature_schema_hash",
        )
    }
    return fingerprint


def fingerprint_hash(fingerprint: Mapping[str, object]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_matching_tokenizer(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    label: str = "tokenizer",
) -> None:
    """Raise when two fingerprints disagree on any token-semantic field."""
    mismatched = [
        key
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]
    if mismatched:
        details = {key: (expected.get(key), actual.get(key)) for key in mismatched}
        raise ValueError(f"{label} fingerprint mismatch at {mismatched}: {details}")


def operator_metadata(
    *,
    token_spec: TokenSpec,
    tokenizer_metadata: Mapping[str, object] | None,
    model_config: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Checkpoint metadata for an MTS operator or transport model."""
    tokenizer = tokenizer_fingerprint(tokenizer_metadata)
    metadata: dict[str, object] = {
        "mts_contract_version": MTS_CONTRACT_VERSION,
        "token_spec": token_spec.as_dict(),
        "token_spec_hash": token_spec.fingerprint(),
        "tokenizer": tokenizer,
        "tokenizer_hash": fingerprint_hash(tokenizer),
        "model_config": dict(model_config or {}),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def validate_operator_metadata(
    metadata: Mapping[str, object],
    *,
    token_spec: TokenSpec | None = None,
    tokenizer_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate a stored operator metadata block against the live tokenizer."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Operator checkpoint is missing its MTS metadata block")
    version = int(metadata.get("mts_contract_version", 0))
    if version != MTS_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported MTS contract version {version}; expected {MTS_CONTRACT_VERSION}"
        )
    stored_spec = metadata.get("token_spec")
    if not isinstance(stored_spec, Mapping):
        raise ValueError("Operator checkpoint is missing token_spec")
    if token_spec is not None and dict(stored_spec) != token_spec.as_dict():
        raise ValueError("Operator checkpoint token_spec does not match the current tokenizer")
    if tokenizer_metadata is not None:
        actual = tokenizer_fingerprint(tokenizer_metadata)
        stored = metadata.get("tokenizer")
        if not isinstance(stored, Mapping):
            raise ValueError("Operator checkpoint is missing its tokenizer fingerprint")
        require_matching_tokenizer(dict(stored), actual, label="operator tokenizer")
        if metadata.get("tokenizer_hash") != fingerprint_hash(actual):
            raise ValueError("Operator checkpoint tokenizer hash is inconsistent")
    return dict(metadata)


def normalize_mask_mixture(mixture: Mapping[str, float]) -> dict[str, float]:
    """Validates and normalizes a mask-kind mixture into a distribution."""
    unknown = sorted(set(mixture) - set(MASK_KINDS))
    if unknown:
        raise ValueError(f"Unknown mask kinds {unknown}; expected {list(MASK_KINDS)}")
    values = {key: float(value) for key, value in mixture.items()}
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Mask mixture must have positive total weight")
    return {key: value / total for key, value in values.items()}


__all__ = [
    "MASK_KINDS",
    "MTS_CONTRACT_VERSION",
    "TokenSpec",
    "TransportOutput",
    "fingerprint_hash",
    "masked_cross_entropy",
    "masked_nll_from_probs",
    "masked_mean",
    "draw_device",
    "normalize_mask_mixture",
    "operator_metadata",
    "require_frame_mask",
    "require_matching_tokenizer",
    "tokenizer_fingerprint",
    "validate_operator_metadata",
]
