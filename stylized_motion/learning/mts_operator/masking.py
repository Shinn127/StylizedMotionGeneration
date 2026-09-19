"""Masking strategies for the style-free base transport.

The base generator is trained as a masked-content model: a subset of tokens is
hidden, the transport predicts them, and :func:`contract.masked_cross_entropy`
scores only the hidden, valid positions.  The five kinds here are the plan's
list; the mixture decides how often each one is drawn.

Every sampled mask keeps at least one supervised token per sample, so a batch
can never hand the trainer an empty (but finite) loss by accident.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from .contract import MASK_KINDS, TokenSpec, draw_device, normalize_mask_mixture
from .layout_adapter import LayoutAdapter


@dataclass(frozen=True)
class MaskConfig:
    """How often each kind is drawn and how strong each kind is."""

    mixture: Mapping[str, float] = field(
        default_factory=lambda: {
            "random_coordinate": 0.20,
            "stream": 0.25,
            "temporal_span": 0.20,
            "spatiotemporal_block": 0.20,
            "full_generation": 0.15,
        }
    )
    coordinate_ratio: float = 0.30
    stream_ratio: float = 0.30
    span_ratio: float = 0.50
    block_frames: int = 16
    block_coordinates: int = 8

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> MaskConfig:
        """Accepts both ``{"mixture": {...}}`` and the plan's flat kind keys."""
        if not value:
            return cls()
        payload = dict(value)
        mixture: dict[str, float] = dict(payload.pop("mixture", {}) or {})
        for kind in MASK_KINDS:
            if kind in payload:
                mixture[kind] = float(payload.pop(kind))
        options = {
            key: payload.pop(key)
            for key in (
                "coordinate_ratio",
                "stream_ratio",
                "span_ratio",
                "block_frames",
                "block_coordinates",
            )
            if key in payload
        }
        unknown = sorted(payload)
        if unknown:
            raise ValueError(f"Unknown masking options {unknown}")
        config = cls(mixture=mixture if mixture else cls().mixture, **options)  # type: ignore[arg-type]
        config.validate()
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "mixture": dict(self.mixture),
            "coordinate_ratio": self.coordinate_ratio,
            "stream_ratio": self.stream_ratio,
            "span_ratio": self.span_ratio,
            "block_frames": self.block_frames,
            "block_coordinates": self.block_coordinates,
        }

    def normalized_mixture(self) -> dict[str, float]:
        return normalize_mask_mixture(self.mixture)

    def validate(self) -> None:
        normalize_mask_mixture(self.mixture)
        for name, ratio in (
            ("coordinate_ratio", self.coordinate_ratio),
            ("stream_ratio", self.stream_ratio),
            ("span_ratio", self.span_ratio),
        ):
            if not 0.0 < float(ratio) <= 1.0:
                raise ValueError(f"masking.{name} must be in (0, 1], got {ratio}")
        if int(self.block_frames) <= 0 or int(self.block_coordinates) <= 0:
            raise ValueError("masking block sizes must be positive")


@dataclass
class MaskBatch:
    """One sampled mask: ``visible_mask`` says which tokens the model may see."""

    visible_mask: torch.Tensor  # [B, T, 40] bool
    kind: str
    config: MaskConfig | None = None

    def __post_init__(self) -> None:
        if self.visible_mask.dtype != torch.bool or self.visible_mask.ndim != 3:
            raise ValueError("visible_mask must be a boolean [B, T, 40] tensor")
        if self.kind not in MASK_KINDS:
            raise ValueError(f"Unknown mask kind {self.kind!r}")

    @property
    def supervision_mask(self) -> torch.Tensor:
        """Positions the loss is applied to: hidden, not merely absent."""
        return ~self.visible_mask

    def summary(self) -> dict[str, Any]:
        total = int(self.visible_mask.numel())
        hidden = int(self.supervision_mask.sum())
        return {
            "kind": self.kind,
            "hidden_tokens": hidden,
            "hidden_fraction": hidden / max(total, 1),
        }


class MaskGenerator:
    """Samples masks from a mixture, deterministically given a generator."""

    def __init__(self, config: MaskConfig | Mapping[str, object] | None = None) -> None:
        if config is None:
            config = MaskConfig()
        elif isinstance(config, Mapping):
            config = MaskConfig.from_mapping(config)
        self.config = config
        self.config.validate()
        self.mixture = self.config.normalized_mixture()
        self.kinds = tuple(MASK_KINDS)

    def sample(
        self,
        batch: int,
        frames: int,
        *,
        adapter: LayoutAdapter | None = None,
        spec: TokenSpec | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> MaskBatch:
        kind = self._draw_kind(generator)
        return self.sample_kind(
            kind, batch, frames, adapter=adapter, spec=spec, generator=generator, device=device
        )

    def _draw_kind(self, generator: torch.Generator | None) -> str:
        weights = torch.tensor(
            [float(self.mixture.get(kind, 0.0)) for kind in self.kinds], dtype=torch.float32
        )
        index = int(torch.multinomial(weights, 1, generator=generator).item())
        return self.kinds[index]

    def sample_kind(
        self,
        kind: str,
        batch: int,
        frames: int,
        *,
        adapter: LayoutAdapter | None = None,
        spec: TokenSpec | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> MaskBatch:
        if kind not in MASK_KINDS:
            raise ValueError(f"Unknown mask kind {kind!r}; expected {list(MASK_KINDS)}")
        batch, frames = int(batch), int(frames)
        if batch <= 0 or frames <= 0:
            raise ValueError("batch and frames must be positive")
        coordinates = int(spec.num_coordinates) if spec is not None else 40
        stream_ids = None
        if adapter is not None:
            coordinates = adapter.num_coordinates
            stream_ids = adapter.coordinate_stream_ids(device=device)
        if kind in {"stream", "full_generation"} and stream_ids is None and kind == "stream":
            raise ValueError("The stream mask requires a LayoutAdapter for stream ownership")
        visible = self._visible_for(
            kind, batch, frames, coordinates, stream_ids, generator, device
        )
        self._ensure_supervised(visible, generator)
        return MaskBatch(visible_mask=visible, kind=kind, config=self.config)

    # -- per-kind masks ----------------------------------------------------
    def _visible_for(
        self,
        kind: str,
        batch: int,
        frames: int,
        coordinates: int,
        stream_ids: torch.Tensor | None,
        generator: torch.Generator | None,
        device: torch.device | None,
    ) -> torch.Tensor:
        visible = torch.ones((batch, frames, coordinates), dtype=torch.bool, device=device)
        if kind == "full_generation":
            return torch.zeros_like(visible)
        if kind == "random_coordinate":
            draw = draw_device(generator, device or visible.device)
            hidden = (
                torch.rand((batch, frames, coordinates), generator=generator, device=draw)
                .to(visible.device)
                < float(self.config.coordinate_ratio)
            )
            return visible & ~hidden
        if kind == "stream":
            assert stream_ids is not None
            streams = int(stream_ids.max()) + 1
            ratio = float(self.config.stream_ratio)
            per_sample = max(1, int(round(ratio * streams)))
            draw = draw_device(generator, device or visible.device)
            for row in range(batch):
                order = torch.randperm(streams, generator=generator, device=draw).to(visible.device)
                hidden_streams = order[:per_sample]
                hide = (stream_ids.view(1, -1) == hidden_streams.view(-1, 1)).any(dim=0)
                visible[row, :, hide] = False
            return visible
        if kind == "temporal_span":
            span = max(1, int(round(float(self.config.span_ratio) * frames)))
            draw = draw_device(generator, device or visible.device)
            for row in range(batch):
                start = int(torch.randint(0, frames - span + 1, (1,), generator=generator, device=draw).item())
                visible[row, start : start + span] = False
            return visible
        if kind == "spatiotemporal_block":
            block_frames = min(int(self.config.block_frames), frames)
            block_coordinates = min(int(self.config.block_coordinates), coordinates)
            draw = draw_device(generator, device or visible.device)
            for row in range(batch):
                start = int(
                    torch.randint(0, frames - block_frames + 1, (1,), generator=generator, device=draw).item()
                )
                columns = torch.randperm(coordinates, generator=generator, device=draw).to(
                    visible.device
                )[:block_coordinates]
                visible[row, start : start + block_frames, columns] = False
            return visible
        raise ValueError(f"Unknown mask kind {kind!r}")

    @staticmethod
    def _ensure_supervised(visible: torch.Tensor, generator: torch.Generator | None) -> None:
        """Never hand the trainer a sample with nothing to predict."""
        fully_visible = visible.all(dim=(1, 2))
        draw = draw_device(generator, visible.device)
        for row in torch.nonzero(fully_visible).flatten().tolist():
            coordinate = int(
                torch.randint(0, visible.shape[2], (1,), generator=generator, device=draw).item()
            )
            frame = int(
                torch.randint(0, visible.shape[1], (1,), generator=generator, device=draw).item()
            )
            visible[row, frame, coordinate] = False


def apply_hard_support(
    mask: MaskBatch, support: torch.Tensor, *, mode: str = "restrict"
) -> MaskBatch:
    """Combines a sampled mask with a region support ``[T, 40]``.

    ``restrict`` supervises only inside the support (style authoring inside a
    region); ``expand`` hides everything outside it (content must be preserved
    there).  Both keep at least one supervised position per sample, and the
    fallback position is drawn from *the region that mode supervises*: the old
    code always hid a position outside the support, so ``restrict`` could quietly
    supervise a token it had promised not to touch.
    """
    if mode not in {"restrict", "expand"}:
        raise ValueError(f"Unsupported support mode {mode!r}")
    visible = mask.visible_mask
    support = support.to(visible.device).bool()
    if support.shape != visible.shape[1:]:
        raise ValueError(
            f"support must be [T, {visible.shape[-1]}], got {tuple(support.shape)}"
        )
    region = support.view(1, *support.shape)
    visible = (visible | ~region) if mode == "restrict" else (visible & region)
    supervision = ~visible
    allowed = torch.nonzero(support if mode == "restrict" else ~support)
    if allowed.numel() == 0:
        raise ValueError(
            f"A support mask that covers every position cannot supervise anything in {mode!r} mode"
        )
    missing = torch.nonzero(~supervision.any(dim=(1, 2))).flatten().tolist()
    if missing:
        visible = visible.clone()
        pick = allowed[0]
        for row in missing:
            # Hide a position the caller asked to supervise; deterministic so
            # the same mask always yields the same batch.
            visible[row, int(pick[0]), int(pick[1])] = False
    return MaskBatch(visible_mask=visible, kind=mask.kind, config=mask.config)


__all__ = ["MaskBatch", "MaskConfig", "MaskGenerator", "apply_hard_support"]
