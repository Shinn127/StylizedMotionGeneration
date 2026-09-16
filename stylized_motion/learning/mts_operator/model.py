"""Top-level MTS model: frozen transport + global style descriptor + operator.

This is the pipeline of plan Phase 4 in one place::

    target tokens + mask + content
            |
    frozen transport -> base logits, H
    reference tokens
            |
    global style encoder -> s
            |
    (base logits, H, s, hard mask, strength)
            |
    operator -> styled probabilities -> masked target-token NLL

The model owns no tokenizer: it consumes the 40x9 alphabet through
:class:`LayoutAdapter` and never writes tokens itself.  The transport and the
style encoder can be frozen or trained, but the tokenizer is always frozen
upstream — that is what makes "the operator changed the motion" a statement about
the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .contract import TokenSpec, masked_cross_entropy
from .layout_adapter import LayoutAdapter
from .operators import OperatorInputs, OperatorOutput, StyleOperator
from .sampling import sample_tokens
from .style_encoder import StyleIDEncoder
from .transport import MotionTransportTransformer


@dataclass
class OperatorBatch:
    """One training or evaluation batch of reference-conditioned edits."""

    target_tokens: torch.Tensor  # [B, T, 40]
    reference_tokens: torch.Tensor | None = None  # [B, Tr, 40]
    visible_mask: torch.Tensor | None = None  # [B, T, 40], defaults to all visible
    hard_mask: torch.Tensor | None = None  # [T, 40] or [B, T, 40]
    strength: torch.Tensor | float = 1.0
    target_valid_mask: torch.Tensor | None = None  # [B, T]
    reference_valid_mask: torch.Tensor | None = None  # [B, Tr]
    content_condition: Any | None = None
    style_ids: torch.Tensor | None = None  # Phase 3 style-ID sandbox
    kind: str = "unknown"

    def supervision_mask(self, spec: TokenSpec) -> torch.Tensor:
        """Positions the loss scores: hidden inside the support."""
        target = spec.validate_tokens(self.target_tokens)
        if self.visible_mask is None:
            visible = torch.ones_like(target, dtype=torch.bool)
        else:
            visible = spec.validate_mask(
                self.visible_mask.to(target.device).bool(),
                name="visible_mask",
                batch=target.shape[0],
                frames=target.shape[1],
            )
        supervision = ~visible
        if self.hard_mask is not None:
            support = self.hard_mask.to(target.device).bool()
            if support.ndim == 2:
                support = support.unsqueeze(0)
            supervision = supervision & support
        return supervision


@dataclass
class OperatorResult:
    probabilities: torch.Tensor  # [B, T, 40, 9]
    base_probabilities: torch.Tensor  # [B, T, 40, 9]
    style_embedding: torch.Tensor  # [B, Ds]
    logits: torch.Tensor | None = None
    rates: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


class MtsStyleOperator(nn.Module):
    """Reference-conditioned style operator over the NEF token alphabet."""

    def __init__(
        self,
        adapter: LayoutAdapter,
        *,
        transport: MotionTransportTransformer,
        style_encoder: nn.Module,
        operator: StyleOperator,
        freeze_transport: bool = True,
        freeze_style_encoder: bool = False,
        strength: float = 1.0,
    ) -> None:
        super().__init__()
        if transport.spec.as_dict() != adapter.token_spec().as_dict():
            raise ValueError("transport and layout adapter disagree about the token alphabet")
        if operator.num_levels != adapter.num_levels:
            raise ValueError(
                f"operator has {operator.num_levels} levels but the tokenizer has {adapter.num_levels}"
            )
        self.adapter = adapter
        self.spec = adapter.token_spec()
        self.transport = transport
        self.style_encoder = style_encoder
        self.operator = operator
        self.freeze_transport = bool(freeze_transport)
        self.freeze_style_encoder = bool(freeze_style_encoder)
        self.default_strength = float(strength)
        if self.freeze_transport:
            for parameter in self.transport.parameters():
                parameter.requires_grad_(False)
            self.transport.eval()
        if self.freeze_style_encoder:
            for parameter in self.style_encoder.parameters():
                parameter.requires_grad_(False)
            self.style_encoder.eval()

    # -- helpers -----------------------------------------------------------
    @property
    def uses_style_ids(self) -> bool:
        return isinstance(self.style_encoder, StyleIDEncoder)

    def _reference_embedding(
        self, batch: OperatorBatch
    ) -> torch.Tensor:
        if self.uses_style_ids:
            if batch.style_ids is None:
                raise ValueError("This style encoder needs batch.style_ids")
            return self.style_encoder(batch.style_ids)
        if batch.reference_tokens is None:
            raise ValueError("Reference-conditioned training needs batch.reference_tokens")
        if self.freeze_style_encoder:
            with torch.no_grad():
                return self.style_encoder(
                    batch.reference_tokens, valid_mask=batch.reference_valid_mask
                )
        return self.style_encoder(batch.reference_tokens, valid_mask=batch.reference_valid_mask)

    def _base(self, batch: OperatorBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the (possibly frozen) transport and returns logits and hidden."""
        tokens = self.spec.validate_tokens(batch.target_tokens)
        visible = (
            self.spec.validate_mask(
                batch.visible_mask.to(tokens.device).bool(),
                name="visible_mask",
                batch=tokens.shape[0],
                frames=tokens.shape[1],
            )
            if batch.visible_mask is not None
            else torch.ones_like(tokens, dtype=torch.bool)
        )
        call = self.transport.__call__
        if self.freeze_transport:
            with torch.no_grad():
                output = call(
                    tokens,
                    visible,
                    content_condition=batch.content_condition,
                    valid_mask=batch.target_valid_mask,
                )
        else:
            output = call(
                tokens,
                visible,
                content_condition=batch.content_condition,
                valid_mask=batch.target_valid_mask,
            )
        hidden = output.stream_hidden
        if hidden is None:
            raise ValueError("The transport must return stream hidden states for the operator")
        return output.logits, hidden

    def operator_inputs(
        self, batch: OperatorBatch, *, style_embedding: torch.Tensor | None = None
    ) -> OperatorInputs:
        logits, hidden = self._base(batch)
        embedding = (
            self._reference_embedding(batch) if style_embedding is None else style_embedding
        )
        return OperatorInputs(
            base_logits=logits,
            style_embedding=embedding,
            strength=batch.strength,
            hard_mask=batch.hard_mask,
            # The operator edits exactly the positions the loss scores: the
            # observed tokens stay evidence.
            edit_mask=batch.supervision_mask(self.spec),
            visible_mask=batch.visible_mask,
            stream_hidden=hidden if self.operator.stream_dim else None,
            valid_mask=batch.target_valid_mask,
            coordinate_stream_ids=self.adapter.coordinate_stream_ids(device=logits.device),
        )

    # -- forward -----------------------------------------------------------
    def forward(self, batch: OperatorBatch) -> OperatorResult:
        inputs = self.operator_inputs(batch)
        output: OperatorOutput = self.operator(inputs)
        return OperatorResult(
            probabilities=output.probabilities,
            base_probabilities=inputs.base_probabilities(),
            style_embedding=inputs.style_embedding,
            logits=output.logits,
            rates=output.rates,
            diagnostics=dict(output.diagnostics),
        )

    def loss(
        self,
        batch: OperatorBatch,
        *,
        content_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Masked target NLL plus an optional content-preservation term."""
        target = self.spec.validate_tokens(batch.target_tokens)
        result = self(batch)
        supervision = batch.supervision_mask(self.spec).to(target.device)
        if not bool(supervision.any()):
            raise ValueError(
                "The batch supervises nothing: every masked position is outside the support"
            )
        loss = masked_cross_entropy(
            result.probabilities, target, valid_mask=batch.target_valid_mask,
            coordinate_mask=supervision,
        )
        metrics: dict[str, torch.Tensor] = {
            "nll": loss.detach(),
            "supervision_fraction": supervision.to(torch.float32).mean().detach(),
        }
        if content_weight > 0.0:
            outside = ~supervision
            if bool(outside.any()):
                content = masked_cross_entropy(
                    result.base_probabilities, target,
                    valid_mask=batch.target_valid_mask, coordinate_mask=outside,
                )
                metrics["base_nll"] = content.detach()
                loss = loss + float(content_weight) * content
        for key, value in result.diagnostics.items():
            metrics[f"operator/{key}"] = value.detach() if isinstance(value, torch.Tensor) else value
        return loss, metrics

    # -- generation --------------------------------------------------------
    @torch.no_grad()
    def generate_edit(
        self,
        batch: OperatorBatch,
        *,
        sampler: Any | None = None,
        generator: torch.Generator | None = None,
        crn: Any | None = None,
        crn_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """Samples the styled distribution inside the support, locking the rest.

        Positions outside ``hard_mask`` are copied from ``target_tokens`` and are
        never resampled, which is the plan's ``locked_edit`` mode.
        """
        target = self.spec.validate_tokens(batch.target_tokens)
        result = self(batch)
        probabilities = result.probabilities.to(target.device)
        if crn is not None:
            # Coupled draw: identical uniforms across conditions.
            drawn = crn.sample(probabilities)
        else:
            drawn = sample_tokens(probabilities, generator=generator)
        if batch.hard_mask is not None:
            support = batch.hard_mask.to(target.device).bool()
            if support.ndim == 2:
                support = support.unsqueeze(0)
            # ``drawn`` and ``support`` are both indexed by (frame, coordinate).
            drawn = torch.where(support, drawn, target)
        if crn_shape is not None and tuple(crn_shape) != tuple(drawn.shape):
            raise ValueError(
                f"crn_shape {tuple(crn_shape)} does not match the token shape {tuple(drawn.shape)}"
            )
        return drawn

    def trainable_parameters(self) -> list[nn.Parameter]:
        for parameter in self.operator.parameters():
            yield parameter
        if not self.freeze_style_encoder:
            for parameter in self.style_encoder.parameters():
                yield parameter
        if not self.freeze_transport:
            for parameter in self.transport.parameters():
                yield parameter

    def describe(self) -> dict[str, Any]:
        return {
            "token_spec": self.spec.as_dict(),
            "operator": self.operator.describe(),
            "transport": self.transport.config(),
            "style_encoder": self.style_encoder.config()
            if hasattr(self.style_encoder, "config")
            else {"kind": type(self.style_encoder).__name__},
            "freeze_transport": self.freeze_transport,
            "freeze_style_encoder": self.freeze_style_encoder,
            "strength": self.default_strength,
            "uses_style_ids": self.uses_style_ids,
        }


__all__ = ["MtsStyleOperator", "OperatorBatch", "OperatorResult"]
