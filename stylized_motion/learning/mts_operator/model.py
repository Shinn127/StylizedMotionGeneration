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

from .contract import TokenSpec, masked_nll_from_probs
from .layout_adapter import LayoutAdapter
from .operators import OperatorInputs, OperatorOutput, StyleOperator
from .sampling import monotonic_fill_steps, sample_tokens
from .style_encoder import ConstantStyleEncoder, StyleIDEncoder
from .transport import MotionTransportTransformer


def _broadcast_hard_mask(hard_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``[B, T, 40]`` bool hard mask, accepting ``[T, 40]`` and ``[1, T, 40]``."""
    support = hard_mask.to(target.device).bool()
    if support.ndim == 2:
        support = support.unsqueeze(0)
    if support.ndim != 3 or support.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"hard_mask must be [T, {target.shape[1]}] or [B, T, {target.shape[1]}], "
            f"got {tuple(support.shape)}"
        )
    if support.shape[0] not in (1, target.shape[0]):
        raise ValueError(
            f"hard_mask batch {support.shape[0]} matches neither 1 nor {target.shape[0]}"
        )
    return support.expand_as(target)


@dataclass
class OperatorBatch:
    """One training or evaluation batch of reference-conditioned edits."""

    target_tokens: torch.Tensor  # [B, T, 40]
    reference_tokens: torch.Tensor | None = None  # [B, Tr, 40]
    #: Tokens the model may observe.  Required by the loss path: omitting it used
    #: to mean "everything is visible", which silently supervised nothing.
    visible_mask: torch.Tensor | None = None  # [B, T, 40]
    hard_mask: torch.Tensor | None = None  # [T, 40] or [B, T, 40]
    #: Positions inside the hard mask that stay observed anyway (explicit anchors).
    anchor_mask: torch.Tensor | None = None  # [B, T, 40]
    strength: torch.Tensor | float = 1.0
    target_valid_mask: torch.Tensor | None = None  # [B, T]
    reference_valid_mask: torch.Tensor | None = None  # [B, Tr]
    content_condition: Any | None = None
    style_ids: torch.Tensor | None = None  # Phase 3 style-ID sandbox
    #: The *reference's* style id per row (N05b's auxiliary seen-style CE).  The
    #: target's style id lives in ``style_ids``; this one describes the clip the
    #: descriptor was computed from.
    reference_style_ids: torch.Tensor | None = None
    #: Per-sample provenance (clip id, style, action, actor, ...).  Not a tensor,
    #: so the trainers' device moves leave it alone; evaluation reads it to group
    #: metrics without re-deriving them from array positions.
    sample_metadata: Any | None = None
    kind: str = "unknown"

    def _visible(self, spec: TokenSpec, target: torch.Tensor, *, require_visible: bool) -> torch.Tensor:
        if self.visible_mask is None:
            if require_visible:
                raise ValueError(
                    "OperatorBatch.visible_mask is required: the loss and forward edit "
                    "paths no longer treat a missing mask as 'everything is visible' "
                    "(that supervised nothing). Pass the observed tokens explicitly; a "
                    "full-generation batch states visible_mask=zeros_like(target_tokens)."
                )
            # The generation path states its default unmistakably: everything the
            # operator may edit is unobserved, everything else is evidence.
            visible = (
                ~_broadcast_hard_mask(self.hard_mask, target)
                if self.hard_mask is not None
                else torch.zeros_like(target, dtype=torch.bool)
            )
        else:
            visible = spec.validate_mask(
                self.visible_mask.to(target.device).bool(),
                name="visible_mask",
                batch=target.shape[0],
                frames=target.shape[1],
            )
        if self.anchor_mask is not None:
            anchor = spec.validate_mask(
                self.anchor_mask.to(target.device).bool(),
                name="anchor_mask",
                batch=target.shape[0],
                frames=target.shape[1],
            )
            visible = visible | anchor
        return visible

    def effective_edit_mask(
        self, spec: TokenSpec, *, require_visible: bool = True
    ) -> torch.Tensor:
        """``[B, T, 40]`` bool: the one definition of what the operator may edit.

        ``hard_mask & ~visible & valid``, with anchors forced to stay observed.
        Every entry point (loss, forward, generation) uses this, so a position can
        never be an edit target in one place and evidence in another.
        """
        target = spec.validate_tokens(self.target_tokens)
        edit = ~self._visible(spec, target, require_visible=require_visible)
        if self.hard_mask is not None:
            edit = edit & _broadcast_hard_mask(self.hard_mask, target)
        if self.target_valid_mask is not None:
            valid = self.target_valid_mask.to(target.device).bool()
            if valid.shape != target.shape[:2]:
                raise ValueError(
                    f"target_valid_mask must be {tuple(target.shape[:2])}, got {tuple(valid.shape)}"
                )
            edit = edit & valid.unsqueeze(-1)
        return edit

    def supervision_mask(self, spec: TokenSpec) -> torch.Tensor:
        """Positions the loss scores: the effective edit mask, visible required."""
        return self.effective_edit_mask(spec, require_visible=True)


@dataclass
class OperatorResult:
    probabilities: torch.Tensor  # [B, T, 40, 9]
    base_probabilities: torch.Tensor  # [B, T, 40, 9]
    style_embedding: torch.Tensor  # [B, Ds]
    logits: torch.Tensor | None = None
    rates: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


def _pick(result: "OperatorResult", use_base: bool) -> torch.Tensor:
    """The distribution a draw should come from: styled, or the frozen base."""
    return result.base_probabilities if use_base else result.probabilities


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

    @property
    def uses_constant_descriptor(self) -> bool:
        """True for the no-reference control: the descriptor reads no style input."""
        return isinstance(self.style_encoder, ConstantStyleEncoder)

    def _reference_embedding(
        self, batch: OperatorBatch
    ) -> torch.Tensor:
        if self.uses_constant_descriptor:
            # No reference tokens and no style id: only the batch size is read, so
            # the control cannot depend on the style input in any way.
            return self.style_encoder(int(batch.target_tokens.shape[0]))
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

    def _base(
        self, batch: OperatorBatch, *, require_visible: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the (possibly frozen) transport and returns logits and hidden."""
        tokens = self.spec.validate_tokens(batch.target_tokens)
        visible = batch._visible(self.spec, tokens, require_visible=require_visible)
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
        self,
        batch: OperatorBatch,
        *,
        style_embedding: torch.Tensor | None = None,
        require_visible: bool = True,
    ) -> OperatorInputs:
        logits, hidden = self._base(batch, require_visible=require_visible)
        embedding = (
            self._reference_embedding(batch) if style_embedding is None else style_embedding
        )
        target = self.spec.validate_tokens(batch.target_tokens)
        edit_mask = batch.effective_edit_mask(self.spec, require_visible=require_visible)
        # The transport observes exactly the tokens the operator may not edit, so a
        # locked edit is conditioned on evidence and a generation step is not
        # conditioned on the answers.
        visible = ~edit_mask
        return OperatorInputs(
            base_logits=logits,
            style_embedding=embedding,
            strength=batch.strength,
            hard_mask=batch.hard_mask,
            edit_mask=edit_mask,
            visible_mask=visible,
            stream_hidden=hidden if self.operator.stream_dim else None,
            valid_mask=batch.target_valid_mask,
            coordinate_stream_ids=self.adapter.coordinate_stream_ids(device=logits.device),
        )

    def effective_edit_mask(self, batch: OperatorBatch, *, require_visible: bool = True) -> torch.Tensor:
        """The batch's edit mask under the model's token spec."""
        return batch.effective_edit_mask(self.spec, require_visible=require_visible)

    # -- forward -----------------------------------------------------------
    def forward(self, batch: OperatorBatch, *, require_visible: bool = True) -> OperatorResult:
        inputs = self.operator_inputs(batch, require_visible=require_visible)
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
        """Masked target NLL from the styled probabilities.

        The operator emits probabilities (the CTMC and kernel families have no
        logits at all), so the objective is ``masked_nll_from_probs``; feeding
        probabilities into the logits CE computed a different quantity.  Metrics
        carry ``nll_sum`` and ``supervised_tokens`` so accumulation across batches
        is exact; an all-empty supervision set yields a graph-connected zero and
        ``supervised_tokens = 0`` for the trainer to skip.
        """
        if float(content_weight) != 0.0:
            raise ValueError(
                "content_weight is not implemented in revision 2: the frozen-base content "
                "term was removed because it was measured against the base distribution "
                "rather than against content.  Leave it at 0."
            )
        target = self.spec.validate_tokens(batch.target_tokens)
        result = self(batch)
        supervision = batch.supervision_mask(self.spec).to(target.device)
        valid = batch.target_valid_mask
        loss = masked_nll_from_probs(
            result.probabilities,
            target,
            valid_mask=valid,
            coordinate_mask=supervision,
        )
        nll_sum = masked_nll_from_probs(
            result.probabilities,
            target,
            valid_mask=valid,
            coordinate_mask=supervision,
            reduction="sum",
        )
        with torch.no_grad():
            frame_mask = torch.ones_like(supervision)
            if valid is not None:
                frame_mask = frame_mask & valid.to(target.device).bool().unsqueeze(-1)
            supervised_mask = supervision & frame_mask
            supervised = int(supervised_mask.sum())
            correct_tokens = int(
                ((result.probabilities.argmax(dim=-1) == target) & supervised_mask).sum()
            )
        metrics: dict[str, torch.Tensor] = {
            "nll": loss.detach(),
            "nll_sum": nll_sum.detach(),
            "supervised_tokens": torch.tensor(supervised, device=target.device),
            "correct_tokens": torch.tensor(correct_tokens, device=target.device),
            "supervision_fraction": supervision.to(torch.float32).mean().detach(),
        }
        for key, value in result.diagnostics.items():
            metrics[f"operator/{key}"] = value.detach() if isinstance(value, torch.Tensor) else value
        # The descriptor itself, with grad: the trainer's optional auxiliary style CE
        # reads it from here.  Validation ignores it (the trainer detaches nothing for
        # the NLL path, and no_grad callers never see the graph).
        metrics["style_embedding"] = result.style_embedding
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
        sample_id: int = 0,
        step_id: int = 0,
        steps: int = 1,
        use_base: bool = False,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Samples the styled distribution inside the edit region, locking the rest.

        The region is :meth:`OperatorBatch.effective_edit_mask`; with no explicit
        ``visible_mask`` it defaults to the hard mask, and positions marked by
        ``anchor_mask`` stay observed.  Everything outside the region is copied
        from ``target_tokens`` and never resampled (the plan's ``locked_edit``).

        ``sample_id``/``step_id`` name the CRN cell, so several draws of the same
        batch are independent while the same cell reproduces exactly.  An empty
        region returns the source tokens without drawing at all.

        ``use_base`` draws from the frozen base transport's own distribution
        instead of the styled one.  That -- not ``strength=0`` -- is the base
        reference: for families without an identity anchor the zero-strength
        distribution is uniform or an identity map, so comparing against it would
        compare the operator with itself.

        ``sampler``, when given, is called as ``sampler(probabilities, generator=...)``
        and replaces the built-in inverse-CDF draw; ``crn`` still takes precedence
        so a paired sweep cannot accidentally mix two couplings.
        """
        target = self.spec.validate_tokens(batch.target_tokens)
        edit_mask = batch.effective_edit_mask(self.spec, require_visible=False)
        if not bool(edit_mask.any()):
            return (target.clone(), []) if return_trace else target.clone()
        if int(steps) > 1:
            # Iterative generation: at every step the model is re-run with the tokens
            # committed so far, and the next block of positions is drawn from that
            # updated context.  Every position is committed exactly once (monotonic
            # filling), the observed set only grows, and positions outside the region
            # are never even looked at.
            import dataclasses

            remaining = edit_mask.to(target.device).clone()
            drawn = target.clone()
            commits: list[torch.Tensor] = []
            schedule = monotonic_fill_steps(remaining, int(steps))
            for step, commit in enumerate(schedule):
                if not bool(commit.any()):
                    commits.append(commit)
                    continue
                # Everything already committed (plus the locked outside) is observed;
                # the block being drawn now is *not*.  Adding ``commit`` here made the
                # block visible before it was sampled, so its distribution was the
                # frozen base's (the operator only edits hidden positions) and a
                # multi-step styled draw came back bitwise equal to a multi-step base
                # draw -- a silently unstyled generation.
                visible_now = ~remaining
                step_batch = dataclasses.replace(
                    batch,
                    target_tokens=drawn,
                    visible_mask=visible_now | (~edit_mask.to(target.device)),
                )
                result = self(step_batch, require_visible=False)
                probabilities = _pick(result, use_base).to(target.device)
                step_tokens = sample_tokens(
                    probabilities, generator=generator, crn=crn,
                    sample_id=sample_id, step_id=step,
                )
                drawn = torch.where(commit.to(target.device), step_tokens.to(target.device), drawn)
                remaining = remaining & ~commit
                commits.append(commit)
            if bool(remaining.any()):
                raise RuntimeError("Iterative generation left unfilled positions")
            return (drawn, commits) if return_trace else drawn
        result = self(batch, require_visible=False)
        probabilities = _pick(result, use_base).to(target.device)
        if crn is not None:
            # Coupled draw: identical uniforms across conditions.
            drawn = crn.sample(probabilities, sample_id=sample_id, step_id=step_id)
        elif sampler is not None:
            drawn = sampler(probabilities, generator=generator)
        else:
            drawn = sample_tokens(
                probabilities, generator=generator, sample_id=sample_id, step_id=step_id
            )
        # Everything outside the effective edit region is copied, so an empty
        # region returns the source tokens exactly.
        drawn = torch.where(edit_mask.to(target.device), drawn, target)
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
