"""Style operators acting on FSQ level probabilities.

Three families, in the order the plan wants them compared (plan §6):

``logit_field``
    Additive logit field.  The baseline: zero-initialized, so ``strength=0`` is
    exactly the base distribution.
``arbitrary_kernel``
    A learned row-stochastic kernel applied to the base probabilities.  It has
    the most freedom and none of the structure — no generator, no semigroup, and
    therefore no identity anchor at ``strength=0``, which is exactly why it is
    the control condition rather than the proposal.
``birth_death``
    A birth-death CTMC on the level axis.  Rates live only on adjacent levels,
    the generator is built from them, and the styled distribution is
    ``exp(Q) p0`` computed by adaptive uniformization.

Every operator shares one support rule (plan §6.4)::

    effective = strength * hard_mask * visibility

where ``visibility`` can only *reduce* an already-visible position and never
escape ``hard_mask``.  ``hard_mask == 0`` therefore implies ``Q == 0`` and
``styled == base`` for every family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

OPERATOR_NAMES = ("logit_field", "arbitrary_kernel", "birth_death")


@dataclass
class OperatorInputs:
    """Everything an operator may look at, all of it already localized."""

    base_logits: torch.Tensor  # [B, T, 40, L]
    style_embedding: torch.Tensor  # [B, Ds]
    strength: torch.Tensor | float = 1.0
    hard_mask: torch.Tensor | None = None  # [T, 40] or [B, T, 40] bool
    #: Positions the operator may edit (the plan's ``visibility``).  When it is
    #: omitted, the complement of ``visible_mask`` is used: a token the transport
    #: was given is evidence, not an edit target.
    edit_mask: torch.Tensor | None = None  # [B, T, 40] bool
    visible_mask: torch.Tensor | None = None  # [B, T, 40] bool, observed tokens
    stream_hidden: torch.Tensor | None = None  # [B, T, 13, D]
    valid_mask: torch.Tensor | None = None  # [B, T] bool
    coordinate_stream_ids: torch.Tensor | None = None  # [40]

    def __post_init__(self) -> None:
        if self.base_logits.ndim != 4:
            raise ValueError(
                f"base_logits must be [B, T, 40, L], got {tuple(self.base_logits.shape)}"
            )
        if self.style_embedding.ndim != 2:
            raise ValueError(
                f"style_embedding must be [B, Ds], got {tuple(self.style_embedding.shape)}"
            )
        if self.style_embedding.shape[0] != self.base_logits.shape[0]:
            raise ValueError("style_embedding batch must match base_logits")
        for name, mask in (
            ("hard_mask", self.hard_mask),
            ("visible_mask", self.visible_mask),
            ("edit_mask", self.edit_mask),
        ):
            if mask is None:
                continue
            if mask.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean tensor")
            if mask.shape not in (
                self.base_logits.shape[:3],
                self.base_logits.shape[1:3],
            ):
                raise ValueError(
                    f"{name} must be [B, T, 40] or [T, 40], got {tuple(mask.shape)}"
                )
        if self.stream_hidden is not None and self.stream_hidden.ndim != 4:
            raise ValueError(
                f"stream_hidden must be [B, T, S, D], got {tuple(self.stream_hidden.shape)}"
            )
        if self.valid_mask is not None and self.valid_mask.shape != self.base_logits.shape[:2]:
            raise ValueError(
                f"valid_mask must be [B, T] = {tuple(self.base_logits.shape[:2])}, "
                f"got {tuple(self.valid_mask.shape)}"
            )

    @property
    def batch(self) -> int:
        return int(self.base_logits.shape[0])

    @property
    def frames(self) -> int:
        return int(self.base_logits.shape[1])

    @property
    def coordinates(self) -> int:
        return int(self.base_logits.shape[2])

    @property
    def num_levels(self) -> int:
        return int(self.base_logits.shape[3])

    def base_probabilities(self) -> torch.Tensor:
        return self.base_logits.softmax(dim=-1)


@dataclass
class OperatorOutput:
    probabilities: torch.Tensor  # [B, T, 40, L]
    logits: torch.Tensor | None = None
    rates: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 4:
            raise ValueError("operator probabilities must be [B, T, 40, L]")
        if bool((self.probabilities < 0).any()):
            raise ValueError("operator returned negative probabilities")
        if bool((self.probabilities.sum(dim=-1) - 1.0).abs().max() > 1e-4):
            raise ValueError("operator probabilities do not sum to one")


class StyleOperator(nn.Module):
    """Shared support arithmetic and context assembly."""

    name = "operator"

    def __init__(self, *, num_levels: int = 9, coordinate_dim: int = 16,
                 hidden_dim: int = 64, style_dim: int | None = None,
                 stream_dim: int = 0) -> None:
        super().__init__()
        if int(num_levels) <= 1:
            raise ValueError("num_levels must be at least two")
        if int(stream_dim) < 0:
            raise ValueError("stream_dim must be non-negative")
        self.num_levels = int(num_levels)
        self.hidden_dim = int(hidden_dim)
        # Width of the transport's per-stream hidden state; 0 means the operator
        # works from the base logits and style vector alone.
        self.stream_dim = int(stream_dim)
        self.coordinate_embedding = nn.Embedding(40, int(coordinate_dim))
        nn.init.normal_(self.coordinate_embedding.weight, std=0.02)
        if style_dim is not None:
            self.style_projection = nn.Linear(int(style_dim), self.hidden_dim)
        else:
            self.style_projection = None

    # -- shared arithmetic -------------------------------------------------
    def _strength_tensor(self, inputs: OperatorInputs) -> torch.Tensor:
        """``[B, 1, 1]`` strength, validated non-negative."""
        value = inputs.strength
        if isinstance(value, torch.Tensor):
            tensor = value.to(inputs.base_logits.device).float()
            if tensor.ndim == 0:
                tensor = tensor.expand(inputs.batch)
            if tensor.ndim != 1 or tensor.shape[0] != inputs.batch:
                raise ValueError(f"strength must be a scalar or [B], got {tuple(tensor.shape)}")
            if bool((tensor < 0).any()):
                raise ValueError("strength must be non-negative")
            return tensor.view(inputs.batch, 1, 1)
        scalar = float(value)
        if scalar < 0.0:
            raise ValueError("strength must be non-negative")
        return inputs.base_logits.new_full((inputs.batch, 1, 1), scalar)

    def editability(self, inputs: OperatorInputs) -> torch.Tensor:
        """``[B, T, 40]`` bool: positions the operator is allowed to edit.

        ``edit_mask`` states it outright; otherwise the complement of
        ``visible_mask`` is used (the tokens the transport had to predict);
        otherwise everything is editable.
        """
        if inputs.edit_mask is not None:
            return inputs.edit_mask.to(inputs.base_logits.device).bool()
        if inputs.visible_mask is not None:
            return ~inputs.visible_mask.to(inputs.base_logits.device).bool()
        return torch.ones(
            (inputs.batch, inputs.frames, inputs.coordinates),
            dtype=torch.bool,
            device=inputs.base_logits.device,
        )

    def support(self, inputs: OperatorInputs) -> torch.Tensor:
        """``[B, T, 40]`` float support: strength x hard mask x editability."""
        strength = self._strength_tensor(inputs)
        allowed = self.editability(inputs)
        support = allowed.to(inputs.base_logits.dtype)
        hard = inputs.hard_mask
        if hard is not None:
            hard_tensor = hard.to(inputs.base_logits.device).bool()
            if hard_tensor.ndim == 2:
                hard_tensor = hard_tensor.unsqueeze(0)
            # The hard region can only narrow the edit set, never widen it.
            support = support * hard_tensor.to(support.dtype)
        if inputs.valid_mask is not None:
            valid = inputs.valid_mask.to(inputs.base_logits.device).bool().unsqueeze(-1)
            support = support * valid.to(support.dtype)
        return support * strength

    @property
    def context_width(self) -> int:
        """Feature width :meth:`condition` produces."""
        return self.stream_dim + self.coordinate_embedding.embedding_dim + self.hidden_dim

    def coordinate_context(self, inputs: OperatorInputs) -> torch.Tensor:
        """Per-coordinate context: the owning stream's hidden state plus identity."""
        parts: list[torch.Tensor] = []
        if inputs.stream_hidden is not None:
            hidden = inputs.stream_hidden
            if int(hidden.shape[-1]) != self.stream_dim:
                raise ValueError(
                    f"stream_hidden has width {hidden.shape[-1]}, but this operator was "
                    f"built with stream_dim={self.stream_dim}"
                )
            if hidden.shape[0] != inputs.batch or hidden.shape[1] != inputs.frames:
                raise ValueError("stream_hidden must share batch and frame axes with base_logits")
            stream_ids = inputs.coordinate_stream_ids
            if stream_ids is None:
                raise ValueError("stream_hidden requires coordinate_stream_ids")
            stream_ids = stream_ids.to(hidden.device).long()
            gathered = hidden.index_select(2, stream_ids)  # [B, T, 40, D]
            parts.append(gathered)
        identity = self.coordinate_embedding.weight.view(
            1, 1, self.coordinate_embedding.num_embeddings, -1
        ).expand(inputs.batch, inputs.frames, -1, -1)
        parts.append(identity.to(inputs.base_logits.dtype))
        return torch.cat(parts, dim=-1)

    def _style_context(self, inputs: OperatorInputs) -> torch.Tensor:
        """``[B, T, 40, hidden]`` style broadcast to every coordinate."""
        embedding = inputs.style_embedding
        if self.style_projection is not None:
            projected = self.style_projection(embedding)
        else:
            projected = embedding
            if projected.shape[-1] != self.hidden_dim:
                raise ValueError(
                    "style_embedding width must equal hidden_dim when style_dim is unset"
                )
        return projected.view(inputs.batch, 1, 1, -1).expand(
            inputs.batch, inputs.frames, inputs.coordinates, -1
        )

    def condition(self, inputs: OperatorInputs) -> torch.Tensor:
        """Operator context: ``[B, T, 40, C + hidden]``."""
        return torch.cat(
            (self.coordinate_context(inputs), self._style_context(inputs)), dim=-1
        )

    def _mlp(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, out_features),
        )

    def _identity_probabilities(self, inputs: OperatorInputs) -> torch.Tensor:
        return inputs.base_probabilities()

    # -- interface ---------------------------------------------------------
    def forward(self, inputs: OperatorInputs) -> OperatorOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """Constructor arguments, so a checkpoint can rebuild this operator.

        This is what a checkpoint replays; ``describe()`` adds the read-only
        facts (level count, context width, structural flags) that are derived.
        """
        payload: dict[str, Any] = {
            "hidden_dim": self.hidden_dim,
            "coordinate_dim": int(self.coordinate_embedding.embedding_dim),
            "style_dim": None
            if self.style_projection is None
            else int(self.style_projection.in_features),
            "stream_dim": self.stream_dim,
        }
        for name in ("identity_mix", "max_rate", "uniformization_tolerance", "max_terms"):
            if hasattr(self, name):
                payload[name] = getattr(self, name)
        return payload

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config(),
            "num_levels": self.num_levels,
            "hidden_dim": self.hidden_dim,
            "stream_dim": self.stream_dim,
            "context_width": self.context_width,
            "has_identity_at_zero": self.HAS_IDENTITY_AT_ZERO,
            "structured": self.STRUCTURED,
        }

    HAS_IDENTITY_AT_ZERO = False
    STRUCTURED = False


class AdditiveLogitField(StyleOperator):
    """``styled = softmax(logits + strength * mask * delta(context, style))``."""

    name = "logit_field"
    HAS_IDENTITY_AT_ZERO = True
    STRUCTURED = False

    def __init__(self, *, num_levels: int = 9, coordinate_dim: int = 16,
                 hidden_dim: int = 64, style_dim: int | None = None,
                 stream_dim: int = 0) -> None:
        super().__init__(
            num_levels=num_levels,
            coordinate_dim=coordinate_dim,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            stream_dim=stream_dim,
        )
        # The delta head is zero-initialized: strength=0 is exactly the base
        # distribution even before any training.
        self.delta_head = self._mlp(self.context_width, self.num_levels)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, inputs: OperatorInputs) -> OperatorOutput:
        support = self.support(inputs)
        delta = self.delta_head(self.condition(inputs))
        styled_logits = inputs.base_logits + support.unsqueeze(-1) * delta
        return OperatorOutput(
            probabilities=styled_logits.softmax(dim=-1),
            logits=styled_logits,
            diagnostics={
                "support_fraction": support.mean(),
                "delta_abs_mean": delta.abs().mean(),
            },
        )


class ArbitraryKernelOperator(StyleOperator):
    """``styled = base @ kernel`` with a learned row-stochastic kernel.

    Deliberately *without* an identity anchor: at ``strength=0`` the kernel
    logits vanish, the kernel becomes uniform and the styled distribution is
    uniform rather than the base.  That is the control condition — an
    unconstrained operator family that has to learn identity from data.
    """

    name = "arbitrary_kernel"
    HAS_IDENTITY_AT_ZERO = False
    STRUCTURED = False

    def __init__(self, *, num_levels: int = 9, coordinate_dim: int = 16,
                 hidden_dim: int = 64, style_dim: int | None = None,
                 stream_dim: int = 0, identity_mix: bool = False) -> None:
        super().__init__(
            num_levels=num_levels,
            coordinate_dim=coordinate_dim,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            stream_dim=stream_dim,
        )
        self.identity_mix = bool(identity_mix)
        self.kernel_head = self._mlp(self.context_width, self.num_levels * self.num_levels)
        nn.init.zeros_(self.kernel_head[-1].weight)
        nn.init.zeros_(self.kernel_head[-1].bias)

    def forward(self, inputs: OperatorInputs) -> OperatorOutput:
        support = self.support(inputs)
        levels = self.num_levels
        kernel_logits = self.kernel_head(self.condition(inputs)).reshape(
            inputs.batch, inputs.frames, inputs.coordinates, levels, levels
        )
        # strength scales how far the kernel may move away from uniform.
        kernel = (kernel_logits * support.unsqueeze(-1).unsqueeze(-1)).softmax(dim=-1)
        if self.identity_mix:
            identity = torch.eye(levels, device=kernel.device, dtype=kernel.dtype)
            kernel = (1.0 - support).unsqueeze(-1).unsqueeze(-1) * identity + support.unsqueeze(
                -1
            ).unsqueeze(-1) * kernel
        base = inputs.base_probabilities()
        styled = torch.einsum("btki,btkij->btkj", base, kernel)
        return OperatorOutput(
            probabilities=styled,
            diagnostics={
                "support_fraction": support.mean(),
                "kernel_offdiagonal_mass": (
                    kernel - kernel.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
                )
                .clamp_min(0.0)
                .sum(dim=(-1, -2))
                .mean(),
            },
        )


def birth_death_generator(
    up_rate: torch.Tensor, down_rate: torch.Tensor
) -> torch.Tensor:
    """Builds ``Q`` from adjacent-level rates.

    ``up_rate[..., i]`` is the ``i -> i+1`` rate for ``i < L-1`` (the last entry
    is ignored) and ``down_rate[..., i]`` the ``i -> i-1`` rate for ``i > 0``.
    """
    if up_rate.shape != down_rate.shape or up_rate.ndim < 1:
        raise ValueError("up_rate and down_rate must share a shape")
    levels = int(up_rate.shape[-1])
    generator = torch.zeros(
        (*up_rate.shape[:-1], levels, levels),
        dtype=up_rate.dtype,
        device=up_rate.device,
    )
    rows = torch.arange(levels - 1, device=up_rate.device)
    generator[..., rows, rows + 1] = up_rate[..., : levels - 1]
    generator[..., rows + 1, rows] = down_rate[..., 1:]
    off_diagonal = generator.sum(dim=-1)
    generator[..., torch.arange(levels, device=up_rate.device), torch.arange(levels, device=up_rate.device)] = (
        -off_diagonal
    )
    return generator


def poisson_term_count(nu: float, *, tolerance: float, max_terms: int = 256) -> tuple[int, float]:
    """Smallest term count whose Poisson tail falls below ``tolerance``."""
    if nu <= 0.0:
        return 1, 0.0
    total = math.exp(-nu)
    terms = 1
    log_factorial = 0.0
    log_nu = math.log(nu)
    while terms < max_terms:
        log_factorial += math.log(terms)
        total += math.exp(-nu + terms * log_nu - log_factorial)
        terms += 1
        if 1.0 - total <= tolerance:
            break
    return terms, max(0.0, 1.0 - total)


def uniformization_expm_apply(
    generator: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    tolerance: float = 1e-10,
    max_terms: int = 256,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Applies the CTMC one unit of time: ``exp(Q)\u1d40 p``.

    ``Q`` is ``[..., L, L]`` with **row sums zero** (``Q[i, j]`` is the rate
    ``i -> j``, the convention :func:`birth_death_generator` builds).  A
    distribution over levels then evolves as ``p -> p\u1d40 exp(Q)``, so the operator
    applies the transpose; using ``exp(Q) p`` would leak probability mass, which
    the ``matrix_exp`` reference check catches.

    Uniformization keeps the row scale per element and chooses the term count
    from the Poisson tail.  A zero row scale means ``Q = 0`` and the input is
    returned unchanged, which is what keeps a fully masked token exactly equal
    to its base distribution.
    """
    if generator.shape[:-2] != probabilities.shape[:-1]:
        raise ValueError("generator and probabilities must share leading shape")
    levels = probabilities.shape[-1]
    if generator.shape[-2:] != (levels, levels):
        raise ValueError("generator must be square over the level axis")
    # p -> p^T exp(Q) == exp(Q^T) p: transpose once so the series acts on the
    # column convention this operator uses.
    transposed = generator.transpose(-1, -2)
    diagonal = transposed.diagonal(dim1=-2, dim2=-1)
    nu = (-diagonal).clamp_min(0.0).amax(dim=-1)  # [...]
    terms, tail = poisson_term_count(
        float(nu.detach().max()), tolerance=tolerance, max_terms=max_terms
    )
    active = nu > 0.0
    scale = torch.where(active, nu, torch.ones_like(nu))
    # Poisson(k; nu) per element: the series is elementwise, so a shared weight
    # vector would be wrong for every element whose row scale is not the max.
    log_nu = torch.log(scale)
    log_weights = []
    log_factorial = 0.0
    for index in range(terms):
        if index > 0:
            log_factorial += math.log(index)
        log_weights.append(-nu + index * log_nu - log_factorial)
    weights = torch.stack(log_weights, dim=-1).exp().unsqueeze(-1)  # [..., terms, 1]

    step = probabilities
    accumulated = weights[..., 0, :] * probabilities
    for index in range(1, terms):
        step = step + (transposed @ step.unsqueeze(-1)).squeeze(-1) / scale.unsqueeze(-1)
        accumulated = accumulated + weights[..., index, :] * step
    if not bool(active.any()):
        # Q is exactly zero here (fully masked or strength 0): the base
        # distribution is returned bit for bit.
        return probabilities.clone(), {
            "uniformization_terms": generator.new_tensor(0.0),
            "poisson_tail": generator.new_tensor(0.0),
            "mass_error": generator.new_tensor(0.0),
            "min_probability_before_clamp": generator.new_tensor(float(probabilities.min())),
        }
    # A truncated series can leave a small negative or a mass error; both are
    # reported instead of claiming an exact semigroup application.
    mass_error = (accumulated.sum(dim=-1) - 1.0).abs().amax()
    negative_before = float(accumulated.detach().min())
    clamped = accumulated.clamp_min(0.0)
    normalized = clamped / clamped.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # Only rows that actually moved are renormalized; untouched rows stay exact.
    accumulated = torch.where(active.unsqueeze(-1), normalized, probabilities)
    return accumulated, {
        "uniformization_terms": generator.new_tensor(float(terms)),
        "poisson_tail": generator.new_tensor(float(tail)),
        "mass_error": mass_error.detach(),
        "min_probability_before_clamp": generator.new_tensor(negative_before),
    }


class BirthDeathCTMCOperator(StyleOperator):
    """Style injection as a birth-death CTMC on the FSQ level axis."""

    name = "birth_death"
    HAS_IDENTITY_AT_ZERO = True
    STRUCTURED = True

    def __init__(self, *, num_levels: int = 9, coordinate_dim: int = 16,
                 hidden_dim: int = 64, style_dim: int | None = None,
                 stream_dim: int = 0, max_rate: float = 2.0,
                 uniformization_tolerance: float = 1e-10,
                 max_terms: int = 256) -> None:
        super().__init__(
            num_levels=num_levels,
            coordinate_dim=coordinate_dim,
            hidden_dim=hidden_dim,
            style_dim=style_dim,
            stream_dim=stream_dim,
        )
        if max_rate <= 0.0:
            raise ValueError("max_rate must be positive")
        self.max_rate = float(max_rate)
        self.uniformization_tolerance = float(uniformization_tolerance)
        self.max_terms = int(max_terms)
        # Two rate heads: up (L-1 useful entries) and down (L-1 useful entries).
        self.rate_head = self._mlp(self.context_width, 2 * self.num_levels)
        nn.init.zeros_(self.rate_head[-1].weight)
        nn.init.zeros_(self.rate_head[-1].bias)

    def rates(self, inputs: OperatorInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(up_rate, down_rate, support)`` with rates already masked."""
        support = self.support(inputs)
        raw = self.rate_head(self.condition(inputs))
        up = raw[..., : self.num_levels]
        down = raw[..., self.num_levels :]
        # Bounded rates: the sigmoid keeps them finite and the support scales
        # them to exactly zero outside the edit region.
        up_rate = self.max_rate * torch.sigmoid(up) * support.unsqueeze(-1)
        down_rate = self.max_rate * torch.sigmoid(down) * support.unsqueeze(-1)
        return up_rate, down_rate, support

    def forward(self, inputs: OperatorInputs) -> OperatorOutput:
        up_rate, down_rate, support = self.rates(inputs)
        generator = birth_death_generator(up_rate, down_rate)
        styled, diagnostics = uniformization_expm_apply(
            generator,
            inputs.base_probabilities(),
            tolerance=self.uniformization_tolerance,
            max_terms=self.max_terms,
        )
        diagnostics["support_fraction"] = support.mean()
        diagnostics["max_up_rate"] = up_rate.max()
        diagnostics["max_down_rate"] = down_rate.max()
        return OperatorOutput(
            probabilities=styled,
            rates=torch.stack((up_rate, down_rate), dim=-2),
            diagnostics=diagnostics,
        )

    def reference_expm(self, inputs: OperatorInputs) -> torch.Tensor:
        """``torch.matrix_exp`` reference for tests and spot checks."""
        up_rate, down_rate, _ = self.rates(inputs)
        generator = birth_death_generator(up_rate, down_rate)
        applied = inputs.base_probabilities().unsqueeze(-2) @ torch.matrix_exp(generator)
        return applied.squeeze(-2)


def build_operator(name: str, **kwargs: Any) -> StyleOperator:
    if name not in OPERATOR_NAMES:
        raise ValueError(f"Unknown operator {name!r}; expected {list(OPERATOR_NAMES)}")
    classes = {
        "logit_field": AdditiveLogitField,
        "arbitrary_kernel": ArbitraryKernelOperator,
        "birth_death": BirthDeathCTMCOperator,
    }
    return classes[name](**kwargs)


__all__ = [
    "OPERATOR_NAMES",
    "AdditiveLogitField",
    "ArbitraryKernelOperator",
    "BirthDeathCTMCOperator",
    "OperatorInputs",
    "OperatorOutput",
    "StyleOperator",
    "birth_death_generator",
    "build_operator",
    "poisson_term_count",
    "uniformization_expm_apply",
]
