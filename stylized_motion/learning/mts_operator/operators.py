"""Style operators acting on FSQ level probabilities.

Three families, in the order the plan wants them compared (plan §6):

``logit_field``
    Additive logit field.  The baseline: zero-initialized, so ``strength=0`` is
    exactly the base distribution.
``arbitrary_kernel``
    A learned row-stochastic kernel applied to the base probabilities.  It has
    the most freedom and none of the structure — no generator, no semigroup, and
    therefore no identity anchor at ``strength=0`` (the kernel becomes uniform
    inside its region), which is exactly why it is the control condition rather
    than the proposal.  ``identity_mix=True`` is the variant where ``strength`` is
    a mixture weight in ``[0, 1]`` instead: ``0`` is the base, ``1`` the learned
    kernel.
``birth_death``
    A birth-death CTMC on the level axis.  Rates live only on adjacent levels,
    the generator is built from them, and the styled distribution is
    ``exp(Q) p0`` computed by adaptive uniformization.

Every operator shares one region rule (plan §1.1)::

    eligible  = hard_mask & editability & valid_mask[..., None]      (bool)
    strength  = how far the operator moves *inside* that region      (float)

``editability`` can only reduce the region and never escapes ``hard_mask``, so
``eligible == 0`` implies ``styled == base`` bit for bit in every family — the
final blend is ``where(eligible, transformed, base)``, not a side effect of
``strength`` being zero.  Keeping the bool region and the float strength apart is
what makes ``lambda = 0`` comparability work: the arbitrary kernel has no
identity anchor, so at ``lambda = 0`` it *is* uniform inside its region and the
base outside it, while the logit field and the CTMC are exactly the base.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

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
        # Finiteness first: a NaN fails every ``<``/``>`` comparison, so the
        # negative and mass checks below would wave it through.
        if not bool(torch.isfinite(self.probabilities).all()):
            raise ValueError("operator probabilities must be finite")
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

    def eligible(self, inputs: OperatorInputs) -> torch.Tensor:
        """``[B, T, 40]`` bool region the operator may act on at all.

        ``hard_mask & editability & valid_mask[..., None]``: the plan's
        ``effective_edit``.  Deliberately independent of ``strength`` — a region
        is a set, ``lambda`` is a magnitude, and conflating them made an empty
        region indistinguishable from ``lambda = 0``.
        """
        eligible = self.editability(inputs)
        hard = inputs.hard_mask
        if hard is not None:
            hard_tensor = hard.to(inputs.base_logits.device).bool()
            if hard_tensor.ndim == 2:
                hard_tensor = hard_tensor.unsqueeze(0)
            if hard_tensor.shape[0] not in (1, inputs.batch):
                raise ValueError(
                    f"hard_mask batch {hard_tensor.shape[0]} matches neither 1 nor {inputs.batch}"
                )
            # The hard region can only narrow the edit set, never widen it.
            eligible = eligible & hard_tensor
        if inputs.valid_mask is not None:
            valid = inputs.valid_mask.to(inputs.base_logits.device).bool().unsqueeze(-1)
            eligible = eligible & valid
        return eligible

    def support(self, inputs: OperatorInputs) -> torch.Tensor:
        """``[B, T, 40]`` float support: eligibility scaled by ``strength``."""
        return self.eligible(inputs).to(inputs.base_logits.dtype) * self._strength_tensor(inputs)

    def region_identity(
        self, inputs: OperatorInputs, transformed: torch.Tensor
    ) -> torch.Tensor:
        """Keeps ``transformed`` inside the region and the base distribution outside."""
        return torch.where(
            self.eligible(inputs).unsqueeze(-1), transformed, inputs.base_probabilities()
        )

    @staticmethod
    def _offdiagonal_mass(kernel: torch.Tensor) -> torch.Tensor:
        """Sum of the off-diagonal kernel entries, ``[...]`` per position.

        Subtracting the diagonal *value* from every entry of its row (the old
        formula) is not an off-diagonal sum; it measured a different number.  For
        a row-stochastic kernel this is ``levels - trace`` per position, so divide
        by ``levels`` to read it as a fraction of the moved mass.
        """
        diagonal = kernel.diagonal(dim1=-2, dim2=-1)
        return kernel.sum(dim=(-1, -2)) - diagonal.sum(dim=-1)

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
        if getattr(self, "shuffled_adjacency", False):
            payload["level_order"] = list(self.level_order)
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
        eligible = self.eligible(inputs)
        support = self.support(inputs)
        delta = self.delta_head(self.condition(inputs))
        styled_logits = inputs.base_logits + support.unsqueeze(-1) * delta
        # Outside the region the logits are the base logits, whatever the delta
        # head produced there (including a non-finite value); inside, lambda = 0
        # still leaves the base logits untouched.
        styled_logits = torch.where(eligible.unsqueeze(-1), styled_logits, inputs.base_logits)
        return OperatorOutput(
            probabilities=styled_logits.softmax(dim=-1),
            logits=styled_logits,
            diagnostics={
                "support_fraction": eligible.float().mean(),
                "strength_mean": self._strength_tensor(inputs).mean(),
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
        eligible = self.eligible(inputs)
        strength = self._strength_tensor(inputs)
        if self.identity_mix and bool((strength > 1.0).any()):
            raise ValueError(
                "identity_mix=True requires 0 <= strength <= 1 so the kernel stays a "
                f"mixture, got up to {float(strength.max()):.3f}"
            )
        levels = self.num_levels
        kernel_logits = self.kernel_head(self.condition(inputs)).reshape(
            inputs.batch, inputs.frames, inputs.coordinates, levels, levels
        )
        if self.identity_mix:
            # Mixture semantics: the learned kernel is the target and strength is
            # the weight of identity, so lambda = 0 is exactly the base and
            # lambda = 1 the learned kernel (validated to [0, 1] above).
            learned = kernel_logits.softmax(dim=-1)
            identity = torch.eye(levels, device=kernel_logits.device, dtype=kernel_logits.dtype)
            mix = strength.unsqueeze(-1).unsqueeze(-1)
            kernel = (1.0 - mix) * identity + mix * learned
        else:
            # No identity anchor: strength scales how far the kernel may move away
            # from uniform, and the region is applied by the final blend, never by
            # zeroing the logits.
            kernel = (kernel_logits * strength.unsqueeze(-1).unsqueeze(-1)).softmax(dim=-1)
        base = inputs.base_probabilities()
        transformed = torch.einsum("btki,btkij->btkj", base, kernel)
        styled = self.region_identity(inputs, transformed)
        offdiagonal = self._offdiagonal_mass(kernel)
        return OperatorOutput(
            probabilities=styled,
            diagnostics={
                "support_fraction": eligible.float().mean(),
                "strength_mean": strength.mean(),
                "kernel_offdiagonal_mass": (
                    offdiagonal * eligible
                ).sum()
                / eligible.sum().clamp_min(1),
                "kernel_offdiagonal_mass_all": offdiagonal.mean(),
                "kernel_offdiagonal_ratio": offdiagonal.mean() / levels,
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
    if not bool(torch.isfinite(generator).all()):
        raise ValueError("generator contains non-finite rates")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities contain non-finite values")
    # p -> p^T exp(Q) == exp(Q^T) p: transpose once so the series acts on the
    # column convention this operator uses.
    transposed = generator.transpose(-1, -2)
    diagonal = transposed.diagonal(dim1=-2, dim2=-1)
    nu = (-diagonal).clamp_min(0.0).amax(dim=-1)  # [...]
    terms, tail = poisson_term_count(
        float(nu.detach().max()), tolerance=tolerance, max_terms=max_terms
    )
    if tail > tolerance:
        # A truncated series that keeps only part of the Poisson mass cannot be
        # repaired by renormalizing: that would hide how much of the operator was
        # dropped.  Fail with the numbers instead.
        raise ValueError(
            f"uniformization truncated: Poisson tail {tail:.3e} still exceeds tolerance "
            f"{tolerance:.3e} after max_terms={max_terms} at row scale "
            f"{float(nu.max()):.3f}; raise max_terms or the tolerance"
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
    weights = torch.stack(log_weights, dim=-1).exp()
    if not bool(active.all()):
        # An inactive element has row scale zero: its Poisson weight vector is
        # (1, 0, ..., 0) exactly.  Leaving 1/k! there inflated the accumulated
        # mass by ~e and made the mass-error diagnostic meaningless.
        first_only = torch.zeros_like(weights)
        first_only[..., 0] = 1.0
        weights = torch.where(active.unsqueeze(-1), weights, first_only)
    weights = weights.unsqueeze(-1)  # [..., terms, 1]

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
            "mass_tolerance": generator.new_tensor(
                float(1e-9 if probabilities.dtype == torch.float64 else 1e-5)
            ),
            "min_probability_before_clamp": generator.new_tensor(float(probabilities.min())),
        }
    # A truncated series can leave a small negative or a mass error; both are
    # checked against a dtype-aware tolerance and reported separately from the
    # theoretical Poisson tail instead of claiming an exact semigroup.
    mass_error = (accumulated.sum(dim=-1) - 1.0).abs().amax()
    negative_before = float(accumulated.detach().min())
    mass_tolerance = 1e-9 if probabilities.dtype == torch.float64 else 1e-5
    if float(mass_error.detach()) > mass_tolerance or negative_before < -mass_tolerance:
        raise ValueError(
            f"uniformization error exceeds the {mass_tolerance:.0e} tolerance for "
            f"{probabilities.dtype}: mass error {float(mass_error):.3e}, "
            f"most negative entry {negative_before:.3e}"
        )
    clamped = accumulated.clamp_min(0.0)
    normalized = clamped / clamped.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # Only rows that actually moved are renormalized; untouched rows stay exact.
    accumulated = torch.where(active.unsqueeze(-1), normalized, probabilities)
    return accumulated, {
        "uniformization_terms": generator.new_tensor(float(terms)),
        "poisson_tail": generator.new_tensor(float(tail)),
        "mass_error": mass_error.detach(),
        "mass_tolerance": generator.new_tensor(float(mass_tolerance)),
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
                 max_terms: int = 256, level_order: Sequence[int] | None = None) -> None:
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
        # `level_order` defines which levels count as neighbours: the identity
        # order is the design (adjacent FSQ levels), a shuffled order is the
        # geometry control that asks whether the adjacency prior itself helps.
        if level_order is None:
            order = tuple(range(self.num_levels))
        else:
            order = tuple(int(value) for value in level_order)
            if sorted(order) != list(range(self.num_levels)):
                raise ValueError(
                    f"level_order must be a permutation of 0..{self.num_levels - 1}, got {order}"
                )
        self.level_order = order
        self.shuffled_adjacency = order != tuple(range(self.num_levels))
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

    def _level_generator(self, up_rate: torch.Tensor, down_rate: torch.Tensor) -> torch.Tensor:
        """Generator over FSQ levels, optionally with a permuted adjacency.

        ``level_order`` is the chain *in level space*: the edge ``order[i] ->
        order[i + 1]`` carries ``up_rate[i]``.  ``birth_death_generator`` builds
        the chain in visit coordinates, so mapping it into level space needs the
        **inverse** permutation ``order^-1``; indexing with ``order`` itself left
        the level-space graph unchanged (still ``i -> i + 1``) and only relabelled
        which rate sat on which edge, so the geometry control was not a geometry
        control at all.
        """
        generator = birth_death_generator(up_rate, down_rate)
        if not self.shuffled_adjacency:
            return generator
        order = torch.as_tensor(self.level_order, device=generator.device)
        inverse = torch.argsort(order)
        return generator[..., inverse][..., inverse, :]

    def forward(self, inputs: OperatorInputs) -> OperatorOutput:
        up_rate, down_rate, support = self.rates(inputs)
        eligible = self.eligible(inputs)
        generator = self._level_generator(up_rate, down_rate)
        styled, diagnostics = uniformization_expm_apply(
            generator,
            inputs.base_probabilities(),
            tolerance=self.uniformization_tolerance,
            max_terms=self.max_terms,
        )
        styled = self.region_identity(inputs, styled)
        diagnostics["support_fraction"] = eligible.float().mean()
        diagnostics["strength_mean"] = self._strength_tensor(inputs).mean()
        diagnostics["max_up_rate"] = up_rate.max()
        diagnostics["max_down_rate"] = down_rate.max()
        diagnostics["shuffled_adjacency"] = up_rate.new_tensor(
            1.0 if self.shuffled_adjacency else 0.0
        )
        return OperatorOutput(
            probabilities=styled,
            rates=torch.stack((up_rate, down_rate), dim=-2),
            diagnostics=diagnostics,
        )

    def reference_expm(self, inputs: OperatorInputs) -> torch.Tensor:
        """``torch.matrix_exp`` reference for tests and spot checks.

        Returns the same region-blended object as :meth:`forward`, so the two are
        directly comparable outside the region too.
        """
        up_rate, down_rate, _ = self.rates(inputs)
        generator = self._level_generator(up_rate, down_rate)
        applied = inputs.base_probabilities().unsqueeze(-2) @ torch.matrix_exp(generator)
        return self.region_identity(inputs, applied.squeeze(-2))


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
