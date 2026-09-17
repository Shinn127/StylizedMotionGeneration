"""Sampling from styled level probabilities, including common random numbers.

Two things live here:

* ``inverse_cdf_sample`` — the standard inverse-CDF draw that consumes an
  explicit uniform tensor, so sampling can be *paired* across conditions;
* ``CommonRandomNumbers`` — a cache of uniforms keyed by shape, which is what
  makes a strength sweep or a base-vs-styled comparison a paired measurement
  instead of two independent draws.

The paired ``changed_token_ratio`` is a coupling statistic, not a property of
the operator alone: two different distributions can score 0 there if they agree
under the shared uniforms, and two identical distributions score 0 by
construction.  Reports must say so (plan §8.2), which is why
:func:`paired_comparison` labels the numbers it returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .layout_adapter import LayoutAdapter


def inverse_cdf_sample(
    probabilities: torch.Tensor, uniforms: torch.Tensor, *, num_levels: int | None = None
) -> torch.Tensor:
    """Draws one level per position by inverse CDF.

    ``probabilities`` is ``[..., L]`` and ``uniforms`` broadcasts to
    ``probabilities.shape[:-1]``.  Ties and numerical drift are handled by
    clamping the final index into range.
    """
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a trailing level axis")
    levels = int(probabilities.shape[-1]) if num_levels is None else int(num_levels)
    if probabilities.shape[-1] != levels:
        raise ValueError("num_levels does not match the probability width")
    cdf = probabilities.cumsum(dim=-1)
    expanded = uniforms.to(probabilities.device).unsqueeze(-1)
    if expanded.shape != (*probabilities.shape[:-1], 1):
        try:
            expanded = expanded.expand(*probabilities.shape[:-1], 1)
        except RuntimeError as exc:  # pragma: no cover - guarded by callers
            raise ValueError(
                f"uniforms must broadcast to {tuple(probabilities.shape[:-1])}, "
                f"got {tuple(uniforms.shape)}"
            ) from exc
    drawn = (cdf < expanded).sum(dim=-1)
    return drawn.clamp_(0, levels - 1).long()


@dataclass
class CommonRandomNumbers:
    """Uniform draws shared across conditions of one experiment."""

    seed: int = 3407
    uniforms_cache: dict[tuple[int, ...], torch.Tensor] = field(default_factory=dict)
    draws: int = 0

    def uniforms(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """A stable ``shape`` uniform tensor (cached per shape and device)."""
        key = (*tuple(int(value) for value in shape), -1 if device is None else id(device))
        cached = self.uniforms_cache.get(key)
        if cached is None:
            local = generator or torch.Generator(device="cpu").manual_seed(
                self.seed + len(self.uniforms_cache)
            )
            cached = torch.rand(tuple(shape), generator=local)
            if device is not None:
                cached = cached.to(device)
            self.uniforms_cache[key] = cached
        self.draws += 1
        return cached

    def sample(self, probabilities: torch.Tensor) -> torch.Tensor:
        uniforms = self.uniforms(probabilities.shape[:-1], device=probabilities.device)
        return inverse_cdf_sample(probabilities, uniforms)

    def clear(self) -> None:
        self.uniforms_cache.clear()
        self.draws = 0


def sample_tokens(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    crn: CommonRandomNumbers | None = None,
) -> torch.Tensor:
    """Samples tokens either with fresh randomness or with shared uniforms."""
    if crn is not None:
        return crn.sample(probabilities)
    # A CPU generator (the reproducible default) with CUDA probabilities is the
    # normal case: draw on the generator's device and move the *result*.
    draw = probabilities.device if generator is None else generator.device
    uniforms = torch.rand(probabilities.shape[:-1], generator=generator, device=draw)
    return inverse_cdf_sample(probabilities, uniforms.to(probabilities.device))


def paired_comparison(
    base_probabilities: torch.Tensor,
    styled_probabilities: torch.Tensor,
    *,
    crn: CommonRandomNumbers | None = None,
    generator: torch.Generator | None = None,
    hard_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Paired change statistics between a base and a styled distribution.

    ``changed_token_ratio`` counts positions whose *sampled* token differs under
    shared uniforms; ``total_variation`` measures the distributions themselves
    and does not depend on the coupling.  Both are reported so a reader can tell
    which one is being quoted.
    """
    if base_probabilities.shape != styled_probabilities.shape:
        raise ValueError("base and styled probabilities must share a shape")
    coupling = crn or CommonRandomNumbers()
    uniforms = coupling.uniforms(base_probabilities.shape[:-1], device=base_probabilities.device)
    if generator is not None and crn is None:
        draw = generator.device
        uniforms = torch.rand(
            base_probabilities.shape[:-1], generator=generator, device=draw
        ).to(base_probabilities.device)
    base_tokens = inverse_cdf_sample(base_probabilities.detach(), uniforms)
    styled_tokens = inverse_cdf_sample(styled_probabilities.detach(), uniforms)
    changed = base_tokens != styled_tokens
    total_variation = (
        0.5 * (base_probabilities.detach() - styled_probabilities.detach()).abs().sum(dim=-1)
    )
    result: dict[str, Any] = {
        "changed_token_ratio": float(changed.float().mean()),
        "changed_token_count": int(changed.sum()),
        "positions": int(changed.numel()),
        "total_variation": float(total_variation.mean()),
        "total_variation_max": float(total_variation.max()),
        "coupled": True,
        "note": (
            "changed_token_ratio is a common-random-number coupling statistic: it "
            "depends on the shared uniforms as well as on the operator, while "
            "total_variation does not."
        ),
    }
    if hard_mask is not None:
        leading = base_probabilities.shape[:-1]
        mask = hard_mask.to(changed.device).bool()
        if mask.ndim == 2 and len(leading) == 3:
            mask = mask.unsqueeze(0)
        if mask.shape != leading:
            raise ValueError(f"hard_mask must have shape {leading}, got {tuple(hard_mask.shape)}")
        result["outside_support_changed"] = int(changed[~mask].sum())
    return result


def region_support_mask(
    adapter: LayoutAdapter,
    regions: Any,
    *,
    graph_radius: int = 0,
    frame_range: tuple[int, int] | None = None,
    length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``[T, 40]`` support mask for one edit region (thin layout delegation)."""
    return adapter.hard_mask(
        regions,
        graph_radius=graph_radius,
        frame_range=frame_range,
        length=length,
        device=device,
    )


__all__ = [
    "CommonRandomNumbers",
    "inverse_cdf_sample",
    "paired_comparison",
    "region_support_mask",
    "sample_tokens",
]
