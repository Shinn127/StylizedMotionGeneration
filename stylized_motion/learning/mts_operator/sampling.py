"""Sampling from styled level probabilities, including common random numbers.

Two things live here:

* ``inverse_cdf_sample`` — the standard inverse-CDF draw that consumes an
  explicit uniform tensor, so sampling can be *paired* across conditions;
* ``CommonRandomNumbers`` — a cache of uniforms keyed by
  ``(sample_id, step_id, shape)`` with a seed derived by SHA-256 from the run seed
  and that key.  The key is explicit, so a strength sweep or a base-vs-styled
  comparison is a paired measurement instead of two independent draws, and the
  same key reproduces the same uniforms in a later process.  An earlier version
  keyed the cache by ``id(device)`` and seeded by cache insertion order, so
  whether two identical calls were coupled depended on CPython object reuse.

The paired ``changed_token_ratio`` is a coupling statistic, not a property of
the operator alone: two different distributions can score 0 there if they agree
under the shared uniforms, and two identical distributions score 0 by
construction.  Reports must say so (plan §8.2), which is why
:func:`paired_comparison` labels the numbers it returns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .layout_adapter import LayoutAdapter


def inverse_cdf_sample(
    probabilities: torch.Tensor, uniforms: torch.Tensor, *, num_levels: int | None = None
) -> torch.Tensor:
    """Draws one level per position by inverse CDF.

    ``probabilities`` is ``[..., L]`` and ``uniforms`` broadcasts to
    ``probabilities.shape[:-1]``.  The search is for the first level whose CDF is
    *greater* than ``u``, so a zero-mass bin can never be selected — ``u = 0`` on
    a one-hot at level 7 returns 7, where the old ``(cdf < u).sum()`` returned 0.

    ``u`` is canonicalised to ``[0, 1)``: exactly ``1.0`` is clamped to the
    largest representable value below one (the CDF of a level can be exactly 1,
    and ``cdf <= 1`` would step past the last level), while ``u < 0`` or ``u > 1``
    is rejected rather than silently clamped.
    """
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a trailing level axis")
    levels = int(probabilities.shape[-1]) if num_levels is None else int(num_levels)
    if probabilities.shape[-1] != levels:
        raise ValueError("num_levels does not match the probability width")
    expanded = uniforms.to(probabilities.device).unsqueeze(-1)
    if expanded.shape != (*probabilities.shape[:-1], 1):
        try:
            expanded = expanded.expand(*probabilities.shape[:-1], 1)
        except RuntimeError as exc:
            raise ValueError(
                f"uniforms must broadcast to {tuple(probabilities.shape[:-1])}, "
                f"got {tuple(uniforms.shape)}"
            ) from exc
    if bool((expanded < 0.0).any()) or bool((expanded > 1.0).any()):
        raise ValueError(
            "uniforms must lie in [0, 1]; clamp to the open upper bound yourself if "
            "you need a value arbitrarily close to one"
        )
    below_one = torch.nextafter(
        torch.ones_like(expanded), torch.zeros_like(expanded)
    )
    expanded = torch.where(expanded >= 1.0, below_one, expanded)
    cdf = probabilities.cumsum(dim=-1)
    drawn = (cdf <= expanded).sum(dim=-1)
    return drawn.clamp_(0, levels - 1).long()


def uniform_seed(seed: int, sample_id: int, step_id: int, shape: Sequence[int]) -> int:
    """Deterministic generator seed for one ``(sample, step, shape)`` cell.

    SHA-256 rather than ``hash()`` so the value is identical in every process and
    platform, and explicit in the key rather than dependent on insertion order.
    """
    payload = "{}|{}|{}|{}".format(
        int(seed), int(sample_id), int(step_id), ",".join(str(int(v)) for v in shape)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@dataclass
class CommonRandomNumbers:
    """Uniform draws shared across conditions of one experiment.

    The cache key is ``(sample_id, step_id, shape)`` — device-independent, so the
    uniforms are generated once on CPU and moved on demand.  Two calls with the
    same key are coupled by construction; two different ``sample_id`` values are
    independent draws.
    """

    seed: int = 3407
    uniforms_cache: dict[tuple[int, int, tuple[int, ...]], torch.Tensor] = field(
        default_factory=dict
    )
    draws: int = 0
    device_transfers: int = 0

    def key(self, shape: Sequence[int], sample_id: int, step_id: int) -> tuple[int, int, tuple[int, ...]]:
        return (int(sample_id), int(step_id), tuple(int(value) for value in shape))

    def uniforms(
        self,
        shape: Sequence[int],
        *,
        sample_id: int = 0,
        step_id: int = 0,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """The uniform tensor of one ``(sample, step, shape)`` cell.

        ``generator`` overrides the derived seed for callers that need their own
        stream, but the cache key stays the same, so the coupling is still stated
        by the key rather than by call order.
        """
        key = self.key(shape, sample_id, step_id)
        cached = self.uniforms_cache.get(key)
        if cached is None:
            local = generator or torch.Generator(device="cpu").manual_seed(
                uniform_seed(self.seed, *key)
            )
            cached = torch.rand(key[2], generator=local, device="cpu")
            self.uniforms_cache[key] = cached
        self.draws += 1
        if device is None:
            return cached
        target = torch.device(device)
        if cached.device == target:
            return cached
        self.device_transfers += 1
        return cached.to(target)

    def sample(
        self, probabilities: torch.Tensor, *, sample_id: int = 0, step_id: int = 0
    ) -> torch.Tensor:
        uniforms = self.uniforms(
            probabilities.shape[:-1],
            sample_id=sample_id,
            step_id=step_id,
            device=probabilities.device,
        )
        return inverse_cdf_sample(probabilities, uniforms)

    def clear(self) -> None:
        self.uniforms_cache.clear()
        self.draws = 0
        self.device_transfers = 0


def sample_tokens(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    crn: CommonRandomNumbers | None = None,
    sample_id: int = 0,
    step_id: int = 0,
) -> torch.Tensor:
    """Samples tokens either with fresh randomness or with shared uniforms."""
    if crn is not None:
        return crn.sample(probabilities, sample_id=sample_id, step_id=step_id)
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
    sample_id: int = 0,
    step_id: int = 0,
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
    uniforms = coupling.uniforms(
        base_probabilities.shape[:-1],
        sample_id=sample_id,
        step_id=step_id,
        device=base_probabilities.device,
    )
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


def monotonic_fill_steps(remaining: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """Splits the not-yet-drawn positions into ``steps`` commits, in a fixed order.

    ``remaining`` is ``[B, T, K]`` bool (positions still to fill).  Every commit
    takes ``ceil(remaining_at_that_step / steps_left)`` positions per sample in
    (time, coordinate) order, so each position is committed exactly once, the
    schedule is deterministic, and a ``steps`` larger than the number of positions
    is legal (later commits are empty).  Returned masks are ``[B, T, K]``.
    """
    if remaining.ndim != 3:
        raise ValueError(f"remaining must be [B, T, K], got {tuple(remaining.shape)}")
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    left = remaining.clone().bool()
    commits: list[torch.Tensor] = []
    for step in range(steps):
        steps_left = steps - step
        counts = left.flatten(1).sum(dim=1)
        take = torch.ceil(counts.to(torch.float64) / float(steps_left)).to(torch.long)
        commit = torch.zeros_like(left)
        # (time, coordinate) order: flatten is already row-major over the last two axes.
        for row in range(left.shape[0]):
            wanted = int(take[row].item())
            if wanted <= 0:
                continue
            flat = torch.nonzero(left[row].flatten(), as_tuple=False).flatten()
            chosen = flat[:wanted]
            commit[row].flatten()[chosen] = True
        commits.append(commit)
        left = left & ~commit
    if bool(left.any()):
        raise RuntimeError("The schedule did not cover every remaining position")
    return commits


def fill_remaining(
    probabilities: torch.Tensor,
    remaining: torch.Tensor,
    *,
    steps: int,
    crn: "CommonRandomNumbers | None" = None,
    generator: torch.Generator | None = None,
    sample_id: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Draws the tokens of a partially known window, committing blocks at a time.

    Returns the drawn tokens and the per-step commit masks, so a caller can replay
    or audit the schedule.  The draws are keyed by ``(sample_id, step)`` in the
    CRN, so two conditions share the same randomness step by step while the
    contexts are allowed to diverge.
    """
    if probabilities.shape[:-1] != remaining.shape:
        raise ValueError(
            f"probabilities {tuple(probabilities.shape[:-1])} and remaining "
            f"{tuple(remaining.shape)} must describe the same positions"
        )
    commits = monotonic_fill_steps(remaining, steps)
    drawn = torch.zeros(remaining.shape, dtype=torch.long, device=remaining.device)
    for step, commit in enumerate(commits):
        if not bool(commit.any()):
            continue
        step_tokens = sample_tokens(
            probabilities, generator=generator, crn=crn, sample_id=sample_id, step_id=step
        )
        drawn = torch.where(commit.to(drawn.device), step_tokens.to(drawn.device), drawn)
    return drawn, commits


__all__ = [
    "CommonRandomNumbers",
    "fill_remaining",
    "inverse_cdf_sample",
    "monotonic_fill_steps",
    "paired_comparison",
    "region_support_mask",
    "sample_tokens",
    "uniform_seed",
]
