"""Metrics for the MTS operator study (plan §8).

Two groups, kept apart on purpose:

* **operator metrics** — style response, reference sensitivity, content
  preservation, changed-token ratio, support locality and the physical cost of
  an edit.  All of them are computed from the model's own distributions and from
  decoded motion, so they are reproducible from the artifacts the other scripts
  write;
* **representation metrics** — reconstruction, FK/root error, contact, foot
  slide, level perplexity, adjacent/far geometry, decoder influence width.  These
  already exist in ``nef_eval`` and ``nef_probe``; :func:`representation_metrics`
  documents where each one comes from instead of reimplementing it.

Metrics that need an external model (motion FID, text-motion alignment, a style
classifier) are listed by :func:`unavailable_metrics` rather than approximated:
a number produced by a stand-in would be worse than no number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .contract import masked_nll_from_probs
from .eval_protocol import retrieval_report
from .model import MtsStyleOperator, OperatorBatch
from stylized_motion.learning.losses import (
    integrate_root_trajectory,
    reconstruct_joint_positions,
)
from stylized_motion.anim import quat
from stylized_motion.learning.nef_eval import contacts_from_toe_motion
from .sampling import CommonRandomNumbers, paired_comparison


def unavailable_metrics() -> dict[str, str]:
    """Metrics this module deliberately does not fake."""
    return {
        "motion_fid": "needs a coupling-model feature extractor over decoded motion",
        "text_motion_alignment": "needs a text-motion model; the operator is not claimed to use text",
        "style_classifier_accuracy": "needs a trained style classifier on a held-out split",
    }


# ---------------------------------------------------------------------------
# operator metrics


def reference_sensitivity(
    model: MtsStyleOperator,
    batch: OperatorBatch,
    *,
    wrong_reference: torch.Tensor | None = None,
    random_reference: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Correct / wrong / random reference comparison (plan Phase 4 exit).

    Reports the masked NLL under each reference and the paired distribution
    change between them, so "the reference changes the style" is a number rather
    than an impression.
    """
    correct = _nll(model, batch)
    report: dict[str, float] = {"nll_correct": correct}
    if wrong_reference is not None:
        wrong_batch = _with_reference(batch, wrong_reference)
        report["nll_wrong"] = _nll(model, wrong_batch)
    if random_reference is not None:
        random_batch = _with_reference(batch, random_reference)
        report["nll_random"] = _nll(model, random_batch)
    if "nll_wrong" in report:
        report["nll_correct_minus_wrong"] = report["nll_correct"] - report["nll_wrong"]
    if "nll_random" in report:
        report["nll_correct_minus_random"] = report["nll_correct"] - report["nll_random"]

    crn = CommonRandomNumbers(seed=3407)
    with torch.no_grad():
        result = model(batch)
    correct_probs = result.probabilities
    if wrong_reference is not None:
        with torch.no_grad():
            wrong_result = model(_with_reference(batch, wrong_reference))
        comparison = paired_comparison(correct_probs, wrong_result.probabilities, crn=crn)
        report["changed_token_ratio_wrong_reference"] = comparison["changed_token_ratio"]
        report["total_variation_wrong_reference"] = comparison["total_variation"]
    if random_reference is not None:
        with torch.no_grad():
            random_result = model(_with_reference(batch, random_reference))
        comparison = paired_comparison(correct_probs, random_result.probabilities, crn=crn)
        report["changed_token_ratio_random_reference"] = comparison["changed_token_ratio"]
        report["total_variation_random_reference"] = comparison["total_variation"]
    return report


def strength_response(
    model: MtsStyleOperator,
    batch: OperatorBatch,
    *,
    strengths: Sequence[float],
) -> list[dict[str, float]]:
    """Strength sweep: how far the styled distribution moves from the base."""
    crn = CommonRandomNumbers(seed=3407)
    with torch.no_grad():
        # The base is the *frozen transport's* distribution.  `strength=0` is only
        # the base for families with an identity anchor; the arbitrary kernel is
        # uniform there by design, so using it as the reference would measure the
        # kernel against itself.
        base_result = model(batch)
    base_probs = base_result.base_probabilities
    base_argmax = base_probs.argmax(dim=-1)
    curve: list[dict[str, float]] = []
    for strength in strengths:
        with torch.no_grad():
            result = model(_with_strength(batch, float(strength)))
        identity_probs = (
            result.probabilities
            if float(strength) == 0.0
            else model(_with_strength(batch, 0.0)).probabilities
        )
        comparison = paired_comparison(result.probabilities, base_probs, crn=crn)
        styled_argmax = result.probabilities.argmax(dim=-1)
        curve.append(
            {
                "strength": float(strength),
                "comparison": "base_transport_vs_styled",
                "total_variation": comparison["total_variation"],
                "changed_token_ratio": comparison["changed_token_ratio"],
                # Named for what it is: an argmax-vs-sampled diagnostic, not a
                # style effect.
                "base_argmax_vs_styled_argmax_ratio": float(
                    (styled_argmax != base_argmax).float().mean().detach()
                ),
                "refresh_only_tv": float(
                    (0.5 * (identity_probs - base_probs).abs().sum(dim=-1).mean()).detach()
                ),
                "nll": _nll(model, _with_strength(batch, float(strength))),
                "styled_entropy": float(
                    -(result.probabilities.clamp_min(1e-9).log() * result.probabilities).sum(-1).mean()
                ),
            }
        )
    return curve


def support_locality(
    model: MtsStyleOperator,
    batch: OperatorBatch,
    *,
    crn: CommonRandomNumbers | None = None,
) -> dict[str, float]:
    """How much of the edit escapes the requested support.

    Token leakage is measured under common random numbers (a coupling
    statistic); the tokenizer's own leakage is measured separately by
    ``scripts/evaluate_nef_locality.py``, which decodes both sides.
    """
    with torch.no_grad():
        result = model(batch)
    support = batch.hard_mask
    if support is None:
        return {"support_fraction": 1.0, "leakage": 0.0, "note": "no hard mask was given"}
    mask = support.to(result.probabilities.device).bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    coupling = crn or CommonRandomNumbers(seed=3407)
    base_tokens = model.generate_edit(_with_strength(batch, 0.0), crn=coupling)
    styled_tokens = model.generate_edit(batch, crn=coupling)
    changed = base_tokens != styled_tokens
    inside = mask.expand_as(changed)
    return {
        "support_fraction": float(mask.float().mean()),
        "changed_token_ratio_inside": float(changed[inside].float().mean())
        if bool(inside.any())
        else 0.0,
        "leakage": float(changed[~inside].float().mean()) if bool((~inside).any()) else 0.0,
        "leaked_tokens": int(changed[~inside].sum()) if bool((~inside).any()) else 0,
    }


def style_retrieval(
    model: MtsStyleOperator,
    targets: OperatorBatch,
    *,
    candidate_sets: Sequence[Sequence[torch.Tensor]],
    positive_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Multi-positive top-1 style retrieval.

    ``candidate_sets[i]`` holds the candidate reference tensors for target ``i``
    and ``positive_indices[i]`` lists *every* candidate that really shares the
    target's style: with several same-style references, each of them is a correct
    answer, not just the first.  The score is the masked target-token NLL under
    that reference, so the accuracy needs no external style classifier.

    The result is ``hits`` and ``count``, never a rounded batch accuracy, so
    aggregating across batches is exact.
    """
    batch_size = int(targets.target_tokens.shape[0])
    if len(candidate_sets) != batch_size or len(positive_indices) != batch_size:
        raise ValueError(
            f"one candidate set and one positive list per target are required, "
            f"got {len(candidate_sets)}/{len(positive_indices)} for {batch_size} targets"
        )
    scores: list[list[float]] = []
    positives: list[list[int]] = []
    for row in range(batch_size):
        candidates = candidate_sets[row]
        row_positives = [int(index) for index in positive_indices[row]]
        if not candidates:
            scores.append([])
            positives.append([])
            continue
        row_batch = _row(targets, row)
        scores.append(
            [
                _nll(
                    model,
                    _with_reference(
                        row_batch, reference if reference.ndim == 3 else reference.unsqueeze(0)
                    ),
                )
                for reference in candidates
            ]
        )
        positives.append([index for index in row_positives if 0 <= index < len(candidates)])
    return retrieval_report(scores, positives)


def token_likelihood_diagnostics(
    model: MtsStyleOperator,
    batch: OperatorBatch,
) -> dict[str, Any]:
    """How the styled distribution scores the *target tokens*.

    This is a token-likelihood proxy, not a content-recognition score: it asks
    whether the styled model still assigns probability to the tokens the frozen
    transport was given, and how far it moved from the frozen base.  The keys say
    so (``target_token_nll_*``), because "content preserved" is a claim this number
    cannot support on its own.
    """
    with torch.no_grad():
        result = model(batch)
    target = model.spec.validate_tokens(batch.target_tokens)
    supervision = batch.supervision_mask(model.spec).to(target.device)
    base_nll = float(
        masked_nll_from_probs(
            result.base_probabilities, target,
            valid_mask=batch.target_valid_mask, coordinate_mask=supervision,
        ).detach()
    )
    styled_nll = float(
        masked_nll_from_probs(
            result.probabilities, target,
            valid_mask=batch.target_valid_mask, coordinate_mask=supervision,
        ).detach()
    )
    base_tokens = result.base_probabilities.argmax(dim=-1)
    styled_tokens = result.probabilities.argmax(dim=-1)
    return {
        "target_token_nll_base": base_nll,
        "target_token_nll_styled": styled_nll,
        "target_token_nll_delta": styled_nll - base_nll,
        "argmax_change_ratio": float((base_tokens != styled_tokens).float().mean()),
        "note": (
            "token-likelihood proxy against the frozen base transport; not an "
            "independent content-recognition score"
        ),
    }


#: Kept so old call sites keep working; the name was the misleading part.
content_preservation = token_likelihood_diagnostics


def token_likelihood_per_sample(
    model: MtsStyleOperator,
    batch: OperatorBatch,
) -> list[dict[str, Any]]:
    """The same proxy as :func:`token_likelihood_diagnostics`, one row per sample.

    A batch mean hides which sample moved, and the per-sample rows are what the
    evaluator writes to ``eval_rows.jsonl``.  A sample with no supervised
    position reports ``None`` instead of a zero that would read as "perfect".
    """
    with torch.no_grad():
        result = model(batch)
    target = model.spec.validate_tokens(batch.target_tokens)
    supervision = batch.supervision_mask(model.spec).to(target.device)
    base_nll, counts = _per_sample_nll(
        result.base_probabilities, target, supervision, valid_mask=batch.target_valid_mask
    )
    styled_nll, _ = _per_sample_nll(
        result.probabilities, target, supervision, valid_mask=batch.target_valid_mask
    )
    rows: list[dict[str, Any]] = []
    for index in range(int(target.shape[0])):
        count = int(counts[index])
        rows.append(
            {
                "supervised_tokens": count,
                "target_token_nll_base": None if count == 0 else float(base_nll[index]),
                "target_token_nll_styled": None if count == 0 else float(styled_nll[index]),
                "target_token_nll_delta": None
                if count == 0
                else float(styled_nll[index] - base_nll[index]),
            }
        )
    return rows


def _per_sample_nll(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    coordinate_mask: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    clamp_min: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``([B] mean NLL, [B] supervised count)`` over each sample's masked positions."""
    picked = probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    nll = -picked.clamp_min(float(clamp_min)).log()
    mask = coordinate_mask.to(torch.bool)
    if valid_mask is not None:
        mask = mask & valid_mask.to(mask.device).bool().unsqueeze(-1)
    weights = mask.to(nll.dtype)
    counts = weights.sum(dim=(1, 2))
    totals = (nll * weights).sum(dim=(1, 2))
    return totals / counts.clamp_min(1.0), counts


def ordinal_level_mass(
    probabilities: torch.Tensor,
    base_probabilities: torch.Tensor,
    *,
    supervision: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    """How the styled mass sits relative to the base mode on the ordered axis.

    ``adjacent_mass_fraction`` is ``p(base mode) + p(mode - 1) + p(mode + 1)`` with
    the neighbours gathered **only when they exist**: the E05 script clamped the
    index instead, so at level 0 or level 8 the boundary bin was counted twice and
    the "fraction" could reach 2.  A fraction above 1 is not a fraction.

    ``expected_level_displacement`` is the styled mass's mean ``|level - base mode|``
    -- a property of the styled distribution, not of any kernel, so it does not
    pretend to be a transition displacement.  A birth-death kernel's own
    displacement has its own name (``kernel_level_displacement``) and is reported
    nowhere here.
    """
    if probabilities.shape != base_probabilities.shape:
        raise ValueError("probabilities and base_probabilities must share a shape")
    levels = probabilities.shape[-1]
    index = torch.arange(levels, device=probabilities.device, dtype=torch.float32)
    base_mode = base_probabilities.argmax(dim=-1, keepdim=True)
    base_mode_float = base_mode.to(torch.float32)
    displacement = (probabilities * (index.view(1, 1, 1, -1) - base_mode_float).abs()).sum(-1)
    adjacent = probabilities.gather(-1, base_mode).squeeze(-1)
    for offset in (-1, 1):
        neighbour = base_mode + offset
        inside = (neighbour >= 0) & (neighbour <= levels - 1)
        gathered = probabilities.gather(-1, neighbour.clamp(0, levels - 1)).squeeze(-1)
        adjacent = adjacent + torch.where(inside.squeeze(-1), gathered, torch.zeros_like(gathered))
    tv = 0.5 * (probabilities - base_probabilities).abs().sum(-1)
    mask = torch.ones_like(displacement, dtype=torch.bool)
    if supervision is not None:
        mask = mask & supervision.to(mask.device).bool()
    if valid_mask is not None:
        mask = mask & valid_mask.to(mask.device).bool().unsqueeze(-1)
    count = int(mask.sum())
    if count == 0:
        return {
            "supervised_positions": 0,
            "adjacent_mass_fraction": None,
            "expected_level_displacement": None,
            "tv_from_base": None,
        }
    return {
        "supervised_positions": count,
        "adjacent_mass_fraction": float(adjacent[mask].mean()),
        "expected_level_displacement": float(displacement[mask].mean()),
        "tv_from_base": float(tv[mask].mean()),
    }


def wasserstein_1d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """``W1 = sum_i |CDF_left(i) - CDF_right(i)|`` over the last axis.

    The one distance that compares a diffusion kernel's marginal against a logit
    arm's marginal on the same ordered axis; the kernel's own transition
    displacement (``sum p0(i) K(i, j) |i - j|``) is a different quantity and is
    never mixed into it.  A one-level shift gives exactly 1.0 and equal
    distributions give exactly 0.0.
    """
    if left.shape != right.shape:
        raise ValueError("Wasserstein 1D needs two distributions of the same shape")
    if left.shape[-1] < 2:
        raise ValueError("The level axis must have at least two entries")
    left_cdf = left.cumsum(dim=-1)[..., :-1]
    right_cdf = right.cumsum(dim=-1)[..., :-1]
    return (left_cdf - right_cdf).abs().sum(dim=-1)


def third_difference_jerk(positions: torch.Tensor, dt: float) -> torch.Tensor:
    """``d^3 position / dt^3`` over the time axis, in ``m/s^3``.

    A constant-velocity trajectory gives exactly zero and a cubic ``a t^3`` gives
    exactly ``6a``, which is what makes the number checkable against an analytic
    value instead of only "bigger after an edit".
    """
    if positions.ndim < 3:
        raise ValueError("positions must be [B, T, J, 3]")
    if positions.shape[1] < 4:
        raise ValueError("jerk needs at least four frames")
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    third = (
        positions[:, 3:]
        - 3.0 * positions[:, 2:-1]
        + 3.0 * positions[:, 1:-2]
        - positions[:, :-3]
    )
    return third / dt**3


def physics_metrics(
    *,
    baseline_motion: torch.Tensor,
    edited_motion: torch.Tensor,
    kinematic: Any,
    edit_interval: tuple[int, int] | None = None,
) -> dict[str, float]:
    """FK, root, contact, foot-slide and jerk cost of one edit.

    ``baseline_motion`` and ``edited_motion`` are decoded, normalized motion
    feature tensors ``[B, T, motion_dim]`` that share everything except the edit.
    Every unit is stated: jerk is ``m/s^3`` from FK world positions, foot slide is
    ``m/s`` from FK world positions, and contact prevalence reports the gate it
    used.  Metrics that cannot be computed are ``None`` with a count, never a
    flattering zero: a foot-slide average over zero contact frames is not 0.0, it
    does not exist.
    """
    if baseline_motion.shape != edited_motion.shape:
        raise ValueError("baseline and edited motion must share a shape")
    if baseline_motion.ndim != 3:
        raise ValueError("motion tensors must be [B, T, motion_dim]")
    positions = reconstruct_joint_positions(
        torch.cat((baseline_motion, edited_motion), dim=0),
        kinematic.feature_offset,
        kinematic.feature_scale,
        kinematic.ref_pos,
        kinematic.parents,
        kinematic.dt,
        world_space=True,
    )
    batch = baseline_motion.shape[0]
    baseline_positions, edited_positions = positions[:batch], positions[batch:]
    joint_change = (edited_positions - baseline_positions).norm(dim=-1)  # [B, T, J]

    baseline_root, baseline_rot = integrate_root_trajectory(
        baseline_motion, kinematic.feature_offset, kinematic.feature_scale, kinematic.dt
    )
    edited_root, edited_rot = integrate_root_trajectory(
        edited_motion, kinematic.feature_offset, kinematic.feature_scale, kinematic.dt
    )
    metrics = {
        "fk_change_mean": float(joint_change.mean()),
        "fk_change_max": float(joint_change.max()),
        "root_position_change": float((edited_root[:, 1:] - baseline_root[:, 1:]).abs().mean()),
        # integrate_root_trajectory returns quaternions, so the geodesic angle
        # comes from the quaternion helper (the 6D helper is for feature tensors).
        "root_rotation_change": float(
            quat.torch_quat_angle(edited_rot[:, 1:], baseline_rot[:, 1:]).mean()
        ),
    }
    toe_indices = kinematic.toe_indices
    if toe_indices is not None and baseline_motion.shape[1] >= 2:
        baseline_contacts = contacts_from_toe_motion(
            baseline_positions, toe_indices, kinematic.dt, threshold=kinematic.contact_threshold
        )
        edited_contacts = contacts_from_toe_motion(
            edited_positions, toe_indices, kinematic.dt, threshold=kinematic.contact_threshold
        )
        contact_gate = baseline_contacts[:, 1:].float() * baseline_contacts[:, :-1].float()
        foot_velocity = (edited_positions[:, 1:, toe_indices] - edited_positions[:, :-1, toe_indices])[
            ..., (0, 2)
        ].abs().mean(dim=-1) / float(kinematic.dt)
        baseline_velocity = (
            baseline_positions[:, 1:, toe_indices] - baseline_positions[:, :-1, toe_indices]
        )[..., (0, 2)].abs().mean(dim=-1) / float(kinematic.dt)
        gate_frames = int(contact_gate.sum())
        metrics["contact_gate_frames"] = gate_frames
        metrics["contact_threshold"] = float(kinematic.contact_threshold)
        metrics["foot_slide_unit"] = "m/s"
        if gate_frames > 0:
            metrics.update(
                {
                    "contact_flip_rate": float(
                        (edited_contacts != baseline_contacts).any(dim=-1).float().mean()
                    ),
                    "foot_slide_edited": float(
                        (foot_velocity * contact_gate).sum() / gate_frames
                    ),
                    "foot_slide_baseline": float(
                        (baseline_velocity * contact_gate).sum() / gate_frames
                    ),
                    "foot_skate_ratio": float(
                        (foot_velocity * contact_gate).sum()
                        / (baseline_velocity * contact_gate).sum().clamp_min(1e-6)
                    ),
                }
            )
        else:
            # No contact frame: the average does not exist, and reporting 0.0 would
            # look like a perfectly clean slide.
            metrics.update(
                {
                    "contact_flip_rate": None,
                    "foot_slide_edited": None,
                    "foot_slide_baseline": None,
                    "foot_skate_ratio": None,
                    "foot_slide_reason": "no_contact_frames",
                }
            )
    # Third derivative of the FK world positions, in m/s^3.  A constant-velocity
    # trajectory has jerk exactly 0 and a cubic has the analytic 6a; a "jerk" that
    # is really a first difference of features is a different quantity with a
    # different unit, so it keeps its own name (``feature_delta_change_max``).
    dt = float(kinematic.dt)
    if dt <= 0.0:
        raise ValueError("kinematic.dt must be positive")
    if baseline_motion.shape[1] >= 4:
        baseline_jerk = third_difference_jerk(baseline_positions, dt)
        edited_jerk = third_difference_jerk(edited_positions, dt)
        # Keep the comparison in the same measurement region for both sides.
        magnitude = edited_jerk.norm(dim=-1)  # [B, T-3, J]
        baseline_magnitude = baseline_jerk.norm(dim=-1)
        metrics["jerk_mean"] = float(magnitude.mean())
        metrics["jerk_max"] = float(magnitude.max())
        metrics["jerk_mean_baseline"] = float(baseline_magnitude.mean())
        metrics["jerk_change_mean"] = float((magnitude - baseline_magnitude).abs().mean())
        if edit_interval is not None:
            start, stop = int(edit_interval[0]), int(edit_interval[1])
            # The interval applies to the *signal*: a third difference at index i
            # uses frames i..i+3, so the boundary neighbourhood is the frames whose
            # stencil touches the edit boundary.  An anomaly far outside the region
            # must not leak into these numbers.
            neighbourhood = 3
            inner = (max(start, 0), min(max(stop - 3, start), magnitude.shape[1]))
            metrics["jerk_mean_inside"] = (
                float(magnitude[:, inner[0] : inner[1]].mean()) if inner[1] > inner[0] else None
            )
            left = (max(start - neighbourhood, 0), max(start, 0))
            right = (max(stop - 3, 0), min(stop + neighbourhood - 3, magnitude.shape[1]))
            boundary = []
            if left[1] > left[0]:
                boundary.append(magnitude[:, left[0] : left[1]])
            if right[1] > right[0]:
                boundary.append(magnitude[:, right[0] : right[1]])
            metrics["jerk_max_boundary"] = (
                float(torch.cat(boundary, dim=1).max()) if boundary else None
            )
        if edit_interval is not None:
            start, stop = int(edit_interval[0]), int(edit_interval[1])
            velocity = (edited_motion[:, 1:] - edited_motion[:, :-1]) - (
                baseline_motion[:, 1:] - baseline_motion[:, :-1]
            )
            step = velocity.abs().amax(dim=(0, 2))
            lower = max(start - 1, 0)
            upper = min(max(stop, lower + 1), step.numel())
            metrics["feature_delta_change_max"] = (
                float(step[lower:upper].max()) if upper > lower else None
            )
    return metrics


def comparison_physics(
    *,
    source_motion: torch.Tensor,
    base_motion: torch.Tensor,
    styled_motion: torch.Tensor,
    kinematic: Any,
    edit_interval: tuple[int, int] | None = None,
) -> dict[str, dict[str, float]]:
    """Physics for the three comparisons the plan asks for, kept apart.

    ``source->base`` is what sampling the base costs, ``base->styled`` is the style
    edit proper, and ``source->styled`` is the end-to-end effect.  Reporting one
    mixed number would hide which of the two steps moved the motion.
    """
    return {
        "source_to_base": {
            "comparison": "source_to_base",
            **physics_metrics(
                baseline_motion=source_motion, edited_motion=base_motion,
                kinematic=kinematic, edit_interval=edit_interval,
            ),
        },
        "base_to_styled": {
            "comparison": "base_to_styled",
            **physics_metrics(
                baseline_motion=base_motion, edited_motion=styled_motion,
                kinematic=kinematic, edit_interval=edit_interval,
            ),
        },
        "source_to_styled": {
            "comparison": "source_to_styled",
            **physics_metrics(
                baseline_motion=source_motion, edited_motion=styled_motion,
                kinematic=kinematic, edit_interval=edit_interval,
            ),
        },
    }


# ---------------------------------------------------------------------------
# representation metrics


def representation_metrics() -> dict[str, str]:
    """Where each representation metric of plan §8.1 is produced."""
    return {
        "feature_reconstruction": "stylized_motion.learning.nef_eval.run_report (overall.recon)",
        "fk_joint_position_error": "nef_eval.run_report (overall.fk_world)",
        "root_trajectory_error": "nef_eval.run_report (overall.root_pos / root_rot)",
        "contact_precision_recall": "nef_eval.run_report + contacts_from_toe_motion",
        "foot_slide_ratio": "nef_eval.run_report (streams / foot metrics)",
        "level_perplexity": "nef_eval.run_report (per stream) / MotionFSQ usage stats",
        "adjacent_far_ratio": "scripts/probe_nef_geometry.py (level geometry probe)",
        "token_temporal_change_rate": "nef_eval.run_report (coordinate_change_rate)",
        "decoder_temporal_influence": "scripts/probe_nef_geometry.py (temporal_influence)",
        "off_target_leakage": "scripts/evaluate_nef_locality.py",
    }


# ---------------------------------------------------------------------------
# helpers


def _with_reference(batch: OperatorBatch, reference: torch.Tensor) -> OperatorBatch:
    payload = dict(batch.__dict__)
    payload["reference_tokens"] = reference
    return OperatorBatch(**payload)


def _with_strength(batch: OperatorBatch, strength: Any) -> OperatorBatch:
    payload = dict(batch.__dict__)
    payload["strength"] = strength
    return OperatorBatch(**payload)


#: Fields whose first axis is the *batch*, so a row slice is valid.  Everything
#: else keeps its shape: ``hard_mask`` is ``[T, K]`` or ``[B, T, K]``, and slicing
#: it because ``T`` happens to equal ``B`` silently changes which tokens the row
#: may edit.
_BATCH_AXIS_FIELDS = frozenset(
    {
        "target_tokens",
        "reference_tokens",
        "visible_mask",
        "anchor_mask",
        "target_valid_mask",
        "reference_valid_mask",
        "style_ids",
        "reference_style_ids",
        "content_condition",
        "strength",
        "sample_metadata",
    }
)


def _row(batch: OperatorBatch, row: int) -> OperatorBatch:
    """One row of a batch, with per-field semantics instead of shape guessing.

    ``hard_mask`` is kept when it is ``[T, K]`` (a shared region) and sliced only
    when it is ``[B, T, K]``; the metadata list is sliced with the tensors so the
    row keeps its own provenance.
    """
    import dataclasses

    batch_size = int(batch.target_tokens.shape[0])
    payload: dict[str, Any] = {}
    for name, value in batch.__dict__.items():
        if name not in _BATCH_AXIS_FIELDS:
            payload[name] = value
            continue
        if isinstance(value, list):
            payload[name] = [value[row]] if 0 <= row < len(value) else []
            continue
        if isinstance(value, tuple):
            payload[name] = (value[row],) if 0 <= row < len(value) else ()
            continue
        if not isinstance(value, torch.Tensor):
            payload[name] = value
            continue
        if value.ndim == 0 or (value.shape and value.shape[0] != batch_size):
            payload[name] = value
            continue
        payload[name] = value[row : row + 1]
    if batch.hard_mask is not None and batch.hard_mask.ndim == 3:
        payload["hard_mask"] = batch.hard_mask[row : row + 1]
    return dataclasses.replace(batch, **payload)


def _nll(model: MtsStyleOperator, batch: OperatorBatch) -> float:
    with torch.no_grad():
        loss, _ = model.loss(batch)
    return float(loss.detach())


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean/median/max/std over a list of per-sample metric dicts."""
    keys = sorted({key for row in rows for key in row})
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        # Rows also carry descriptive columns (split, regions, frame_range); only
        # numeric metrics are summarized instead of failing on a label.
        collected: list[float] = []
        for row in rows:
            if key not in row:
                continue
            try:
                collected.append(float(row[key]))
            except (TypeError, ValueError):
                continue
        values = np.asarray(collected, dtype=np.float64)
        if values.size == 0:
            continue
        result[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "std": float(values.std()),
            "n": int(values.size),
        }
    return result


__all__ = [
    "comparison_physics",
    "token_likelihood_diagnostics",
    "aggregate",
    "content_preservation",
    "ordinal_level_mass",
    "physics_metrics",
    "reference_sensitivity",
    "representation_metrics",
    "strength_response",
    "style_retrieval",
    "support_locality",
    "unavailable_metrics",
    "wasserstein_1d",
]
