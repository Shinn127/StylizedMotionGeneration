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

from .contract import masked_cross_entropy
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
        base_result = model(_with_strength(batch, 0.0))
    base_probs = base_result.probabilities
    curve: list[dict[str, float]] = []
    for strength in strengths:
        with torch.no_grad():
            result = model(_with_strength(batch, float(strength)))
        comparison = paired_comparison(result.probabilities, base_probs, crn=crn)
        curve.append(
            {
                "strength": float(strength),
                "total_variation": comparison["total_variation"],
                "changed_token_ratio": comparison["changed_token_ratio"],
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
    correct_index: Sequence[int],
) -> dict[str, float]:
    """Top-1 style retrieval: is the correct reference the best explanation?

    ``candidate_sets[i]`` holds the candidate reference tensors for target ``i``
    (``[T, 40]`` or ``[1, T, 40]``) and ``correct_index[i]`` marks which one really
    shares its style.  Each pair is scored by the masked NLL of the target under
    that reference, so the accuracy needs no external style classifier — the
    model ranks its own references.
    """
    batch_size = int(targets.target_tokens.shape[0])
    if len(candidate_sets) != batch_size or len(correct_index) != batch_size:
        raise ValueError(
            f"one candidate set and one correct index per target are required, "
            f"got {len(candidate_sets)}/{len(correct_index)} for {batch_size} targets"
        )
    hits = []
    for row in range(batch_size):
        candidates = candidate_sets[row]
        if not candidates:
            continue
        row_batch = _row(targets, row)
        scores = [
            _nll(
                model,
                _with_reference(
                    row_batch, reference if reference.ndim == 3 else reference.unsqueeze(0)
                ),
            )
            for reference in candidates
        ]
        hits.append(int(np.argmin(scores)) == int(correct_index[row]))
    chance = 1.0 / max(len(candidate_sets[0]), 1) if candidate_sets else 0.0
    return {
        "top1_accuracy": float(np.mean(hits)) if hits else 0.0,
        "targets": float(len(hits)),
        "candidates": float(max((len(candidates) for candidates in candidate_sets), default=0)),
        "chance": chance,
    }


def content_preservation(
    model: MtsStyleOperator,
    batch: OperatorBatch,
) -> dict[str, float]:
    """How much of the target's content survives the edit.

    ``base_nll`` is the frozen transport's own masked likelihood (the content
    ceiling); ``styled_nll`` is the operator's.  The difference is the content
    price of the style edit, and it is reported next to the masked change so a
    reader cannot mistake "changed a lot" for "preserved a lot".
    """
    with torch.no_grad():
        result = model(batch)
    target = model.spec.validate_tokens(batch.target_tokens)
    supervision = batch.supervision_mask(model.spec).to(target.device)
    base_nll = float(
        masked_cross_entropy(
            result.base_probabilities, target,
            valid_mask=batch.target_valid_mask, coordinate_mask=supervision,
        ).detach()
    )
    styled_nll = float(
        masked_cross_entropy(
            result.probabilities, target,
            valid_mask=batch.target_valid_mask, coordinate_mask=supervision,
        ).detach()
    )
    base_tokens = result.base_probabilities.argmax(dim=-1)
    styled_tokens = result.probabilities.argmax(dim=-1)
    return {
        "base_nll": base_nll,
        "styled_nll": styled_nll,
        "nll_increase": styled_nll - base_nll,
        "argmax_change_ratio": float((base_tokens != styled_tokens).float().mean()),
    }


def physics_metrics(
    *,
    baseline_motion: torch.Tensor,
    edited_motion: torch.Tensor,
    kinematic: Any,
    edit_interval: tuple[int, int] | None = None,
) -> dict[str, float]:
    """FK, root, contact, foot-slide and boundary-jerk cost of one edit.

    ``baseline_motion`` and ``edited_motion`` are decoded, normalized motion
    feature tensors ``[B, T, motion_dim]`` that share everything except the edit.
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
        weighted_gate = contact_gate.sum().clamp_min(1.0)
        metrics.update(
            {
                "contact_flip_rate": float(
                    (edited_contacts != baseline_contacts).any(dim=-1).float().mean()
                ),
                "foot_slide_edited": float((foot_velocity * contact_gate).sum() / weighted_gate),
                "foot_slide_baseline": float(
                    (baseline_velocity * contact_gate).sum() / weighted_gate
                ),
                "foot_skate_ratio": float(
                    (foot_velocity * contact_gate).sum()
                    / (baseline_velocity * contact_gate).sum().clamp_min(1e-6)
                ),
            }
        )
    if edit_interval is not None:
        start, stop = int(edit_interval[0]), int(edit_interval[1])
        velocity = (edited_motion[:, 1:] - edited_motion[:, :-1]) - (
            baseline_motion[:, 1:] - baseline_motion[:, :-1]
        )
        step = velocity.abs().amax(dim=(0, 2))
        lower = max(start - 1, 0)
        upper = min(max(stop, lower + 1), step.numel())
        metrics["boundary_jerk_max"] = float(step[lower:upper].max()) if upper > lower else 0.0
    return metrics


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


def _row(batch: OperatorBatch, row: int) -> OperatorBatch:
    payload = {}
    for name, value in batch.__dict__.items():
        if isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[0] == batch.target_tokens.shape[0] or isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] == batch.target_tokens.shape[0]:
            payload[name] = value[row : row + 1]
        else:
            payload[name] = value
    return OperatorBatch(**payload)


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
    "aggregate",
    "content_preservation",
    "physics_metrics",
    "reference_sensitivity",
    "representation_metrics",
    "strength_response",
    "style_retrieval",
    "support_locality",
    "unavailable_metrics",
]
