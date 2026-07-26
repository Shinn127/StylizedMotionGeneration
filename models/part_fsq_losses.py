from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from models.part_layout import FEATURE_GROUP_NAMES, GROUP_NAMES, PartFSQLayout


@dataclass
class PartReuseLosses:
    loss: torch.Tensor
    group_losses: torch.Tensor
    group_innovation: torch.Tensor
    group_gate_mean: torch.Tensor


def _reuse_thresholds(
    thresholds: float | Sequence[float] | Mapping[str, float] | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if thresholds is None:
        values = [1.0] * len(GROUP_NAMES)
    elif isinstance(thresholds, Mapping):
        missing = set(GROUP_NAMES) - set(thresholds)
        if missing:
            raise ValueError(f"reuse thresholds are missing groups: {sorted(missing)}")
        values = [float(thresholds[group]) for group in GROUP_NAMES]
    elif isinstance(thresholds, (int, float)):
        values = [float(thresholds)] * len(GROUP_NAMES)
    else:
        values = [float(value) for value in thresholds]
        if len(values) != len(GROUP_NAMES):
            raise ValueError(f"Expected {len(GROUP_NAMES)} reuse thresholds, got {len(values)}")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"reuse thresholds must be positive, got {values}")
    return reference.new_tensor(values)


def adaptive_part_fsq_reuse_loss(
    codes: torch.Tensor,
    target_motion: torch.Tensor,
    layout: PartFSQLayout,
    thresholds: float | Sequence[float] | Mapping[str, float] | None = None,
    feature_weights: torch.Tensor | None = None,
    contact_values: torch.Tensor | None = None,
    contact_transition_threshold: float = 0.25,
    level_step: float | None = None,
    charbonnier_eps: float = 1e-3,
) -> PartReuseLosses:
    """Penalize code changes only when the corresponding target region is quiet.

    The old 64-frame dataset supervises every frame, so this loss intentionally
    uses all 63 adjacent frame pairs and has no target-window mask.
    """
    if codes.ndim != 3 or target_motion.ndim != 3:
        raise ValueError("codes and target_motion must both have shape [B, T, ...]")
    if codes.shape[:2] != target_motion.shape[:2]:
        raise ValueError("codes and target_motion must have matching batch and time dimensions")
    if codes.shape[-1] != layout.num_coordinates:
        raise ValueError(f"Expected {layout.num_coordinates} Part-FSQ coordinates, got {codes.shape[-1]}")
    layout.validate_motion_dim(target_motion.shape[-1])
    if level_step is None:
        level_step = 2.0 / 8.0
    if level_step <= 0.0:
        raise ValueError(f"level_step must be positive, got {level_step}")
    if charbonnier_eps <= 0.0:
        raise ValueError(f"charbonnier_eps must be positive, got {charbonnier_eps}")

    zero = codes.new_zeros(())
    if codes.shape[1] < 2:
        zeros = codes.new_zeros((len(GROUP_NAMES),))
        return PartReuseLosses(loss=zero, group_losses=zeros, group_innovation=zeros, group_gate_mean=zeros)

    target_delta = (target_motion[:, 1:] - target_motion[:, :-1]).detach().abs()
    if feature_weights is not None:
        if feature_weights.ndim != 1 or feature_weights.shape[0] != target_motion.shape[-1]:
            raise ValueError("feature_weights must have one value per motion feature")
        feature_weights = feature_weights.to(target_delta.device, dtype=target_delta.dtype)
    thresholds_tensor = _reuse_thresholds(thresholds, codes)
    feature_indices = layout.feature_indices(target_motion.shape[-1])

    innovations: dict[str, torch.Tensor] = {}
    for group in FEATURE_GROUP_NAMES:
        index = feature_indices[group].to(target_delta.device)
        values = target_delta.index_select(-1, index)
        if feature_weights is None:
            innovations[group] = values.mean(dim=-1)
        else:
            weights = feature_weights.index_select(0, index)
            innovations[group] = (values * weights).sum(dim=-1) / weights.sum().clamp_min(1e-7)

    if contact_values is None:
        contact_values = target_motion[..., -2:]
    if contact_values.shape != (*target_motion.shape[:2], 2):
        raise ValueError("contact_values must have shape [B, T, 2]")
    contact_transitions = (
        (contact_values[:, 1:].to(codes.device) - contact_values[:, :-1].to(codes.device)).detach().abs()
        > float(contact_transition_threshold)
    )

    normalized = {
        group: innovations[group] / thresholds_tensor[GROUP_NAMES.index(group)] for group in FEATURE_GROUP_NAMES
    }
    sync_normalized = torch.stack([normalized[group] for group in FEATURE_GROUP_NAMES], dim=-1).amax(dim=-1)

    group_losses = []
    group_innovation = []
    group_gate_mean = []
    for group in GROUP_NAMES:
        if group == "sync":
            innovation = sync_normalized
            innovation_log = sync_normalized
        else:
            innovation = normalized[group]
            innovation_log = innovations[group]
        gate = F.relu(1.0 - innovation).detach()
        if group in {"global", "sync"}:
            gate = gate.masked_fill(contact_transitions.any(dim=-1), 0.0)
        elif group == "left_leg":
            gate = gate.masked_fill(contact_transitions[..., 0], 0.0)
        elif group == "right_leg":
            gate = gate.masked_fill(contact_transitions[..., 1], 0.0)

        group_slice = layout.group_slices[group]
        level_delta = (codes[:, 1:, group_slice] - codes[:, :-1, group_slice]) / float(level_step)
        coordinate_penalty = torch.sqrt(level_delta.square() + float(charbonnier_eps) ** 2) - float(charbonnier_eps)
        group_losses.append((coordinate_penalty.mean(dim=-1) * gate).mean())
        group_innovation.append(innovation_log.mean())
        group_gate_mean.append(gate.mean())

    losses = torch.stack(group_losses)
    return PartReuseLosses(
        loss=losses.mean(),
        group_losses=losses,
        group_innovation=torch.stack(group_innovation),
        group_gate_mean=torch.stack(group_gate_mean),
    )
