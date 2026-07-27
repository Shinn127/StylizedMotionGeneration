from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from models.part_layout import FEATURE_GROUP_NAMES, PartFSQLayout


@dataclass
class BaseReuseLoss:
    loss: torch.Tensor
    innovation: torch.Tensor
    gate_mean: torch.Tensor


def adaptive_base_reuse_loss(
    base_codes: torch.Tensor,
    target_motion: torch.Tensor,
    layout: PartFSQLayout,
    thresholds: float | Mapping[str, float] = 1.0,
    feature_weights: torch.Tensor | None = None,
    contact_values: torch.Tensor | None = None,
    contact_transition_threshold: float = 0.25,
    level_step: float | None = None,
    charbonnier_eps: float = 1e-3,
) -> BaseReuseLoss:
    """Smooth holistic base codes only when every anatomical region is quiet."""
    if base_codes.ndim != 3 or target_motion.ndim != 3:
        raise ValueError("base_codes and target_motion must have shape [B, T, D]")
    if base_codes.shape[:2] != target_motion.shape[:2]:
        raise ValueError("base_codes and target_motion must share batch/time dimensions")
    layout.validate_motion_dim(target_motion.shape[-1])
    if level_step is None:
        level_step = 2.0 / 8.0
    if level_step <= 0.0 or charbonnier_eps <= 0.0:
        raise ValueError("level_step and charbonnier_eps must be positive")
    if base_codes.shape[1] < 2:
        zero = base_codes.new_zeros(())
        return BaseReuseLoss(zero, zero, zero)

    if isinstance(thresholds, Mapping):
        missing = set(FEATURE_GROUP_NAMES) - set(thresholds)
        if missing:
            raise ValueError(f"base reuse thresholds are missing groups: {sorted(missing)}")
        threshold_values = {group: float(thresholds[group]) for group in FEATURE_GROUP_NAMES}
    else:
        threshold_values = {group: float(thresholds) for group in FEATURE_GROUP_NAMES}
    if any(value <= 0.0 for value in threshold_values.values()):
        raise ValueError("base reuse thresholds must be positive")

    delta = (target_motion[:, 1:] - target_motion[:, :-1]).detach().abs()
    indices = layout.feature_indices(target_motion.shape[-1])
    if feature_weights is not None:
        if feature_weights.ndim != 1 or feature_weights.shape[0] != target_motion.shape[-1]:
            raise ValueError("feature_weights must have one value per motion feature")
        feature_weights = feature_weights.to(delta.device, delta.dtype)
    normalized = []
    for group in FEATURE_GROUP_NAMES:
        index = indices[group].to(delta.device)
        values = delta.index_select(-1, index)
        if feature_weights is None:
            innovation = values.mean(dim=-1)
        else:
            weights = feature_weights.index_select(0, index)
            innovation = (values * weights).sum(dim=-1) / weights.sum().clamp_min(1e-7)
        normalized.append(innovation / threshold_values[group])
    holistic_innovation = torch.stack(normalized, dim=-1).amax(dim=-1)
    gate = F.relu(1.0 - holistic_innovation).detach()

    if contact_values is None:
        contact_values = target_motion[..., -2:]
    if contact_values.shape != (*target_motion.shape[:2], 2):
        raise ValueError("contact_values must have shape [B, T, 2]")
    contact_transition = (
        (contact_values[:, 1:].to(base_codes.device) - contact_values[:, :-1].to(base_codes.device))
        .detach()
        .abs()
        .gt(float(contact_transition_threshold))
        .any(dim=-1)
    )
    gate = gate.masked_fill(contact_transition, 0.0)
    level_delta = (base_codes[:, 1:] - base_codes[:, :-1]) / float(level_step)
    penalty = torch.sqrt(level_delta.square() + float(charbonnier_eps) ** 2) - float(charbonnier_eps)
    loss = (penalty.mean(dim=-1) * gate).mean()
    return BaseReuseLoss(loss=loss, innovation=holistic_innovation.mean(), gate_mean=gate.mean())
