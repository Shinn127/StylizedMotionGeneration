from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from stylized_motion.anim import quat
from stylized_motion.learning.part_layout import FEATURE_GROUP_NAMES, GROUP_NAMES, PART_NAMES, PartFSQLayout


@dataclass
class MotionReconstructionLosses:
    loss: torch.Tensor
    recon: torch.Tensor
    delta: torch.Tensor
    commit: torch.Tensor
    root_pos: torch.Tensor
    root_rot: torch.Tensor
    joint: torch.Tensor
    contact: torch.Tensor
    foot_slide: torch.Tensor
    foot_height: torch.Tensor
    target_contact: torch.Tensor


VQVAELosses = MotionReconstructionLosses


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average weighted values over valid elements."""
    if weights is not None:
        weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1)
        if values.shape[-1] != weights.numel():
            raise ValueError(
                f"Weights have length {weights.numel()}, expected {values.shape[-1]}"
            )
        values = values * weights.view(*([1] * (values.ndim - 1)), -1)
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def integrate_root_trajectory(
    motion: torch.Tensor,
    feature_offset: torch.Tensor,
    feature_scale: torch.Tensor,
    dt: float,
    return_positions: bool = True,
    return_rotations: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not return_positions and not return_rotations:
        raise ValueError("At least one of return_positions or return_rotations must be True")

    lin_local = None
    if return_positions:
        lin_local = motion[..., 0:3] * feature_scale[0:3] + feature_offset[0:3]
    ang_local = motion[..., 3:6] * feature_scale[3:6] + feature_offset[3:6]
    batch_size, seq_len, _ = ang_local.shape

    positions = []
    rotations = []
    pos = ang_local.new_zeros((batch_size, 3))
    rot = ang_local.new_zeros((batch_size, 4))
    rot[:, 0] = 1.0

    if return_positions:
        positions.append(pos)
    if return_rotations:
        rotations.append(rot)
    for frame in range(1, seq_len):
        world_ang = quat.torch_mul_vec(rot, ang_local[:, frame])
        if return_positions:
            world_lin = quat.torch_mul_vec(rot, lin_local[:, frame])
            pos = pos + float(dt) * world_lin
        rot_delta = quat.torch_from_scaled_angle_axis(float(dt) * world_ang)
        rot = quat.torch_normalize(quat.torch_mul(rot_delta, rot))
        if return_positions:
            positions.append(pos)
        if return_rotations:
            rotations.append(rot)

    positions_out = torch.stack(positions, dim=1) if return_positions else None
    rotations_out = torch.stack(rotations, dim=1) if return_rotations else None
    return positions_out, rotations_out


def root_trajectory_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_offset: torch.Tensor,
    feature_scale: torch.Tensor,
    dt: float,
    compute_pos: bool,
    compute_rot: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_pos, pred_rot = integrate_root_trajectory(
        pred,
        feature_offset,
        feature_scale,
        dt,
        return_positions=compute_pos,
        return_rotations=compute_rot,
    )
    with torch.no_grad():
        target_pos, target_rot = integrate_root_trajectory(
            target,
            feature_offset,
            feature_scale,
            dt,
            return_positions=compute_pos,
            return_rotations=compute_rot,
        )

    root_pos_loss = F.l1_loss(pred_pos[:, 1:], target_pos[:, 1:]) if compute_pos else pred.new_zeros(())
    root_rot_loss = quat.torch_quat_angle(pred_rot[:, 1:], target_rot[:, 1:]).mean() if compute_rot else pred.new_zeros(())
    return root_pos_loss, root_rot_loss


def denormalize_motion_features(
    motion: torch.Tensor,
    feature_offset: torch.Tensor,
    feature_scale: torch.Tensor,
) -> torch.Tensor:
    return motion * feature_scale.view(1, 1, -1) + feature_offset.view(1, 1, -1)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Converts the first two rotation-matrix columns to orthonormal matrices."""
    first = F.normalize(rotation_6d[..., :, 0], dim=-1)
    second_raw = rotation_6d[..., :, 1]
    second = F.normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quat.torch_normalize(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = quaternion.new_tensor(2.0)
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def forward_kinematics(
    local_positions: torch.Tensor,
    local_rotations: torch.Tensor,
    parents: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes global rotations and positions for a parent-ordered skeleton."""
    if isinstance(parents, torch.Tensor):
        if parents.ndim != 1:
            raise ValueError("parents must have shape [num_joints] matching local transforms")
        parent_indices = tuple(int(parent) for parent in parents.detach().cpu().tolist())
    else:
        parent_indices = tuple(int(parent) for parent in parents)
    if len(parent_indices) != local_positions.shape[2]:
        raise ValueError("parents must have shape [num_joints] matching local transforms")
    global_rotations = []
    global_positions = []
    for joint, parent in enumerate(parent_indices):
        if parent < 0:
            global_rotations.append(local_rotations[:, :, joint])
            global_positions.append(local_positions[:, :, joint])
            continue
        parent_rotation = global_rotations[parent]
        parent_position = global_positions[parent]
        global_rotations.append(parent_rotation @ local_rotations[:, :, joint])
        global_positions.append(
            parent_position + (parent_rotation @ local_positions[:, :, joint].unsqueeze(-1)).squeeze(-1)
        )
    return torch.stack(global_rotations, dim=2), torch.stack(global_positions, dim=2)


def reconstruct_local_joint_positions(
    motion_raw: torch.Tensor,
    ref_pos: torch.Tensor,
    parents: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Runs FK in the root-local frame from already denormalized features."""
    batch_size, seq_len, _ = motion_raw.shape
    num_joints = int(ref_pos.shape[0])
    expected_dim = 3 + 3 + 3 + (num_joints - 1) * 6 + 3 + (num_joints - 1) * 3 + 2
    if motion_raw.shape[-1] != expected_dim:
        raise ValueError(f"Expected motion feature dim {expected_dim}, got {motion_raw.shape[-1]}")

    rotation_end = 9 + (num_joints - 1) * 6
    local_positions = ref_pos.view(1, 1, num_joints, 3).expand(batch_size, seq_len, -1, -1).clone()
    local_positions[:, :, 0] = 0.0
    local_positions[:, :, 1] = motion_raw[:, :, 6:9]

    identity = torch.eye(3, device=motion_raw.device, dtype=motion_raw.dtype)
    local_rotations = identity.view(1, 1, 1, 3, 3).expand(batch_size, seq_len, num_joints, -1, -1).clone()
    rotations_6d = motion_raw[:, :, 9:rotation_end].reshape(batch_size, seq_len, num_joints - 1, 3, 2)
    local_rotations[:, :, 1:] = rotation_6d_to_matrix(rotations_6d)
    _, positions = forward_kinematics(local_positions, local_rotations, parents)
    return positions


def root_local_to_world_positions(
    local_positions: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
) -> torch.Tensor:
    """Applies a root trajectory to root-local FK positions.

    A world-space FK is exactly this rigid transform of the root-local FK, so
    reusing the latter avoids a second full skeleton traversal.
    """
    if root_positions.shape != (*local_positions.shape[:2], 3):
        raise ValueError("root_positions must have shape [B, T, 3]")
    if root_rotations.shape != (*local_positions.shape[:2], 4):
        raise ValueError("root_rotations must have shape [B, T, 4]")
    root_matrices = quaternion_to_matrix(root_rotations)
    return (root_matrices.unsqueeze(2) @ local_positions.unsqueeze(-1)).squeeze(-1) + root_positions.unsqueeze(2)


def reconstruct_joint_positions(
    motion: torch.Tensor,
    feature_offset: torch.Tensor,
    feature_scale: torch.Tensor,
    ref_pos: torch.Tensor,
    parents: Sequence[int] | torch.Tensor,
    dt: float,
    world_space: bool,
) -> torch.Tensor:
    """Recovers FK joint positions from normalized feature vectors."""
    motion_raw = denormalize_motion_features(motion, feature_offset, feature_scale)
    local_positions = reconstruct_local_joint_positions(motion_raw, ref_pos, parents)
    if world_space:
        root_positions, root_rotations = integrate_root_trajectory(
            motion,
            feature_offset,
            feature_scale,
            dt,
            return_positions=True,
            return_rotations=True,
        )
        return root_local_to_world_positions(local_positions, root_positions, root_rotations)
    return local_positions


def compute_motion_reconstruction_losses(
    batch_motion: torch.Tensor,
    output: dict[str, torch.Tensor],
    feature_weights: torch.Tensor,
    feature_offset: torch.Tensor,
    feature_scale: torch.Tensor,
    delta_weight: float,
    commit_weight: float,
    root_pos_weight: float,
    root_rot_weight: float,
    root_dt: float,
    joint_weight: float = 0.0,
    contact_weight: float = 0.0,
    foot_slide_weight: float = 0.0,
    foot_height_weight: float = 0.0,
    contact_temperature: float = 10.0,
    ref_pos: torch.Tensor | None = None,
    parents: Sequence[int] | torch.Tensor | None = None,
    joint_weights: torch.Tensor | None = None,
    foot_indices: tuple[int, int] | None = None,
    loss_mask: torch.Tensor | None = None,
) -> MotionReconstructionLosses:
    recon = output["recon_state"]
    if loss_mask is None:
        loss_mask = torch.ones(batch_motion.shape[:2], device=batch_motion.device, dtype=torch.bool)
    if loss_mask.shape != batch_motion.shape[:2]:
        raise ValueError(f"loss_mask must have shape {tuple(batch_motion.shape[:2])}, got {tuple(loss_mask.shape)}")
    loss_mask = loss_mask.to(device=batch_motion.device, dtype=torch.bool)

    feature_weights = feature_weights.to(device=batch_motion.device, dtype=batch_motion.dtype).reshape(-1)
    feature_offset = feature_offset.to(batch_motion.device, dtype=batch_motion.dtype)
    feature_scale = feature_scale.to(batch_motion.device, dtype=batch_motion.dtype)

    recon_loss = _masked_weighted_mean(
        torch.abs(recon - batch_motion), loss_mask, feature_weights
    )
    pair_mask = loss_mask[:, 1:] & loss_mask[:, :-1]
    delta_loss = _masked_weighted_mean(
        torch.abs((recon[:, 1:] - recon[:, :-1]) - (batch_motion[:, 1:] - batch_motion[:, :-1])),
        pair_mask,
        feature_weights,
    )
    commit_loss = output["commit_loss"]
    motion_raw_pred = denormalize_motion_features(recon, feature_offset, feature_scale)
    motion_raw_target = denormalize_motion_features(batch_motion, feature_offset, feature_scale)
    target_contact = motion_raw_target[..., -2:].clamp(0.0, 1.0)

    compute_root_pos = root_pos_weight > 0.0
    compute_root_rot = root_rot_weight > 0.0
    compute_joint = joint_weight > 0.0
    compute_foot = foot_slide_weight > 0.0 or foot_height_weight > 0.0
    need_root_positions = compute_root_pos or compute_foot
    need_root_rotations = compute_root_rot or compute_foot
    if need_root_positions or need_root_rotations:
        pred_root_positions, pred_root_rotations = integrate_root_trajectory(
            recon,
            feature_offset=feature_offset,
            feature_scale=feature_scale,
            dt=root_dt,
            return_positions=need_root_positions,
            return_rotations=need_root_rotations,
        )
        with torch.no_grad():
            target_root_positions, target_root_rotations = integrate_root_trajectory(
                batch_motion,
                feature_offset=feature_offset,
                feature_scale=feature_scale,
                dt=root_dt,
                return_positions=need_root_positions,
                return_rotations=need_root_rotations,
            )
        root_pos_loss = (
            _masked_weighted_mean(torch.abs(pred_root_positions[:, 1:] - target_root_positions[:, 1:]), loss_mask[:, 1:])
            if compute_root_pos
            else recon.new_zeros(())
        )
        root_rot_loss = (
            _masked_weighted_mean(quat.torch_quat_angle(pred_root_rotations[:, 1:], target_root_rotations[:, 1:]), loss_mask[:, 1:])
            if compute_root_rot
            else recon.new_zeros(())
        )
    else:
        root_pos_loss = recon.new_zeros(())
        root_rot_loss = recon.new_zeros(())

    if compute_joint or compute_foot:
        if ref_pos is None or parents is None:
            raise ValueError("ref_pos and parents are required for joint or foot kinematic losses")
        ref_pos = ref_pos.to(batch_motion.device, dtype=batch_motion.dtype)
        # parents are topology metadata, not tensor data for FK. Keep the
        # common trainer path on CPU to avoid GPU-to-CPU synchronization here.
        if isinstance(parents, torch.Tensor):
            parents = tuple(int(parent) for parent in parents.detach().cpu().tolist())

    if compute_joint or compute_foot:
        # One batched root-local FK replaces separate pred/target and
        # local/world FK traversals. World positions are a rigid transform of
        # the same root-local result.
        batch_size = batch_motion.shape[0]
        local_positions = reconstruct_local_joint_positions(
            torch.cat((motion_raw_pred, motion_raw_target.detach()), dim=0), ref_pos, parents
        )
        pred_joint_positions, target_joint_positions = local_positions.split(batch_size, dim=0)

    if compute_joint:
        if joint_weights is None:
            weights = torch.ones(ref_pos.shape[0], device=batch_motion.device, dtype=batch_motion.dtype)
        else:
            weights = joint_weights.to(batch_motion.device, dtype=batch_motion.dtype).clone()
        weights[0] = 0.0
        joint_error = F.smooth_l1_loss(pred_joint_positions, target_joint_positions, reduction="none").mean(dim=-1)
        joint_loss = _masked_weighted_mean(joint_error, loss_mask, weights)
    else:
        joint_loss = recon.new_zeros(())

    if contact_weight > 0.0:
        contact_logits = float(contact_temperature) * (motion_raw_pred[..., -2:] - 0.5)
        contact_loss = _masked_weighted_mean(
            F.binary_cross_entropy_with_logits(contact_logits, target_contact, reduction="none"), loss_mask
        )
    else:
        contact_loss = recon.new_zeros(())

    if compute_foot:
        if foot_indices is None:
            raise ValueError("foot_indices are required for foot losses")
        world_positions = root_local_to_world_positions(
            local_positions,
            torch.cat((pred_root_positions, target_root_positions), dim=0),
            torch.cat((pred_root_rotations, target_root_rotations), dim=0),
        )
        pred_world_positions, target_world_positions = world_positions.split(batch_motion.shape[0], dim=0)
        pred_feet = pred_world_positions[:, :, list(foot_indices)]
        target_feet = target_world_positions[:, :, list(foot_indices)]
        contact_gate = target_contact[:, 1:] * target_contact[:, :-1]
        foot_velocity = (pred_feet[:, 1:] - pred_feet[:, :-1]) / float(root_dt)
        horizontal_speed = foot_velocity[..., (0, 2)].abs().mean(dim=-1)
        valid_contact_gate = contact_gate * pair_mask.unsqueeze(-1).to(contact_gate.dtype)
        foot_slide_loss = _masked_weighted_mean(horizontal_speed, valid_contact_gate)
        foot_height_error = (pred_feet[..., 1] - target_feet[..., 1]).abs()
        valid_target_contact = target_contact * loss_mask.unsqueeze(-1).to(target_contact.dtype)
        foot_height_loss = _masked_weighted_mean(foot_height_error, valid_target_contact)
    else:
        foot_slide_loss = recon.new_zeros(())
        foot_height_loss = recon.new_zeros(())

    loss = (
        recon_loss
        + float(delta_weight) * delta_loss
        + float(root_pos_weight) * root_pos_loss
        + float(root_rot_weight) * root_rot_loss
        + float(joint_weight) * joint_loss
        + float(contact_weight) * contact_loss
        + float(foot_slide_weight) * foot_slide_loss
        + float(foot_height_weight) * foot_height_loss
        + float(commit_weight) * commit_loss
    )
    return MotionReconstructionLosses(
        loss=loss,
        recon=recon_loss,
        delta=delta_loss,
        commit=commit_loss,
        root_pos=root_pos_loss,
        root_rot=root_rot_loss,
        joint=joint_loss,
        contact=contact_loss,
        foot_slide=foot_slide_loss,
        foot_height=foot_height_loss,
        target_contact=target_contact,
    )


def compute_vqvae_losses(*args, **kwargs) -> MotionReconstructionLosses:
    return compute_motion_reconstruction_losses(*args, **kwargs)


def _quiet_gate(target_motion: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    if target_motion.shape[1] < 2:
        return target_motion.new_zeros((target_motion.shape[0], 0))
    delta = (target_motion[:, 1:] - target_motion[:, :-1]).detach().abs().mean(dim=-1)
    return (1.0 - delta / float(threshold)).clamp_min(0.0)


def _code_reuse_loss(codes: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if codes.shape[1] < 2:
        return codes.new_zeros(())
    changes = codes[:, 1:] - codes[:, :-1]
    penalty = torch.sqrt(changes.square() + 1e-6) - 1e-3
    return _masked_weighted_mean(penalty, gate)


def compute_part_representation_losses(
    output: Mapping[str, object],
    batch: Mapping[str, object],
    layout: PartFSQLayout,
) -> dict[str, torch.Tensor]:
    codes = output["codes"]
    motion = batch["motion"]
    if not isinstance(codes, torch.Tensor) or not isinstance(motion, torch.Tensor):
        raise TypeError("Part representation loss requires tensor codes and motion")
    if codes.shape[-1] != layout.num_coordinates:
        raise ValueError("Part representation codes do not match the coordinate layout")
    gate = _quiet_gate(motion, float(batch.get("reuse_threshold", 1.0)))
    loss_mask = batch.get("loss_mask")
    if isinstance(loss_mask, torch.Tensor):
        gate = gate * (loss_mask[:, 1:] & loss_mask[:, :-1]).to(gate.dtype)
    return {"reuse": _code_reuse_loss(codes, gate) * float(batch.get("reuse_weight", 0.01))}


def compute_residual_representation_losses(
    output: Mapping[str, object],
    batch: Mapping[str, object],
    layout: PartFSQLayout,
) -> dict[str, torch.Tensor]:
    base_codes = output.get("base_codes")
    motion = batch["motion"]
    if not isinstance(base_codes, torch.Tensor) or not isinstance(motion, torch.Tensor):
        raise TypeError("Residual representation loss requires base codes and motion")
    gate = _quiet_gate(motion, float(batch.get("base_reuse_threshold", 1.0)))
    loss_mask = batch.get("loss_mask")
    if isinstance(loss_mask, torch.Tensor):
        gate = gate * (loss_mask[:, 1:] & loss_mask[:, :-1]).to(gate.dtype)
    return {
        "base_reuse": _code_reuse_loss(base_codes, gate) * float(batch.get("base_reuse_weight", 0.0025)),
    }


def compute_latent_residual_representation_losses(
    output: Mapping[str, object],
    batch: Mapping[str, object],
    layout: PartFSQLayout,
) -> dict[str, torch.Tensor]:
    residual_energy = output.get("latent_residual_energy")
    if not isinstance(residual_energy, torch.Tensor):
        raise TypeError("Latent Residual representation output is missing latent_residual_energy")
    loss_mask = batch.get("loss_mask")
    if isinstance(loss_mask, torch.Tensor):
        if residual_energy.shape != loss_mask.shape:
            raise ValueError(
                "latent_residual_energy must have shape [B,T] when loss_mask is provided"
            )
        mask = loss_mask.to(device=residual_energy.device, dtype=residual_energy.dtype)
        residual_energy = _masked_weighted_mean(residual_energy, mask)
    elif residual_energy.ndim != 0:
        residual_energy = residual_energy.mean()
    return {
        "latent_energy": residual_energy * float(batch.get("latent_energy_weight", 0.01)),
    }


def compute_latent_residual_v2_representation_losses(
    output: Mapping[str, object],
    batch: Mapping[str, object],
    layout: PartFSQLayout,
) -> dict[str, torch.Tensor]:
    motion = batch.get("motion")
    base_recon = output.get("base_recon_state")
    edit_recon = output.get("edit_recon_state")
    edit_part = output.get("edit_part")
    donor_permutation = output.get("donor_permutation")
    if not isinstance(motion, torch.Tensor) or not isinstance(base_recon, torch.Tensor):
        raise TypeError("Latent Residual-FSQ V2 training requires motion and base_recon_state")
    feature_weights = batch.get("feature_weights")
    if not isinstance(feature_weights, torch.Tensor):
        feature_weights = torch.ones(motion.shape[-1], device=motion.device, dtype=motion.dtype)
    feature_weights = feature_weights.to(device=motion.device, dtype=motion.dtype)
    loss_mask = batch.get("loss_mask")
    if not isinstance(loss_mask, torch.Tensor):
        loss_mask = torch.ones(motion.shape[:2], device=motion.device, dtype=torch.bool)
    if loss_mask.shape != motion.shape[:2]:
        raise ValueError(
            "V2 representation loss_mask must have shape "
            f"{tuple(motion.shape[:2])}, got {tuple(loss_mask.shape)}"
        )
    base_loss = _masked_weighted_mean(
        torch.abs(base_recon - motion), loss_mask, feature_weights
    )
    losses = {"base_recon": base_loss * float(batch.get("base_recon_weight", 0.1))}

    if edit_recon is None:
        return losses
    if (
        not isinstance(edit_recon, torch.Tensor)
        or not isinstance(edit_part, str)
        or not isinstance(donor_permutation, torch.Tensor)
    ):
        raise TypeError("V2 edit output is incomplete")
    if edit_part not in PART_NAMES:
        raise ValueError(f"Unknown V2 edit part {edit_part!r}")
    donor = motion.index_select(0, donor_permutation.to(device=motion.device, dtype=torch.long))
    part_index = layout.feature_indices(motion.shape[-1])[edit_part].to(motion.device)
    part_mask = torch.zeros(motion.shape[-1], device=motion.device, dtype=torch.bool)
    part_mask[part_index] = True
    transfer = _masked_weighted_mean(
        torch.abs(edit_recon[..., part_mask] - donor[..., part_mask]),
        loss_mask,
        feature_weights[part_mask],
    )
    preserve = _masked_weighted_mean(
        torch.abs(edit_recon[..., ~part_mask] - motion[..., ~part_mask]),
        loss_mask,
        feature_weights[~part_mask],
    )
    edit_weight = float(batch.get("edit_weight", 0.25))
    losses["part_edit_transfer"] = transfer * edit_weight
    losses["part_edit_preserve"] = preserve * edit_weight * float(batch.get("edit_preserve_weight", 1.0))
    return losses
