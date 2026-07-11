from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from preprocess import quat


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


VQVAELosses = MotionReconstructionLosses


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
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes global rotations and positions for a parent-ordered skeleton."""
    if parents.ndim != 1 or parents.shape[0] != local_positions.shape[2]:
        raise ValueError("parents must have shape [num_joints] matching local transforms")
    global_rotations = []
    global_positions = []
    for joint, parent in enumerate(parents.detach().cpu().tolist()):
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


def reconstruct_joint_positions(
    motion: torch.Tensor,
    feature_offset: torch.Tensor,
    feature_scale: torch.Tensor,
    ref_pos: torch.Tensor,
    parents: torch.Tensor,
    dt: float,
    world_space: bool,
) -> torch.Tensor:
    """Recovers FK joint positions from normalized feature vectors."""
    motion_raw = denormalize_motion_features(motion, feature_offset, feature_scale)
    batch_size, seq_len, _ = motion_raw.shape
    num_joints = int(ref_pos.shape[0])
    expected_dim = 3 + 3 + 3 + (num_joints - 1) * 6 + 3 + (num_joints - 1) * 3 + 2
    if motion_raw.shape[-1] != expected_dim:
        raise ValueError(f"Expected motion feature dim {expected_dim}, got {motion_raw.shape[-1]}")

    rotation_end = 9 + (num_joints - 1) * 6
    local_positions = ref_pos.view(1, 1, num_joints, 3).expand(batch_size, seq_len, -1, -1).clone()
    local_positions[:, :, 1] = motion_raw[:, :, 6:9]

    identity = torch.eye(3, device=motion.device, dtype=motion.dtype)
    local_rotations = identity.view(1, 1, 1, 3, 3).expand(batch_size, seq_len, num_joints, -1, -1).clone()
    rotations_6d = motion_raw[:, :, 9:rotation_end].reshape(batch_size, seq_len, num_joints - 1, 3, 2)
    local_rotations[:, :, 1:] = rotation_6d_to_matrix(rotations_6d)

    if world_space:
        root_positions, root_rotations = integrate_root_trajectory(
            motion,
            feature_offset,
            feature_scale,
            dt,
            return_positions=True,
            return_rotations=True,
        )
        local_positions[:, :, 0] = root_positions
        local_rotations[:, :, 0] = quaternion_to_matrix(root_rotations)
    else:
        local_positions[:, :, 0] = 0.0

    _, global_positions = forward_kinematics(local_positions, local_rotations, parents)
    return global_positions


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
    parents: torch.Tensor | None = None,
    joint_weights: torch.Tensor | None = None,
    foot_indices: tuple[int, int] | None = None,
) -> MotionReconstructionLosses:
    recon = output["recon_state"]
    feature_weights = feature_weights.view(1, 1, -1).to(batch_motion.device)
    feature_offset = feature_offset.to(batch_motion.device, dtype=batch_motion.dtype)
    feature_scale = feature_scale.to(batch_motion.device, dtype=batch_motion.dtype)

    recon_loss = torch.mean(feature_weights * torch.abs(recon - batch_motion))
    delta_loss = F.l1_loss(recon[:, 1:] - recon[:, :-1], batch_motion[:, 1:] - batch_motion[:, :-1])
    commit_loss = output["commit_loss"]

    compute_root_pos = root_pos_weight > 0.0
    compute_root_rot = root_rot_weight > 0.0
    if compute_root_pos or compute_root_rot:
        root_pos_loss, root_rot_loss = root_trajectory_losses(
            pred=recon,
            target=batch_motion,
            feature_offset=feature_offset,
            feature_scale=feature_scale,
            dt=root_dt,
            compute_pos=compute_root_pos,
            compute_rot=compute_root_rot,
        )
    else:
        root_pos_loss = recon.new_zeros(())
        root_rot_loss = recon.new_zeros(())

    compute_joint = joint_weight > 0.0
    compute_foot = foot_slide_weight > 0.0 or foot_height_weight > 0.0
    if compute_joint or compute_foot:
        if ref_pos is None or parents is None:
            raise ValueError("ref_pos and parents are required for joint or foot kinematic losses")
        ref_pos = ref_pos.to(batch_motion.device, dtype=batch_motion.dtype)
        parents = parents.to(batch_motion.device, dtype=torch.long)

    if compute_joint:
        pred_joint_positions = reconstruct_joint_positions(
            recon, feature_offset, feature_scale, ref_pos, parents, root_dt, world_space=False
        )
        with torch.no_grad():
            target_joint_positions = reconstruct_joint_positions(
                batch_motion, feature_offset, feature_scale, ref_pos, parents, root_dt, world_space=False
            )
        if joint_weights is None:
            weights = torch.ones(ref_pos.shape[0], device=batch_motion.device, dtype=batch_motion.dtype)
        else:
            weights = joint_weights.to(batch_motion.device, dtype=batch_motion.dtype).clone()
        weights[0] = 0.0
        joint_error = F.smooth_l1_loss(pred_joint_positions, target_joint_positions, reduction="none").sum(dim=-1)
        joint_loss = (joint_error * weights.view(1, 1, -1)).sum() / (
            weights.sum().clamp_min(1.0) * batch_motion.shape[0] * batch_motion.shape[1]
        )
    else:
        joint_loss = recon.new_zeros(())

    motion_raw_pred = denormalize_motion_features(recon, feature_offset, feature_scale)
    motion_raw_target = denormalize_motion_features(batch_motion, feature_offset, feature_scale)
    target_contact = motion_raw_target[..., -2:].clamp(0.0, 1.0)
    if contact_weight > 0.0:
        contact_logits = float(contact_temperature) * (motion_raw_pred[..., -2:] - 0.5)
        contact_loss = F.binary_cross_entropy_with_logits(contact_logits, target_contact)
    else:
        contact_loss = recon.new_zeros(())

    if compute_foot:
        if foot_indices is None:
            raise ValueError("foot_indices are required for foot losses")
        pred_world_positions = reconstruct_joint_positions(
            recon, feature_offset, feature_scale, ref_pos, parents, root_dt, world_space=True
        )
        with torch.no_grad():
            target_world_positions = reconstruct_joint_positions(
                batch_motion, feature_offset, feature_scale, ref_pos, parents, root_dt, world_space=True
            )
        pred_feet = pred_world_positions[:, :, list(foot_indices)]
        target_feet = target_world_positions[:, :, list(foot_indices)]
        contact_gate = target_contact[:, 1:] * target_contact[:, :-1]
        foot_velocity = (pred_feet[:, 1:] - pred_feet[:, :-1]) / float(root_dt)
        horizontal_speed = foot_velocity[..., (0, 2)].abs().sum(dim=-1)
        foot_slide_loss = (horizontal_speed * contact_gate).sum() / contact_gate.sum().clamp_min(1.0)
        foot_height_error = (pred_feet[..., 1] - target_feet[..., 1]).abs()
        foot_height_loss = (foot_height_error * target_contact).sum() / target_contact.sum().clamp_min(1.0)
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
    )


def compute_vqvae_losses(*args, **kwargs) -> MotionReconstructionLosses:
    return compute_motion_reconstruction_losses(*args, **kwargs)
