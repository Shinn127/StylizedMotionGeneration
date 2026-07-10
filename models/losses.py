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
) -> MotionReconstructionLosses:
    recon = output["recon_state"]
    feature_weights = feature_weights.view(1, 1, -1).to(batch_motion.device)
    feature_offset = feature_offset.to(batch_motion.device)
    feature_scale = feature_scale.to(batch_motion.device)

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

    loss = (
        recon_loss
        + float(delta_weight) * delta_loss
        + float(root_pos_weight) * root_pos_loss
        + float(root_rot_weight) * root_rot_loss
        + float(commit_weight) * commit_loss
    )
    return MotionReconstructionLosses(
        loss=loss,
        recon=recon_loss,
        delta=delta_loss,
        commit=commit_loss,
        root_pos=root_pos_loss,
        root_rot=root_rot_loss,
    )


def compute_vqvae_losses(*args, **kwargs) -> MotionReconstructionLosses:
    return compute_motion_reconstruction_losses(*args, **kwargs)
