import torch
import pytest

from stylized_motion.learning.fsq import FSQMotionAutoencoder, MotionFSQ
from stylized_motion.learning.losses import (
    compute_motion_reconstruction_losses,
    denormalize_motion_features,
    forward_kinematics,
    integrate_root_trajectory,
    quaternion_to_matrix,
    reconstruct_joint_positions,
    rotation_6d_to_matrix,
)
from stylized_motion.learning.runner import _joint_weights_from_feature_weights


def _identity_rotation_6d() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def _simple_motion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    # Skeleton: Simulation -> Hips -> LeftToeBase, RightToeBase.
    parents = torch.tensor([-1, 0, 1, 1])
    ref_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.1, -1.0, 0.1], [0.1, -1.0, 0.1]]
    )
    motion_dim = 3 + 3 + 3 + (len(parents) - 1) * 6 + 3 + (len(parents) - 1) * 3 + 2
    motion = torch.zeros(1, 3, motion_dim)
    rotation_start = 9
    for joint in range(1, len(parents)):
        motion[:, :, rotation_start + (joint - 1) * 6 : rotation_start + joint * 6] = _identity_rotation_6d()
    motion[..., -2:] = 1.0
    return motion, ref_pos, parents, (2, 3)


def test_fsq_codes_roundtrip_and_ste_gradient():
    torch.manual_seed(3)
    model = FSQMotionAutoencoder(motion_dim=12, code_dim=16, width=16, num_coordinates=5)
    model.eval()
    motion = torch.randn(2, 64, 12, requires_grad=True)
    output = model(motion)

    assert output["fsq_codes"].shape == (2, 64, 5)
    assert output["indices"].shape == (2, 64, 5)
    assert output["indices"].dtype == torch.long
    assert int(output["indices"].min()) >= 0
    assert int(output["indices"].max()) <= 8
    torch.testing.assert_close(model.decode_from_codes(output["fsq_codes"]), output["recon_state"])
    torch.testing.assert_close(model.decode_from_indices(output["indices"]), output["recon_state"])

    output["fsq_codes"].sum().backward()
    assert motion.grad is not None
    assert torch.count_nonzero(motion.grad) > 0


def test_motion_fsq_can_skip_sequence_statistics():
    quantizer = MotionFSQ(code_dim=8, num_coordinates=5, num_levels=9)
    quantizer._sequence_stats = lambda _indices: pytest.fail("sequence statistics should be skipped")
    z = torch.randn(2, 8, 64)
    result = quantizer(z, collect_stats=True, collect_sequence_stats=False)
    assert result[4].isfinite()
    assert result[10].item() == 0.0
    assert result[11].item() == 0.0
    assert result[12].item() == 0.0


def test_delta_loss_uses_feature_weights():
    motion = torch.zeros(1, 3, 4)
    recon = motion.clone()
    recon[:, 1:, 0] = torch.tensor([1.0, 3.0])
    common = dict(
        batch_motion=motion,
        feature_offset=torch.zeros(4),
        feature_scale=torch.ones(4),
        delta_weight=1.0,
        commit_weight=0.0,
        root_pos_weight=0.0,
        root_rot_weight=0.0,
        root_dt=1.0 / 60.0,
        joint_weight=0.0,
        contact_weight=0.0,
        foot_slide_weight=0.0,
        foot_height_weight=0.0,
    )
    unweighted = compute_motion_reconstruction_losses(
        output={"recon_state": recon, "commit_loss": motion.new_zeros(())},
        feature_weights=torch.ones(4),
        **common,
    )
    weighted = compute_motion_reconstruction_losses(
        output={"recon_state": recon, "commit_loss": motion.new_zeros(())},
        feature_weights=torch.tensor([2.0, 1.0, 1.0, 1.0]),
        **common,
    )
    torch.testing.assert_close(unweighted.delta, torch.tensor(0.375))
    torch.testing.assert_close(weighted.delta, torch.tensor(0.75))


def test_joint_weights_are_projected_from_rotation_feature_weights():
    feature_weights = torch.ones(32)
    feature_weights[9:15] = 2.0
    feature_weights[15:21] = 4.0
    projected = _joint_weights_from_feature_weights(feature_weights, num_joints=3)
    torch.testing.assert_close(projected, torch.tensor([0.0, 2.0, 4.0]))


def test_joint_and_foot_losses_use_target_contact_gates():
    motion, ref_pos, parents, foot_indices = _simple_motion()
    output = {"recon_state": motion.clone(), "commit_loss": motion.new_zeros(())}
    feature_weights = torch.ones(motion.shape[-1])
    offset = torch.zeros(motion.shape[-1])
    scale = torch.ones(motion.shape[-1])
    joint_weights = torch.ones(len(parents))

    losses = compute_motion_reconstruction_losses(
        batch_motion=motion,
        output=output,
        feature_weights=feature_weights,
        feature_offset=offset,
        feature_scale=scale,
        delta_weight=0.0,
        commit_weight=0.0,
        root_pos_weight=0.0,
        root_rot_weight=0.0,
        root_dt=1.0 / 60.0,
        joint_weight=1.0,
        contact_weight=0.0,
        foot_slide_weight=1.0,
        foot_height_weight=1.0,
        ref_pos=ref_pos,
        parents=parents,
        joint_weights=joint_weights,
        foot_indices=foot_indices,
    )
    torch.testing.assert_close(losses.joint, torch.zeros_like(losses.joint))
    torch.testing.assert_close(losses.foot_slide, torch.zeros_like(losses.foot_slide))
    torch.testing.assert_close(losses.foot_height, torch.zeros_like(losses.foot_height))

    moved = motion.clone()
    moved[:, 1:, 0] = 1.0
    positions = reconstruct_joint_positions(moved, offset, scale, ref_pos, parents, 1.0 / 60.0, world_space=True)
    assert torch.linalg.vector_norm(positions[:, 1:, list(foot_indices), :] - positions[:, :-1, list(foot_indices), :]) > 0


def test_world_fk_is_equivalent_to_root_transform_of_local_fk():
    motion, ref_pos, parents, _ = _simple_motion()
    motion = motion.repeat(2, 1, 1)
    motion[0, :, 0] = torch.tensor([0.2, -0.1, 0.3])
    motion[1, :, 0] = torch.tensor([0.0, 0.1, -0.2])
    motion[:, :, 3] = 0.15
    offset = torch.zeros(motion.shape[-1])
    scale = torch.ones(motion.shape[-1])

    actual = reconstruct_joint_positions(motion, offset, scale, ref_pos, parents, 1.0 / 60.0, world_space=True)

    # Reference: the pre-optimization world-space FK construction.
    raw = denormalize_motion_features(motion, offset, scale)
    batch_size, seq_len, _ = raw.shape
    num_joints = ref_pos.shape[0]
    local_positions = ref_pos.view(1, 1, num_joints, 3).expand(batch_size, seq_len, -1, -1).clone()
    local_positions[:, :, 1] = raw[:, :, 6:9]
    identity = torch.eye(3).view(1, 1, 1, 3, 3).expand(batch_size, seq_len, num_joints, -1, -1).clone()
    rotations = raw[:, :, 9 : 9 + (num_joints - 1) * 6].reshape(batch_size, seq_len, num_joints - 1, 3, 2)
    identity[:, :, 1:] = rotation_6d_to_matrix(rotations)
    root_positions, root_rotations = integrate_root_trajectory(motion, offset, scale, 1.0 / 60.0)
    local_positions[:, :, 0] = root_positions
    identity[:, :, 0] = quaternion_to_matrix(root_rotations)
    _, expected = forward_kinematics(local_positions, identity, parents)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
