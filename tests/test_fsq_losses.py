import torch

from models.fsq import FSQMotionAutoencoder
from models.losses import compute_motion_reconstruction_losses, reconstruct_joint_positions


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
