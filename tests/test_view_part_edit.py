from types import SimpleNamespace

import numpy as np
import torch

from stylized_motion import run
from stylized_motion.anim.view_part_edit import _coordinate_slice, _edit_pair, _read_range_window
from stylized_motion.learning.latent_residual_fsq import LatentResidualPartFSQMotionAutoencoder
from stylized_motion.learning.latent_residual_fsq_v2 import LatentResidualPartFSQV2MotionAutoencoder
from stylized_motion.learning.part_fsq import HierarchicalPartFSQMotionAutoencoder
from stylized_motion.learning.residual_part_fsq import ResidualPartFSQMotionAutoencoder


NAMES = [
    "Simulation", "Hips", "Spine", "LeftUpLeg", "LeftToeBase", "RightUpLeg",
    "RightToeBase", "LeftShoulder", "LeftHand", "RightShoulder", "RightHand",
]
PARENTS = [-1, 0, 1, 1, 3, 1, 5, 2, 7, 2, 9]
MOTION_DIM = 9 * len(NAMES) + 5


def _metadata(model):
    if hasattr(model, "layout") and "sync" in getattr(model.layout, "group_slices", {}):
        order = ("global", "sync", "torso", "left_leg", "right_leg", "left_arm", "right_arm")
        counts = {name: model.layout.group_slices[name].stop - model.layout.group_slices[name].start for name in order}
    else:
        order = tuple(model.group_slices)
        counts = {name: model.group_slices[name].stop - model.group_slices[name].start for name in order}
    return {"coordinate_order": list(order), "coordinate_counts": counts}


def _models():
    common = {"names": NAMES, "parents": PARENTS, "motion_dim": MOTION_DIM}
    return (
        ("part_fsq", HierarchicalPartFSQMotionAutoencoder(**common, stream_dim=8)),
        ("residual_part_fsq", ResidualPartFSQMotionAutoencoder(
            **common, base_code_dim=16, base_width=16, part_state_dim=8,
            residual_decoder_dim=16, residual_decoder_width=16, residual_hidden_dim=8,
            residual_group_dropout=0.0,
        )),
        ("latent_residual_fsq", LatentResidualPartFSQMotionAutoencoder(
            **common, base_code_dim=16, base_width=16, part_state_dim=8,
            part_predictor_hidden_dim=8, latent_projector_hidden_dim=8,
            part_latent_dims=[4, 3, 3, 3, 3],
        )),
        ("latent_residual_fsq_v2", LatentResidualPartFSQV2MotionAutoencoder(
            **common, base_code_dim=16, base_width=16, part_state_dim=8,
            part_predictor_hidden_dim=8, part_encoder_width=8,
        )),
    )


def test_dispatch_registers_part_edit_visualizer():
    assert run.COMMANDS[("visualize", "part-edit")] == "stylized_motion.anim.view_part_edit"


def test_coordinate_slice_uses_representation_metadata():
    metadata = {
        "coordinate_order": ["base", "torso", "left_leg", "right_leg", "left_arm", "right_arm"],
        "coordinate_counts": {"base": 20, "torso": 6, "left_leg": 4, "right_leg": 4, "left_arm": 3, "right_arm": 3},
    }
    assert _coordinate_slice(metadata, "left_arm") == slice(34, 37)


def test_range_reader_uses_range_to_shard_mapping_and_left_pads(tmp_path):
    shard = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    shard_path = tmp_path / "shard.npy"
    np.save(shard_path, shard)
    store = SimpleNamespace(
        range_names=("clip",),
        range_starts=np.asarray([5]),
        range_stops=np.asarray([15]),
        range_shard_indices=np.asarray([0]),
        range_mirror=np.asarray([False]),
        motion_files=[shard_path],
        motion_dim=4,
    )
    window, source, metadata = _read_range_window(store, 0, 1, 3, history=4)
    assert window.shape == (7, 4)
    np.testing.assert_array_equal(window[:3], np.repeat(shard[5:6], 3, axis=0))
    np.testing.assert_array_equal(source, shard[6:9])
    assert metadata["absolute_start"] == 6


def test_edit_pair_supports_all_part_fsq_families():
    torch.manual_seed(17)
    pair = torch.randn(2, 64, MOTION_DIM)
    for family, model in _models():
        model.family = family
        model.eval()
        with torch.no_grad():
            result = _edit_pair(model, pair, "left_arm", _metadata(model))
        assert result["target_recon_state"].shape == (1, 64, MOTION_DIM)
        assert result["donor_recon_state"].shape == (1, 64, MOTION_DIM)
        assert result["edited_recon_state"].shape == (1, 64, MOTION_DIM)
        assert result["target_indices"].shape == (1, 64, 40)
        if family in {"part_fsq", "residual_part_fsq"}:
            part_slice = _coordinate_slice(_metadata(model), "left_arm")
            torch.testing.assert_close(
                result["edited_indices"][..., part_slice],
                result["donor_indices"][..., part_slice],
            )
        else:
            assert result["edited_indices"] is None
