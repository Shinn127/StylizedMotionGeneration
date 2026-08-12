from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stylized_motion.learning.representation import (
    FLAT_FSQ_FAMILY,
    LATENT_RESIDUAL_FSQ_FAMILY,
    LATENT_RESIDUAL_FSQ_V2_FAMILY,
    PART_FSQ_FAMILY,
    RESIDUAL_PART_FSQ_FAMILY,
    build_representation,
    load_representation_checkpoint,
    representation_spec,
)
from stylized_motion.learning.runner import (
    RepresentationRunner,
    _matches_requested_device,
    load_experiment_config,
)
from stylized_motion.learning.checkpoint import CheckpointManager


def _skeleton() -> tuple[list[str], list[int]]:
    names = [
        "Simulation", "Hips", "Spine", "Spine1", "Neck",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeEnd",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeEnd",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFinger",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFinger",
    ]
    parents = [
        -1, 0, 1, 2, 3,
        1, 5, 6, 7, 8,
        1, 10, 11, 12, 13,
        3, 15, 16, 17, 18,
        3, 20, 21, 22, 23,
    ]
    return names, parents


def _config(name: str, *, small: bool = True) -> dict[str, object]:
    config = load_experiment_config(Path(__file__).parents[1] / "data" / "configs" / name)
    names, parents = _skeleton()
    representation = dict(config["representation"])
    model_config = dict(representation["config"])
    if representation["family"] == FLAT_FSQ_FAMILY:
        model_config.update({"motion_dim": 230, "code_dim": 8, "width": 8})
    else:
        model_config.update({"motion_dim": 230, "names": names, "parents": parents})
        if small:
            if representation["family"] == PART_FSQ_FAMILY:
                model_config.update({"stream_dim": 8})
            else:
                model_config.update({"base_code_dim": 16, "base_width": 16, "part_state_dim": 8})
            if representation["family"] == RESIDUAL_PART_FSQ_FAMILY:
                model_config.update({"residual_decoder_dim": 16, "residual_decoder_width": 16, "residual_hidden_dim": 8})
            if representation["family"] == LATENT_RESIDUAL_FSQ_FAMILY:
                model_config.update({"part_predictor_hidden_dim": 8, "latent_projector_hidden_dim": 8, "part_latent_dims": [4, 3, 3, 3, 3]})
            if representation["family"] == LATENT_RESIDUAL_FSQ_V2_FAMILY:
                model_config.update({"part_predictor_hidden_dim": 8, "part_encoder_width": 8})
    representation["config"] = model_config
    config["representation"] = representation
    return config


def test_canonical_configs_and_specs_are_explicit():
    expected = {
        "flat_fsq_40x9.yaml": (FLAT_FSQ_FAMILY, "flat", "flat_fsq_40x9"),
        "part_fsq_40x9.yaml": (PART_FSQ_FAMILY, "hierarchical", "part_fsq_40x9"),
        "residual_part_fsq_40x9.yaml": (RESIDUAL_PART_FSQ_FAMILY, "default", "residual_part_fsq_40x9"),
        "latent_residual_fsq_40x9.yaml": (LATENT_RESIDUAL_FSQ_FAMILY, "v2", "latent_residual_fsq_40x9"),
        "latent_residual_fsq_v2_40x9.yaml": (LATENT_RESIDUAL_FSQ_V2_FAMILY, "v2", "latent_residual_fsq_v2_40x9"),
    }
    for filename, identity in expected.items():
        spec = representation_spec(_config(filename))
        assert (spec.family, spec.variant, spec.representation_id) == identity
        assert (spec.num_coordinates, spec.num_levels) == (40, 9)
        assert (spec.receptive_field, spec.lookahead_frames, spec.history_frames) == (64, 0, 63)


def test_unindexed_cuda_request_accepts_the_current_cuda_device():
    assert _matches_requested_device(torch.device("cuda:0"), torch.device("cuda"))
    assert not _matches_requested_device(torch.device("cuda:1"), torch.device("cuda:0"))


def test_builder_roundtrip_contract_for_all_four_lines():
    torch.manual_seed(9)
    motion = torch.randn(1, 64, 230)
    for filename in (
        "flat_fsq_40x9.yaml",
        "part_fsq_40x9.yaml",
        "residual_part_fsq_40x9.yaml",
        "latent_residual_fsq_40x9.yaml",
        "latent_residual_fsq_v2_40x9.yaml",
    ):
        representation = build_representation(_config(filename))
        output = representation(motion, collect_metrics=False)
        assert output["recon_state"].shape == motion.shape
        assert output["codes"].shape == (1, 64, 40)
        assert output["indices"].shape == (1, 64, 40)
        assert output["indices"].dtype == torch.long
        assert int(output["indices"].min()) >= 0
        assert int(output["indices"].max()) < 9
        torch.testing.assert_close(
            representation.decode_from_indices(output["indices"]),
            output["recon_state"],
            rtol=0.0,
            atol=0.0,
        )
        metadata = representation.representation_metadata()
        assert metadata["lookahead_frames"] == 0
        assert metadata["receptive_field"] == 64
        assert sum(metadata["coordinate_counts"].values()) == 40


def test_checkpoint_metadata_is_complete_and_load_is_strict(tmp_path):
    config = _config("flat_fsq_40x9.yaml")
    representation = build_representation(config)
    feature_schema = {
        "name": "motion_feature_v2",
        "motion_dim": 230,
        "joint_subset": "pruned",
        "names_sha256": "names",
        "stats_sha256": "stats",
    }
    runner = RepresentationRunner(
        representation,
        family=FLAT_FSQ_FAMILY,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
        metric_suite={},
        checkpoint_manager=CheckpointManager(tmp_path),
        config=config,
        feature_schema=feature_schema,
        feature_stats={"offset": np.zeros(230, dtype=np.float32), "scale": np.ones(230, dtype=np.float32)},
        device=torch.device("cpu"),
        epochs=1,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(runner.checkpoint_payload(epoch=1, metrics={}), path)
    checkpoint, loaded = load_representation_checkpoint(path, torch.device("cpu"), feature_schema=feature_schema)
    assert checkpoint["schema_version"] == 2
    assert checkpoint["representation"]["representation_id"] == "flat_fsq_40x9"
    assert checkpoint["representation"]["feature_schema"] == feature_schema
    assert loaded.representation_id == "flat_fsq_40x9"


def test_checkpoint_loader_rebuilds_part_layout_from_model_config(tmp_path):
    config = _config("part_fsq_40x9.yaml")
    representation = build_representation(config)
    feature_schema = {
        "name": "motion_feature_v2",
        "motion_dim": 230,
        "joint_subset": "pruned",
        "names_sha256": "names",
        "stats_sha256": "stats",
    }
    runner = RepresentationRunner(
        representation,
        family=PART_FSQ_FAMILY,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
        metric_suite={},
        checkpoint_manager=CheckpointManager(tmp_path),
        config=config,
        feature_schema=feature_schema,
        feature_stats={"offset": np.zeros(230, dtype=np.float32), "scale": np.ones(230, dtype=np.float32)},
        device=torch.device("cpu"),
        epochs=1,
    )
    path = tmp_path / "part_checkpoint.pt"
    torch.save(runner.checkpoint_payload(epoch=1, metrics={}), path)
    _, loaded = load_representation_checkpoint(path, torch.device("cpu"), feature_schema=feature_schema)
    assert loaded.representation_id == "part_fsq_40x9"
    assert loaded.module.layout.names[0] == "Simulation"


def test_common_runner_owns_train_validate_test_lifecycle(tmp_path):
    config = _config("flat_fsq_40x9.yaml")
    representation = build_representation(config)
    loader = DataLoader([{"motion": torch.zeros(64, 230)} for _ in range(2)], batch_size=1)
    runner = RepresentationRunner(
        representation,
        family=FLAT_FSQ_FAMILY,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
        metric_suite={},
        checkpoint_manager=CheckpointManager(tmp_path),
        config=config,
        feature_schema={"name": "motion_feature_v2", "motion_dim": 230},
        feature_stats={"offset": np.zeros(230, dtype=np.float32), "scale": np.ones(230, dtype=np.float32)},
        device=torch.device("cpu"),
        epochs=1,
        optimizer=torch.optim.SGD(representation.parameters(), lr=0.001),
    )
    train = runner.run("train")
    validate = runner.run("validate")
    test = runner.run("test")
    assert train["global_step"] == 2
    assert validate["mode"] == "val"
    assert test["mode"] == "test"


def test_compact_output_preserves_the_uniform_training_contract():
    torch.manual_seed(21)
    motion = torch.randn(1, 64, 230)
    for filename in (
        "flat_fsq_40x9.yaml",
        "part_fsq_40x9.yaml",
        "residual_part_fsq_40x9.yaml",
        "latent_residual_fsq_40x9.yaml",
        "latent_residual_fsq_v2_40x9.yaml",
    ):
        representation = build_representation(_config(filename))
        with torch.no_grad():
            full = representation(motion, collect_metrics=False)
            compact = representation(motion, collect_metrics=False, compact_output=True)
        for key in ("recon_state", "codes", "indices", "commit_loss", "representation_metrics"):
            assert key in compact
        torch.testing.assert_close(compact["recon_state"], full["recon_state"])
        torch.testing.assert_close(compact["codes"], full["codes"])
        torch.testing.assert_close(compact["indices"], full["indices"])
        assert "fsq_codes" not in compact
        assert "group_codes" not in compact
        assert "group_indices" not in compact
        assert "part_codes" not in compact
        assert "part_indices" not in compact
        assert "part_residuals" not in compact
        assert "part_latent_residuals" not in compact


class _MetricIntervalProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))
        self.metric_flags: list[bool] = []

    def forward(self, motion, *, collect_metrics, compact_output):
        self.metric_flags.append(bool(collect_metrics))
        output = {
            "recon_state": motion + self.scale,
            "codes": torch.zeros((*motion.shape[:2], 40), device=motion.device),
            "indices": torch.zeros((*motion.shape[:2], 40), dtype=torch.long, device=motion.device),
            "commit_loss": motion.new_zeros(()),
        }
        if collect_metrics:
            output["representation_metrics"] = {"probe": motion.new_ones(())}
        return output


def test_training_metrics_interval_samples_without_changing_loss_frequency(tmp_path):
    probe = _MetricIntervalProbe()
    loader = DataLoader([{"motion": torch.zeros(64, 230)} for _ in range(5)], batch_size=1)
    runner = RepresentationRunner(
        probe,
        family=FLAT_FSQ_FAMILY,
        train_loader=loader,
        val_loader=None,
        test_loader=None,
        loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
        metric_suite={},
        checkpoint_manager=CheckpointManager(tmp_path),
        config={"evaluation": {"metrics_interval": 2}},
        feature_schema={},
        feature_stats={},
        device=torch.device("cpu"),
        epochs=1,
        optimizer=torch.optim.SGD(probe.parameters(), lr=0.001),
    )
    result = runner.train_epoch(1)
    assert probe.metric_flags == [True, False, True, False, True]
    assert result["samples"] == 5.0
    assert result["valid_frames"] == 5.0 * 64.0
    assert "loss" in result
    assert "representation/probe" in result
