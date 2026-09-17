"""NEF-FSQ v1.1 staged physical objective: schedule, identity and warmup parity.

The physical ablation must not change the representation contract, and its
warmup epochs must reproduce the v1 objective exactly.  Both are checked here
without a dataset: the loss context and loss closure are pure functions of the
config plus the store statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from stylized_motion.learning.nef_fsq import NEFMotionAutoencoder
from stylized_motion.learning.nef_layout import GENO_SKELETON
from stylized_motion.learning.representation import representation_spec
from stylized_motion.learning.runner import (
    RepresentationRunner,
    build_loss_context,
    build_loss_fn,
    effective_loss_weights,
    load_experiment_config,
    physical_schedule_scale,
)

CONFIG_DIR = Path(__file__).parents[1] / "data" / "configs"
BASELINE = CONFIG_DIR / "nef_fsq_40x9.yaml"
PHYSICAL = CONFIG_DIR / "nef_fsq_40x9_physical.yaml"
PHYSICAL_FT = CONFIG_DIR / "nef_fsq_soma_packed_40x9_physical_ft.yaml"


def skeleton_from_spec(spec) -> tuple[list[str], list[int]]:
    index: dict[str, int] = {}
    names: list[str] = []
    parents: list[int] = []
    for chain in spec.chains:
        for position, name in enumerate(chain):
            if name not in index:
                index[name] = len(names)
                names.append(name)
                parents.append(-1 if position == 0 else index[chain[position - 1]])
            elif position > 0:
                assert parents[index[name]] == index[chain[position - 1]]
    return names, parents


class _StoreStub:
    """Just enough store surface for build_loss_context/build_loss_fn."""

    def __init__(self, names: list[str], parents: list[int]) -> None:
        motion_dim = 9 * len(names) + 5
        self.names = names
        self.parents = torch.tensor(parents)
        self.motion_dim = motion_dim
        self.stats = type(
            "Stats",
            (),
            {
                "offset": np.zeros(motion_dim, dtype=np.float32),
                "scale": np.full(motion_dim, 0.5, dtype=np.float32),
                "ref_pos": np.tile(np.array([0.0, 0.9, 0.0], dtype=np.float32), (len(names), 1)),
            },
        )()

    def model_feature_weights(self):
        return np.ones(self.motion_dim, dtype=np.float32)


def _loss(model, config, store, *, physical_scale: float, motion=None):
    context = build_loss_context(config, store, torch.device("cpu"))
    context["physical_scale"] = physical_scale
    loss_fn = build_loss_fn(model, context, config)
    if motion is None:
        torch.manual_seed(3)
        motion = torch.randn(2, 64, model.motion_dim) * 0.1
    batch = {
        "motion": motion,
        "loss_mask": torch.ones(2, 64, dtype=torch.bool),
        "_all_frames_valid": True,
    }
    output = model(motion, collect_metrics=False)
    return loss_fn(output, batch), context


def test_physical_config_keeps_the_representation_and_data_identity():
    baseline = load_experiment_config(BASELINE)
    physical = load_experiment_config(PHYSICAL)
    assert physical["representation"] == baseline["representation"]
    assert physical["data"] == baseline["data"]
    assert physical["sampling"] == baseline["sampling"]
    assert physical["loader"] == baseline["loader"]
    assert representation_spec(physical) == representation_spec(baseline)
    assert physical["training"]["output_dir"] != baseline["training"]["output_dir"]
    assert physical["training"]["seed"] == baseline["training"]["seed"]


def test_physical_config_declares_the_designed_weights_and_schedule():
    training = load_experiment_config(PHYSICAL)["training"]
    assert training["objective_variant"] == "recon_delta_physical_warmup"
    assert training["delta_weight"] == 3.0
    assert training["physical_warmup_epochs"] == 10
    assert training["physical_ramp_epochs"] == 20
    assert {
        key: float(training[key])
        for key in (
            "root_pos_weight",
            "root_rot_weight",
            "joint_weight",
            "contact_weight",
            "foot_slide_weight",
            "foot_height_weight",
        )
    } == {
        "root_pos_weight": 0.05,
        "root_rot_weight": 0.05,
        "joint_weight": 0.10,
        "contact_weight": 0.03,
        "foot_slide_weight": 0.05,
        "foot_height_weight": 0.02,
    }
    # The v1 baseline stays flat: no schedule keys, all physical weights zero.
    baseline = load_experiment_config(BASELINE)["training"]
    assert "objective_variant" not in baseline
    assert float(baseline["joint_weight"]) == 0.0


def test_physical_schedule_waits_then_ramps_then_holds():
    assert physical_schedule_scale(1, 10, 20) == 0.0
    assert physical_schedule_scale(10, 10, 20) == 0.0
    assert physical_schedule_scale(11, 10, 20) == pytest.approx(1 / 20)
    assert physical_schedule_scale(20, 10, 20) == pytest.approx(0.5)
    assert physical_schedule_scale(30, 10, 20) == 1.0
    assert physical_schedule_scale(31, 10, 20) == 1.0
    # A zero ramp is a step at the warmup boundary, never a division by zero.
    assert physical_schedule_scale(3, 2, 0) == 1.0
    assert physical_schedule_scale(2, 2, 0) == 0.0
    assert physical_schedule_scale(1, 0, 0) == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        physical_schedule_scale(1, -1, 0)


def test_effective_weights_scale_only_the_physical_terms():
    context = {
        "delta_weight": 3.0,
        "root_pos_weight": 0.05,
        "root_rot_weight": 0.05,
        "joint_weight": 0.10,
        "contact_weight": 0.03,
        "foot_slide_weight": 0.05,
        "foot_height_weight": 0.02,
        "physical_scale": 0.4,
    }
    weights = effective_loss_weights(context)
    assert weights["delta_weight"] == 3.0
    assert weights["joint_weight"] == pytest.approx(0.04)
    assert weights["root_pos_weight"] == pytest.approx(0.02)
    context["physical_scale"] = 0.0
    assert effective_loss_weights(context)["joint_weight"] == 0.0
    assert effective_loss_weights(context)["delta_weight"] == 3.0


def test_warmup_scale_reproduces_the_v1_objective_exactly():
    names, parents = skeleton_from_spec(GENO_SKELETON)
    store = _StoreStub(names, parents)
    torch.manual_seed(5)
    model = NEFMotionAutoencoder(names, parents, stream_dim=16).eval()
    torch.manual_seed(11)
    motion = torch.randn(2, 64, model.motion_dim) * 0.2

    baseline_values, _ = _loss(
        model, load_experiment_config(BASELINE), store, physical_scale=1.0, motion=motion
    )
    warmup_values, _ = _loss(
        model, load_experiment_config(PHYSICAL), store, physical_scale=0.0, motion=motion
    )
    for name in ("loss", "recon", "delta", "root_pos", "joint", "contact", "foot_slide", "foot_height"):
        torch.testing.assert_close(
            warmup_values[name], baseline_values[name], rtol=0.0, atol=0.0
        )
    assert float(baseline_values["root_pos"]) == 0.0
    assert float(baseline_values["joint"]) == 0.0

    full_values, _ = _loss(
        model, load_experiment_config(PHYSICAL), store, physical_scale=1.0, motion=motion
    )
    for name in ("root_pos", "root_rot", "joint", "contact", "foot_slide", "foot_height"):
        assert float(full_values[name].detach()) > 0.0, name
    assert float(full_values["loss"]) > float(warmup_values["loss"])
    expected = (
        float(full_values["recon"])
        + 3.0 * float(full_values["delta"])
        + 0.05 * float(full_values["root_pos"])
        + 0.05 * float(full_values["root_rot"])
        + 0.10 * float(full_values["joint"])
        + 0.03 * float(full_values["contact"])
        + 0.05 * float(full_values["foot_slide"])
        + 0.02 * float(full_values["foot_height"])
    )
    assert float(full_values["loss"]) == pytest.approx(expected, rel=1e-5)

    half_values, _ = _loss(
        model, load_experiment_config(PHYSICAL), store, physical_scale=0.5, motion=motion
    )
    half_expected = float(half_values["recon"]) + 3.0 * float(half_values["delta"]) + 0.5 * (
        float(full_values["loss"]) - float(warmup_values["loss"])
    )
    assert float(half_values["loss"]) == pytest.approx(half_expected, rel=1e-5)


def test_runner_applies_the_schedule_to_the_loss_context():
    config = load_experiment_config(PHYSICAL)
    names, parents = skeleton_from_spec(GENO_SKELETON)
    store = _StoreStub(names, parents)
    context = build_loss_context(config, store, torch.device("cpu"))
    model = NEFMotionAutoencoder(names, parents, stream_dim=8).eval()
    runner = RepresentationRunner(
        model,
        family="nef_fsq",
        train_loader=None,
        val_loader=None,
        test_loader=None,
        loss_fn=build_loss_fn(model, context, config),
        metric_suite={},
        checkpoint_manager=None,  # type: ignore[arg-type]
        config=config,
        feature_schema={},
        feature_stats={},
        device=torch.device("cpu"),
        epochs=100,
        optimizer=None,
        loss_context=context,
    )
    # A freshly built runner is already scheduled for its starting epoch.
    assert context["physical_scale"] == 0.0
    assert runner.apply_objective_schedule(1) == 0.0
    assert context["physical_scale"] == 0.0
    assert runner.apply_objective_schedule(20) == pytest.approx(0.5)
    assert context["physical_scale"] == pytest.approx(0.5)
    assert runner.apply_objective_schedule(40) == 1.0


def test_staged_objective_requires_a_loss_context_and_a_known_variant():
    config = load_experiment_config(PHYSICAL)
    names, parents = skeleton_from_spec(GENO_SKELETON)
    model = NEFMotionAutoencoder(names, parents, stream_dim=8).eval()
    with pytest.raises(ValueError, match="loss context"):
        RepresentationRunner(
            model,
            family="nef_fsq",
            train_loader=None,
            val_loader=None,
            test_loader=None,
            loss_fn=lambda output, batch: {"loss": torch.zeros(())},
            metric_suite={},
            checkpoint_manager=None,  # type: ignore[arg-type]
            config=config,
            feature_schema={},
            feature_stats={},
            device=torch.device("cpu"),
            epochs=1,
            optimizer=None,
        )
    broken = dict(config)
    training = dict(config["training"])
    training["objective_variant"] = "something_else"
    broken["training"] = training
    with pytest.raises(ValueError, match="objective_variant"):
        RepresentationRunner(
            model,
            family="nef_fsq",
            train_loader=None,
            val_loader=None,
            test_loader=None,
            loss_fn=lambda output, batch: {"loss": torch.zeros(())},
            metric_suite={},
            checkpoint_manager=None,  # type: ignore[arg-type]
            config=broken,
            feature_schema={},
            feature_stats={},
            device=torch.device("cpu"),
            epochs=1,
            optimizer=None,
        )


def test_loss_context_rejects_an_unknown_objective_variant():
    config = load_experiment_config(PHYSICAL)
    names, parents = skeleton_from_spec(GENO_SKELETON)
    store = _StoreStub(names, parents)
    training = dict(config["training"])
    training["objective_variant"] = "physical_only"
    config["training"] = training
    with pytest.raises(ValueError, match="objective_variant"):
        build_loss_context(config, store, torch.device("cpu"))

def test_physical_schedule_is_relative_to_the_run_start():
    """A resumed fine-tune ramps from its own start, not from the checkpoint's epoch."""
    config = load_experiment_config(PHYSICAL)
    names, parents = skeleton_from_spec(GENO_SKELETON)
    store = _StoreStub(names, parents)
    context = build_loss_context(config, store, torch.device("cpu"))
    model = NEFMotionAutoencoder(names, parents, stream_dim=8).eval()

    def runner_with(start_epoch: int) -> RepresentationRunner:
        runner = RepresentationRunner(
            model,
            family="nef_fsq",
            train_loader=None,
            val_loader=None,
            test_loader=None,
            loss_fn=build_loss_fn(model, context, config),
            metric_suite={},
            checkpoint_manager=None,  # type: ignore[arg-type]
            config=config,
            feature_schema={},
            feature_stats={},
            device=torch.device("cpu"),
            epochs=100,
            optimizer=None,
            loss_context=context,
        )
        runner.start_epoch = start_epoch
        return runner

    # PHYSICAL uses warmup 10 / ramp 20.
    fresh = runner_with(1)
    assert fresh.apply_objective_schedule(10) == 0.0
    assert fresh.apply_objective_schedule(20) == pytest.approx(0.5)
    assert fresh.apply_objective_schedule(30) == 1.0
    # Resuming at epoch 72 counts as this run's epoch 1: the ramp restarts there
    # instead of treating epoch 73 as already past a 10+20 schedule.
    resumed = runner_with(73)
    assert resumed.apply_objective_schedule(73) == 0.0
    assert resumed.apply_objective_schedule(82) == 0.0
    assert resumed.apply_objective_schedule(83) == pytest.approx(1 / 20)
    assert resumed.apply_objective_schedule(92) == pytest.approx(0.5)
    assert resumed.apply_objective_schedule(102) == 1.0


def test_reset_scheduler_flag_is_read_from_training_config():
    config = load_experiment_config(PHYSICAL_FT)
    assert config["training"]["reset_scheduler_on_resume"] is True
    assert config["training"]["physical_warmup_epochs"] == 2
    assert config["training"]["physical_ramp_epochs"] == 6
    assert config["training"]["objective_variant"] == "recon_delta_physical_warmup"
    assert load_experiment_config(BASELINE)["training"].get("reset_scheduler_on_resume") is None
    # The fine-tune ramps over 2 + 6 epochs counted from the resume point.
    ft = load_experiment_config(PHYSICAL_FT)
    assert {
        epoch: physical_schedule_scale(epoch, ft["training"]["physical_warmup_epochs"],
                                       ft["training"]["physical_ramp_epochs"])
        for epoch in (1, 2, 3, 5, 8, 9)
    } == {1: 0.0, 2: 0.0, 3: pytest.approx(1 / 6), 5: pytest.approx(0.5), 8: 1.0, 9: 1.0}
    # Representation identity is unchanged: the fine-tune stays a NEF-FSQ 40x9.
    assert representation_spec(config) == representation_spec(load_experiment_config(BASELINE))
