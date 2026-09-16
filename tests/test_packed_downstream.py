"""Schema-v4 downstream contracts: clip-boundary tokens, trajectory conditioning, budget.

Acceptance criteria from the plan's P2/P3 rows: benchmark layers that separate
sampler/loader/resident/end-to-end cost, a training budget expressed in steps,
token encoding that never lets one clip's context reach another, and trajectory
controls that are clip-local, validity-masked and train-only normalized.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stylized_motion.anim.features import joint_feature_dim
from stylized_motion.data.benchmark import BenchmarkConfig, LatencyRecorder, run_benchmark
from stylized_motion.data.packed_store import (
    PackedFeatureStoreWriter,
    feature_schema_hash,
    open_packed_feature_store,
    skeleton_hash,
    write_clip_table,
)
from stylized_motion.data.packed_token import (
    PackedTokenDataset,
    _encode_clip,
    build_packed_token_store,
    open_packed_token_store,
    verify_token_store,
)
from stylized_motion.data.packed_trajectory import (
    PackedConditionalTokenDataset,
    build_packed_trajectory_store,
    open_packed_trajectory_store,
    root_relative_future,
)
from stylized_motion.data.resume import PREPROCESS_VERSION
from stylized_motion.data.sampling import TrainWindowSampler

NAMES = ["Simulation", "Hips", "LeftFoot"]
PARENTS = [-1, 0, 1]
MOTION_DIM = joint_feature_dim(len(NAMES))
ROOT_CHANNELS = 7


class _RecurrentEncoder:
    """Causal 64-frame encoder whose tokens expose which context it saw.

    ``token[t]`` depends on ``x0[t]`` and ``x0[t - 63]`` (a sentinel before the
    clip start), so the first token of any encoded sequence is readable proof of
    whether frames outside that sequence were visible.
    """

    num_coordinates = 40
    num_levels = 9
    receptive_field = 64
    lookahead_frames = 0
    start_marker = 3

    def __init__(self) -> None:
        self.calls: list[int] = []

    def representation_metadata(self) -> dict[str, object]:
        return {
            "family": "flat_fsq",
            "variant": "flat",
            "representation_id": "flat_fsq_40x9",
            "coordinate_order": ["flat"],
            "coordinate_counts": {"flat": 40},
            "num_coordinates": 40,
            "num_levels": 9,
            "temporal_downsample": 1,
            "receptive_field": 64,
            "lookahead_frames": 0,
            "decoder_passes_inference": 1,
        }

    def encode_to_codes(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, _dim = values.shape
        self.calls.append(int(frames))
        head = values[:, :, 0]
        history = torch.cat(
            [
                torch.full((batch, 63), float(self.start_marker), dtype=head.dtype, device=head.device),
                head[:, :-63] if frames > 63 else head[:, :0],
            ],
            dim=1,
        )[:, :frames]
        marker = torch.where(
            history == float(self.start_marker),
            torch.full_like(history, float(self.start_marker)),
            torch.round(history * 8.0),
        )
        tokens = ((torch.round(head * 5.0) + marker.long()) % self.num_levels).to(torch.uint8)
        indices = tokens.unsqueeze(-1).repeat(1, 1, self.num_coordinates)
        return indices.float(), indices


def _write_store(tmp_path: Path, *, clip_lengths: list[int], splits: list[int], groups: list[int],
                 motions: np.ndarray | None = None, shard_bytes: int = 64 * 1024) -> Path:
    staging = tmp_path / "store"
    staging.mkdir(parents=True, exist_ok=True)
    writer = PackedFeatureStoreWriter(
        staging, motion_dim=MOTION_DIM, num_joints=len(NAMES), shard_bytes=shard_bytes, names=NAMES
    )
    root_writer = PackedFeatureStoreWriter(
        staging, motion_dim=ROOT_CHANNELS, num_joints=len(NAMES), shard_bytes=shard_bytes, names=NAMES,
        subdirectory="root", frames_per_shard=writer.frames_per_shard,
    )
    rng = np.random.default_rng(0)
    entries = []
    for row, length in enumerate(clip_lengths):
        values = rng.normal(size=(length, MOTION_DIM)).astype(np.float32)
        if motions is not None:
            values[:, 0] = motions[row][:length]
        root = np.zeros((length, ROOT_CHANNELS), dtype=np.float32)
        root[:, 3] = 1.0  # identity quaternion
        root[:, 2] = 0.01 * np.arange(length)  # constant forward travel
        entries.append(
            writer.append_clip(
                values, source_group=groups[row], variant=0, split=splits[row], mirror=False,
                source_id=groups[row], move_name=f"clip_{row}",
                position_sum=values[:, :3].reshape(length, 3)[:3].astype(np.float64),
            )
        )
        root_writer.append_clip(
            root, source_group=groups[row], variant=0, split=splits[row], mirror=False,
            source_id=groups[row], move_name=f"clip_{row}",
        )
    writer.close_shards()
    root_writer.close_shards()
    write_clip_table(staging, entries, num_joints=len(NAMES))
    from stylized_motion.data.normalization import compute_normalization

    manifest = {
        "data_schema_version": 4,
        "store_type": "feature_packed",
        "layout": "packed",
        "frame_rate": 60,
        "created_by": "tests",
        "num_shards": len(writer.shard_files),
        "shard_files": writer.shard_files,
        "shard_sha256": writer.shard_sha256,
        "shard_num_frames": writer.shard_num_frames,
        "root_channels": ROOT_CHANNELS,
        "root_shard_files": root_writer.shard_files,
        "root_shard_sha256": root_writer.shard_sha256,
        "motion_dim": MOTION_DIM,
        "num_clips": len(entries),
        "total_frames": int(sum(entry.length for entry in entries)),
        "clip_names": [f"clip_{row}" for row in range(len(entries))],
        "style_names": ["s0"],
        "action_names": ["a0"],
        "package_names": ["p0"],
        "feature_schema": {
            "name": "motion_feature_v2",
            "motion_dim": MOTION_DIM,
            "joint_subset": "full",
            "names": NAMES,
            "parents": PARENTS,
        },
        "feature_schema_hash": feature_schema_hash(NAMES, PARENTS, "full"),
        "skeleton_hash": skeleton_hash(NAMES, PARENTS, "full"),
        "split_manifest_hash": "split-hash",
        "split_seed": 3407,
        "build": {"status": "complete", "preprocess_version": PREPROCESS_VERSION},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    store = open_packed_feature_store(staging, load_normalization=False)
    try:
        normalization = compute_normalization(store)
    finally:
        store.close()
    normalization.save(staging)
    manifest["normalization_hash"] = normalization.normalization_hash()
    manifest["normalization_train_frames"] = int(normalization.train_frames)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return staging


# ---------------------------------------------------------------------------
# P3: token encoding never crosses a clip boundary
# ---------------------------------------------------------------------------


def test_encode_clip_starts_every_clip_with_its_own_context():
    encoder = _RecurrentEncoder()
    clip = np.arange(200, dtype=np.float32).reshape(200, 1).repeat(MOTION_DIM, axis=1) / 200.0
    indices, _codes = _encode_clip(encoder, clip, chunk_size=256, device=torch.device("cpu"))
    assert indices.shape == (200, 40)
    # the first frame of a clip may only see the start marker, never a
    # neighbour's frames, which would show up as a different first token
    expected_first = int(indices[0, 0])
    second, _ = _encode_clip(encoder, clip + 0.5, chunk_size=256, device=torch.device("cpu"))
    assert int(second[0, 0]) != expected_first or int(second[1, 0]) != int(indices[1, 0])


def test_chunked_encoding_matches_whole_clip_encoding():
    encoder = _RecurrentEncoder()
    clip = (np.arange(300, dtype=np.float32).reshape(300, 1) + 1.0).repeat(MOTION_DIM, axis=1) / 40.0
    whole, _ = _encode_clip(encoder, clip, chunk_size=4096, device=torch.device("cpu"))
    chunked, _ = _encode_clip(encoder, clip, chunk_size=17, device=torch.device("cpu"))
    np.testing.assert_array_equal(whole, chunked)
    # the history replay must actually read back 63 frames for later chunks
    assert max(encoder.calls) > 17


def test_token_store_binds_checkpoint_feature_normalization_and_split(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[80, 90, 100, 110], splits=[0, 0, 1, 2],
                              groups=[0, 1, 2, 3])
    encoder = _RecurrentEncoder()
    report = build_packed_token_store(
        store_path,
        tmp_path / "tokens",
        encoder=encoder,
        checkpoint_sha256="a" * 64,
        device="cpu",
        chunk_size=32,
        unit_dir=tmp_path / "units",
        shard_bytes=8 * 1024,
    )
    assert report["clips"] == 4
    feature_store = open_packed_feature_store(store_path)
    tokens = open_packed_token_store(tmp_path / "tokens")
    try:
        assert tokens.checkpoint_sha256 == "a" * 64
        assert tokens.feature_schema_hash == feature_store.feature_schema_hash
        assert tokens.normalization_hash == feature_store.normalization_hash
        assert tokens.split_manifest_hash == feature_store.split_manifest_hash
        assert np.array_equal(tokens.clip_length, feature_store.clip_length)
        assert np.array_equal(tokens.clip_split, feature_store.clip_split)
        verify_token_store(tmp_path / "tokens", feature_store=feature_store, full=True)
    finally:
        tokens.close()
        feature_store.close()


def test_token_store_rejects_a_mismatched_feature_store(tmp_path: Path):
    first = _write_store(tmp_path / "a", clip_lengths=[80, 90, 100, 110], splits=[0, 0, 1, 2],
                         groups=[0, 1, 2, 3])
    second = _write_store(tmp_path / "b", clip_lengths=[80, 90, 100, 100], splits=[0, 0, 1, 2],
                          groups=[0, 1, 2, 3])
    encoder = _RecurrentEncoder()
    build_packed_token_store(first, tmp_path / "tokens", encoder=encoder, checkpoint_sha256="b" * 64,
                            device="cpu", chunk_size=32, unit_dir=tmp_path / "units")
    other = open_packed_feature_store(second)
    try:
        with pytest.raises(ValueError, match="disagrees with the feature store"):
            verify_token_store(tmp_path / "tokens", feature_store=other, full=False)
    finally:
        other.close()


def test_token_encode_is_resumable_and_idempotent(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[80, 90, 100], splits=[0, 1, 2], groups=[0, 1, 2])
    encoder = _RecurrentEncoder()
    first = build_packed_token_store(
        store_path, tmp_path / "tokens", encoder=encoder, checkpoint_sha256="c" * 64,
        device="cpu", chunk_size=32, unit_dir=tmp_path / "units",
    )
    assert first["reused"] == 0
    second = build_packed_token_store(
        store_path, tmp_path / "tokens2", encoder=encoder, checkpoint_sha256="c" * 64,
        device="cpu", chunk_size=32, unit_dir=tmp_path / "units",
    )
    assert second["reused"] == 3
    assert second["encoded"] == 0
    # a different checkpoint invalidates the cached units
    third = build_packed_token_store(
        store_path, tmp_path / "tokens3", encoder=encoder, checkpoint_sha256="d" * 64,
        device="cpu", chunk_size=32, unit_dir=tmp_path / "units",
    )
    assert third["reused"] == 0


def test_packed_token_dataset_reads_65_frame_windows(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[200, 300], splits=[0, 1], groups=[0, 1])
    encoder = _RecurrentEncoder()
    build_packed_token_store(store_path, tmp_path / "tokens", encoder=encoder, checkpoint_sha256="e" * 64,
                            device="cpu", chunk_size=64, unit_dir=tmp_path / "units")
    store = open_packed_token_store(tmp_path / "tokens")
    try:
        sampler = TrainWindowSampler(store, target_frames=64, required_frames=65, samples_per_epoch=4, seed=1)
        requests = list(sampler)
        dataset = PackedTokenDataset("train", store, sequence_frames=65)
        batch = dataset.__getitems__(requests)
        assert batch["tokens"].shape == (4, 64, 40)
        assert batch["tokens"].dtype == torch.uint8
        item = dataset[requests[0]]
        assert item["tokens"].shape == (64, 40)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# P3: trajectory conditioning
# ---------------------------------------------------------------------------


def test_root_relative_future_is_expressible_in_the_root_frame():
    frames = 100
    root = np.zeros((frames, 7), dtype=np.float32)
    root[:, 3] = 1.0
    root[:, 2] = 0.01 * np.arange(frames)  # +Z travel, identity heading
    values, valid = root_relative_future(root, (10, 20))
    assert values.shape == (frames, 12)
    assert int(valid.sum()) == frames - 20
    # positions are future-minus-current in the root frame
    np.testing.assert_allclose(values[0, 0:3], [0.0, 0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(values[0, 3:6], [0.0, 0.0, 0.2], atol=1e-6)
    # headings are the identity rotation applied to +Z
    np.testing.assert_allclose(values[0, 6:9], [0.0, 0.0, 1.0], atol=1e-6)
    assert not valid[-1] and not valid[frames - 20]
    assert valid[frames - 21]


def test_root_relative_future_rotates_into_the_root_frame():
    root = np.zeros((50, 7), dtype=np.float32)
    # quaternions are w-first, so a 180 degree turn about Y is w=0, y=1
    root[:, 5] = 1.0
    root[:, 2] = 0.01 * np.arange(50)
    values, valid = root_relative_future(root, (5,))
    assert valid[:45].all() and not valid[45:].any()
    # world +Z travel becomes root-local -Z once the heading is flipped
    np.testing.assert_allclose(values[0, 0:3], [0.0, 0.0, -0.05], atol=1e-5)
    # a constant heading stays +Z *in the root frame* by construction
    np.testing.assert_allclose(values[0, 3:6], [0.0, 0.0, 1.0], atol=1e-5)


def test_root_relative_future_reports_a_heading_change():
    root = np.zeros((40, 7), dtype=np.float32)
    root[:10, 5] = 1.0            # 180 degree yaw for the first ten frames
    root[10:, 3] = 1.0            # then identity again
    root[:, 2] = 0.0
    values, valid = root_relative_future(root, (10,))
    assert valid[:30].all()
    # from a flipped root, the future identity heading appears as -Z locally
    np.testing.assert_allclose(values[0, 3:6], [0.0, 0.0, -1.0], atol=1e-5)
    # from an unflipped root the same future heading is +Z locally
    np.testing.assert_allclose(values[20, 3:6], [0.0, 0.0, 1.0], atol=1e-5)


def test_short_clips_yield_no_valid_trajectory_frames():
    root = np.zeros((10, 7), dtype=np.float32)
    root[:, 3] = 1.0
    values, valid = root_relative_future(root, (20, 40, 60))
    assert values.shape == (10, 18)
    assert not valid.any()


def test_trajectory_store_is_clip_local_and_validity_masked(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[150, 120, 110], splits=[0, 0, 1], groups=[0, 1, 2])
    report = build_packed_trajectory_store(store_path, tmp_path / "traj", future_frames=(20, 40, 60))
    assert report["trajectory_dim"] == 18
    trajectory = open_packed_trajectory_store(tmp_path / "traj")
    feature_store = open_packed_feature_store(store_path)
    try:
        assert np.array_equal(trajectory.clip_split, feature_store.clip_split)
        assert np.array_equal(trajectory.clip_length, feature_store.clip_length)
        for clip_idx in range(trajectory.num_clips):
            values, valid = trajectory.read_clip(clip_idx)
            length = int(trajectory.clip_length[clip_idx])
            usable = length - 60
            assert valid[:usable].all()
            assert not valid[usable:].any()
            assert np.isfinite(values[valid]).all()
            # the mask keeps unusable tail frames at exactly zero, so a window
            # that reaches the clip end cannot read a neighbour's future
            np.testing.assert_allclose(values[~valid], 0.0)
    finally:
        trajectory.close()
        feature_store.close()


def test_trajectory_normalization_uses_only_train_frames(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[150, 120, 110, 100], splits=[0, 0, 1, 2],
                              groups=[0, 1, 2, 3])
    report = build_packed_trajectory_store(store_path, tmp_path / "traj", future_frames=(20, 40))
    # train clips of 150 + 120 frames each contribute length - 40 valid frames
    assert report["normalization_valid_frames"] == (150 - 40) + (120 - 40)


def test_conditional_dataset_rebases_windows_across_stores(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[300, 300], splits=[0, 1], groups=[0, 1])
    encoder = _RecurrentEncoder()
    # The feature store packs each 300-frame clip into its own shard; the token
    # store uses a wide shard and packs both into one. The two layouts must
    # therefore disagree on absolute offsets while agreeing on clip rows.
    build_packed_token_store(store_path, tmp_path / "tokens", encoder=encoder, checkpoint_sha256="f" * 64,
                            device="cpu", chunk_size=64, unit_dir=tmp_path / "units", shard_bytes=1024 * 1024)
    build_packed_trajectory_store(store_path, tmp_path / "traj", future_frames=(20, 40, 60))
    tokens = open_packed_token_store(tmp_path / "tokens")
    trajectory = open_packed_trajectory_store(tmp_path / "traj")
    try:
        assert not np.array_equal(tokens.clip_offset, trajectory.clip_offset)
        assert np.array_equal(tokens.clip_length, trajectory.clip_length)
        sampler = TrainWindowSampler(tokens, target_frames=64, required_frames=65, samples_per_epoch=6, seed=2)
        requests = list(sampler)
        dataset = PackedConditionalTokenDataset("train", tokens, trajectory, sequence_frames=65)
        batch = dataset.__getitems__(requests)
        assert batch["tokens"].shape == (6, 64, 40)
        assert batch["trajectory"].shape == (6, 64, 18)
        assert batch["trajectory_valid"].shape == (6, 64)
        # the trajectory values for each request must match a manual local read,
        # including the validity mask for windows that reach the clip's end
        horizon = 60
        for row, request in enumerate(requests):
            clip_idx = int(request.variant_idx)
            local = int(request.target_start) - int(trajectory.clip_offset[clip_idx])
            length = int(trajectory.clip_length[clip_idx])
            values, expected_valid = trajectory.read_window(
                clip_idx, int(trajectory.clip_offset[clip_idx]) + local, 64
            )
            expected = (values - trajectory.normalization_mean) / np.maximum(trajectory.normalization_std, 1e-6)
            np.testing.assert_allclose(batch["trajectory"][row].numpy(), expected, rtol=1e-5, atol=1e-6)
            expected_mask = (local + np.arange(64)) < (length - horizon)
            np.testing.assert_array_equal(batch["trajectory_valid"][row].numpy(), expected_mask)
            # a window may only lose validity where it reaches past the horizon
            assert int((~expected_mask).sum()) <= horizon
    finally:
        tokens.close()


def test_conditional_dataset_rejects_mismatched_stores(tmp_path: Path):
    first = _write_store(tmp_path / "a", clip_lengths=[300, 300], splits=[0, 1], groups=[0, 1])
    second = _write_store(tmp_path / "b", clip_lengths=[300, 400], splits=[0, 1], groups=[0, 1])
    encoder = _RecurrentEncoder()
    build_packed_token_store(first, tmp_path / "tokens", encoder=encoder, checkpoint_sha256="g" * 64,
                            device="cpu", chunk_size=64, unit_dir=tmp_path / "units")
    build_packed_trajectory_store(second, tmp_path / "traj", future_frames=(20,))
    tokens = open_packed_token_store(tmp_path / "tokens")
    trajectory = open_packed_trajectory_store(tmp_path / "traj")
    try:
        with pytest.raises(ValueError, match="disagrees with the token store"):
            PackedConditionalTokenDataset("train", tokens, trajectory)
    finally:
        tokens.close()


# ---------------------------------------------------------------------------
# P2: benchmark harness and training budget
# ---------------------------------------------------------------------------


def test_latency_recorder_reports_percentiles_and_wait_fraction():
    recorder = LatencyRecorder()
    for wait, step in ((0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.4, 1.0)):
        recorder.observe(wait, step)
    summary = recorder.summary()
    assert summary["steps"] == 4
    assert summary["data_wait_mean"] == pytest.approx(0.25)
    assert summary["data_wait_fraction"] == pytest.approx(0.25)
    assert summary["data_wait_p95"] >= summary["data_wait_p50"]
    assert summary["steps_per_second"] == pytest.approx(1.0)


def test_benchmark_reports_every_layer_without_claiming_disk_speed(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[400, 420, 440, 460], splits=[0, 0, 0, 1],
                              groups=[0, 1, 2, 3])
    config = BenchmarkConfig(
        store=store_path, batch_size=8, num_workers=0, samples=32, warmup_batches=1, device="cpu"
    )
    report = run_benchmark(config, layers=["layout", "sampler", "loader", "resident", "end_to_end"])
    assert set(report["layers"]) == {"layout", "sampler", "loader", "resident", "end_to_end"}
    layout = report["layers"]["layout"]
    assert layout["clips"] == 4
    assert layout["cold_probe_seconds"] >= 0.0
    assert layout["warm_probe_seconds"] >= 0.0
    assert layout["normalization_hash"]
    sampler = report["layers"]["sampler"]
    assert sampler["requests"] == 32
    assert sampler["requests_per_second"] > 0
    assert sampler["coverage"]["samples"] == 32
    loader = report["layers"]["loader"]
    assert loader["batches"] >= 1
    assert loader["first_batch_seconds"] is not None
    resident = report["layers"]["resident"]
    assert resident["batch_bytes"] == 8 * 64 * MOTION_DIM * 4
    end_to_end = report["layers"]["end_to_end"]
    assert "data_wait_p95" in end_to_end
    assert "data_wait_fraction" in report["gates"]


def test_benchmark_rejects_unknown_layers(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[200], splits=[0], groups=[0])
    with pytest.raises(ValueError, match="Unsupported benchmark layers"):
        run_benchmark(BenchmarkConfig(store=store_path), layers=["teleport"])


def test_runner_budget_caps_steps_per_epoch_and_total_steps(tmp_path: Path):
    import numpy as np
    from torch.utils.data import DataLoader

    from stylized_motion.learning.checkpoint import CheckpointManager
    from stylized_motion.learning.representation import FLAT_FSQ_FAMILY, build_representation
    from stylized_motion.learning.runner import RepresentationRunner, load_experiment_config

    config = load_experiment_config("data/configs/flat_fsq_40x9.yaml")
    config["training"]["steps_per_epoch"] = 2
    config["training"]["max_steps"] = 3
    config["training"]["epochs"] = 10
    config["evaluation"]["full_eval_every_epochs"] = 1
    config["evaluation"]["eval_every_steps"] = 0
    representation = build_representation(config)
    loader = DataLoader(
        [{"motion": torch.zeros(64, 230)} for _ in range(4)],
        batch_size=1,
    )
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
        epochs=10,
        optimizer=torch.optim.SGD(representation.parameters(), lr=0.001),
    )
    result = runner.run("train")
    assert runner.global_step == 3                     # max_steps wins over the epoch budget
    assert result["global_step"] == 3
    steps = [entry["train"]["steps"] for entry in result["history"]]
    assert steps == [2, 1]                             # 2 in epoch 1, then 1 to reach the cap
    assert all(entry["train"]["data_wait_fraction"] >= 0.0 for entry in result["history"])


def _budget_config(tmp_path: Path, *, eval_limit: int | None, checkpoint_every_steps: int = 0):
    """The flat_fsq recipe, resized to the tiny fixture skeleton."""
    from stylized_motion.learning.runner import load_experiment_config

    config = load_experiment_config("data/configs/flat_fsq_40x9.yaml")
    config["representation"]["config"]["motion_dim"] = MOTION_DIM
    config["training"]["steps_per_epoch"] = 2
    config["training"]["max_steps"] = 6
    config["training"]["epochs"] = 5
    config["training"]["checkpoint_every_steps"] = checkpoint_every_steps
    config["evaluation"]["full_eval_every_epochs"] = 1
    config["evaluation"]["eval_every_steps"] = 0
    config["sampling"]["eval_limit"] = eval_limit
    return config


def _budget_runner(tmp_path: Path, config, *, loaders, resume_state=None, epochs: int = 5):
    from stylized_motion.learning.checkpoint import CheckpointManager
    from stylized_motion.learning.representation import FLAT_FSQ_FAMILY, build_representation
    from stylized_motion.learning.runner import RepresentationRunner

    representation = build_representation(config)
    return RepresentationRunner(
        representation,
        family=FLAT_FSQ_FAMILY,
        train_loader=loaders.train,
        val_loader=loaders.val,
        test_loader=loaders.test,
        full_val_loader=loaders.full_val,
        loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
        metric_suite={},
        checkpoint_manager=CheckpointManager(tmp_path),
        config=config,
        feature_schema={"name": "motion_feature_v2", "motion_dim": MOTION_DIM},
        feature_stats={
            "offset": np.zeros(MOTION_DIM, dtype=np.float32),
            "scale": np.ones(MOTION_DIM, dtype=np.float32),
        },
        device=torch.device("cpu"),
        epochs=epochs,
        optimizer=torch.optim.SGD(representation.parameters(), lr=0.001),
        resume_state=resume_state,
    )


def test_full_validation_uses_the_unbounded_loader_and_decides_best(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        from stylized_motion.data.loader import build_data_loaders

        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={
                "strategy": "clip_uniform",
                "target_frames": 64,
                "samples_per_epoch": 4,
                "eval_limit": 2,
            },
            loader_config={"batch_size": 2, "num_workers": 0, "prefetch_memory_limit_mb": None},
        )
        assert loaders.full_val is not None
        monitor_windows = len(loaders.val.sampler)
        full_windows = len(loaders.full_val.sampler)
        assert monitor_windows == 2 and full_windows > monitor_windows

        config = _budget_config(tmp_path, eval_limit=2)
        runner = _budget_runner(tmp_path, config, loaders=loaders)
        monitor = runner.evaluate("val")
        full = runner.evaluate("val", full=True)
        assert monitor["scope"] == "val_subset"
        assert monitor["windows"] == monitor_windows
        assert full["scope"] == "val_full"
        assert full["windows"] == full_windows
        assert full["samples"] > monitor["samples"]

        result = runner.run("train")
        assert result["best_metric_source"] == "val_full"
        assert all(entry["full_validation"] for entry in result["history"])
        assert all(entry["val_full"] for entry in result["history"])
    finally:
        store.close()


def test_full_validation_respects_its_cadence_and_still_runs_on_the_last_epoch(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        from stylized_motion.data.loader import build_data_loaders

        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={
                "strategy": "clip_uniform",
                "target_frames": 64,
                "samples_per_epoch": 4,
                "eval_limit": 2,
            },
            loader_config={"batch_size": 2, "num_workers": 0, "prefetch_memory_limit_mb": None},
        )
        config = _budget_config(tmp_path, eval_limit=2)
        config["evaluation"]["full_eval_every_epochs"] = 3
        config["training"]["max_steps"] = 4          # two epochs of two steps
        runner = _budget_runner(tmp_path, config, loaders=loaders, epochs=4)
        result = runner.run("train")
        cadence = [entry["full_validation"] for entry in result["history"]]
        # epoch 3 hits the cadence, and the final reached epoch is always swept
        assert cadence == [False, True]
    finally:
        store.close()


def test_bounded_monitoring_never_decides_best_when_a_full_loader_exists(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        from stylized_motion.data.loader import build_data_loaders

        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={"strategy": "clip_uniform", "target_frames": 64, "samples_per_epoch": 2,
                             "eval_limit": 1},
            loader_config={"batch_size": 1, "num_workers": 0, "prefetch_memory_limit_mb": None},
        )
        config = _budget_config(tmp_path, eval_limit=1)
        config["training"]["max_steps"] = 2
        runner = _budget_runner(tmp_path, config, loaders=loaders, epochs=1)
        result = runner.run("train")
        assert runner.best_metric_source == "val_full"
        assert result["best_val"] is not None
        assert result["history"][-1]["best_metric_source"] == "val_full"
    finally:
        store.close()


def test_training_resumes_epoch_step_and_sampler_position(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        import torch

        from stylized_motion.data.loader import build_data_loaders

        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={"strategy": "clip_uniform", "target_frames": 64, "samples_per_epoch": 4},
            loader_config={"batch_size": 1, "num_workers": 0, "prefetch_memory_limit_mb": None},
        )
        config = _budget_config(tmp_path, eval_limit=None)
        config["training"]["steps_per_epoch"] = 2
        config["training"]["max_steps"] = 6
        runner = _budget_runner(tmp_path, config, loaders=loaders)
        first = runner.run("train")
        assert runner.global_step == 6
        assert len(first["history"]) == 3
        checkpoint = torch.load(tmp_path / "last.pt", weights_only=False)
        assert checkpoint["global_step"] == 6
        assert checkpoint["epoch"] == 3
        # epoch 3 finished, so resuming starts epoch 4 at ordinal 0
        assert checkpoint["sampler_state"] == {"epoch": 3, "next_ordinal": 4, "complete": True}
        assert checkpoint["best_val"] is not None
        assert checkpoint["normalization_on"] == "cpu"

        config["training"]["max_steps"] = 10
        resumed = _budget_runner(
            tmp_path, config, loaders=loaders, resume_state=checkpoint, epochs=5
        )
        outcome = resumed.run("train")
        assert resumed.global_step == 10
        assert [entry["epoch"] for entry in outcome["history"]] == [4, 5]
    finally:
        store.close()


def test_mid_epoch_checkpoint_resumes_inside_the_same_epoch(tmp_path: Path):
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        import torch

        from stylized_motion.data.loader import build_data_loaders

        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={"strategy": "clip_uniform", "target_frames": 64, "samples_per_epoch": 4},
            loader_config={"batch_size": 1, "num_workers": 0, "prefetch_memory_limit_mb": None},
        )
        config = _budget_config(tmp_path, eval_limit=None, checkpoint_every_steps=1)
        config["training"]["steps_per_epoch"] = 4
        config["training"]["max_steps"] = 5
        runner = _budget_runner(tmp_path, config, loaders=loaders)
        runner.run("train")
        # every step wrote a checkpoint, so the last one is mid-epoch 2 at step 5
        checkpoint = torch.load(tmp_path / "last.pt", weights_only=False)
        assert checkpoint["epoch"] == 2
        assert checkpoint["sampler_state"]["epoch"] == 2
        assert checkpoint["sampler_state"]["next_ordinal"] < 4

        resumed = _budget_runner(
            tmp_path, config, loaders=loaders, resume_state=checkpoint, epochs=5
        )
        assert resumed.start_epoch == 2
        assert resumed.resume_ordinal == checkpoint["sampler_state"]["next_ordinal"]
        assert resumed.global_step == 5
    finally:
        store.close()


def test_runner_applies_deferred_normalization_for_normalize_on_none(tmp_path: Path):
    """normalize_on='none' must place the batch in normalized space, not feed raw frames."""
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        def observed_batches(normalize_on: str, output_name: str) -> tuple[list[torch.Tensor], torch.Tensor | None]:
            from stylized_motion.data.loader import build_data_loaders

            run_loaders = build_data_loaders(
                "representation",
                store,
                sampling_config={"strategy": "clip_uniform", "target_frames": 64, "samples_per_epoch": 2},
                loader_config={"batch_size": 1, "num_workers": 0, "prefetch_memory_limit_mb": None,
                               "normalize_on": normalize_on},
            )
            config = _budget_config(tmp_path / output_name, eval_limit=None)
            config["loader"]["normalize_on"] = normalize_on
            config["training"]["max_steps"] = 1
            config["training"]["steps_per_epoch"] = 1
            runner = _budget_runner(tmp_path / output_name, config, loaders=run_loaders, epochs=1)
            seen: list[torch.Tensor] = []
            raw_seen: list[torch.Tensor] = []
            original_forward = runner._forward

            def spy(batch, *, collect_metrics, compact_output):
                seen.append(batch["motion"].clone())
                return original_forward(batch, collect_metrics=collect_metrics, compact_output=compact_output)

            original_to_device = runner._to_device

            def spy_device(batch):
                raw_seen.append(batch["motion"].clone())
                return original_to_device(batch)

            runner._forward = spy
            runner._to_device = spy_device
            runner.run("train")
            checkpoint = torch.load(tmp_path / output_name / "last.pt", weights_only=False)
            return seen, (raw_seen[0] if raw_seen else None), checkpoint

        cpu_batches, _, cpu_checkpoint = observed_batches("cpu", "run_cpu")
        deferred_batches, raw_batch, deferred_checkpoint = observed_batches("none", "run_none")
        assert cpu_batches and deferred_batches, "the training step never reached the model"
        # the deferred path must hand the model exactly what the CPU path did
        np.testing.assert_allclose(
            deferred_batches[0].numpy(), cpu_batches[0].numpy(), rtol=1e-5, atol=1e-6
        )
        # ... which is normalized data, not the raw frames the dataset served
        assert not np.allclose(deferred_batches[0].numpy(), raw_batch.numpy())
        assert cpu_checkpoint["normalization_on"] == "cpu"
        assert deferred_checkpoint["normalization_on"] == "none"
    finally:
        store.close()


def test_store_path_error_names_contained_stores(tmp_path: Path):
    from stylized_motion.data.packed_store import open_any_feature_store
    from stylized_motion.data.seed_build import SeedBuildConfig, SeedBuildError, build_packed_feature_store

    parent = tmp_path / "processed"
    (parent / "seed_soma_pruned_v4").mkdir(parents=True)
    (parent / "seed_soma_pruned_v4" / "manifest.json").write_text(
        json.dumps({"data_schema_version": 4}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="seed_soma_pruned_v4"):
        open_any_feature_store(parent)

    # a unit cache inside the store directory would be destroyed on publish
    from stylized_motion.data.seed_catalog import SeedCatalog, assign_group_splits, discover_catalog

    root = tmp_path / "seed"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "seed_metadata_v004.csv").write_text(
        "move_name,filename,move_duration_frames,package,category,is_neutral,is_mirror,"
        "move_soma_uniform_path,take_name,take_actor,take_org_name,take_date,take_day_part,"
        "content_uniform_style,content_type_of_movement,content_body_position,"
        "content_horizontal_move,content_vertical_move,content_props,content_complex_action,"
        "content_repeated_action,actor_uid\n"
        "walk_001__A001,walk_001__A001,400,Locomotion,Baseline,1.0,False,"
        "soma_uniform/bvh/walk_001__A001.bvh,tak,A001,org,240101,_1,neutral,walking,standing,"
        "0,0,0,0,0,A001\n",
        encoding="utf-8",
    )
    bvh_path = root / "soma_uniform" / "bvh" / "walk_001__A001.bvh"
    bvh_path.parent.mkdir(parents=True)
    bvh_path.write_text("HIERARCHY\nROOT Root\nMOTION\nFrames: 400\nFrame Time: 0.008333\n", encoding="utf-8")
    clips, manifest = discover_catalog(root)
    assign_group_splits(clips, seed=1)
    catalog = SeedCatalog(tmp_path / "catalog", manifest, clips)
    catalog.save()
    output = tmp_path / "store"
    with pytest.raises(SeedBuildError, match="inside the store directory"):
        build_packed_feature_store(
            catalog,
            SeedBuildConfig(root=root, output=output, unit_cache=output / "units", verify="quick"),
        )


def test_prefetching_does_not_inflate_the_resume_position(tmp_path: Path):
    """Worker prefetch advances the sampler past what was trained on."""
    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        import torch

        from stylized_motion.data.loader import build_data_loaders

        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={"strategy": "clip_uniform", "target_frames": 64, "samples_per_epoch": 32},
            loader_config={
                "batch_size": 4,
                "num_workers": 2,
                "prefetch_factor": 2,
                "persistent_workers": False,
                "prefetch_memory_limit_mb": None,
            },
        )
        config = _budget_config(tmp_path, eval_limit=None)
        # the global budget stops the run well before the epoch's own budget, so
        # the epoch really is unfinished
        config["training"]["steps_per_epoch"] = 10
        config["training"]["max_steps"] = 2
        runner = _budget_runner(tmp_path, config, loaders=loaders)
        runner.run("train")
        checkpoint = torch.load(tmp_path / "last.pt", weights_only=False)
        position = checkpoint["sampler_state"]
        # two steps of four samples were trained; the sampler's own counter has
        # already run ahead because the workers prefetched
        assert position["epoch"] == 1
        assert position["next_ordinal"] == 8
        assert position["complete"] is False
        assert loaders.train.sampler.state_dict()["next_ordinal"] > 8
    finally:
        store.close()


def test_cli_resume_flag_continues_the_runs_own_checkpoint(tmp_path: Path, monkeypatch):
    """--resume must load <output_dir>/last.pt instead of starting over."""
    from stylized_motion.learning import runner as runner_module

    store_path = _write_store(tmp_path, clip_lengths=[400, 400, 420, 440], splits=[0, 0, 1, 1],
                              groups=[0, 1, 2, 3])
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    config_path = tmp_path / "config.yaml"
    import yaml

    from stylized_motion.learning.runner import load_experiment_config

    config = load_experiment_config("data/configs/flat_fsq_40x9.yaml")
    config["representation"]["config"]["motion_dim"] = MOTION_DIM
    config["data"]["fsq_window_index"] = str(store_path)
    config["data"]["required_data_schema_version"] = 4
    config["training"].update(
        {"epochs": 3, "steps_per_epoch": 2, "max_steps": 2, "output_dir": str(tmp_path / "run")}
    )
    config["sampling"]["samples_per_epoch"] = 8
    config["sampling"]["eval_limit"] = None
    config["loader"].update({"batch_size": 2, "num_workers": 0, "prefetch_memory_limit_mb": None})
    # the fixture skeleton has no toe joints, so the foot/contact terms cannot
    # contribute; the budget and resume behaviour under test is unaffected
    config["training"].update(
        {"joint_weight": 0.0, "foot_slide_weight": 0.0, "foot_height_weight": 0.0, "contact_weight": 0.0}
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import sys

    argv = ["--workflow-mode", "train", "--representation", "flat-fsq", "--config", str(config_path)]
    saved_argv = sys.argv
    sys.argv = ["runner", *argv]
    try:
        runner_module.main(argv)
    finally:
        sys.argv = saved_argv
    first = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert first["global_step"] == 2

    # the second invocation with --resume continues instead of restarting
    config["training"]["max_steps"] = 4
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    argv = [
        "--workflow-mode", "train", "--representation", "flat-fsq",
        "--config", str(config_path), "--resume",
    ]
    sys.argv = ["runner", *argv]
    try:
        runner_module.main(argv)
    finally:
        sys.argv = saved_argv
    resumed = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert resumed["global_step"] == 4
    assert resumed["epoch"] >= first["epoch"]


def test_runner_rejects_invalid_budget_values(tmp_path: Path):
    from stylized_motion.learning.checkpoint import CheckpointManager
    from stylized_motion.learning.representation import FLAT_FSQ_FAMILY, build_representation
    from stylized_motion.learning.runner import RepresentationRunner, load_experiment_config

    config = load_experiment_config("data/configs/flat_fsq_40x9.yaml")
    config["training"]["steps_per_epoch"] = 0
    with pytest.raises(ValueError, match="steps_per_epoch"):
        RepresentationRunner(
            build_representation(config),
            family=FLAT_FSQ_FAMILY,
            train_loader=None,
            val_loader=None,
            test_loader=None,
            loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
            metric_suite={},
            checkpoint_manager=CheckpointManager(tmp_path),
            config=config,
            feature_schema={"name": "motion_feature_v2", "motion_dim": 230},
            feature_stats={"offset": np.zeros(230, dtype=np.float32), "scale": np.ones(230, dtype=np.float32)},
            device=torch.device("cpu"),
            epochs=1,
        )
