from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from stylized_motion.anim.features import MotionFeatureStats
from stylized_motion.data.feature_data import (
    FeatureDataset,
    _stats_sha256,
    canonical_json_bytes,
    open_feature_store,
)
from stylized_motion.data.preprocess import (
    MotionDatabaseWriter,
    STYLE100_CLIPS,
    UNKNOWN_LABEL,
    _SourceClip,
    _clip_tags,
    _discover_source_clips,
    _normalize_motion_shard,
    _save_motion_shard,
    _slice_motion,
    validate_data,
)
import stylized_motion.data.preprocess as preprocess_module
from stylized_motion.data.sampling import SampleRequest


def _motion(nframes: int) -> dict[str, np.ndarray | list[str]]:
    return {
        "positions": np.zeros((nframes, 2, 3), dtype=np.float32),
        "velocities": np.ones((nframes, 2, 3), dtype=np.float32),
        "rotations": np.tile(np.asarray([1, 0, 0, 0], dtype=np.float32), (nframes, 2, 1)),
        "angular_velocities": np.full((nframes, 2, 3), 2.0, dtype=np.float32),
        "contacts": np.zeros((nframes, 2), dtype=np.uint8),
        "parents": np.asarray([-1, 0], dtype=np.int32),
        "names": ["Simulation", "Hips"],
    }


def _write_feature_store(tmp_path: Path) -> None:
    shard_dir = tmp_path / "motion"
    shard_dir.mkdir()
    motion = np.arange(128 * 230, dtype=np.float32).reshape(128, 230)
    shard = shard_dir / "shard_00000.npy"
    np.save(shard, motion)
    stats = MotionFeatureStats(
        offset=np.zeros(230, dtype=np.float32),
        scale=np.ones(230, dtype=np.float32),
        dist=np.ones(230, dtype=np.float32),
        weights=np.ones(230, dtype=np.float32),
        ref_pos=np.zeros((2, 3), dtype=np.float32),
    )
    names = ["Simulation", "Hips"]
    parents = [-1, 0]
    schema_payload = {
        "name": "motion_feature_v2",
        "motion_dim": 230,
        "joint_subset": "full",
        "names_sha256": hashlib.sha256(canonical_json_bytes(names)).hexdigest(),
        "stats_sha256": _stats_sha256(stats),
    }
    schema_hash = hashlib.sha256(canonical_json_bytes(schema_payload)).hexdigest()
    manifest = {
        "data_schema_version": 3,
        "store_type": "feature",
        "frame_rate": 60,
        "num_shards": 1,
        "shard_files": ["motion/shard_00000.npy"],
        "shard_sha256": [hashlib.sha256(shard.read_bytes()).hexdigest()],
        "split_manifest_hash": "split-hash",
        "feature_schema_hash": schema_hash,
        "created_by": "tests",
        "motion_dim": 230,
        "range_names": ["style_action"],
        "source_clip_names": ["style_action"],
        "style_names": ["style"],
        "action_names": ["action"],
        "feature_schema": {**schema_payload, "names": names, "parents": parents},
        "normalization_train_frames": 128,
    }
    np.savez(
        tmp_path / "index.npz",
        shard_num_frames=np.asarray([128], dtype=np.int64),
        clip_ids=np.asarray([0], dtype=np.int32),
        source_clip_ids=np.asarray([0], dtype=np.int32),
        range_shard_indices=np.asarray([0], dtype=np.int32),
        range_starts=np.asarray([0], dtype=np.int64),
        range_stops=np.asarray([128], dtype=np.int64),
        range_mirror=np.asarray([False], dtype=bool),
        split_ids=np.asarray([0], dtype=np.uint8),
        style_ids=np.asarray([0], dtype=np.int32),
        action_ids=np.asarray([0], dtype=np.int32),
        offset=stats.offset,
        scale=stats.scale,
        dist=stats.dist,
        weights=stats.weights,
        ref_pos=stats.ref_pos,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_database_writer_preserves_ranges_and_clips_tags(tmp_path: Path):
    output = tmp_path / "database.npz"
    writer = MotionDatabaseWriter(
        output,
        total_frames=6,
        tags_data=[("clip", "all", 0, None), ("clip", "style", 1, 99)],
        prune_ends_and_fingers=True,
    )
    writer.add("clip", False, _motion(3))
    writer.add("clip", True, _motion(3))
    writer.save()

    with np.load(output, allow_pickle=True) as data:
        np.testing.assert_array_equal(data["range_starts"], [0, 3])
        np.testing.assert_array_equal(data["range_stops"], [3, 6])
        np.testing.assert_array_equal(data["tag_range_starts"], [0, 1, 3, 4])
        np.testing.assert_array_equal(data["tag_range_stops"], [3, 3, 6, 6])
        np.testing.assert_array_equal(data["range_mirror"], [False, True])
        assert data["positions"].shape == (6, 2, 3)
        assert data["joint_subset"].item() == "prune_ends_and_fingers"


def test_slice_motion_crops_only_frame_aligned_values():
    motion = _motion(10)
    motion["positions"][:, 0, 0] = np.arange(10, dtype=np.float32)

    clipped = _slice_motion(motion, 3, 8)

    np.testing.assert_array_equal(clipped["positions"][:, 0, 0], [3, 4, 5, 6, 7])
    assert clipped["parents"] is motion["parents"]
    assert clipped["names"] is motion["names"]


def test_lafan_labels_remain_unclassified_until_taxonomy_is_defined(tmp_path: Path):
    clip = _SourceClip("lafan/walk1_subject1", tmp_path / "walk1_subject1.bvh", 0, 10, UNKNOWN_LABEL, UNKNOWN_LABEL)
    assert _clip_tags(clip) == [("lafan/walk1_subject1", "all", 0, 10)]


def test_100style_action_list_excludes_sparse_tr3():
    assert "TR3" not in STYLE100_CLIPS
    assert "TR2" not in STYLE100_CLIPS
    assert STYLE100_CLIPS[-1] == "TR1"


def test_combined_discovery_uses_dataset_prefixes_and_cut_intervals(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(preprocess_module, "LAFAN_SOURCE", tmp_path / "lafan")
    monkeypatch.setattr(preprocess_module, "STYLE100_SOURCE", tmp_path / "100style")
    (tmp_path / "lafan").mkdir()
    (tmp_path / "lafan" / "walk1_subject1.bvh").touch()
    (tmp_path / "100style" / "Aeroplane").mkdir(parents=True)
    (tmp_path / "100style" / "Frame_Cuts.csv").write_text(
        "STYLE_NAME,BR_START,BR_STOP\nAeroplane,10,70\n", encoding="utf-8"
    )
    (tmp_path / "100style" / "Aeroplane" / "Aeroplane_BR.bvh").touch()
    monkeypatch.setattr(preprocess_module.bvh, "read_frame_count", lambda path: 100)

    clips = _discover_source_clips("combined", None, None)

    assert [clip.range_name for clip in clips] == [
        "lafan/walk1_subject1",
        "100style/Aeroplane_BR",
    ]
    assert [clip.nframes for clip in clips] == [100, 60]
    assert clips[0].style == UNKNOWN_LABEL
    assert _clip_tags(clips[1])[-2:] == [
        ("100style/Aeroplane_BR", "Aeroplane", 0, 60),
        ("100style/Aeroplane_BR", "BR", 0, 60),
    ]


def test_normalize_motion_shard_updates_file_in_chunks(tmp_path: Path):
    path = tmp_path / "motion.npy"
    raw = np.arange(18, dtype=np.float32).reshape(6, 3)
    np.save(path, raw)
    stats = MotionFeatureStats(
        offset=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        scale=np.asarray([2.0, 4.0, 5.0], dtype=np.float32),
        dist=np.ones(3, dtype=np.float32),
        weights=np.ones(3, dtype=np.float32),
        ref_pos=np.zeros((2, 3), dtype=np.float32),
    )

    _normalize_motion_shard(path, stats, chunk_size=2)

    np.testing.assert_allclose(np.load(path), (raw - stats.offset) / stats.scale)


def test_save_motion_shard_creates_directory_under_output(tmp_path: Path):
    motion = np.zeros((4, 230), dtype=np.float32)

    relative = _save_motion_shard(tmp_path / "staging", 3, motion)

    assert relative == "motion/shard_00003.npy"
    assert (tmp_path / "staging" / relative).exists()


def test_schema_v3_feature_store_returns_64_frame_causal_batch(tmp_path: Path):
    _write_feature_store(tmp_path)
    store = open_feature_store(tmp_path)
    try:
        request = SampleRequest(0, 63, 64, 0)
        dataset = FeatureDataset("train", store, requests=[request])
        item = dataset[0]
        batch = dataset.__getitems__([request])
        assert item["motion"].shape == (64, 230)
        assert item["motion"].is_contiguous()
        assert item["loss_mask"].shape == (64,)
        assert int(item["loss_mask"].sum()) == 64
        assert bool(item["loss_mask"].all())
        assert batch["motion"].is_contiguous()
        assert batch["motion"].shape == (1, 64, 230)
        assert validate_data(feature_database=tmp_path, full=True) == {
            "feature": True,
            "token": False,
            "trajectory": False,
            "full": True,
        }
    finally:
        store.close()
