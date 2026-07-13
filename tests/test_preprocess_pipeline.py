from pathlib import Path

import numpy as np

from datasets.feature_dataset import build_feature_store
from motion_features import MotionFeatureStats
from preprocess import build_feature_database
from preprocess.build_database import MotionDatabaseWriter
from preprocess.build_feature_database import _normalize_motion_shard


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


def test_build_processed_data_writes_both_outputs_in_one_stream(tmp_path: Path, monkeypatch):
    specs = [{"range_name": "clip", "mirror": mirror, "nframes": 6} for mirror in [False, True]]
    tags = [("clip", "all", 0, None)]
    calls = []

    monkeypatch.setattr(
        build_feature_database,
        "_build_shard_specs",
        lambda **_kwargs: (specs, tags, [Path("clip.bvh")]),
    )

    def fake_motion_pairs(*_args, **_kwargs):
        calls.append("processed")
        yield "clip", [(False, _motion(6)), (True, _motion(6))]

    monkeypatch.setattr(build_feature_database, "iter_motion_pairs", fake_motion_pairs)
    output_dir = tmp_path / "processed"
    feature_dir = output_dir / "feature_database"
    database_path = output_dir / "database.npz"

    build_feature_database.build_processed_data(
        dataset_name="100style",
        output_dir=output_dir,
        window_size=2,
        workers=4,
    )

    assert calls == ["processed"]
    assert database_path.exists()
    store = build_feature_store(feature_dir)
    assert len(store.motion_files) == 2
    assert store.motion_dim == 23
    assert len(store.split_windows["train"]) == 2
    assert all(np.isfinite(np.load(path)).all() for path in store.motion_files)
