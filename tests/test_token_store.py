from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from stylized_motion.data.preprocess import validate_data
from stylized_motion.data.sampling import SampleRequest
from stylized_motion.data.token_data import TokenDataset, open_token_store


def _write_store(tmp_path, *, token_value: int = 0):
    shard_dir = tmp_path / "indices"
    shard_dir.mkdir()
    shard = shard_dir / "shard_00000.npy"
    np.save(shard, np.full((65, 40), token_value, dtype=np.uint8))
    shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "data_schema_version": 3,
        "store_type": "token",
        "frame_rate": 60,
        "num_shards": 1,
        "shard_files": ["indices/shard_00000.npy"],
        "shard_sha256": [shard_hash],
        "split_manifest_hash": "split-hash",
        "feature_schema_hash": "feature-hash",
        "created_by": "tests",
        "range_names": ["style_action"],
        "source_clip_names": ["style_action"],
        "style_names": ["style"],
        "action_names": ["action"],
        "representation_family": "flat_fsq",
        "representation_variant": "flat",
        "representation_id": "flat_fsq_40x9",
        "model_family_legacy": "fsq",
        "checkpoint_sha256": "checkpoint-hash",
        "motion_dim": 230,
        "num_coordinates": 40,
        "num_levels": 9,
        "temporal_downsample": 1,
        "receptive_field": 64,
        "context_left": 63,
        "lookahead_frames": 0,
        "decoder_passes_inference": 1,
        "coordinate_order": ["flat"],
        "coordinate_counts": {"flat": 40},
        "feature_schema": {"name": "motion_feature_v2", "motion_dim": 230},
    }
    np.savez(
        tmp_path / "index.npz",
        shard_num_frames=np.asarray([65], dtype=np.int64),
        clip_ids=np.asarray([0], dtype=np.int32),
        source_clip_ids=np.asarray([0], dtype=np.int32),
        range_shard_indices=np.asarray([0], dtype=np.int32),
        range_starts=np.asarray([0], dtype=np.int64),
        range_stops=np.asarray([65], dtype=np.int64),
        range_mirror=np.asarray([False], dtype=bool),
        split_ids=np.asarray([0], dtype=np.uint8),
        style_ids=np.asarray([0], dtype=np.int32),
        action_ids=np.asarray([0], dtype=np.int32),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_token_store_opens_and_dataset_reads_schema_v3(tmp_path):
    _write_store(tmp_path)
    store = open_token_store(tmp_path)
    try:
        request = SampleRequest(0, 0, 64, 0, 0)
        dataset = TokenDataset("train", store, requests=[request])
        item = dataset[0]
        assert item["indices"].shape == (65, 40)
        assert item["indices"].dtype == torch.uint8
        assert int(item["indices"].max()) == 0
    finally:
        store.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_sha256": "wrong"},
        {"representation": {"family": "part_fsq"}},
        {"feature_schema": {"feature_schema_hash": "wrong"}},
    ],
)
def test_token_store_rejects_contract_mismatches(tmp_path, kwargs):
    _write_store(tmp_path)
    store = open_token_store(tmp_path)
    try:
        with pytest.raises(ValueError):
            store.validate_contract(**kwargs)
    finally:
        store.close()


def test_token_values_are_range_checked_by_full_validation(tmp_path):
    _write_store(tmp_path, token_value=9)
    with pytest.raises(ValueError, match="out-of-range"):
        validate_data(token_database=tmp_path, full=True)
