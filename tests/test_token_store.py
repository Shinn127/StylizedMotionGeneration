from __future__ import annotations

import json

import numpy as np
import pytest

from stylized_motion.data.token_store import TokenDataset, build_token_store


def _write_store(tmp_path, *, token_value: int = 0):
    (tmp_path / "indices").mkdir()
    np.save(tmp_path / "indices" / "indices_00000.npy", np.full((64, 40), token_value, dtype=np.uint8))
    layout = {"flat": 40}
    windows = np.asarray([[0, 0, 64, 0]], dtype=np.int32)
    np.savez(
        tmp_path / "metadata.npz",
        token_files=np.asarray(["indices/indices_00000.npy"], dtype=object),
        code_files=np.asarray([""], dtype=object),
        num_frames=np.asarray([64], dtype=np.int32),
        range_names=np.asarray(["style_action"], dtype=object),
        range_mirror=np.asarray([False], dtype=bool),
        style_names=np.asarray(["style"], dtype=object),
        action_names=np.asarray(["action"], dtype=object),
        style_ids=np.asarray([0], dtype=np.int32),
        action_ids=np.asarray([0], dtype=np.int32),
        train_windows=windows,
        val_windows=windows,
        test_windows=windows,
        schema_version=np.asarray(2, dtype=np.int32),
        motion_dim=np.asarray(230, dtype=np.int32),
        window_size=np.asarray(64, dtype=np.int32),
        frame_rate=np.asarray(60, dtype=np.int32),
        temporal_downsample=np.asarray(1, dtype=np.int32),
        receptive_field=np.asarray(64, dtype=np.int32),
        context_left=np.asarray(63, dtype=np.int32),
        lookahead_frames=np.asarray(0, dtype=np.int32),
        decoder_passes_inference=np.asarray(1, dtype=np.int32),
        num_coordinates=np.asarray(40, dtype=np.int32),
        num_levels=np.asarray(9, dtype=np.int32),
        checkpoint_path=np.asarray("checkpoint.pt", dtype=object),
        checkpoint_sha256=np.asarray("checkpoint-hash", dtype=object),
        feature_database=np.asarray("feature-database", dtype=object),
        feature_schema_hash=np.asarray("feature-hash", dtype=object),
        names_sha256=np.asarray("names-hash", dtype=object),
        stats_sha256=np.asarray("stats-hash", dtype=object),
        joint_subset=np.asarray("pruned", dtype=object),
        representation_family=np.asarray("flat_fsq", dtype=object),
        representation_variant=np.asarray("flat", dtype=object),
        representation_id=np.asarray("flat_fsq_40x9", dtype=object),
        model_family_legacy=np.asarray("fsq", dtype=object),
        coordinate_order=np.asarray(["flat"], dtype=object),
        coordinate_counts=np.asarray(json.dumps(layout), dtype=object),
    )


def test_token_store_opens_and_dataset_validates_shards(tmp_path):
    _write_store(tmp_path)
    store = build_token_store(tmp_path)
    item = TokenDataset("train", store)[0]
    assert item["indices"].shape == (64, 40)
    assert int(item["indices"].max()) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_sha256": "wrong"},
        {"representation": {"family": "part_fsq"}},
        {"feature_schema": {"names_sha256": "wrong"}},
    ],
)
def test_token_store_rejects_contract_mismatches(tmp_path, kwargs):
    _write_store(tmp_path)
    with pytest.raises(ValueError):
        build_token_store(tmp_path, **kwargs)


def test_token_dataset_rejects_out_of_range_indices_at_open(tmp_path):
    _write_store(tmp_path, token_value=9)
    store = build_token_store(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        TokenDataset("train", store)
