"""NEF-FSQ persistence: checkpoint restore and TokenStore motion-width schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stylized_motion.data.sampling import SampleRequest
from stylized_motion.data.token_data import TokenDataset, open_token_store
from stylized_motion.learning.checkpoint import CheckpointManager
from stylized_motion.learning.nef_layout import NEF_STREAM_COORDINATES, NEF_STREAM_NAMES, NEFLayout
from stylized_motion.learning.representation import (
    NEF_FSQ_FAMILY,
    RepresentationAdapter,
    build_representation,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import RepresentationRunner, load_experiment_config
from stylized_motion.learning.nef_layout import GENO_SKELETON, SOMA_SKELETON


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


def nef_config(spec=GENO_SKELETON, *, stream_dim: int = 8, motion_dim: int | None = None) -> dict[str, object]:
    config = load_experiment_config(Path(__file__).parents[1] / "data" / "configs" / "nef_fsq_40x9.yaml")
    names, parents = skeleton_from_spec(spec)
    representation = dict(config["representation"])
    representation["config"] = {
        **dict(representation["config"]),
        "names": names,
        "parents": parents,
        "stream_dim": stream_dim,
        "motion_dim": 9 * len(names) + 5 if motion_dim is None else motion_dim,
    }
    config["representation"] = representation
    return config


def feature_schema(spec=GENO_SKELETON) -> dict[str, object]:
    names, _parents = skeleton_from_spec(spec)
    motion_dim = 9 * len(names) + 5
    return {
        "name": "motion_feature_v2",
        "motion_dim": motion_dim,
        "joint_subset": "prune_ends_and_fingers",
        "names_sha256": f"names-{spec.name}",
        "stats_sha256": f"stats-{spec.name}",
        "feature_schema_hash": f"hash-{spec.name}",
    }


def _save_checkpoint(tmp_path, spec=GENO_SKELETON, name: str = "nef.pt") -> tuple[Path, RepresentationAdapter, torch.Tensor]:
    config = nef_config(spec)
    representation = build_representation(config)
    schema = feature_schema(spec)
    motion_dim = int(schema["motion_dim"])
    runner = RepresentationRunner(
        representation,
        family=NEF_FSQ_FAMILY,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        loss_fn=lambda output, batch: {"loss": output["recon_state"].square().mean()},
        metric_suite={},
        checkpoint_manager=CheckpointManager(tmp_path),
        config=config,
        feature_schema=schema,
        feature_stats={
            "offset": np.zeros(motion_dim, dtype=np.float32),
            "scale": np.ones(motion_dim, dtype=np.float32),
        },
        device=torch.device("cpu"),
        epochs=1,
        optimizer=torch.optim.SGD(representation.parameters(), lr=0.001),
    )
    path = tmp_path / name
    torch.save(runner.checkpoint_payload(epoch=1, metrics={}), path)
    return path, representation, torch.randn(1, 64, motion_dim)


def _write_token_store(tmp_path, *, family: str, motion_dim: int, representation: dict | None, coordinate_order=None, coordinate_counts=None, variant: str | None = None, representation_id: str | None = None, num_coordinates: int = 40):
    if variant is None:
        variant = {"flat_fsq": "flat", "nef_fsq": "independent"}[family]
    if representation_id is None:
        representation_id = (
            f"{family}_{variant}_{num_coordinates}x9" if family == "nef_fsq" else f"{family}_{num_coordinates}x9"
        )
    order = list(coordinate_order if coordinate_order is not None else NEF_STREAM_NAMES)
    counts = dict(coordinate_counts if coordinate_counts is not None else NEF_STREAM_COORDINATES)
    shard_dir = tmp_path / "indices"
    shard_dir.mkdir(exist_ok=True)
    shard = shard_dir / "shard_00000.npy"
    np.save(shard, np.zeros((65, num_coordinates), dtype=np.uint8))
    manifest = {
        "data_schema_version": 3,
        "store_type": "token",
        "frame_rate": 60,
        "num_shards": 1,
        "shard_files": ["indices/shard_00000.npy"],
        "shard_sha256": [hashlib.sha256(shard.read_bytes()).hexdigest()],
        "split_manifest_hash": "split-hash",
        "feature_schema_hash": "feature-hash",
        "created_by": "tests",
        "range_names": ["style_action"],
        "source_clip_names": ["style_action"],
        "style_names": ["style"],
        "action_names": ["action"],
        "representation_family": family,
        "representation_variant": variant,
        "representation_id": representation_id,
        "model_family_legacy": {"flat_fsq": "fsq", "nef_fsq": "nef_fsq"}[family],
        "checkpoint_sha256": "checkpoint-hash",
        "motion_dim": motion_dim,
        "num_coordinates": num_coordinates,
        "num_levels": 9,
        "temporal_downsample": 1,
        "receptive_field": 64,
        "lookahead_frames": 0,
        "decoder_passes_inference": 1,
        "coordinate_order": order,
        "coordinate_counts": counts,
        "feature_schema": {"name": "motion_feature_v2", "motion_dim": motion_dim},
        "split_policy": "fixed_window_random_v1",
        "window_frames": 64,
    }
    if representation is not None:
        manifest["representation"] = representation
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


def test_geno_checkpoint_restores_layout_schema_and_reconstruction(tmp_path):
    path, representation, motion = _save_checkpoint(tmp_path)
    metadata = representation.representation_metadata()
    assert metadata["family"] == NEF_FSQ_FAMILY
    assert metadata["variant"] == "independent"
    assert metadata["representation_id"] == "nef_fsq_independent_40x9"
    assert metadata["architecture_version"] == 1
    assert metadata["receptive_field"] == 64 and metadata["lookahead_frames"] == 0
    assert metadata["coordinate_order"] == list(NEF_STREAM_NAMES)
    assert metadata["nef_layout"]["skeleton"] == "geno"
    with torch.no_grad():
        expected = representation(motion, collect_metrics=False)["recon_state"]

    checkpoint, restored = load_representation_checkpoint(path, torch.device("cpu"), feature_schema=feature_schema())
    assert isinstance(restored, RepresentationAdapter)
    assert restored.representation_metadata()["nef_layout"] == metadata["nef_layout"]
    with torch.no_grad():
        actual = restored(motion, collect_metrics=False)["recon_state"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_soma_checkpoint_restores_its_own_layout(tmp_path):
    path, representation, motion = _save_checkpoint(tmp_path, SOMA_SKELETON, name="nef_soma.pt")
    assert representation.motion_dim == 248
    with torch.no_grad():
        expected = representation(motion, collect_metrics=False)["recon_state"]
    _, restored = load_representation_checkpoint(path, torch.device("cpu"), feature_schema=feature_schema(SOMA_SKELETON))
    assert restored.motion_dim == 248
    assert restored.representation_metadata()["nef_layout"]["skeleton"] == "soma"
    with torch.no_grad():
        actual = restored(motion, collect_metrics=False)["recon_state"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_checkpoint_loader_rejects_tampered_nef_layout(tmp_path):
    path, _representation, _motion = _save_checkpoint(tmp_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["representation"]["nef_layout"]["stream_slices"]["left_arm_node"] = [0, 4]
    tampered = tmp_path / "tampered.pt"
    torch.save(checkpoint, tampered)
    with pytest.raises(ValueError, match="nef_layout"):
        load_representation_checkpoint(tampered, torch.device("cpu"), feature_schema=feature_schema())


def test_checkpoint_loader_rejects_a_feature_schema_mismatch(tmp_path):
    path, _representation, _motion = _save_checkpoint(tmp_path)
    wrong = dict(feature_schema())
    wrong["stats_sha256"] = "other"
    with pytest.raises(ValueError, match="feature_schema"):
        load_representation_checkpoint(path, torch.device("cpu"), feature_schema=wrong)


def test_nef_token_store_reads_back_fixed_coordinate_order(tmp_path):
    layout = NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON))
    _write_token_store(
        tmp_path,
        family="nef_fsq",
        motion_dim=230,
        representation={"family": "nef_fsq", "variant": "independent", "nef_layout": layout.to_dict()},
    )
    store = open_token_store(tmp_path)
    try:
        assert (store.representation_family, store.representation_id) == ("nef_fsq", "nef_fsq_independent_40x9")
        assert store.coordinate_order == NEF_STREAM_NAMES
        assert store.validate_contract(
            representation={
                "family": "nef_fsq",
                "variant": "independent",
                "nef_layout": layout.to_dict(),
            }
        ) is None
        item = TokenDataset("train", store, requests=[SampleRequest(0, 0, 64, 0)])[0]
        assert item["indices"].shape == (65, 40)
        assert item["indices"].dtype == torch.uint8
    finally:
        store.close()


def test_soma_248_token_store_uses_its_own_schema(tmp_path):
    layout = NEFLayout.from_skeleton(*skeleton_from_spec(SOMA_SKELETON))
    _write_token_store(
        tmp_path,
        family="nef_fsq",
        motion_dim=248,
        representation={"family": "nef_fsq", "variant": "independent", "nef_layout": layout.to_dict()},
    )
    store = open_token_store(tmp_path)
    try:
        assert store.motion_dim == 248
        store.validate_contract(
            representation={
                "family": "nef_fsq",
                "variant": "independent",
                "nef_layout": layout.to_dict(),
            }
        )
    finally:
        store.close()


def test_legacy_geno_230_token_store_still_opens(tmp_path):
    _write_token_store(
        tmp_path,
        family="flat_fsq",
        motion_dim=230,
        representation=None,
        coordinate_order=["flat"],
        coordinate_counts={"flat": 40},
    )
    store = open_token_store(tmp_path)
    try:
        assert (store.motion_dim, store.num_coordinates, store.representation_family) == (230, 40, "flat_fsq")
        store.validate_contract()
    finally:
        store.close()


@pytest.mark.parametrize(
    "kwargs, message",
    [
            ({"motion_dim": 231}, r"9J\+5"),
            ({"motion_dim": 200}, r"9J\+5"),
        ({"coordinate_order": list(reversed(NEF_STREAM_NAMES))}, "coordinate layout"),
        ({"representation_id": "nef_fsq_40x9"}, "representation_id"),
        ({"variant": "shared"}, "representation_variant"),
    ],
)
def test_nef_token_store_rejects_schema_and_layout_mismatches(tmp_path, kwargs, message):
    _write_token_store(
        tmp_path, family="nef_fsq", representation=None, **{"motion_dim": 230, **kwargs}
    )
    with pytest.raises(ValueError, match=message):
        open_token_store(tmp_path)


def test_nef_token_store_rejects_cross_skeleton_donor_decode(tmp_path):
    geno = NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON))
    soma = NEFLayout.from_skeleton(*skeleton_from_spec(SOMA_SKELETON))
    _write_token_store(
        tmp_path,
        family="nef_fsq",
        motion_dim=230,
        representation={"family": "nef_fsq", "variant": "independent", "nef_layout": geno.to_dict()},
    )
    store = open_token_store(tmp_path)
    try:
        store.validate_contract(
            representation={"family": "nef_fsq", "variant": "independent", "nef_layout": geno.to_dict()}
        )
        with pytest.raises(ValueError, match="NEF-FSQ"):
            store.validate_contract(
                representation={"family": "nef_fsq", "variant": "independent", "nef_layout": soma.to_dict()}
            )
        with pytest.raises(ValueError, match="NEF-FSQ"):
            store.validate_contract(
                representation={
                    "family": "nef_fsq",
                    "variant": "independent",
                    "nef_layout": {"names": list(soma.names)},
                }
            )
        with pytest.raises(ValueError, match="node/edge layout"):
            store.validate_contract(representation={"family": "nef_fsq", "variant": "independent"})
    finally:
        store.close()


def test_nef_token_store_rejects_changed_parents_without_hash_comparison(tmp_path):
    geno = NEFLayout.from_skeleton(*skeleton_from_spec(GENO_SKELETON))
    _write_token_store(
        tmp_path,
        family="nef_fsq",
        motion_dim=230,
        representation={"family": "nef_fsq", "variant": "independent", "nef_layout": geno.to_dict()},
    )
    store = open_token_store(tmp_path)
    try:
        changed = geno.to_dict()
        changed["parents"] = list(changed["parents"])
        changed["parents"][changed["names"].index("LeftShoulder")] = changed["names"].index("Spine2")
        with pytest.raises(ValueError, match="chain"):
            store.validate_contract(
                representation={"family": "nef_fsq", "variant": "independent", "nef_layout": changed}
            )
    finally:
        store.close()


def test_nef_token_store_rejects_legacy_family_metadata(tmp_path):
    _write_token_store(
        tmp_path,
        family="nef_fsq",
        motion_dim=230,
        representation=None,
        representation_id="flat_fsq_40x9",
    )
    with pytest.raises(ValueError, match="representation_id"):
        open_token_store(tmp_path)
