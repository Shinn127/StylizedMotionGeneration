"""Schema-v4 data-registration, packed-store, statistics, sampler and resume contracts.

These tests encode the acceptance criteria of
``docs/bones_seed_data_pipeline_plan.md``: the raw/target frame contract, mirror
policy, take-group splits, packed shards decoupled from logical clips, train-only
versioned statistics, compact samplers with equal DDP steps, and a resumable
preprocessing journal.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stylized_motion.anim.features import joint_feature_dim
from stylized_motion.data.loader import build_data_loaders
from stylized_motion.data.normalization import (
    FeatureNormalization,
    FeatureStatsAccumulator,
    compute_normalization,
    compute_scale,
    compute_weights,
)
from stylized_motion.data.packed_store import (
    PACKED_SCHEMA_VERSION,
    PackedFeatureDataset,
    PackedFeatureStoreWriter,
    feature_schema_hash,
    normalize_batch_on_device,
    open_any_feature_store,
    open_packed_feature_store,
    skeleton_hash,
    write_clip_table,
)
from stylized_motion.data.resume import (
    PREPROCESS_VERSION,
    UnitResult,
    WorkJournal,
    atomic_write_json,
    bounded_map,
    build_work_unit,
    fingerprint_file,
    preprocess_signature,
    verify_completed_units,
)
from stylized_motion.data.sampling import (
    FixedWindowSampler,
    SampleRequest,
    TrainWindowSampler,
    sampling_contract,
    store_intervals,
)
from stylized_motion.data.seed_build import (
    ROOT_CHANNELS,
    SeedBuildError,
    order_packed_clips,
    packed_order_key,
    packed_row_order,
    probe_bvh_header,
    window_coverage,
)
from stylized_motion.data.seed_catalog import (
    FrameContract,
    SeedCatalog,
    SeedCatalogError,
    SeedClip,
    UNKNOWN_LABEL,
    VARIANT_OFFICIAL_MIRROR,
    VARIANT_ORIGINAL,
    assign_group_splits,
    build_group_key,
    catalog_frame_audit,
    catalog_tier_report,
    clip_label_value,
    discover_catalog,
    load_temporal_labels,
    refresh_catalog_counts,
    seconds_to_frame,
    select_representative_groups,
    time_range_to_frames,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

NAMES = ["Simulation", "Hips", "LeftFoot"]
PARENTS = [-1, 0, 1]
MOTION_DIM = joint_feature_dim(len(NAMES))


def _write_seed_csv(root: Path, rows: list[dict[str, str]]) -> Path:
    """Write a SEED-shaped metadata CSV plus placeholder BVH files."""
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "move_name",
        "filename",
        "move_duration_frames",
        "package",
        "category",
        "is_neutral",
        "is_mirror",
        "move_soma_uniform_path",
        "take_name",
        "take_actor",
        "take_org_name",
        "take_date",
        "take_day_part",
        "content_uniform_style",
        "content_type_of_movement",
        "content_body_position",
        "content_horizontal_move",
        "content_vertical_move",
        "content_props",
        "content_complex_action",
        "content_repeated_action",
        "actor_uid",
    ]
    csv_path = metadata_dir / "seed_metadata_v004.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in columns}
            writer.writerow(payload)
    for row in rows:
        path = root / row["move_soma_uniform_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("HIERARCHY\nROOT Root\nMOTION\n", encoding="utf-8")
    return csv_path


def _seed_row(
    name: str,
    *,
    frames: int,
    is_mirror: bool,
    package: str = "Locomotion",
    style: str = "neutral",
    actor: str = "A001",
    take_date: str = "240101",
    take_org: str = "base",
) -> dict[str, str]:
    return {
        "move_name": name,
        "filename": name,
        "move_duration_frames": str(frames),
        "package": package,
        "category": "Baseline",
        "is_neutral": "1.0",
        "is_mirror": "True" if is_mirror else "False",
        "move_soma_uniform_path": f"soma_uniform/bvh/{take_date}/{name}.bvh",
        "take_name": f"{take_org}_{actor}",
        "take_actor": actor,
        "take_org_name": take_org,
        "take_date": take_date,
        "take_day_part": "_1",
        "content_uniform_style": style,
        "content_type_of_movement": "walking",
        "content_body_position": "standing",
        "content_horizontal_move": "0",
        "content_vertical_move": "0",
        "content_props": "0",
        "content_complex_action": "0",
        "content_repeated_action": "0",
        "actor_uid": actor,
    }


def _synthetic_clip_rows(count: int = 12) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(count):
        actor = f"A{index % 3:03d}"
        base = _seed_row(
            f"walk_{index:03d}__{actor}",
            frames=400 + index * 8,
            is_mirror=False,
            package=["Locomotion", "Interactions", "Dances"][index % 3],
            style=["neutral", "hurry", "old"][index % 3],
            actor=actor,
            take_date=f"2401{index % 2:02d}",
        )
        rows.append(base)
        if index % 3 != 2:  # a third of the takes have no official mirror
            rows.append({**base, "move_name": base["move_name"] + "_M", "filename": base["filename"] + "_M",
                         "is_mirror": "True", "move_duration_frames": str(int(base["move_duration_frames"]) - 1),
                         "move_soma_uniform_path": base["move_soma_uniform_path"].replace(".bvh", "_M.bvh")})
    return rows


@pytest.fixture
def seed_root(tmp_path: Path) -> Path:
    root = tmp_path / "seed"
    _write_seed_csv(root, _synthetic_clip_rows())
    return root


def _build_store(
    tmp_path: Path,
    *,
    clip_lengths: list[int] | None = None,
    splits: list[int] | None = None,
    groups: list[int] | None = None,
    variants: list[int] | None = None,
    motion_dim: int = MOTION_DIM,
    shard_bytes: int = 4096,
    with_normalization: bool = True,
    seed: int = 3407,
) -> Path:
    """Write a tiny but structurally complete schema-v4 packed store."""
    lengths = list(clip_lengths if clip_lengths is not None else [80, 96, 120, 64, 70, 200])
    default_splits = [0, 0, 0, 1, 1, 2]
    default_groups = [0, 1, 2, 3, 4, 5]
    default_variants = [0, 1, 0, 0, 1, 0]

    def _expand(values: list[int] | None, default: list[int]) -> list[int]:
        if values is not None:
            return list(values)
        if len(lengths) == len(default):
            return list(default)
        return [default[index % len(default)] for index in range(len(lengths))]

    split_ids = _expand(splits, default_splits)
    group_ids = _expand(groups, default_groups)
    variant_ids = _expand(variants, default_variants)
    staging = tmp_path / "store"
    staging.mkdir(parents=True, exist_ok=True)
    writer = PackedFeatureStoreWriter(
        staging, motion_dim=motion_dim, num_joints=len(NAMES), shard_bytes=shard_bytes, names=NAMES
    )
    rng = np.random.default_rng(seed)
    entries = []
    for row, length in enumerate(lengths):
        values = rng.normal(size=(length, motion_dim)).astype(np.float32)
        entries.append(
            writer.append_clip(
                values,
                source_group=group_ids[row],
                variant=variant_ids[row],
                split=split_ids[row],
                mirror=bool(variant_ids[row]),
                source_id=group_ids[row],
                style_id=row % 3,
                action_id=row % 2,
                package_id=row % 3,
                move_name=f"clip_{row}",
                relative_path=f"clips/{row}.bvh",
                position_sum=values[:, :3].reshape(length, 3)[: len(NAMES)].astype(np.float64),
            )
        )
    writer.close_shards()
    write_clip_table(staging, entries, num_joints=len(NAMES))
    manifest = {
        "data_schema_version": PACKED_SCHEMA_VERSION,
        "store_type": "feature_packed",
        "layout": "packed",
        "frame_rate": 60,
        "created_by": "tests",
        "num_shards": len(writer.shard_files),
        "shard_files": writer.shard_files,
        "shard_sha256": writer.shard_sha256,
        "shard_num_frames": writer.shard_num_frames,
        "shard_target_bytes": int(shard_bytes),
        "motion_dim": motion_dim,
        "num_clips": len(entries),
        "total_frames": int(sum(entry.length for entry in entries)),
        "clip_names": [f"clip_{row}" for row in range(len(entries))],
        "style_names": ["s0", "s1", "s2"],
        "action_names": ["a0", "a1"],
        "package_names": ["p0", "p1", "p2"],
        "split_policy": "take_group_v1",
        "split_seed": int(seed),
        "feature_schema": {
            "name": "motion_feature_v2",
            "motion_dim": motion_dim,
            "joint_subset": "full",
            "names": NAMES,
            "parents": PARENTS,
        },
        "feature_schema_hash": feature_schema_hash(NAMES, PARENTS, "full"),
        "skeleton_hash": skeleton_hash(NAMES, PARENTS, "full"),
        "split_manifest_hash": "split-hash",
        "build": {"status": "complete", "preprocess_version": PREPROCESS_VERSION},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if with_normalization:
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
# P0: frame contract, mirror policy, splits
# ---------------------------------------------------------------------------


def test_frame_contract_decimates_120fps_rows_without_using_metadata_as_stop():
    contract = FrameContract(120, 60)
    assert contract.decimation == 2
    assert contract.target_frames(3769) == 1885
    assert contract.target_frames(3768) == 1884
    # the mirror being one raw frame short is a legitimate length difference
    assert contract.target_frames(3769) - contract.target_frames(3768) == 1
    assert contract.target_interval(0, 3769) == (0, 1885)


def test_frame_contract_rejects_non_integer_rates():
    with pytest.raises(SeedCatalogError, match="integer multiple"):
        FrameContract(90, 60)


def test_clip_rejects_metadata_frames_used_as_target_stop(seed_root: Path):
    clips, _manifest = discover_catalog(seed_root)
    clip = clips[0]
    with pytest.raises(SeedCatalogError, match="declares .* target frames"):
        SeedClip(
            clip_id=0,
            group_id=0,
            variant_id=0,
            is_mirror=False,
            dataset="seed",
            relative_path=clip.relative_path,
            move_name="broken",
            canonical_name="broken",
            raw_frames=clip.raw_frames,
            target_frames=clip.raw_frames,  # the bug the plan warns about
            stop=clip.raw_frames,
        )


def test_official_mirror_policy_groups_original_and_mirror(seed_root: Path):
    clips, manifest = discover_catalog(seed_root, mirror_policy="official")
    originals = [clip for clip in clips if clip.variant_id == VARIANT_ORIGINAL]
    mirrors = [clip for clip in clips if clip.variant_id == VARIANT_OFFICIAL_MIRROR]
    assert originals and mirrors
    by_group: dict[int, set[int]] = {}
    for clip in clips:
        by_group.setdefault(clip.group_id, set()).add(clip.variant_id)
    assert all(VARIANT_ORIGINAL in variants for variants in by_group.values())
    assert any(VARIANT_OFFICIAL_MIRROR in variants for variants in by_group.values())
    assert manifest["num_originals"] == len(originals)
    assert manifest["num_official_mirrors"] == len(mirrors)


def test_generate_and_none_policies_drop_official_mirrors(seed_root: Path):
    for policy in ("generate", "none"):
        clips, manifest = discover_catalog(seed_root, mirror_policy=policy)
        assert all(clip.variant_id == VARIANT_ORIGINAL for clip in clips)
        assert manifest["num_official_mirrors"] == 0
    with pytest.raises(SeedCatalogError, match="Unsupported mirror policy"):
        discover_catalog(seed_root, mirror_policy="both")


def test_group_key_uses_take_identity_not_stripped_suffix():
    first = build_group_key("240101", "A001", "org", "walk_001")
    second = build_group_key("240102", "A001", "org", "walk_001")
    assert first != second
    # a derived cut with the same stem but a different take is a different group
    assert build_group_key("240101", "A002", "org", "walk_001") != first


def test_group_split_keeps_every_variant_of_a_take_together(seed_root: Path):
    clips, manifest = discover_catalog(seed_root)
    assign_group_splits(clips, seed=11)
    by_group: dict[int, set[str]] = {}
    for clip in clips:
        by_group.setdefault(clip.group_id, set()).add(clip.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert {clip.split for clip in clips} <= {"train", "val", "test"}


def test_actor_holdout_freezes_whole_actors_into_test(seed_root: Path):
    clips, _manifest = discover_catalog(seed_root)
    summary = assign_group_splits(clips, seed=11, actor_holdout_ratio=0.34)
    holdout = set(summary["split_actor_holdout"])
    assert holdout
    for clip in clips:
        if clip.actor_uid in holdout:
            assert clip.split == "test"


def test_split_is_deterministic_and_hashes_its_assignments(seed_root: Path, tmp_path: Path):
    clips, manifest = discover_catalog(seed_root)
    assign_group_splits(clips, seed=5)
    first = {clip.move_name: clip.split for clip in clips}
    first_hash = SeedCatalog(tmp_path / "a", manifest, clips).split_manifest_hash()

    # rediscovering the same catalogue reproduces the same assignment and hash
    other_clips, other_manifest = discover_catalog(seed_root)
    assign_group_splits(other_clips, seed=5)
    assert {clip.move_name: clip.split for clip in other_clips} == first
    assert SeedCatalog(tmp_path / "b", other_manifest, other_clips).split_manifest_hash() == first_hash

    # a different seed produces a different assignment (and therefore hash)
    shifted, _ = discover_catalog(seed_root)
    assign_group_splits(shifted, seed=99)
    assert SeedCatalog(tmp_path / "c", other_manifest, shifted).split_manifest_hash() != first_hash


def test_catalog_round_trips_through_disk(seed_root: Path, tmp_path: Path):
    clips, manifest = discover_catalog(seed_root)
    assign_group_splits(clips, seed=3)
    catalog = SeedCatalog(tmp_path / "catalog", manifest, clips)
    catalog.save()
    loaded = SeedCatalog.load(tmp_path / "catalog")
    assert [clip.move_name for clip in loaded.clips] == [clip.move_name for clip in clips]
    assert [clip.split for clip in loaded.clips] == [clip.split for clip in clips]
    assert [clip.raw_frames for clip in loaded.clips] == [clip.raw_frames for clip in clips]
    assert np.array_equal(loaded.group_ids(), catalog.group_ids())
    assert loaded.split_manifest_hash() == catalog.split_manifest_hash()
    assert np.array_equal(np.load(tmp_path / "catalog" / "index" / "clip_id.npy"), np.arange(len(clips)))
    labels = json.loads((tmp_path / "catalog" / "labels_actor_uid.json").read_text())
    assert "A001" in labels


def test_labels_keep_raw_values_and_unknowns(seed_root: Path):
    clips, _manifest = discover_catalog(seed_root)
    assert {clip.labels["content_uniform_style"] for clip in clips} <= {"neutral", "hurry", "old", ""}
    clip = clips[0]
    assert clip_label_value(clip, "actor_uid") == clip.actor_uid
    assert clip_label_value(clip, "take_date") == clip.take_date
    assert clip_label_value(clip, "not_a_field") == UNKNOWN_LABEL


def test_temporal_labels_convert_seconds_with_a_versioned_rule(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {
                "filename": "walk_001",
                "num_events": 2,
                "events": [
                    {"start_time": 0.0, "end_time": 1.88, "description": "a"},
                    {"start_time": 1.88, "end_time": 4.83, "description": "b"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = load_temporal_labels(path)
    assert len(labels["walk_001"]) == 2
    assert seconds_to_frame(1.88, 60) == 113
    window = time_range_to_frames(0.0, 1.88, nframes=200, label="a")
    assert (window.start_frame, window.stop_frame) == (0, 113)
    assert window.frames == 113
    # clamping keeps a label inside a short clip instead of producing invalid spans
    clamped = time_range_to_frames(0.0, 10.0, nframes=120)
    assert (clamped.start_frame, clamped.stop_frame) == (0, 120)


def test_frame_audit_reports_variant_lengths(seed_root: Path):
    clips, _manifest = discover_catalog(seed_root)
    audit = catalog_frame_audit(clips)
    assert audit["clips"] == len(clips)
    assert audit["target_hours"] >= 0.0
    assert set(audit["by_variant"]) <= {"original", "official_mirror", "generated_mirror"}


def test_tier_selection_spans_packages_lengths_and_mirror_coverage():
    def clip(index: int, *, package: str, frames: int, mirror: bool) -> SeedClip:
        name = f"take_{index:03d}" + ("_M" if mirror else "")
        return SeedClip(
            clip_id=index,
            group_id=index,
            variant_id=VARIANT_OFFICIAL_MIRROR if mirror else VARIANT_ORIGINAL,
            is_mirror=mirror,
            dataset="seed",
            relative_path=f"clips/{name}.bvh",
            move_name=name,
            canonical_name=f"take_{index:03d}",
            raw_frames=frames * 2,
            target_frames=frames,
            labels={"package": package, "content_uniform_style": "neutral"},
        )

    clips: list[SeedClip] = []
    index = 0
    # three packages, lengths spread over an order of magnitude, and only half
    # of the takes have an official mirror
    for package in ("Locomotion", "Interactions", "Dances"):
        for step in range(10):
            frames = 80 + step * 120
            members = [clip(index, package=package, frames=frames, mirror=False)]
            index += 1
            if step % 2 == 0:
                members.append(clip(index, package=package, frames=frames - 1, mirror=True))
                index += 1
            for member in members:
                member.group_key = members[0].canonical_name
            clips.extend(members)

    selected = select_representative_groups(clips, 9, seed=5)
    assert len(selected) == 9
    chosen = [c for c in clips if c.group_key in set(selected)]
    report = catalog_tier_report(chosen)
    # every package is represented rather than the alphabetically first one
    assert set(report["packages"]) == {"Locomotion", "Interactions", "Dances"}
    # the tier reaches past the shortest length bucket instead of taking the
    # nine shortest takes
    assert report["length_min"] <= 200
    assert report["length_max"] >= 320
    # and it covers both mirror-coverage strata
    assert report["groups_with_official_mirror"] > 0
    assert report["groups_without_official_mirror"] > 0
    # asking for at least as many groups as exist returns every group
    assert len(select_representative_groups(clips, 10_000)) == len({c.group_key for c in clips})


def test_subsetting_a_catalog_requires_renumbering(seed_root: Path, tmp_path: Path):
    from stylized_motion.data.seed_catalog import renumber_clips

    clips, manifest = discover_catalog(seed_root)
    subset = clips[2:6]
    assign_group_splits(subset, seed=4)
    broken = SeedCatalog(tmp_path / "broken", manifest, subset)
    with pytest.raises(SeedCatalogError, match="dense index"):
        broken.save()
    renumber_clips(subset)
    fixed = SeedCatalog(tmp_path / "fixed", refresh_catalog_counts(manifest, subset), subset)
    fixed.save()
    reloaded = SeedCatalog.load(tmp_path / "fixed")
    assert reloaded.num_clips == 4
    assert reloaded.manifest["num_clips"] == 4
    assert [c.clip_id for c in reloaded.clips] == [0, 1, 2, 3]
    assert len(set(reloaded.group_ids().tolist())) == reloaded.manifest["num_groups"]


def test_legacy_dataset_path_refuses_seed():
    from stylized_motion.data.preprocess import _discover_source_clips

    with pytest.raises(ValueError, match="catalogue \\+ packed-store pipeline"):
        _discover_source_clips("seed", None, None)


def test_short_clips_do_not_crash_simulation_root():
    from stylized_motion.data.preprocess import _savgol_filter

    short = np.arange(20, dtype=np.float64).reshape(20, 1)
    # the historical fixed window (61) exceeds a 20-frame clip
    assert _savgol_filter(short, 61, 3).shape == short.shape
    long = np.arange(200, dtype=np.float64).reshape(200, 1)
    assert np.allclose(_savgol_filter(long, 31, 3), __import__("scipy.signal", fromlist=["signal"]).savgol_filter(
        long, 31, 3, axis=0, mode="interp"
    ))


# ---------------------------------------------------------------------------
# P1: packed store
# ---------------------------------------------------------------------------


def test_packed_store_packs_many_clips_into_few_shards(tmp_path: Path):
    # 32 bytes per frame at 32 feature dims, so 32 KiB holds ~256 frames
    store_path = _build_store(tmp_path, clip_lengths=[80] * 20, groups=list(range(20)), shard_bytes=32 * 1024)
    store = open_packed_feature_store(store_path)
    try:
        assert store.num_clips == 20
        assert len(store.shard_files) < 20
        assert store.total_frames == 1600
        assert store.motion_dim == MOTION_DIM
        # rows are packed back to back inside a shard
        assert store.clip_offset[1] == store.clip_offset[0] + store.clip_length[0]
    finally:
        store.close()


def test_packed_store_rejects_windows_outside_a_logical_clip(tmp_path: Path):
    # Both clips must land in the same physical shard for this test to mean
    # anything: the point is that a window cannot reach its neighbour, not that
    # they are separated by shard boundaries.
    store_path = _build_store(tmp_path, clip_lengths=[80, 64], groups=[0, 1], variants=[0, 0], splits=[0, 1],
                              shard_bytes=64 * 1024)
    store = open_packed_feature_store(store_path)
    try:
        offset = int(store.clip_offset[0])
        length = int(store.clip_length[0])
        store.read_window(0, offset, length)
        with pytest.raises(IndexError, match="leaves clip"):
            store.read_window(0, offset - 1, 8)
        with pytest.raises(IndexError, match="leaves clip"):
            store.read_window(0, offset + length - 4, 8)
        # the same physical shard holds a second clip, but the first clip's
        # window cannot reach into it
        assert int(store.clip_shard[0]) == int(store.clip_shard[1])
    finally:
        store.close()


def test_packed_store_rows_align_across_geometry(tmp_path: Path):
    store_path = _build_store(tmp_path, clip_lengths=[80, 96, 120], splits=[0, 1, 2], variants=[0, 1, 0],
                             groups=[0, 0, 2])
    store = open_packed_feature_store(store_path)
    try:
        assert np.array_equal(store.clip_split, np.asarray([0, 1, 2], dtype=np.uint8))
        assert np.array_equal(store.source_clip_ids if hasattr(store, "source_clip_ids") else store.clip_source_id,
                              np.asarray([0, 0, 2], dtype=np.int32))
        assert store.clip_label(1)["mirror"] is True
    finally:
        store.close()


def test_packed_store_clip_position_sums_support_train_only_reference(tmp_path: Path):
    store_path = _build_store(tmp_path, splits=[0, 0, 1, 2], clip_lengths=[70, 90, 110, 130],
                             groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path)
    try:
        train_sum, train_frames = store.split_position_sum("train")
        test_sum, test_frames = store.split_position_sum("test")
        assert train_frames == 160
        assert test_frames == 130
        assert not np.allclose(train_sum, test_sum)
    finally:
        store.close()


def test_open_any_feature_store_rejects_unknown_schema(tmp_path: Path):
    path = tmp_path / "store"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"data_schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported data schema version"):
        open_any_feature_store(path)


def test_normalization_is_separate_and_versioned(tmp_path: Path):
    store_path = _build_store(tmp_path)
    store = open_packed_feature_store(store_path)
    try:
        assert store.normalization is not None
        assert store.normalization_hash == store.manifest["normalization_hash"]
        raw = store.read_clip(0)
        # feature bytes stay un-normalized: the file holds the raw values
        expected = store.read_frames(int(store.clip_shard[0]), int(store.clip_offset[0]), int(store.clip_length[0]))
        np.testing.assert_allclose(raw, expected)
        normalized = store.normalization.normalize(raw)
        assert not np.allclose(normalized, raw)
    finally:
        store.close()
    # refreshing statistics must not touch feature bytes
    import stylized_motion.data.preprocess as preprocess

    before = [
        hashlib.sha256((store_path / relative).read_bytes()).hexdigest()
        for relative in json.loads((store_path / "manifest.json").read_text())["shard_files"]
    ]
    report = preprocess.refresh_normalization(store_path)
    after = [
        hashlib.sha256((store_path / relative).read_bytes()).hexdigest()
        for relative in json.loads((store_path / "manifest.json").read_text())["shard_files"]
    ]
    assert before == after
    assert report["normalization_hash"]
    with pytest.raises(ValueError, match="train split"):
        preprocess.refresh_normalization(store_path, split="val")


def test_train_statistics_ignore_validation_and_test_frames(tmp_path: Path):
    store_path = _build_store(tmp_path, splits=[0, 0, 1, 2], clip_lengths=[70, 90, 110, 130],
                             groups=[0, 1, 2, 3])
    store = open_packed_feature_store(store_path, load_normalization=False)
    try:
        normalization = compute_normalization(store)
        # re-run with the val/test clips' data made extreme: the statistics must
        # not move because those frames are not part of the train scan
        accumulator = FeatureStatsAccumulator(store.motion_dim, store.num_joints)
        for block in store.iter_split_feature_blocks("train", chunk_frames=64):
            accumulator.update(block)
        assert int(accumulator.count) == int(normalization.train_frames)
        assert np.allclose(accumulator.finalize(NAMES).offset, normalization.stats.offset)
    finally:
        store.close()


def test_welford_accumulator_merges_like_a_single_pass():
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(97, 11)).astype(np.float32) for _ in range(7)]
    whole = FeatureStatsAccumulator(11, 2)
    for block in blocks:
        whole.update(block)
    split_accumulator = FeatureStatsAccumulator(11, 2)
    for block in blocks[:3]:
        split_accumulator.update(block)
    other = FeatureStatsAccumulator(11, 2)
    for block in blocks[3:]:
        other.update(block)
    split_accumulator.merge(other)
    assert whole.count == split_accumulator.count
    np.testing.assert_allclose(split_accumulator.mean, whole.mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(split_accumulator.std, whole.std, rtol=1e-10, atol=1e-12)


def test_normalization_rule_matches_the_legacy_v3_pipeline():
    """The v3 accumulator and the v4 streaming rule must agree numerically."""
    from stylized_motion.data.preprocess import _FeatureStatsAccumulator

    names = ["Simulation", "Hips", "LeftFoot"]
    dim = joint_feature_dim(len(names))
    rng = np.random.default_rng(7)
    values = rng.normal(size=(512, dim)).astype(np.float32)
    mask = np.ones(len(values), dtype=bool)

    legacy = _FeatureStatsAccumulator()

    class _Components:
        x = values
        positions = np.zeros((len(values), len(names), 3), dtype=np.float32)

    legacy.update(_Components(), mask)
    legacy_stats = legacy.finalize(names)

    streaming = FeatureStatsAccumulator(dim, len(names))
    streaming.update(values)
    streaming.add_ref_pos_sum(np.zeros((len(names), 3)), len(values))
    streaming_stats = streaming.finalize(names)

    np.testing.assert_allclose(streaming_stats.offset, legacy_stats.offset, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(streaming_stats.scale, legacy_stats.scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(streaming_stats.dist, legacy_stats.dist, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(streaming_stats.weights, legacy_stats.weights)
    # ... and the block-averaged scale rule is the historical one
    std = values.astype(np.float64).std(axis=0)
    np.testing.assert_allclose(compute_scale(names, std), legacy_stats.scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(compute_weights(names), legacy_stats.weights)


def test_normalization_artifact_rejects_tampering(tmp_path: Path):
    store_path = _build_store(tmp_path)
    store = open_packed_feature_store(store_path)
    store.close()
    with np.load(store_path / "normalization.npz", allow_pickle=False) as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    arrays["scale"] = arrays["scale"] * 1.5
    np.savez(store_path / "normalization.npz", **arrays)
    with pytest.raises(ValueError, match="do not match their recorded hash"):
        FeatureNormalization.load(store_path)


def test_packed_dataset_normalizes_on_cpu_or_defers_to_device(tmp_path: Path):
    import torch

    store_path = _build_store(tmp_path)
    store = open_packed_feature_store(store_path)
    try:
        request = SampleRequest(
            shard_idx=int(store.clip_shard[0]), target_start=int(store.clip_offset[0]), target_frames=64, variant_idx=0
        )
        cpu_dataset = PackedFeatureDataset("train", store, normalize_on="cpu")
        cpu_batch = cpu_dataset.__getitems__([request])
        raw = store.read_window(0, int(store.clip_offset[0]), 64)
        np.testing.assert_allclose(cpu_batch["motion"][0].numpy(), store.normalization.normalize(raw), rtol=1e-6)

        deferred = PackedFeatureDataset("train", store, normalize_on="none")
        deferred_batch = deferred.__getitems__([request])
        np.testing.assert_allclose(deferred_batch["motion"][0].numpy(), raw, rtol=1e-6)
        assert "normalization" in deferred_batch
        normalized = normalize_batch_on_device(deferred_batch, torch.device("cpu"))
        np.testing.assert_allclose(normalized["motion"][0].numpy(), cpu_batch["motion"][0].numpy(), rtol=1e-6)
    finally:
        store.close()


def test_packed_dataset_rejects_requests_from_another_split(tmp_path: Path):
    store_path = _build_store(tmp_path, splits=[0, 1], clip_lengths=[80, 80], groups=[0, 1], variants=[0, 0])
    store = open_packed_feature_store(store_path)
    try:
        dataset = PackedFeatureDataset("train", store)
        other = SampleRequest(
            shard_idx=int(store.clip_shard[1]), target_start=int(store.clip_offset[1]), target_frames=64, variant_idx=1
        )
        with pytest.raises(ValueError, match="does not belong to split"):
            dataset.__getitems__([other])
    finally:
        store.close()


def test_build_data_loaders_accepts_a_packed_store(tmp_path: Path):
    store_path = _build_store(tmp_path, clip_lengths=[100, 100, 100], splits=[0, 1, 2], groups=[0, 1, 2],
                             variants=[0, 0, 0])
    store = open_packed_feature_store(store_path)
    try:
        loaders = build_data_loaders(
            "representation",
            store,
            sampling_config={"strategy": "clip_uniform", "target_frames": 64, "samples_per_epoch": 8},
            loader_config={"batch_size": 4, "num_workers": 0, "prefetch_memory_limit_mb": None},
        )
        batch = next(iter(loaders.train))
        assert batch["motion"].shape == (4, 64, MOTION_DIM)
        assert loaders.train.sampler.index.num_groups == 1
    finally:
        store.close()


def test_loader_rejects_gpu_normalization_for_pre_normalized_v3_stores():
    class _FakeStore:
        motion_dim = MOTION_DIM

    with pytest.raises(TypeError):
        build_data_loaders(
            "representation",
            _FakeStore(),  # type: ignore[arg-type]
            sampling_config={"strategy": "clip_uniform"},
            loader_config={"batch_size": 2},
        )


# ---------------------------------------------------------------------------
# P1: resume machinery
# ---------------------------------------------------------------------------


def test_fingerprint_changes_when_a_source_file_changes(tmp_path: Path):
    path = tmp_path / "clip.bvh"
    path.write_bytes(b"HIERARCHY\n" * 100)
    first = fingerprint_file(path)
    path.write_bytes(b"HIERARCHY\n" * 200)
    second = fingerprint_file(path)
    assert first.digest() != second.digest()


def test_work_unit_ids_are_stable_and_signature_sensitive(tmp_path: Path):
    path = tmp_path / "clip.bvh"
    path.write_bytes(b"HIERARCHY\n" * 10)
    fingerprint = fingerprint_file(path)
    first = build_work_unit(signature="sig-a", clip_id=1, variant_id=0, source=path, fingerprint=fingerprint,
                            target_frames=100, start=0, stop=100)
    again = build_work_unit(signature="sig-a", clip_id=1, variant_id=0, source=path, fingerprint=fingerprint,
                            target_frames=100, start=0, stop=100)
    other = build_work_unit(signature="sig-b", clip_id=1, variant_id=0, source=path, fingerprint=fingerprint,
                            target_frames=100, start=0, stop=100)
    assert first.unit_id == again.unit_id
    assert first.unit_id != other.unit_id


def test_preprocess_signature_tracks_configuration():
    base = preprocess_signature(prune_ends_and_fingers=True, mirror_policy="official", target_fps=60)
    assert base == preprocess_signature(prune_ends_and_fingers=True, mirror_policy="official", target_fps=60)
    assert base != preprocess_signature(prune_ends_and_fingers=False, mirror_policy="official", target_fps=60)
    assert base != preprocess_signature(prune_ends_and_fingers=True, mirror_policy="generate", target_fps=60)
    assert base != preprocess_signature(prune_ends_and_fingers=True, mirror_policy="official", target_fps=30)
    assert base != preprocess_signature(prune_ends_and_fingers=True, mirror_policy="official", target_fps=60,
                                        extra={"shard_bytes": 1})


def test_journal_resumes_only_matching_units(tmp_path: Path):
    source = tmp_path / "clip.bvh"
    source.write_bytes(b"HIERARCHY\n" * 50)
    fingerprint = fingerprint_file(source)
    signature = "signature-1"
    unit = build_work_unit(signature=signature, clip_id=0, variant_id=0, source=source, fingerprint=fingerprint,
                           target_frames=100, start=0, stop=100)
    outputs = tmp_path / "units"
    outputs.mkdir()
    np.save(outputs / "unit.npy", np.zeros((100, MOTION_DIM), dtype=np.float32))
    journal = WorkJournal(tmp_path / "journal.jsonl")
    assert verify_completed_units(journal, [unit], signature=signature, store_root=tmp_path)[0] == [unit]
    result = UnitResult(
        unit_id=unit.unit_id,
        clip_id=0,
        variant_id=0,
        output="units/unit.npy",
        sha256=hashlib.sha256((outputs / "unit.npy").read_bytes()).hexdigest(),
        frames=100,
        motion_dim=MOTION_DIM,
        skeleton_hash="sk",
    )
    journal.commit(result, signature=signature)
    pending, completed = verify_completed_units(journal, [unit], signature=signature, store_root=tmp_path)
    assert pending == []
    assert unit.unit_id in completed

    # a different signature invalidates every previously completed unit
    pending, completed = verify_completed_units(journal, [unit], signature="signature-2", store_root=tmp_path)
    assert pending == [unit] and completed == {}

    # so does corrupted output
    np.save(outputs / "unit.npy", np.ones((100, MOTION_DIM), dtype=np.float32))
    pending, completed = verify_completed_units(journal, [unit], signature=signature, store_root=tmp_path)
    assert pending == [unit]


def test_journal_records_failures_and_survives_reload(tmp_path: Path):
    journal = WorkJournal(tmp_path / "journal.jsonl")
    journal.record_failure("unit-x", signature="sig", error="boom", source="/tmp/x.bvh")
    reloaded = WorkJournal(tmp_path / "journal.jsonl")
    assert "unit-x" in reloaded.failures
    assert reloaded.failures["unit-x"]["error"] == "boom"
    assert reloaded.report()["failed"] == 1


def test_bounded_map_respects_in_flight_limits_and_accounts_for_every_task():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    state = {"active": 0, "peak": 0, "peak_bytes": 0, "bytes": 0}
    lock = threading.Lock()

    def call(task: int) -> int:
        with lock:
            state["active"] += 1
            state["bytes"] += 100
            state["peak"] = max(state["peak"], state["active"])
            state["peak_bytes"] = max(state["peak_bytes"], state["bytes"])
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
            state["bytes"] -= 100
        return task

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = [
            value
            for _task, value in bounded_map(
                executor,
                range(12),
                call,
                max_inflight=3,
                max_inflight_bytes=250,
                task_bytes=lambda task: 100,
            )
        ]
    # every task is accounted for exactly once, in completion order
    assert sorted(results) == list(range(12))
    assert state["peak"] <= 3
    # the byte budget admits at most two 100-byte tasks before draining
    assert state["peak_bytes"] <= 250


def test_atomic_write_json_replaces_content(tmp_path: Path):
    target = tmp_path / "report.json"
    atomic_write_json(target, {"a": 1})
    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text())["a"] == 2
    assert not list(tmp_path.glob(".report.json*"))


def test_thread_limits_are_offered_for_worker_pools():
    from stylized_motion.data.resume import limit_thread_environment

    env = limit_thread_environment(2)
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"


# ---------------------------------------------------------------------------
# P1: samplers
# ---------------------------------------------------------------------------


def _store_rows(shard_ids, starts, stops, groups, mirrors, splits):
    return type(
        "Store",
        (),
        {
            "split_ids": np.asarray(splits, dtype=np.uint8),
            "source_clip_ids": np.asarray(groups, dtype=np.int32),
            "range_shard_indices": np.asarray(shard_ids, dtype=np.int32),
            "range_starts": np.asarray(starts, dtype=np.int64),
            "range_stops": np.asarray(stops, dtype=np.int64),
            "range_mirror": np.asarray(mirrors, dtype=bool),
            "style_ids": np.zeros(len(starts), dtype=np.int32),
            "action_ids": np.zeros(len(starts), dtype=np.int32),
        },
    )()


def test_train_sampler_balances_source_clips_not_window_counts():
    store = _store_rows(
        shard_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        starts=[0, 0, 0, 64, 0, 0, 0, 64],
        stops=[64, 64, 64, 128, 64, 64, 64, 128],
        groups=[0, 0, 1, 1, 0, 0, 1, 1],
        mirrors=[False, True, False, True, False, True, False, True],
        splits=[0] * 8,
    )
    sampler = TrainWindowSampler(store, samples_per_epoch=16, seed=3407)
    np.testing.assert_allclose(sampler.group_weights, [0.5, 0.5])


def test_clip_uniform_does_not_double_a_group_weight_when_a_mirror_exists():
    """A mirrored variant must not make its source twice as likely."""
    store = _store_rows(
        shard_ids=[0, 1, 2],
        starts=[0, 0, 0],
        stops=[100, 100, 100],
        groups=[0, 0, 1],          # group 0 has original+mirror, group 1 only original
        mirrors=[False, True, False],
        splits=[0, 0, 0],
    )
    sampler = TrainWindowSampler(store, samples_per_epoch=64, seed=3407, mirror_probability=0.5)
    np.testing.assert_allclose(sampler.group_weights, [0.5, 0.5])
    interval = store_intervals(store, "train")
    index = sampler.index
    # slots: group 0 owns two variant slots, group 1 owns one
    assert int(index.group_slot_stop[0] - index.group_slot_start[0]) == 2
    assert int(index.group_slot_stop[1] - index.group_slot_start[1]) == 1
    assert np.array_equal(interval.clip_id, np.arange(3))


def test_mirror_probability_falls_back_when_a_variant_has_no_windows():
    store = _store_rows(
        shard_ids=[0, 1],
        starts=[0, 0],
        stops=[100, 40],           # the mirror is too short for 64-frame windows
        groups=[0, 0],
        mirrors=[False, True],
        splits=[0, 0],
    )
    sampler = TrainWindowSampler(
        store, target_frames=64, samples_per_epoch=32, seed=1, mirror_probability=1.0
    )
    requests = list(sampler)
    assert len(requests) == 32
    # every request must come from the original: the mirror has no valid window
    assert {request.variant_idx for request in requests} == {0}


def test_frame_uniform_mode_weights_by_valid_start_count():
    store = _store_rows(
        shard_ids=[0, 1],
        starts=[0, 0],
        stops=[100, 600],          # 37 vs 537 valid 64-frame starts
        groups=[0, 1],
        mirrors=[False, False],
        splits=[0, 0],
    )
    clip_uniform = TrainWindowSampler(store, target_frames=64, samples_per_epoch=400, seed=7)
    frame_uniform = TrainWindowSampler(
        store, target_frames=64, samples_per_epoch=400, seed=7, strategy="frame_uniform"
    )
    assert frame_uniform.strategy == "frame_uniform"
    long_uniform = sum(1 for request in clip_uniform if request.variant_idx == 1)
    long_frames = sum(1 for request in frame_uniform if request.variant_idx == 1)
    assert long_uniform < long_frames
    # frame_uniform is a real mode, not an alias of clip_uniform
    assert long_frames > 250


def test_every_strategy_honours_mirror_probability():
    """group_balanced used to bypass variant selection entirely."""
    store = _store_rows(
        shard_ids=list(range(8)),
        starts=[0] * 8,
        stops=[300] * 8,
        groups=[0, 0, 1, 1, 2, 2, 3, 3],
        mirrors=[False, True, False, True, False, True, False, True],
        splits=[0] * 8,
    )
    for strategy, extra in (
        ("clip_uniform", {}),
        ("frame_uniform", {}),
        ("group_balanced", {"balance_key": "style", "balance_mix": 0.5}),
    ):
        wants_mirror = TrainWindowSampler(
            store, target_frames=64, samples_per_epoch=200, seed=11, mirror_probability=1.0,
            strategy=strategy, **extra
        )
        mirror_share = np.mean([bool(store.range_mirror[request.variant_idx]) for request in wants_mirror])
        wants_plain = TrainWindowSampler(
            store, target_frames=64, samples_per_epoch=200, seed=11, mirror_probability=0.0,
            strategy=strategy, **extra
        )
        plain_share = np.mean([not bool(store.range_mirror[request.variant_idx]) for request in wants_plain])
        assert mirror_share == 1.0, f"{strategy} ignored mirror_probability=1.0"
        assert plain_share == 1.0, f"{strategy} ignored mirror_probability=0.0"


def test_group_balanced_falls_back_when_mirrors_have_no_windows():
    store = _store_rows(
        shard_ids=[0, 1, 2, 3],
        starts=[0] * 4,
        stops=[300, 40, 300, 40],          # every mirror is too short for 64 frames
        groups=[0, 0, 1, 1],
        mirrors=[False, True, False, True],
        splits=[0] * 4,
    )
    sampler = TrainWindowSampler(
        store,
        target_frames=64,
        samples_per_epoch=64,
        seed=3,
        mirror_probability=1.0,
        strategy="group_balanced",
        balance_key="style",
    )
    requests = list(sampler)
    assert len(requests) == 64
    assert all(not bool(store.range_mirror[request.variant_idx]) for request in requests)


def test_group_balanced_caps_rare_class_weights():
    groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    styles = [0] * 9 + [1]
    store = type(
        "Store",
        (),
        {
            "split_ids": np.zeros(10, dtype=np.uint8),
            "source_clip_ids": np.asarray(groups, dtype=np.int32),
            "range_shard_indices": np.arange(10, dtype=np.int32),
            "range_starts": np.zeros(10, dtype=np.int64),
            "range_stops": np.full(10, 100, dtype=np.int64),
            "range_mirror": np.zeros(10, dtype=bool),
            "style_ids": np.asarray(styles, dtype=np.int32),
            "action_ids": np.zeros(10, dtype=np.int32),
        },
    )()
    plain = TrainWindowSampler(store, target_frames=64, samples_per_epoch=100, seed=1)
    balanced = TrainWindowSampler(
        store,
        target_frames=64,
        samples_per_epoch=100,
        seed=1,
        strategy="group_balanced",
        balance_key="style",
        balance_mix=1.0,
        balance_max_ratio=2.0,
    )
    rare_plain = float(plain.group_weights[-1])
    rare_balanced = float(balanced.group_weights[-1])
    assert rare_balanced > rare_plain              # the rare style is up-weighted
    assert rare_balanced <= rare_plain * 2.0 + 1e-9  # but capped against the natural weight
    natural = np.full(10, 0.1)
    assert np.all(balanced.group_weights <= natural * 2.0 + 1e-9)
    assert balanced.group_weights.sum() == pytest.approx(1.0)


def test_sampler_reports_coverage_and_repeats():
    store = _store_rows(
        shard_ids=[0, 1, 2, 3],
        starts=[0, 0, 0, 0],
        stops=[100, 100, 100, 100],
        groups=[0, 1, 2, 3],
        mirrors=[False, False, False, False],
        splits=[0, 0, 0, 0],
    )
    sampler = TrainWindowSampler(store, target_frames=64, samples_per_epoch=200, seed=2)
    list(sampler)
    summary = sampler.coverage_summary()
    assert summary["samples"] == 200
    assert summary["groups"] == 4
    assert summary["group_coverage"] == 1.0
    assert 0.0 < summary["normalized_entropy"] <= 1.0
    sampler.reset_coverage()
    assert sampler.coverage_summary()["samples"] == 0


def test_ddp_ranks_get_equal_step_counts_and_never_duplicate_ordinals():
    store = _store_rows(
        shard_ids=list(range(8)),
        starts=[0] * 8,
        stops=[100] * 8,
        groups=list(range(8)),
        mirrors=[False] * 8,
        splits=[0] * 8,
    )
    lengths = [
        len(TrainWindowSampler(store, target_frames=64, samples_per_epoch=33, seed=1, rank=rank, world_size=4))
        for rank in range(4)
    ]
    assert lengths == [8, 8, 8, 8]
    # a padded tail keeps the counts equal even when the budget is not divisible
    padded = [
        len(TrainWindowSampler(store, target_frames=64, samples_per_epoch=34, seed=1, rank=rank,
                               world_size=4, tail="pad"))
        for rank in range(4)
    ]
    assert padded == [9, 9, 9, 9]
    per_rank = []
    for rank in range(4):
        sampler = TrainWindowSampler(store, target_frames=64, samples_per_epoch=16, seed=1, rank=rank, world_size=4)
        per_rank.append([(request.variant_idx, request.target_start) for request in sampler])
    flattened = [item for rows in per_rank for item in rows]
    assert len(flattened) == 16
    assert len(set(flattened)) == len(flattened)
    with pytest.raises(ValueError, match="tail policy"):
        TrainWindowSampler(store, target_frames=64, samples_per_epoch=4, tail="wrap")


def test_sampler_state_supports_resuming_mid_epoch():
    store = _store_rows(
        shard_ids=list(range(4)),
        starts=[0] * 4,
        stops=[200] * 4,
        groups=list(range(4)),
        mirrors=[False] * 4,
        splits=[0] * 4,
    )
    sampler = TrainWindowSampler(store, target_frames=64, samples_per_epoch=8, seed=3)
    assert sampler.state_dict() == {"epoch": 0, "next_ordinal": 0}
    samples = list(sampler)
    assert sampler.state_dict()["next_ordinal"] == 8
    # resuming after the third sample replays the remaining ordinals exactly
    resumed = TrainWindowSampler(store, target_frames=64, samples_per_epoch=8, seed=3)
    resumed.load_state_dict({"epoch": 0, "next_ordinal": 3})
    remaining = list(resumed)
    assert remaining == samples[3:]
    assert len(remaining) == 5


def test_fixed_window_sampler_is_compact_and_bounded():
    store = _store_rows(
        shard_ids=[0, 1],
        starts=[0, 0],
        stops=[200, 100],
        groups=[0, 1],
        mirrors=[False, False],
        splits=[1, 1],
    )
    sampler = FixedWindowSampler(store, "val", target_frames=64, stride=64, include_tail=True)
    assert sampler.index.shape[1] == 4
    windows = list(sampler)
    assert len(windows) == len(sampler)
    assert all(request.target_frames == 64 for request in windows)
    limited = FixedWindowSampler(store, "val", target_frames=64, stride=64, limit=4)
    assert len(list(limited)) == 4


def test_sampling_contract_rejects_contradictory_configuration():
    with pytest.raises(ValueError, match="Unsupported sampling strategy"):
        sampling_contract({"strategy": "random"}, "representation")
    with pytest.raises(ValueError, match="group_balanced"):
        sampling_contract({"strategy": "group_balanced"}, "representation")
    with pytest.raises(ValueError, match="balance_key"):
        sampling_contract({"strategy": "group_balanced", "balance_key": "mood"}, "representation")
    with pytest.raises(ValueError, match="target_frames=64"):
        sampling_contract({"target_frames": 32}, "representation")
    contract = sampling_contract({"strategy": "frame_uniform"}, "representation")
    assert contract["strategy"] == "frame_uniform"
    assert contract["required_frames"] == 64
    assert sampling_contract({}, "generator")["required_frames"] == 65


# ---------------------------------------------------------------------------
# P1/P2: build orchestration helpers
# ---------------------------------------------------------------------------


def test_packed_order_is_split_major_and_mixes_sources():
    class _Plan:
        def __init__(self, split, group, source, variant):
            self.split = split
            self.source_group = group
            self.source_clip_id = source
            self.variant = variant

    plans = [_Plan(index % 3, index, index, 0) for index in range(30)]
    ordered = order_packed_clips(plans)
    splits = [plan.split for plan in ordered]
    assert splits == sorted(splits)  # same-split data stays together
    first_split_groups = [plan.source_group for plan in ordered if plan.split == 0]
    assert first_split_groups != sorted(first_split_groups)  # but sources are mixed
    keys = [packed_order_key(split=p.split, source_group=p.source_group, source_clip_id=p.source_clip_id,
                             variant=p.variant) for p in ordered]
    assert keys == sorted(keys)


def test_packed_row_order_matches_the_planner_order():
    store_path = _build_store(Path(__import__("tempfile").mkdtemp()), clip_lengths=[80, 90, 100],
                             groups=[2, 0, 1], variants=[0, 0, 0], splits=[0, 0, 0])
    store = open_packed_feature_store(store_path)
    try:
        order = packed_row_order(store)
        keys = [
            packed_order_key(split=int(store.clip_split[row]), source_group=int(store.clip_source_group[row]),
                             source_clip_id=int(store.clip_source_id[row]), variant=int(store.clip_variant[row]))
            for row in range(store.num_clips)
        ]
        assert [keys[row] for row in order.tolist()] == sorted(keys)
    finally:
        store.close()


def test_window_coverage_flags_splits_without_a_usable_window(tmp_path: Path):
    store_path = _build_store(tmp_path, clip_lengths=[40, 200], splits=[0, 1], groups=[0, 1], variants=[0, 0])
    store = open_packed_feature_store(store_path)
    try:
        coverage = window_coverage(store, window_frames=64)
        assert coverage["train"]["valid_windows"] == 0   # 40 < 64 frames
        assert coverage["val"]["valid_windows"] == 137
    finally:
        store.close()


def test_probe_bvh_header_reads_only_the_header(tmp_path: Path):
    path = tmp_path / "clip.bvh"
    path.write_text(
        "HIERARCHY\n"
        "ROOT Root\n"
        "{\n"
        "  JOINT Hips\n"
        "  {\n"
        "    End Site\n"
        "    {\n"
        "    }\n"
        "  }\n"
        "}\n"
        "MOTION\n"
        "Frames: 42\n"
        "Frame Time: 0.008333\n"
        + ("0.0 " * 100 + "\n") * 50,
        encoding="utf-8",
    )
    header = probe_bvh_header(path)
    assert header["frames"] == 42
    assert header["joints"] == 3
    assert abs(header["frametime"] - 0.008333) < 1e-9
    assert header["names"][0] == "Root"


def test_seed_build_rejects_units_that_do_not_match_the_skeleton(tmp_path: Path):
    names = list(NAMES)
    parents = list(PARENTS)
    assert skeleton_hash(names, parents, "full") != skeleton_hash(names, parents, "prune_ends_and_fingers")
    assert feature_schema_hash(names, parents, "full") != feature_schema_hash(names + ["Extra"], parents, "full")
    assert ROOT_CHANNELS == 7
    assert SeedBuildError is not None
