"""BONES-SEED -> schema-v4 packed feature-store build.

Plan §4.3. The build is split into stages that can each be interrupted and
resumed:

``inventory``
    Probe every catalogue file's header (frame count, frame time, joint count,
    mirror pairing) and persist a lightweight JSONL report. Nothing is parsed
    twice and mismatches are reported instead of discovered halfway through.
``units``
    One work unit per *(clip, variant)* parses, decimates, applies the SOMA
    conventions and extracts features *inside the worker*, writing a temporary
    ``.npy`` plus a small JSON sidecar. Only a tiny descriptor crosses the
    process boundary, which removes the v3 upload-of-whole-clips bottleneck.
``pack``
    Stream the unit files in split-major order into ~256 MiB packed shards,
    deleting nothing until the store is published.
``stats``
    Scan the published train ranges with float64 Welford accumulators and write
    a separately versioned normalization artifact.

Every unit carries a stable id, an input fingerprint and a preprocessing
signature; success is committed to an append-only journal and a rerun reuses
whatever still matches.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from stylized_motion.anim import bvh
from stylized_motion.anim.features import build_motion_feature_components, joint_feature_dim
from stylized_motion.data.normalization import compute_normalization
from stylized_motion.data.packed_store import (
    CLIP_UNKNOWN_LABEL_ID,
    ClipTableEntry,
    PackedFeatureStoreWriter,
    feature_schema_hash,
    publish_packed_store,
    skeleton_hash,
    write_clip_table,
)
from stylized_motion.data.resume import (
    PREPROCESS_VERSION,
    FileFingerprint,
    UnitResult,
    WorkJournal,
    apply_thread_limits,
    bounded_map,
    build_work_unit,
    fingerprint_file,
    preprocess_signature,
    remove_tree,
    sha256_file,
)
from stylized_motion.data.seed_catalog import (
    MIRROR_POLICIES,
    VARIANT_GENERATED_MIRROR,
    VARIANT_OFFICIAL_MIRROR,
    VARIANT_ORIGINAL,
    SeedCatalog,
    SeedCatalogError,
)

INVENTORY_FILENAME = "inventory.jsonl"
SIDECAR_SUFFIX = ".meta.json"
ROOT_SUFFIX = ".root"


class SeedBuildError(RuntimeError):
    """Raised when a SEED build cannot be completed safely."""


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def probe_bvh_header(path: str | Path) -> dict[str, Any]:
    """Read only the BVH header: joint names, frame count and frame time."""
    target = Path(path)
    names: list[str] = []
    frames: int | None = None
    frametime: float | None = None
    motion_lines = 0
    in_motion = False
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not in_motion:
                if "MOTION" in line:
                    in_motion = True
                    continue
                for keyword in ("ROOT ", "JOINT ", "End Site"):
                    if keyword in line:
                        names.append("EndSite" if keyword == "End Site" else line.split(keyword, 1)[1].strip())
                        break
                continue
            # Only the two header lines after MOTION belong to the header; the
            # rest of the file is frame data and must not be scanned here.
            motion_lines += 1
            if line.startswith("Frames:"):
                frames = int(line.split(":", 1)[1].strip())
            elif line.startswith("Frame Time:"):
                frametime = float(line.split(":", 1)[1].strip())
            if motion_lines >= 4:
                break
    if frames is None:
        raise SeedBuildError(f"Missing Frames header in {target}")
    return {
        "frames": int(frames),
        "frametime": frametime,
        "joints": len(names),
        "names_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "names": names,
    }


def _inventory_task(payload: tuple[int, str, int]) -> dict[str, Any]:
    clip_id, path, expected_frames = payload
    try:
        header = probe_bvh_header(path)
    except Exception as error:  # pragma: no cover - exercised through the report
        return {"clip_id": int(clip_id), "path": path, "status": "error", "error": str(error)[:500]}
    return {
        "clip_id": int(clip_id),
        "path": path,
        "status": "ok",
        "frames": int(header["frames"]),
        "frametime": header["frametime"],
        "joints": int(header["joints"]),
        "names_sha256": header["names_sha256"],
        "frames_match_metadata": int(header["frames"]) == int(expected_frames),
        "is_120fps": header["frametime"] is not None and abs(float(header["frametime"]) - 1.0 / 120.0) < 1e-5,
        "is_60fps": header["frametime"] is not None and abs(float(header["frametime"]) - 1.0 / 60.0) < 1e-5,
    }


def inventory_catalog(
    catalog: SeedCatalog,
    root: str | Path,
    output: str | Path,
    *,
    workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Probe every catalogue file's header and persist the report as JSONL."""
    root = Path(root)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Inventory report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        (clip.clip_id, str(root / clip.relative_path), int(clip.raw_frames)) for clip in catalog.clips
    ]
    summary: dict[str, Any] = {
        "clips": len(tasks),
        "ok": 0,
        "errors": 0,
        "frames_mismatch": 0,
        "wrong_frame_time": 0,
        "joint_counts": {},
        "skeleton_hashes": {},
    }
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        if workers <= 1:
            iterator: Iterable[dict[str, Any]] = (
                _inventory_task(task) for task in tqdm(tasks, desc="Inventory")
            )
            for record in iterator:
                _record_inventory(record, summary)
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        else:
            context = __import__("multiprocessing").get_context("fork")
            with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
                iterator = executor.map(_inventory_task, tasks)
                for record in tqdm(iterator, total=len(tasks), desc="Inventory"):
                    _record_inventory(record, summary)
                    handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    summary["report"] = str(output)
    if summary["errors"]:
        raise SeedBuildError(
            f"{summary['errors']} SEED files could not be read during inventory; see {output}"
        )
    if summary["frames_mismatch"]:
        raise SeedBuildError(
            f"{summary['frames_mismatch']} SEED files disagree with move_duration_frames; see {output}"
        )
    if summary["wrong_frame_time"]:
        raise SeedBuildError(
            f"{summary['wrong_frame_time']} SEED files are neither 120 fps nor 60 fps; see {output}"
        )
    if len(summary["skeleton_hashes"]) != 1:
        raise SeedBuildError(
            f"SEED soma_uniform files do not share one skeleton: {sorted(summary['skeleton_hashes'])}"
        )
    return summary


def _record_inventory(record: Mapping[str, Any], summary: dict[str, Any]) -> None:
    if record["status"] != "ok":
        summary["errors"] = int(summary["errors"]) + 1
        return
    summary["ok"] = int(summary["ok"]) + 1
    if not record["frames_match_metadata"]:
        summary["frames_mismatch"] = int(summary["frames_mismatch"]) + 1
    if not (record["is_120fps"] or record["is_60fps"]):
        summary["wrong_frame_time"] = int(summary["wrong_frame_time"]) + 1
    joints = str(int(record["joints"]))
    summary["joint_counts"][joints] = int(summary["joint_counts"].get(joints, 0)) + 1
    digest = str(record["names_sha256"])
    summary["skeleton_hashes"][digest] = int(summary["skeleton_hashes"].get(digest, 0)) + 1


# ---------------------------------------------------------------------------
# worker units
# ---------------------------------------------------------------------------


ROOT_CHANNELS = 7  # root position xyz + root rotation quaternion wxyz


@dataclass(frozen=True)
class UnitTask:
    """Picklable worker payload for one (clip, variant) pair."""

    unit_id: str
    clip_id: int
    variant_id: int
    source: str
    mirror: bool
    prune_ends_and_fingers: bool
    start: int
    stop: int
    output: str
    sidecar: str
    root_output: str = ""


def _root_channels(cropped: Mapping[str, Any]) -> np.ndarray:
    """Per-frame root position and rotation, saved at feature-build time.

    Trajectory conditioning needs root translation and heading, which the
    feature vector does not carry; capturing them here avoids re-parsing BVH
    files for the trajectory store later (plan §4.6).
    """
    positions = np.asarray(cropped["positions"], dtype=np.float32)
    rotations = np.asarray(cropped["rotations"], dtype=np.float32)
    values = np.concatenate((positions[:, 0], rotations[:, 0]), axis=-1).astype(np.float32)
    if values.ndim != 2 or values.shape[1] != ROOT_CHANNELS:
        raise SeedBuildError(f"Root channels must be [N, {ROOT_CHANNELS}], got {values.shape}")
    return values


def process_unit(task: UnitTask) -> UnitResult:
    """Parse, decimate, SOMA-process and featurize one clip inside the worker."""
    # Imported lazily so the worker module stays importable without pulling the
    # whole preprocessing stack in the parent process.
    from stylized_motion.data.preprocess import _process_motion_data, _slice_motion

    apply_thread_limits(1)
    path = Path(task.source)
    bvh_data = bvh.load(path.as_posix())
    motion = _process_motion_data(
        bvh_data,
        mirror=bool(task.mirror),
        prune_ends_and_fingers=bool(task.prune_ends_and_fingers),
    )
    frames = int(np.asarray(motion["positions"]).shape[0])
    if int(task.stop) > frames:
        raise SeedBuildError(
            f"Clip {path.name} has {frames} processed frames but the unit asks for [{task.start}, {task.stop})"
        )
    cropped = _slice_motion(motion, int(task.start), int(task.stop))
    if len(cropped["positions"]) != int(task.stop) - int(task.start):
        raise SeedBuildError(f"Cropped clip {path.name} does not match its declared frame count")
    components = build_motion_feature_components(cropped)
    values = np.ascontiguousarray(components.x, dtype=np.float32)
    output = Path(task.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, values)
    if task.root_output:
        root_path = Path(task.root_output)
        root_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(root_path, _root_channels(cropped))
    names = [str(name) for name in motion["names"]]
    parents = [int(value) for value in np.asarray(motion["parents"]).tolist()]
    sidecar_payload = {
        "unit_id": task.unit_id,
        "clip_id": int(task.clip_id),
        "variant_id": int(task.variant_id),
        "frames": int(values.shape[0]),
        "motion_dim": int(values.shape[1]),
        "names": names,
        "parents": parents,
        "position_sum": np.asarray(components.positions, dtype=np.float64).sum(axis=0).tolist(),
        "source": str(path),
        "mirror": bool(task.mirror),
        "preprocess_version": PREPROCESS_VERSION,
    }
    sidecar = Path(task.sidecar)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(sidecar_payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256_file(output)
    return UnitResult(
        unit_id=task.unit_id,
        clip_id=int(task.clip_id),
        variant_id=int(task.variant_id),
        output=output.as_posix(),
        sha256=digest,
        frames=int(values.shape[0]),
        motion_dim=int(values.shape[1]),
        skeleton_hash=skeleton_hash(names, parents, "prune_ends_and_fingers" if task.prune_ends_and_fingers else "full"),
        bytes_written=int(values.nbytes),
        extra={"sidecar": sidecar.as_posix(), "mirror": bool(task.mirror)},
    )


def plan_units(
    catalog: SeedCatalog,
    root: str | Path,
    unit_dir: str | Path,
    *,
    signature: str,
    mirror_policy: str,
    prune_ends_and_fingers: bool,
    fingerprints: Mapping[int, FileFingerprint] | None = None,
    emit_root_channels: bool = True,
) -> list[UnitTask]:
    """Expand the catalogue into work units for the selected mirror policy."""
    if str(mirror_policy) not in MIRROR_POLICIES:
        raise SeedCatalogError(f"Unsupported mirror policy {mirror_policy!r}")
    root = Path(root)
    unit_dir = Path(unit_dir)
    tasks: list[UnitTask] = []
    cache: dict[int, FileFingerprint] = dict(fingerprints or {})
    for clip in catalog.clips:
        if clip.variant_id == VARIANT_OFFICIAL_MIRROR and mirror_policy != "official":
            continue
        fingerprint = cache.get(int(clip.clip_id))
        if fingerprint is None:
            fingerprint = fingerprint_file(root / clip.relative_path)
            cache[int(clip.clip_id)] = fingerprint
        mirrors = (False, True) if mirror_policy == "generate" else (bool(clip.is_mirror),)
        for mirror in mirrors:
            if mirror_policy == "none" and mirror:
                continue
            unit = build_work_unit(
                signature=signature,
                clip_id=int(clip.clip_id),
                variant_id=int(VARIANT_GENERATED_MIRROR if mirror_policy == "generate" and mirror else clip.variant_id),
                source=root / clip.relative_path,
                fingerprint=fingerprint,
                target_frames=int(clip.target_frames),
                start=int(clip.start),
                stop=int(clip.stop),
            )
            tasks.append(
                UnitTask(
                    unit_id=unit.unit_id,
                    clip_id=int(clip.clip_id),
                    variant_id=int(unit.variant_id),
                    source=str(unit.source),
                    mirror=bool(mirror),
                    prune_ends_and_fingers=bool(prune_ends_and_fingers),
                    start=int(clip.start),
                    stop=int(clip.stop),
                    output=(unit_dir / f"{unit.unit_id}.npy").as_posix(),
                    sidecar=(unit_dir / f"{unit.unit_id}{SIDECAR_SUFFIX}").as_posix(),
                    root_output=(unit_dir / f"{unit.unit_id}{ROOT_SUFFIX}.npy").as_posix()
                    if emit_root_channels
                    else "",
                )
            )
    return tasks


def run_units(
    tasks: Sequence[UnitTask],
    *,
    journal: WorkJournal,
    signature: str,
    store_root: Path,
    workers: int = 1,
    max_inflight: int | None = None,
    max_inflight_bytes: int | None = None,
    motion_dim_hint: int = 248,
    verify_checksums: bool = True,
    desc: str = "Building units",
) -> tuple[dict[str, UnitResult], dict[str, Any]]:
    """Run pending units, reusing journaled results, and commit atomically."""
    from stylized_motion.data.resume import _sha256_file

    completed: dict[str, UnitResult] = dict(journal.committed)
    pending: list[UnitTask] = []
    reused = 0
    for task in tasks:
        record = completed.get(task.unit_id)
        if record is None:
            pending.append(task)
            continue
        path = Path(record.output)
        if not path.exists() or (verify_checksums and _sha256_file(path) != record.sha256):
            completed.pop(task.unit_id, None)
            pending.append(task)
            continue
        reused += 1
    # Long clips first: with completion-order intake the tail of the run is
    # otherwise dominated by whichever long clip started last.
    pending.sort(key=lambda task: (-(int(task.stop) - int(task.start)), task.unit_id))
    report: dict[str, Any] = {
        "units": len(tasks),
        "reused": reused,
        "pending": len(pending),
        "failed": 0,
        "failures": [],
    }
    if not pending:
        return completed, report
    inflight = int(max_inflight) if max_inflight is not None else max(2, int(workers) * 2)
    bytes_budget = (
        int(max_inflight_bytes)
        if max_inflight_bytes is not None
        else max(256 * 1024 * 1024, inflight * motion_dim_hint * 4 * 4096)
    )
    if workers <= 1:
        for task in tqdm(pending, desc=desc):
            try:
                result = process_unit(task)
            except Exception as error:
                journal.record_failure(task.unit_id, signature=signature, error=str(error), source=task.source)
                report["failed"] = int(report["failed"]) + 1
                report["failures"].append({"unit_id": task.unit_id, "source": task.source, "error": str(error)[:200]})
                continue
            journal.commit(result, signature=signature)
            completed[result.unit_id] = result
        return completed, report
    import multiprocessing as mp

    # Forked workers inherit this environment, which is what actually keeps
    # BLAS from spawning `workers x cores` threads on a 30 GiB host.
    apply_thread_limits(1)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
        for task, outcome in bounded_map(
            executor,
            pending,
            _bounded_process_unit,
            max_inflight=inflight,
            max_inflight_bytes=bytes_budget,
            task_bytes=lambda item: (int(item.stop) - int(item.start)) * int(motion_dim_hint) * 4,
        ):
            if isinstance(outcome, Exception):
                journal.record_failure(
                    task.unit_id, signature=signature, error=str(outcome), source=task.source
                )
                report["failed"] = int(report["failed"]) + 1
                report["failures"].append(
                    {"unit_id": task.unit_id, "source": task.source, "error": str(outcome)[:200]}
                )
                continue
            journal.commit(outcome, signature=signature)
            completed[outcome.unit_id] = outcome
    return completed, report


def _bounded_process_unit(task: UnitTask) -> Any:
    """``process_unit`` that reports failures as values so the pool keeps going."""
    try:
        return process_unit(task)
    except Exception as error:  # noqa: BLE001 - failures are journaled by the caller
        return error


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


@dataclass
class PackedClipPlan:
    """One clip row of the packed store, resolved before any bytes are written."""

    clip_id: int
    source_clip_id: int
    unit_id: str
    source_group: int
    variant: int
    split: int
    mirror: bool
    length: int
    move_name: str
    relative_path: str
    style_id: int
    action_id: int
    package_id: int
    performer_id: int
    position_sum: np.ndarray


def _label_ids(
    catalog: SeedCatalog,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    def build(field: str) -> dict[str, int]:
        values: list[str] = []
        for clip in catalog.clips:
            raw = clip.labels.get(field, "") or "__unknown__"
            if raw not in values:
                values.append(raw)
        return {name: index for index, name in enumerate(sorted(values))}

    # Actor labels come from the catalogue's take metadata, not from the CSV
    # label block: they are what a performer holdout needs, and the store cannot
    # recover them later without re-reading the catalogue.
    actors = sorted({str(clip.take_actor or clip.actor_uid) for clip in catalog.clips if (clip.take_actor or clip.actor_uid)})
    return (
        build("content_uniform_style"),
        build("package"),
        build("category"),
        {name: index for index, name in enumerate(actors)},
    )


def plan_packed_clips(
    catalog: SeedCatalog,
    results: Mapping[str, UnitResult],
    *,
    unit_dir: Path,
    mirror_policy: str,
) -> list[PackedClipPlan]:
    """Resolve user-facing clip rows into packed-store rows.

    Under the ``official`` policy every official file is one row, including the
    official mirrors. Under ``generate`` the synthetic mirror is a second row in
    the *same* source group, marked with the generated-mirror variant, so the
    sampler keeps treating the group as one source.
    """
    style_ids, package_ids, category_ids, performer_ids = _label_ids(catalog)
    index: dict[tuple[int, bool], UnitResult] = {}
    for result in results.values():
        index[(int(result.clip_id), bool(result.extra.get("mirror")))] = result
    plans: list[PackedClipPlan] = []
    for clip in catalog.clips:
        if clip.variant_id == VARIANT_OFFICIAL_MIRROR and mirror_policy != "official":
            continue
        variants: list[tuple[bool, int, str]] = []
        if mirror_policy == "generate":
            for mirror in (False, True):
                base = index.get((int(clip.clip_id), bool(mirror)))
                if base is not None:
                    variants.append(
                        (mirror, VARIANT_GENERATED_MIRROR if mirror else VARIANT_ORIGINAL, base.unit_id)
                    )
        else:
            base = index.get((int(clip.clip_id), bool(clip.is_mirror)))
            if base is not None:
                variants.append((bool(clip.is_mirror), int(clip.variant_id), base.unit_id))
        for mirror, variant_id, unit_id in variants:
            record = results[unit_id]
            sidecar = json.loads(
                (unit_dir / f"{unit_id}{SIDECAR_SUFFIX}").read_text(encoding="utf-8")
            )
            plans.append(
                PackedClipPlan(
                    clip_id=len(plans),
                    source_clip_id=int(clip.clip_id),
                    unit_id=unit_id,
                    source_group=int(clip.group_id),
                    variant=int(variant_id),
                    split={"train": 0, "val": 1, "test": 2}[clip.split],
                    mirror=bool(mirror),
                    length=int(record.frames),
                    move_name=clip.move_name,
                    relative_path=clip.relative_path,
                    style_id=style_ids.get(clip.labels.get("content_uniform_style", "") or "__unknown__", 0),
                    action_id=category_ids.get(clip.labels.get("category", "") or "__unknown__", 0),
                    package_id=package_ids.get(clip.labels.get("package", "") or "__unknown__", 0),
                    performer_id=performer_ids.get(
                        str(clip.take_actor or clip.actor_uid), CLIP_UNKNOWN_LABEL_ID
                    ),
                    position_sum=np.asarray(sidecar["position_sum"], dtype=np.float64),
                )
            )
    return plans


def packed_order_key(
    *,
    split: int,
    source_group: int,
    source_clip_id: int,
    variant: int,
    seed: int = 3407,
) -> tuple[Any, ...]:
    """Deterministic packed-store row order, computable from the clip table alone.

    Splits stay contiguous so same-split data lands together, while the
    within-split order is a hash of the source group, which mixes recording
    dates inside a shard. Depending on nothing but ``(split, group, source,
    variant)`` lets the token store reproduce the feature store's order row for
    row without having to re-derive it from the catalogue.
    """
    digest = hashlib.sha256(f"{seed}:{int(source_group)}:{int(source_clip_id)}".encode("utf-8")).digest()
    return (int(split), digest, int(source_clip_id), int(variant))


def order_packed_clips(
    plans: Sequence[PackedClipPlan],
    *,
    seed: int = 3407,
) -> list[PackedClipPlan]:
    """Order packed rows with :func:`packed_order_key`."""
    return sorted(
        plans,
        key=lambda plan: packed_order_key(
            split=plan.split,
            source_group=plan.source_group,
            source_clip_id=plan.source_clip_id,
            variant=plan.variant,
            seed=seed,
        ),
    )


def packed_row_order(store: Any, *, seed: int = 3407) -> np.ndarray:
    """Return the row permutation that :func:`order_packed_clips` would produce."""
    keys = [
        packed_order_key(
            split=int(store.clip_split[row]),
            source_group=int(store.clip_source_group[row]),
            source_clip_id=int(store.clip_source_id[row]),
            variant=int(store.clip_variant[row]),
            seed=seed,
        )
        for row in range(store.num_clips)
    ]
    return np.asarray(sorted(range(len(keys)), key=lambda row: keys[row]), dtype=np.int64)


def read_unit_frames(unit_path: Path) -> np.ndarray:
    return np.load(unit_path, mmap_mode="r", allow_pickle=False)


def pack_clips(
    plans: Sequence[PackedClipPlan],
    unit_dir: Path,
    staging: Path,
    *,
    motion_dim: int,
    num_joints: int,
    names: Sequence[str],
    shard_bytes: int,
    purge_units: bool = False,
    desc: str = "Packing shards",
) -> tuple[list[ClipTableEntry], dict[str, Any]]:
    """Stream unit features into packed shards in the planned order."""
    writer = PackedFeatureStoreWriter(
        staging, motion_dim=motion_dim, num_joints=num_joints, shard_bytes=shard_bytes, names=names
    )
    # Root position/heading rides along as a parallel packed array with the same
    # shard/offset/length layout, so trajectory conditioning never has to
    # re-parse BVH files (plan §4.6).
    root_writer = PackedFeatureStoreWriter(
        staging,
        motion_dim=ROOT_CHANNELS,
        num_joints=num_joints,
        shard_bytes=shard_bytes,
        names=names,
        subdirectory="root",
        frames_per_shard=writer.frames_per_shard,
    )
    entries: list[ClipTableEntry] = []
    progress = tqdm(total=len(plans), desc=desc)
    try:
        for plan in plans:
            unit_path = unit_dir / f"{plan.unit_id}.npy"
            values = read_unit_frames(unit_path)
            if values.shape[0] != plan.length or values.shape[1] != motion_dim:
                raise SeedBuildError(
                    f"Unit {plan.unit_id} has shape {values.shape}, expected ({plan.length}, {motion_dim})"
                )
            root_path = unit_dir / f"{plan.unit_id}{ROOT_SUFFIX}.npy"
            if root_path.exists():
                root_values = np.asarray(read_unit_frames(root_path))
                if root_values.shape != (plan.length, ROOT_CHANNELS):
                    raise SeedBuildError(
                        f"Unit {plan.unit_id} root channels have shape {root_values.shape}, "
                        f"expected ({plan.length}, {ROOT_CHANNELS})"
                    )
                root_writer.append_clip(
                    root_values,
                    source_group=plan.source_group,
                    variant=plan.variant,
                    split=plan.split,
                    mirror=plan.mirror,
                    source_id=plan.source_clip_id,
                    style_id=plan.style_id,
                    action_id=plan.action_id,
                    package_id=plan.package_id,
                    performer_id=plan.performer_id,
                    move_name=plan.move_name,
                    relative_path=plan.relative_path,
                    position_sum=plan.position_sum,
                )
            entry = writer.append_clip(
                np.asarray(values),
                source_group=plan.source_group,
                variant=plan.variant,
                split=plan.split,
                mirror=plan.mirror,
                source_id=plan.source_clip_id,
                style_id=plan.style_id,
                action_id=plan.action_id,
                package_id=plan.package_id,
                performer_id=plan.performer_id,
                move_name=plan.move_name,
                relative_path=plan.relative_path,
                position_sum=plan.position_sum,
            )
            entries.append(entry)
            progress.update()
            if purge_units:
                remove_tree(unit_path)
                remove_tree(root_path)
                remove_tree(unit_dir / f"{plan.unit_id}{SIDECAR_SUFFIX}")
    finally:
        progress.close()
        writer.close_shards()
        root_writer.close_shards()
    return entries, {
        "shards": len(writer.shard_files),
        "bytes": writer.bytes_on_disk,
        "root_shards": len(root_writer.shard_files),
        "root_shard_files": list(root_writer.shard_files),
        "root_shard_sha256": list(root_writer.shard_sha256),
    }


VERIFY_LEVELS = ("quick", "checksum", "full")


def verify_split_isolation(store: Any) -> None:
    """Assert that no source group straddles two splits."""
    splits_by_group: dict[int, set[int]] = {}
    for group, split in zip(store.clip_source_group.tolist(), store.clip_split.tolist()):
        splits_by_group.setdefault(int(group), set()).add(int(split))
    leaked = [group for group, splits in splits_by_group.items() if len(splits) > 1]
    if leaked:
        raise SeedBuildError(f"{len(leaked)} source groups span multiple splits (first: {leaked[:3]})")


def window_coverage(store: Any, *, window_frames: int = 64) -> dict[str, dict[str, int]]:
    """Count how many windows of ``window_frames`` each split can actually serve.

    A split that has clips but no valid window would train or evaluate on
    nothing; the publish check turns that into a hard error instead of a
    silently empty run.
    """
    window_frames = int(window_frames)
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    report: dict[str, dict[str, int]] = {}
    for name, index in (("train", 0), ("val", 1), ("test", 2)):
        rows = np.flatnonzero(store.clip_split == index)
        if len(rows) == 0:
            report[name] = {"clips": 0, "frames": 0, "valid_windows": 0}
            continue
        lengths = np.asarray(store.clip_length, dtype=np.int64)[rows]
        valid = np.maximum(lengths - window_frames + 1, 0)
        report[name] = {
            "clips": int(len(rows)),
            "frames": int(lengths.sum()),
            "valid_windows": int(valid.sum()),
        }
    return report


def verify_packed_store(
    database: Path,
    *,
    level: str = "full",
    window_frames: int = 64,
) -> dict[str, Any]:
    """Re-open a packed store and validate it at the requested level.

    ``quick`` checks the manifest, shapes and the clip table; ``checksum``
    additionally re-hashes every shard; ``full`` also scans every frame for
    finiteness and confirms that no source group straddles two splits. The
    publish step runs ``full`` so a silently corrupted build can never be
    mistaken for a successful one.
    """
    from stylized_motion.data.packed_store import open_packed_feature_store

    if str(level) not in VERIFY_LEVELS:
        raise ValueError(f"Unsupported verification level {level!r}; expected one of {VERIFY_LEVELS}")
    store = open_packed_feature_store(database, load_normalization=False)
    try:
        report = {
            "level": str(level),
            "clips": store.num_clips,
            "shards": len(store.shard_files),
            "frames": store.total_frames,
            "motion_dim": store.motion_dim,
            "clips_without_tail": int(
                (
                    np.asarray(store.shard_num_frames, dtype=np.int64)[store.clip_shard]
                    - (store.clip_offset + store.clip_length)
                ).sum()
            ),
            "splits": {
                name: int((store.clip_split == index).sum())
                for name, index in (("train", 0), ("val", 1), ("test", 2))
            },
            "window_coverage": window_coverage(store, window_frames=int(window_frames)),
        }
        empty = [
            name
            for name, entry in report["window_coverage"].items()
            if entry["clips"] > 0 and entry["valid_windows"] == 0
        ]
        if empty:
            raise SeedBuildError(
                f"Splits {empty} contain clips but no {int(window_frames)}-frame window"
            )
        if level in {"checksum", "full"}:
            manifest = store.manifest
            for relative, digest in zip(manifest["shard_files"], manifest["shard_sha256"]):
                if sha256_file(database / relative) != digest:
                    raise SeedBuildError(f"Packed shard checksum mismatch: {relative}")
        if level == "full":
            for name in ("train", "val", "test"):
                for clip_idx in store.split_clip_indices(name).tolist():
                    values = store.read_clip(int(clip_idx))
                    if not np.isfinite(values).all():
                        raise SeedBuildError(f"Packed store contains non-finite values at clip {clip_idx}")
            verify_split_isolation(store)
        return report
    finally:
        store.close()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


@dataclass
class SeedBuildConfig:
    """Everything that a resume must match to reuse earlier work."""

    root: Path
    output: Path
    mirror_policy: str = "official"
    prune_ends_and_fingers: bool = True
    shard_bytes: int = 256 * 1024 * 1024
    workers: int = 4
    max_inflight: int | None = None
    max_inflight_bytes: int | None = None
    unit_cache: Path | None = None
    purge_units: bool = False
    verify: str = "full"
    window_frames: int = 64
    limit_clips: int | None = None


def build_packed_feature_store(
    catalog: SeedCatalog,
    config: SeedBuildConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, normalize and publish a schema-v4 packed feature store."""
    output = Path(config.output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Packed store already exists: {output}; pass overwrite=True to replace it")
    root = Path(config.root)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        remove_tree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    signature = preprocess_signature(
        prune_ends_and_fingers=bool(config.prune_ends_and_fingers),
        mirror_policy=str(config.mirror_policy),
        target_fps=60,
        extra={
            "shard_bytes": int(config.shard_bytes),
            "dataset": catalog.manifest.get("dataset", ""),
            "temporal_label_version": catalog.manifest.get("temporal_label_version", ""),
        },
    )
    unit_dir = Path(config.unit_cache) if config.unit_cache is not None else staging / "units"
    if config.unit_cache is not None:
        # Publishing replaces the output directory atomically, so a unit cache
        # inside it would be created first and then destroyed on publish. The
        # cache must be a sibling that survives rebuilds.
        resolved_units = unit_dir.resolve()
        resolved_output = output.resolve()
        if resolved_units == resolved_output or resolved_output in resolved_units.parents:
            raise SeedBuildError(
                f"--unit-cache {unit_dir} is inside the store directory {output}; "
                "publishing replaces that directory and would delete the cache. "
                f"Use a sibling path such as {output.parent / (output.name + '_units')}"
            )
    unit_dir.mkdir(parents=True, exist_ok=True)
    journal = WorkJournal(unit_dir / "journal.jsonl")
    plan_signature = f"{signature}:{catalog.split_manifest_hash()[:16]}"
    tasks = plan_units(
        catalog,
        root,
        unit_dir,
        signature=signature,
        mirror_policy=str(config.mirror_policy),
        prune_ends_and_fingers=bool(config.prune_ends_and_fingers),
    )
    if config.limit_clips is not None:
        keep = {int(clip.clip_id) for clip in catalog.clips[: int(config.limit_clips)]}
        tasks = [task for task in tasks if int(task.clip_id) in keep]
    if not tasks:
        raise SeedBuildError("No work units were planned for this catalogue")
    completed, unit_report = run_units(
        tasks,
        journal=journal,
        signature=plan_signature,
        store_root=unit_dir,
        workers=int(config.workers),
        max_inflight=config.max_inflight,
        max_inflight_bytes=config.max_inflight_bytes,
    )
    if unit_report["failed"]:
        raise SeedBuildError(
            f"{unit_report['failed']} units failed; rerun to retry them. First failures: "
            f"{unit_report['failures'][:3]}"
        )
    if len(completed) != len(tasks):
        raise SeedBuildError(
            f"{len(tasks) - len(completed)} units are missing after the run; rerun to complete the build"
        )
    first = next(iter(completed.values()))
    sidecar = json.loads((unit_dir / f"{first.unit_id}{SIDECAR_SUFFIX}").read_text(encoding="utf-8"))
    names = [str(value) for value in sidecar["names"]]
    parents = [int(value) for value in sidecar["parents"]]
    motion_dim = joint_feature_dim(len(names))
    if int(first.motion_dim) != motion_dim:
        raise SeedBuildError(
            f"Worker features have width {first.motion_dim} but the skeleton implies {motion_dim}"
        )
    skeleton_digest = skeleton_hash(names, parents, "prune_ends_and_fingers" if config.prune_ends_and_fingers else "full")
    mismatched = [result.unit_id for result in completed.values() if result.skeleton_hash != skeleton_digest]
    if mismatched:
        raise SeedBuildError(f"{len(mismatched)} units disagree on the skeleton hash")
    plans = plan_packed_clips(catalog, completed, unit_dir=unit_dir, mirror_policy=str(config.mirror_policy))
    if not plans:
        raise SeedBuildError("No clips could be resolved from the completed units")
    plans = order_packed_clips(plans)
    entries, pack_report = pack_clips(
        plans,
        unit_dir,
        staging,
        motion_dim=motion_dim,
        num_joints=len(names),
        names=names,
        shard_bytes=int(config.shard_bytes),
        purge_units=bool(config.purge_units),
    )
    write_clip_table(staging, entries, num_joints=len(names))
    style_ids, package_ids, category_ids, performer_ids = _label_ids(catalog)
    styles = [name for name, _index in sorted(style_ids.items(), key=lambda item: item[1])]
    packages = [name for name, _index in sorted(package_ids.items(), key=lambda item: item[1])]
    categories = [name for name, _index in sorted(category_ids.items(), key=lambda item: item[1])]
    performers = [name for name, _index in sorted(performer_ids.items(), key=lambda item: item[1])]
    shard_files = sorted(path.relative_to(staging).as_posix() for path in (staging / "motion").glob("shard_*.npy"))
    if len(shard_files) != int(pack_report["shards"]):
        raise SeedBuildError(
            f"Packed {pack_report['shards']} shards but found {len(shard_files)} on disk"
        )
    manifest: dict[str, Any] = {
        "data_schema_version": 4,
        "store_type": "feature_packed",
        "layout": "packed",
        "frame_rate": 60,
        "created_by": "stylized_motion.data.seed_build",
        "preprocess_version": PREPROCESS_VERSION,
        "motion_dim": motion_dim,
        "num_shards": len(shard_files),
        "shard_files": shard_files,
        "shard_target_bytes": int(config.shard_bytes),
        "total_frames": int(sum(entry.length for entry in entries)),
        "num_clips": len(entries),
        "clip_names": [plan.move_name for plan in plans],
        "style_names": styles,
        "action_names": categories,
        "package_names": packages,
        "performer_names": performers,
        "dataset": catalog.manifest.get("dataset", ""),
        "metadata_csv": catalog.manifest.get("metadata_csv", ""),
        # SEED's labels do not mean what the 100STYLE style/action fields meant,
        # so the mapping is recorded instead of being left to downstream guesswork.
        "label_semantics": {
            "style": "content_uniform_style",
            "action": "category",
            "package": "package",
            "performer": "take_actor (actor_uid fallback); empty when the catalogue has neither",
            "source_group": "take group: take_date + take_actor + take_org_name + canonical move name",
            "variant": "0=original, 1=official mirror, 2=generated mirror",
        },
        "temporal_label_version": catalog.manifest.get("temporal_label_version", ""),
        "mirror_policy": str(config.mirror_policy),
        "split_policy": catalog.manifest.get("split_policy", ""),
        "split_seed": int(catalog.manifest.get("split_seed", 0)),
        "split_ratios": catalog.manifest.get("split_ratios", {}),
        "split_actor_holdout": list(catalog.manifest.get("split_actor_holdout", [])),
        "split_coverage": catalog.manifest.get("split_coverage", {}),
        "split_manifest_hash": catalog.split_manifest_hash(),
        "skeleton_hash": skeleton_digest,
        "feature_schema": {
            "name": "motion_feature_v2",
            "motion_dim": motion_dim,
            "joint_subset": "prune_ends_and_fingers" if config.prune_ends_and_fingers else "full",
            "names": names,
            "parents": parents,
        },
        "unit_report": {
            "units": unit_report["units"],
            "reused": unit_report["reused"],
            "completed": len(completed),
        },
        "build": {"status": "complete", "preprocess_version": PREPROCESS_VERSION, "staging": False},
    }
    manifest["shard_num_frames"] = [
        int(np.load(staging / relative, mmap_mode="r", allow_pickle=False).shape[0])
        for relative in manifest["shard_files"]
    ]
    manifest["shard_sha256"] = [sha256_file(staging / relative) for relative in manifest["shard_files"]]
    if pack_report.get("root_shard_files"):
        root_files = list(pack_report["root_shard_files"])
        root_frames = [
            int(np.load(staging / relative, mmap_mode="r", allow_pickle=False).shape[0])
            for relative in root_files
        ]
        if root_frames != manifest["shard_num_frames"]:
            raise SeedBuildError("Root channel shards are not aligned with the feature shards")
        manifest["root_channels"] = int(ROOT_CHANNELS)
        manifest["root_shard_files"] = root_files
        manifest["root_shard_sha256"] = list(pack_report["root_shard_sha256"])
    manifest["feature_schema_hash"] = feature_schema_hash(
        names, parents, manifest["feature_schema"]["joint_subset"]
    )
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    # Statistics are computed on the staged store, then written next to it and
    # bound into the manifest through their own hash.
    from stylized_motion.data.packed_store import open_packed_feature_store

    staged_store = open_packed_feature_store(staging, load_normalization=False)
    try:
        normalization = compute_normalization(staged_store, split="train", source=f"{output.name}:train")
    finally:
        staged_store.close()
    normalization.save(staging)
    manifest["normalization_hash"] = normalization.normalization_hash()
    manifest["normalization_train_frames"] = int(normalization.train_frames)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    report = verify_packed_store(
        staging, level=str(config.verify), window_frames=int(config.window_frames)
    )
    publish_packed_store(staging, output, overwrite=overwrite)
    if config.purge_units and config.unit_cache is None:
        remove_tree(unit_dir)
    report.update(
        {
            "output": str(output),
            "normalization_hash": normalization.normalization_hash(),
            "train_frames": int(normalization.train_frames),
            "units_reused": unit_report["reused"],
            "bytes": pack_report["bytes"],
            "feature_schema_hash": manifest["feature_schema_hash"],
            "skeleton_hash": skeleton_digest,
        }
    )
    return report


__all__ = [
    "INVENTORY_FILENAME",
    "ROOT_CHANNELS",
    "ROOT_SUFFIX",
    "SIDECAR_SUFFIX",
    "PackedClipPlan",
    "SeedBuildConfig",
    "SeedBuildError",
    "UnitTask",
    "build_packed_feature_store",
    "inventory_catalog",
    "order_packed_clips",
    "pack_clips",
    "plan_packed_clips",
    "plan_units",
    "probe_bvh_header",
    "process_unit",
    "run_units",
    "verify_packed_store",
]
