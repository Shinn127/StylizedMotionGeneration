"""BONES-SEED clip catalog: discovery, frame contract, mirror policy, group splits.

This is the schema-v4 data-registration layer described in
``docs/bones_seed_data_pipeline_plan.md`` §4.1. It is deliberately independent
of feature extraction: it answers *what* clips exist, at which frame rate, how
they relate to their mirrors, and which split they belong to.

Contract highlights
-------------------
* ``seed_metadata_v004.csv`` is the authoritative first-phase directory; the
  parquet file and the temporal-label JSONL are optional extensions.
* Frame arithmetic is explicit. SOMA uniform BVHs are 120 fps while the
  pipeline contract is 60 fps, so a clip carries *both* its raw frame interval
  and its decimated target interval. The metadata ``move_duration_frames``
  column counts 120 fps rows and must never be used as a 60 fps ``stop``.
* Mirror handling is one explicit policy: ``official`` (keep every official
  file, group it with its sibling), ``generate`` (originals only, the
  preprocessor synthesises mirrors), ``none`` (originals only, no mirrors).
* Splits are assigned per take group, so an original, its official mirror and
  any derived cut of the same take always land in the same split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from stylized_motion.data.feature_data import canonical_json_bytes


SEED_DATASET = "seed"
SEED_METADATA_CSV = ("metadata", "seed_metadata_v004.csv")
SEED_METADATA_PARQUET = ("metadata", "seed_metadata_v004.parquet")
SEED_TEMPORAL_LABELS = ("metadata", "seed_metadata_v002_temporal_labels.jsonl")

SOURCE_FPS = 120
TARGET_FPS = 60
MIRROR_POLICIES = ("official", "generate", "none")

VARIANT_ORIGINAL = 0
VARIANT_OFFICIAL_MIRROR = 1
VARIANT_GENERATED_MIRROR = 2
VARIANT_NAMES = {
    VARIANT_ORIGINAL: "original",
    VARIANT_OFFICIAL_MIRROR: "official_mirror",
    VARIANT_GENERATED_MIRROR: "generated_mirror",
}
MIRROR_SUFFIX = "_M"

# Temporal labels are given in seconds. Frame intervals use the half-open rule
# ``[start, stop)`` over target-rate frames with round-half-up on both
# endpoints, so a boundary that lands exactly between two frames joins the
# later one. Any change to that rule must bump this version string, which is
# persisted in the catalog manifest and hashed into the split manifest.
TEMPORAL_LABEL_RULE = "half_open_round_half_up"
TEMPORAL_LABEL_VERSION = "temporal_labels_v002@60fps+half_open_round_half_up"

CATALOG_SCHEMA_VERSION = 1
SPLIT_POLICY = "take_group_v1"

_LABEL_FIELDS = (
    "package",
    "category",
    "content_uniform_style",
    "content_type_of_movement",
    "content_body_position",
    "content_horizontal_move",
    "content_vertical_move",
    "content_props",
    "content_complex_action",
    "content_repeated_action",
    "take_name",
    "take_org_name",
    "take_day_part",
)
_ID_FIELDS = ("package", "category", "content_uniform_style", "actor_uid", "take_date", "take_actor")

UNKNOWN_LABEL = "__unknown__"


class SeedCatalogError(ValueError):
    """Raised when SEED metadata or a catalog artifact violates the contract."""


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no", ""}:
        return False
    raise SeedCatalogError(f"Cannot parse boolean metadata value {value!r}")


def _parse_float(value: object) -> float:
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError as error:
        raise SeedCatalogError(f"Cannot parse numeric metadata value {value!r}") from error


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little", signed=False)


@dataclass(frozen=True)
class FrameContract:
    """Explicit raw/target frame arithmetic for one source rate."""

    source_fps: int = SOURCE_FPS
    target_fps: int = TARGET_FPS

    def __post_init__(self) -> None:
        if int(self.source_fps) <= 0 or int(self.target_fps) <= 0:
            raise SeedCatalogError("Frame rates must be positive")
        factor = int(self.source_fps) / int(self.target_fps)
        if abs(factor - round(factor)) > 1e-9:
            raise SeedCatalogError(
                f"Source rate {self.source_fps} fps is not an integer multiple of {self.target_fps} fps"
            )
        if int(self.source_fps) % int(self.target_fps) != 0:
            raise SeedCatalogError("Source rate must be an integer multiple of the target rate")

    @property
    def decimation(self) -> int:
        """Number of source frames per target frame."""
        return int(self.source_fps) // int(self.target_fps)

    def target_frames(self, raw_frames: int) -> int:
        """Target frame count of ``[0, raw_frames)`` decimated by ``decimation``."""
        raw_frames = int(raw_frames)
        if raw_frames < 0:
            raise SeedCatalogError("raw_frames must be non-negative")
        step = self.decimation
        return (raw_frames + step - 1) // step

    def target_interval(self, raw_start: int, raw_stop: int) -> tuple[int, int]:
        """Map a half-open raw interval onto the decimated frame grid.

        A raw frame ``f`` survives decimation when ``f % step == 0``, so the
        decimated interval is ``[ceil(raw_start/step), ceil(raw_stop/step))``.
        """
        raw_start, raw_stop = int(raw_start), int(raw_stop)
        if raw_start < 0 or raw_stop < raw_start:
            raise SeedCatalogError(f"Invalid raw interval [{raw_start}, {raw_stop})")
        step = self.decimation
        return (raw_start + step - 1) // step, (raw_stop + step - 1) // step


@dataclass(frozen=True)
class TimeRange:
    """One labelled time interval in both seconds and target frames."""

    start_seconds: float
    stop_seconds: float
    label: str
    description: str = ""
    start_frame: int = 0
    stop_frame: int = 0

    @property
    def frames(self) -> int:
        return int(self.stop_frame) - int(self.start_frame)


def seconds_to_frame(seconds: float, fps: int = TARGET_FPS) -> int:
    """Round a timestamp to a target-rate frame index (half up, ties to later)."""
    value = float(seconds) * int(fps)
    if not math.isfinite(value):
        raise SeedCatalogError(f"Cannot map non-finite timestamp {seconds!r} to a frame")
    return int(math.floor(value + 0.5))


def time_range_to_frames(
    start_seconds: float,
    stop_seconds: float,
    *,
    label: str = "",
    description: str = "",
    nframes: int,
    fps: int = TARGET_FPS,
) -> TimeRange:
    """Convert a labelled second interval into a clamped target frame interval."""
    if float(stop_seconds) <= float(start_seconds):
        raise SeedCatalogError(f"Temporal label interval is empty: [{start_seconds}, {stop_seconds})")
    start_frame = seconds_to_frame(start_seconds, fps)
    stop_frame = seconds_to_frame(stop_seconds, fps)
    start_frame = max(0, min(start_frame, int(nframes)))
    stop_frame = max(0, min(stop_frame, int(nframes)))
    return TimeRange(
        start_seconds=float(start_seconds),
        stop_seconds=float(stop_seconds),
        label=str(label),
        description=str(description),
        start_frame=start_frame,
        stop_frame=stop_frame,
    )


@dataclass
class SeedClip:
    """One catalogue entry: a single BVH file plus its metadata and split."""

    clip_id: int
    group_id: int
    variant_id: int
    is_mirror: bool
    dataset: str
    relative_path: str
    move_name: str
    canonical_name: str
    raw_frames: int
    target_frames: int
    source_fps: int = SOURCE_FPS
    target_fps: int = TARGET_FPS
    raw_start: int = 0
    raw_stop: int = 0
    start: int = 0
    stop: int = 0
    actor_uid: str = UNKNOWN_LABEL
    take_date: str = UNKNOWN_LABEL
    take_actor: str = UNKNOWN_LABEL
    take_org_name: str = UNKNOWN_LABEL
    group_key: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    split: str = ""
    skeleton_hash: str = ""
    preprocess_version: str = ""

    def __post_init__(self) -> None:
        if not self.group_key:
            self.group_key = build_group_key(self.take_date, self.take_actor, self.take_org_name, self.canonical_name)
        if self.raw_stop == 0:
            self.raw_stop = int(self.raw_frames)
        contract = FrameContract(int(self.source_fps), int(self.target_fps))
        expected = contract.target_frames(int(self.raw_stop))
        if int(self.target_frames) != expected:
            raise SeedCatalogError(
                f"Clip {self.move_name!r} declares {self.target_frames} target frames but "
                f"[0, {self.raw_stop}) at {self.source_fps}->{self.target_fps} fps yields {expected}"
            )
        if self.stop == 0:
            self.stop = int(self.target_frames)
        if self.start < 0 or self.stop <= self.start or self.stop > int(self.target_frames):
            raise SeedCatalogError(
                f"Clip {self.move_name!r} has an invalid target interval [{self.start}, {self.stop}) "
                f"for {self.target_frames} target frames"
            )

    @property
    def nframes(self) -> int:
        """Frames the pipeline will actually materialise for this clip."""
        return int(self.stop) - int(self.start)

    @property
    def variant_name(self) -> str:
        return VARIANT_NAMES.get(int(self.variant_id), f"variant_{int(self.variant_id)}")

    def as_manifest_record(self) -> dict[str, Any]:
        return {
            "clip_id": int(self.clip_id),
            "group_id": int(self.group_id),
            "variant_id": int(self.variant_id),
            "variant": self.variant_name,
            "is_mirror": bool(self.is_mirror),
            "dataset": self.dataset,
            "relative_path": self.relative_path,
            "move_name": self.move_name,
            "canonical_name": self.canonical_name,
            "group_key": self.group_key,
            "source_fps": int(self.source_fps),
            "target_fps": int(self.target_fps),
            "raw_start": int(self.raw_start),
            "raw_stop": int(self.raw_stop),
            "start": int(self.start),
            "stop": int(self.stop),
            "target_frames": int(self.target_frames),
            "actor_uid": self.actor_uid,
            "take_date": self.take_date,
            "take_actor": self.take_actor,
            "take_org_name": self.take_org_name,
            "split": self.split,
            "skeleton_hash": self.skeleton_hash,
            "preprocess_version": self.preprocess_version,
            "labels": dict(self.labels),
        }


def build_group_key(take_date: str, take_actor: str, take_org_name: str, canonical_name: str) -> str:
    """Group key from date + actor + original take identity + canonical move name.

    The plan forbids keying groups on a stripped ``_M`` suffix or on the file
    stem alone: two different takes can share a stem, and a derived cut keeps a
    stem while belonging to a different group. Combining the take identity with
    the canonical (non-mirror) move name keeps originals, their mirrors and
    same-take derivatives in one group.
    """
    return "\x1f".join(
        (
            str(take_date).strip() or UNKNOWN_LABEL,
            str(take_actor).strip() or UNKNOWN_LABEL,
            str(take_org_name).strip() or UNKNOWN_LABEL,
            str(canonical_name).strip(),
        )
    )


def clip_label_value(clip: "SeedClip", field_name: str) -> str:
    """Read one label field, falling back to the dedicated clip attributes."""
    if field_name in clip.labels:
        return str(clip.labels[field_name]) or UNKNOWN_LABEL
    attribute = {"actor_uid": "actor_uid", "take_date": "take_date", "take_actor": "take_actor"}.get(field_name)
    if attribute is not None:
        return str(getattr(clip, attribute)) or UNKNOWN_LABEL
    return UNKNOWN_LABEL


def _canonical_move_name(move_name: str, is_mirror: bool) -> str:
    name = str(move_name).strip()
    if is_mirror and name.endswith(MIRROR_SUFFIX) and len(name) > len(MIRROR_SUFFIX):
        return name[: -len(MIRROR_SUFFIX)]
    return name


@dataclass(frozen=True)
class _SeedRow:
    move_name: str
    filename: str
    raw_frames: int
    is_mirror: bool
    relative_path: str
    canonical_name: str
    labels: dict[str, str]
    actor_uid: str
    take_date: str
    take_actor: str
    take_org_name: str

    @property
    def group_key(self) -> str:
        return build_group_key(self.take_date, self.take_actor, self.take_org_name, self.canonical_name)


def _iter_csv_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _row_to_seed_row(row: Mapping[str, str], row_index: int) -> _SeedRow:
    move_name = str(row.get("move_name", "")).strip()
    if not move_name:
        raise SeedCatalogError(f"SEED metadata row {row_index} has an empty move_name")
    filename = str(row.get("filename", "")).strip() or move_name
    path_value = str(row.get("move_soma_uniform_path", "")).strip()
    if not path_value:
        raise SeedCatalogError(f"SEED metadata row {row_index + 2} has no move_soma_uniform_path")
    try:
        raw_frames = int(float(str(row.get("move_duration_frames", "")).strip()))
    except ValueError as error:
        raise SeedCatalogError(f"SEED metadata row {row_index} has an invalid move_duration_frames") from error
    if raw_frames <= 0:
        raise SeedCatalogError(f"SEED metadata row {row_index} has a non-positive move_duration_frames")
    is_mirror = _parse_bool(row.get("is_mirror", "False"))
    canonical_name = _canonical_move_name(move_name, is_mirror)
    labels = {key: str(row.get(key, "")).strip() for key in _LABEL_FIELDS}
    labels["is_neutral"] = f"{_parse_float(row.get('is_neutral', 'nan')):g}"
    return _SeedRow(
        move_name=move_name,
        filename=filename,
        raw_frames=raw_frames,
        is_mirror=is_mirror,
        relative_path=path_value,
        canonical_name=canonical_name,
        labels=labels,
        actor_uid=str(row.get("actor_uid", "")).strip() or UNKNOWN_LABEL,
        take_date=str(row.get("take_date", "")).strip() or UNKNOWN_LABEL,
        take_actor=str(row.get("take_actor", "")).strip() or UNKNOWN_LABEL,
        take_org_name=str(row.get("take_org_name", "")).strip() or UNKNOWN_LABEL,
    )


def read_parquet_rows(parquet_path: Path) -> Iterator[dict[str, str]]:
    """Read the optional parquet mirror of the CSV metadata.

    Parquet support is an extension hook: the CSV stays authoritative for the
    first phase, and this reader only exists so a future column set can be
    loaded without changing the catalog contract.
    """
    try:
        import pyarrow.parquet as parquet  # type: ignore
    except ImportError as error:  # pragma: no cover - optional dependency
        raise SeedCatalogError("Reading SEED parquet metadata requires pyarrow") from error
    table = parquet.read_table(parquet_path)
    columns = table.column_names
    for row_index, row in enumerate(table.to_pylist()):
        yield {str(name): "" if row[name] is None else str(row[name]) for name in columns}


def load_temporal_labels(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Load ``seed_metadata_v002_temporal_labels.jsonl`` keyed by filename."""
    labels: dict[str, list[tuple[float, float, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise SeedCatalogError(f"Invalid temporal label JSON on line {line_number}") from error
            filename = str(payload.get("filename", "")).strip()
            if not filename:
                continue
            events = []
            for event in payload.get("events", []) or []:
                events.append(
                    (
                        float(event["start_time"]),
                        float(event["end_time"]),
                        str(event.get("description", "")),
                    )
                )
            labels[filename] = events
    return labels


def discover_catalog(
    seed_root: str | Path,
    *,
    mirror_policy: str = "official",
    metadata_csv: str | Path | None = None,
    temporal_labels: str | Path | None = None,
    max_clips: int | None = None,
    max_groups: int | None = None,
    verify_files: bool = True,
) -> tuple[list[SeedClip], dict[str, Any]]:
    """Build the SEED clip catalogue from the authoritative CSV metadata.

    ``max_clips``/``max_groups`` cap the catalogue for the benchmark tiers.
    Both select *whole* take groups, never a partial group, so a truncated
    catalogue still satisfies the original/mirror co-location contract.

    Returns the clip list and a manifest describing the frame contract, mirror
    policy, temporal-label version and any inventory warnings.
    """
    policy = str(mirror_policy)
    if policy not in MIRROR_POLICIES:
        raise SeedCatalogError(f"Unsupported mirror policy {policy!r}; expected one of {MIRROR_POLICIES}")
    root = Path(seed_root)
    csv_path = Path(metadata_csv) if metadata_csv is not None else root.joinpath(*SEED_METADATA_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"SEED metadata CSV not found: {csv_path}")
    contract = FrameContract()

    rows: list[_SeedRow] = []
    warnings: list[str] = []
    seen_move_names: set[str] = set()
    for row_index, raw_row in enumerate(_iter_csv_rows(csv_path)):
        row = _row_to_seed_row(raw_row, row_index)
        if row.move_name in seen_move_names:
            raise SeedCatalogError(f"Duplicate move_name {row.move_name!r} in {csv_path}")
        seen_move_names.add(row.move_name)
        rows.append(row)

    temporal: dict[str, list[tuple[float, float, str]]] = {}
    labels_path = Path(temporal_labels) if temporal_labels is not None else root.joinpath(*SEED_TEMPORAL_LABELS)
    if labels_path.exists():
        temporal = load_temporal_labels(labels_path)

    rows_by_name = {row.move_name: row for row in rows}
    # Grouping: canonical (non-mirror) rows own the group; a mirror joins the
    # group of its canonical sibling when that file is present in the catalog,
    # otherwise it forms its own group so it is never silently dropped.
    groups: dict[str, list[_SeedRow]] = {}
    collision_guard: dict[tuple[str, bool], str] = {}
    for row in rows:
        owner_name = row.canonical_name
        owner = rows_by_name.get(owner_name)
        if owner is not None:
            group_key = owner.group_key
        else:
            group_key = row.group_key
            if row.is_mirror:
                warnings.append(
                    f"Mirror {row.move_name!r} has no canonical sibling in the catalog; "
                    "it forms its own group"
                )
        key = (group_key, row.is_mirror)
        previous = collision_guard.get(key)
        if previous is not None:
            raise SeedCatalogError(
                f"Group collision: {row.move_name!r} and {previous!r} share group {group_key!r} "
                f"with is_mirror={row.is_mirror}"
            )
        collision_guard[key] = row.move_name
        groups.setdefault(group_key, []).append(row)

    for group_key, members in groups.items():
        mirrors = [row for row in members if row.is_mirror]
        for mirror in mirrors:
            if mirror.canonical_name not in {row.move_name for row in members if not row.is_mirror}:
                warnings.append(
                    f"Group {group_key!r} contains mirror {mirror.move_name!r} without its canonical file"
                )

    ordered_group_keys = sorted(groups, key=lambda value: (_stable_u64(value), value))
    group_budget = int(max_groups) if max_groups is not None else None
    clip_budget = int(max_clips) if max_clips is not None else None
    if group_budget is not None and group_budget <= 0:
        raise SeedCatalogError("max_groups must be positive")
    if clip_budget is not None and clip_budget <= 0:
        raise SeedCatalogError("max_clips must be positive")

    selected: list[_SeedRow] = []
    for group_key in ordered_group_keys:
        if group_budget is not None and len({row.group_key for row in selected}) >= group_budget:
            break
        if clip_budget is not None and len(selected) >= clip_budget:
            break
        members = sorted(groups[group_key], key=lambda row: (row.is_mirror, row.move_name))
        originals = [row for row in members if not row.is_mirror]
        mirrors = [row for row in members if row.is_mirror]
        # ``official`` keeps every official file (originals and official
        # mirrors); ``generate``/``none`` keep originals only and differ in
        # whether the preprocessor synthesises the mirror later.
        selected.extend(originals)
        if policy == "official":
            selected.extend(mirrors)

    group_order: list[str] = []
    for row in selected:
        owner = rows_by_name.get(row.canonical_name)
        key = owner.group_key if owner is not None else row.group_key
        if key not in group_order:
            group_order.append(key)
    group_ids = {key: index for index, key in enumerate(group_order)}

    clips: list[SeedClip] = []
    missing_files = 0
    for row in selected:
        owner = rows_by_name.get(row.canonical_name)
        group_key = owner.group_key if owner is not None else row.group_key
        variant_id = VARIANT_ORIGINAL
        if row.is_mirror:
            variant_id = VARIANT_OFFICIAL_MIRROR if policy == "official" else VARIANT_GENERATED_MIRROR
        target_frames = contract.target_frames(row.raw_frames)
        label = temporal.get(row.filename)
        clip_labels = dict(row.labels)
        if label is not None:
            first = label[0]
            last = label[-1]
            clip_labels["temporal_label_count"] = str(len(label))
            clip_labels["temporal_first_event"] = first[2]
            clip_labels["temporal_last_event"] = last[2]
        clip = SeedClip(
            clip_id=len(clips),
            group_id=group_ids[group_key],
            variant_id=variant_id,
            is_mirror=bool(row.is_mirror),
            dataset=SEED_DATASET,
            relative_path=row.relative_path,
            move_name=row.move_name,
            canonical_name=row.canonical_name,
            raw_frames=int(row.raw_frames),
            target_frames=target_frames,
            source_fps=contract.source_fps,
            target_fps=contract.target_fps,
            raw_start=0,
            raw_stop=int(row.raw_frames),
            start=0,
            stop=target_frames,
            actor_uid=row.actor_uid,
            take_date=row.take_date,
            take_actor=row.take_actor,
            take_org_name=row.take_org_name,
            group_key=group_key,
            labels=clip_labels,
        )
        if verify_files and not (root / clip.relative_path).exists():
            missing_files += 1
        clips.append(clip)

    if not clips:
        raise SeedCatalogError(f"No SEED clips were selected under mirror policy {policy!r}")
    if missing_files:
        raise FileNotFoundError(
            f"{missing_files} of {len(clips)} catalogued SEED BVH files are missing under {root}"
        )

    manifest = {
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "dataset": SEED_DATASET,
        # Recorded so later stages can resolve clip paths without re-deriving
        # the root from wherever the metadata CSV happened to live.
        "seed_root": str(root),
        "metadata_csv": str(csv_path),
        "metadata_parquet": str(root.joinpath(*SEED_METADATA_PARQUET)),
        "temporal_labels": str(labels_path) if temporal else "",
        "temporal_label_rule": TEMPORAL_LABEL_RULE,
        "temporal_label_version": TEMPORAL_LABEL_VERSION,
        "source_fps": contract.source_fps,
        "target_fps": contract.target_fps,
        "decimation": contract.decimation,
        "mirror_policy": policy,
        "num_clips": len(clips),
        "num_groups": len(group_order),
        "num_originals": sum(1 for clip in clips if clip.variant_id == VARIANT_ORIGINAL),
        "num_official_mirrors": sum(1 for clip in clips if clip.variant_id == VARIANT_OFFICIAL_MIRROR),
        "truncated": bool(len(group_order) < len(groups)),
        "raw_frames_total": int(sum(clip.raw_frames for clip in clips)),
        "target_frames_total": int(sum(clip.nframes for clip in clips)),
        "warnings": warnings,
    }
    return clips, manifest


class SeedCatalog:
    """A persisted, mmap-friendly catalogue plus its group-split assignment."""

    def __init__(self, directory: Path, manifest: dict[str, Any], clips: list[SeedClip]) -> None:
        self.directory = Path(directory)
        self.manifest = manifest
        self.clips = clips

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    def group_ids(self) -> np.ndarray:
        return np.asarray([clip.group_id for clip in self.clips], dtype=np.int32)

    def split_ids(self) -> np.ndarray:
        mapping = {"train": 0, "val": 1, "test": 2}
        values = []
        for clip in self.clips:
            if clip.split not in mapping:
                raise SeedCatalogError(f"Clip {clip.move_name!r} has no split assignment")
            values.append(mapping[clip.split])
        return np.asarray(values, dtype=np.uint8)

    def split_manifest_hash(self) -> str:
        payload = {
            "policy": str(self.manifest.get("split_policy", "")),
            "seed": int(self.manifest.get("split_seed", 0)),
            "ratios": self.manifest.get("split_ratios", {}),
            "actor_holdout": list(self.manifest.get("split_actor_holdout", [])),
            "assignments": sorted(
                (clip.group_key, clip.split) for clip in self.clips if clip.variant_id == VARIANT_ORIGINAL
            ),
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def groups_by_split(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        for clip in self.clips:
            if clip.variant_id != VARIANT_ORIGINAL:
                continue
            result[clip.split].append(int(clip.group_id))
        return result

    def save(self, directory: str | Path | None = None, *, overwrite: bool = False) -> Path:
        expected_ids = list(range(len(self.clips)))
        actual_ids = [int(clip.clip_id) for clip in self.clips]
        if actual_ids != expected_ids:
            raise SeedCatalogError(
                "Catalog clip_id must be a dense index into the clip list; "
                "call renumber_clips() after subsetting"
            )
        target = Path(directory) if directory is not None else self.directory
        if target.exists() and not overwrite and any(target.iterdir()):
            raise FileExistsError(f"Catalog directory already exists: {target}")
        target.mkdir(parents=True, exist_ok=True)
        index_dir = target / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        numeric = {
            "clip_id": np.asarray([clip.clip_id for clip in self.clips], dtype=np.int32),
            "group_id": np.asarray([clip.group_id for clip in self.clips], dtype=np.int32),
            "variant_id": np.asarray([clip.variant_id for clip in self.clips], dtype=np.uint8),
            "is_mirror": np.asarray([clip.is_mirror for clip in self.clips], dtype=bool),
            "raw_frames": np.asarray([clip.raw_frames for clip in self.clips], dtype=np.int64),
            "raw_start": np.asarray([clip.raw_start for clip in self.clips], dtype=np.int64),
            "raw_stop": np.asarray([clip.raw_stop for clip in self.clips], dtype=np.int64),
            "target_frames": np.asarray([clip.target_frames for clip in self.clips], dtype=np.int64),
            "start": np.asarray([clip.start for clip in self.clips], dtype=np.int64),
            "stop": np.asarray([clip.stop for clip in self.clips], dtype=np.int64),
            "split_id": self.split_ids(),
        }
        for key, values in numeric.items():
            np.save(index_dir / f"{key}.npy", values)
        label_names: list[str] = []
        for field_name in _ID_FIELDS:
            values: dict[str, int] = {}
            for clip in self.clips:
                raw = clip_label_value(clip, field_name)
                values.setdefault(raw, len(values))
            ordered = [name for name, _index in sorted(values.items(), key=lambda item: item[1])]
            np.save(
                index_dir / f"label_{field_name}.npy",
                np.asarray([values[clip_label_value(clip, field_name)] for clip in self.clips], dtype=np.int32),
            )
            (target / f"labels_{field_name}.json").write_text(
                json.dumps(ordered, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
            )
            label_names.append(field_name)
        with (target / "clips.jsonl").open("w", encoding="utf-8") as handle:
            for clip in self.clips:
                handle.write(json.dumps(clip.as_manifest_record(), ensure_ascii=True, sort_keys=True) + "\n")
        payload = dict(self.manifest)
        payload["split_manifest_hash"] = self.split_manifest_hash()
        payload["label_fields"] = label_names
        (target / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "SeedCatalog":
        target = Path(directory)
        manifest_path = target / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing catalog manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("catalog_schema_version", 0)) != CATALOG_SCHEMA_VERSION:
            raise SeedCatalogError(
                f"Unsupported catalog schema version {manifest.get('catalog_schema_version')!r}"
            )
        index_dir = target / "index"
        names = {
            "clip_id": np.int32,
            "group_id": np.int32,
            "variant_id": np.uint8,
            "is_mirror": bool,
            "raw_frames": np.int64,
            "raw_start": np.int64,
            "raw_stop": np.int64,
            "target_frames": np.int64,
            "start": np.int64,
            "stop": np.int64,
            "split_id": np.uint8,
        }
        arrays = {key: np.load(index_dir / f"{key}.npy") for key in names}
        clips: list[SeedClip] = []
        with (target / "clips.jsonl").open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                index = int(record["clip_id"])
                if index != len(clips):
                    raise SeedCatalogError(
                        f"Catalog clip records are out of order: expected clip_id {len(clips)}, got {index}"
                    )
                clips.append(
                    SeedClip(
                        clip_id=index,
                        group_id=int(arrays["group_id"][index]),
                        variant_id=int(arrays["variant_id"][index]),
                        is_mirror=bool(arrays["is_mirror"][index]),
                        dataset=str(record["dataset"]),
                        relative_path=str(record["relative_path"]),
                        move_name=str(record["move_name"]),
                        canonical_name=str(record["canonical_name"]),
                        raw_frames=int(arrays["raw_frames"][index]),
                        target_frames=int(arrays["target_frames"][index]),
                        source_fps=int(record["source_fps"]),
                        target_fps=int(record["target_fps"]),
                        raw_start=int(arrays["raw_start"][index]),
                        raw_stop=int(arrays["raw_stop"][index]),
                        start=int(arrays["start"][index]),
                        stop=int(arrays["stop"][index]),
                        actor_uid=str(record.get("actor_uid", UNKNOWN_LABEL)),
                        take_date=str(record.get("take_date", UNKNOWN_LABEL)),
                        take_actor=str(record.get("take_actor", UNKNOWN_LABEL)),
                        take_org_name=str(record.get("take_org_name", UNKNOWN_LABEL)),
                        group_key=str(record.get("group_key", "")),
                        labels=dict(record.get("labels", {})),
                        split=str(record.get("split", "")),
                        skeleton_hash=str(record.get("skeleton_hash", "")),
                        preprocess_version=str(record.get("preprocess_version", "")),
                    )
                )
        if len(clips) != int(manifest.get("num_clips", len(clips))):
            raise SeedCatalogError("Catalog clip count does not match its manifest")
        return cls(target, manifest, clips)


def assign_group_splits(
    clips: list[SeedClip],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 3407,
    stratify_key: str = "package",
    actor_holdout: Sequence[str] = (),
    actor_holdout_ratio: float = 0.0,
) -> dict[str, Any]:
    """Assign whole take groups to train/val/test.

    Ratios are applied to groups — never to individual clips — so an original,
    its official mirror and any same-take derivative stay together. Groups are
    walked in a deterministic hashed order and stratified by ``stratify_key``
    (package by default) so rare packages are spread across splits instead of
    being swept into whichever split lands last.
    """
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if min(ratios) < 0.0 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SeedCatalogError("split ratios must be non-negative and sum to one")
    holdout = {str(value) for value in actor_holdout}
    if actor_holdout_ratio < 0.0 or actor_holdout_ratio > 1.0:
        raise SeedCatalogError("actor_holdout_ratio must be in [0, 1]")

    groups: dict[int, list[SeedClip]] = {}
    for clip in clips:
        groups.setdefault(int(clip.group_id), []).append(clip)
    if not groups:
        raise SeedCatalogError("Cannot split an empty catalog")

    def group_label(members: Sequence[SeedClip]) -> str:
        for member in members:
            if member.variant_id == VARIANT_ORIGINAL:
                value = member.labels.get(stratify_key, "") or UNKNOWN_LABEL
                return str(value)
        return str(members[0].labels.get(stratify_key, "") or UNKNOWN_LABEL)

    ordered_groups = sorted(groups)
    strata: dict[str, list[int]] = {}
    for group_id in ordered_groups:
        strata.setdefault(group_label(groups[group_id]), []).append(group_id)

    assignments: dict[int, str] = {}
    held_out_actors: set[str] = set()
    candidates = [
        int(group_id)
        for group_id in ordered_groups
        if all(
            clip.actor_uid in holdout or clip.actor_uid == UNKNOWN_LABEL
            for clip in groups[group_id]
        )
    ]
    if actor_holdout_ratio > 0.0 and not holdout:
        actors = sorted({clip.actor_uid for clip in clips if clip.actor_uid != UNKNOWN_LABEL})
        if actors:
            target = max(1, int(round(len(actors) * float(actor_holdout_ratio))))
            ordered_actors = sorted(actors, key=lambda name: (_stable_u64(f"{seed}:actor:{name}"), name))
            held_out_actors = set(ordered_actors[:target])
            candidates = [
                int(group_id)
                for group_id in ordered_groups
                if all(
                    clip.actor_uid in held_out_actors or clip.actor_uid == UNKNOWN_LABEL
                    for clip in groups[group_id]
                )
            ]
    else:
        held_out_actors = set(holdout)
        candidates = [
            int(group_id)
            for group_id in ordered_groups
            if all(clip.actor_uid in held_out_actors for clip in groups[group_id])
        ]

    frozen: list[int] = []
    if held_out_actors:
        for group_id in candidates:
            assignments[group_id] = "test"
            frozen.append(group_id)

    remaining = [group_id for group_id in ordered_groups if group_id not in assignments]
    ordered: list[int] = []
    for stratum in sorted(strata):
        members = sorted(
            (group_id for group_id in strata[stratum] if group_id not in assignments),
            key=lambda group_id: (_stable_u64(f"{seed}:{stratum}:{group_id}"), group_id),
        )
        ordered.extend(members)
    if not remaining and not ordered:
        raise SeedCatalogError("Actor hold-out consumed every group; reduce the hold-out ratio")

    remaining_count = len(ordered)
    if frozen:
        # When a frozen test set exists the ratios apply to the remaining
        # groups, so the frozen actors are not counted twice.
        train_count = int(np.floor(remaining_count * ratios[0] / max(1e-9, ratios[0] + ratios[1])))
        train_count = min(train_count, remaining_count)
        val_count = remaining_count - train_count
        counts = [train_count, val_count, 0]
    else:
        counts = [int(np.floor(remaining_count * ratio)) for ratio in ratios]
        counts[2] = remaining_count - counts[0] - counts[1]
        if remaining_count >= 3:
            for split_index in (1, 2):
                if counts[split_index] == 0:
                    donor = max(range(3), key=lambda index: counts[index])
                    if counts[donor] > 1:
                        counts[donor] -= 1
                        counts[split_index] += 1
    if sum(counts) != remaining_count or min(counts) < 0:
        raise SeedCatalogError(f"Invalid split counts {counts} for {remaining_count} groups")

    offset = 0
    for split_name, count in zip(("train", "val", "test"), counts):
        for group_id in ordered[offset : offset + count]:
            assignments[group_id] = split_name
        offset += count
    if len(assignments) != len(groups):
        raise SeedCatalogError("Split assignment did not cover every group")

    for clip in clips:
        clip.split = assignments[int(clip.group_id)]

    def coverage(split_name: str) -> dict[str, Any]:
        members = [clip for clip in clips if clip.split == split_name]
        return {
            "clips": len(members),
            "groups": len({clip.group_id for clip in members}),
            "frames": int(sum(clip.nframes for clip in members)),
            "actors": len({clip.actor_uid for clip in members}),
        }

    return {
        "split_policy": SPLIT_POLICY,
        "split_seed": int(seed),
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "split_stratify_key": str(stratify_key),
        "split_actor_holdout": sorted(held_out_actors),
        "split_actor_holdout_ratio": float(actor_holdout_ratio),
        "split_coverage": {name: coverage(name) for name in ("train", "val", "test")},
        "split_counts": counts,
    }


def refresh_catalog_counts(manifest: dict[str, Any], clips: Sequence[SeedClip]) -> dict[str, Any]:
    """Recompute the manifest's catalogue-wide counts from the current clips."""
    manifest["num_clips"] = len(clips)
    manifest["num_groups"] = len({clip.group_id for clip in clips})
    manifest["num_originals"] = sum(1 for clip in clips if int(clip.variant_id) == VARIANT_ORIGINAL)
    manifest["num_official_mirrors"] = sum(
        1 for clip in clips if int(clip.variant_id) == VARIANT_OFFICIAL_MIRROR
    )
    manifest["raw_frames_total"] = int(sum(clip.raw_frames for clip in clips))
    manifest["target_frames_total"] = int(sum(clip.nframes for clip in clips))
    return manifest


def renumber_clips(clips: Sequence[SeedClip]) -> None:
    """Make ``clip_id`` and ``group_id`` dense indices into the given list.

    Catalog arrays are indexed positionally, so any subset (a benchmark tier, a
    filtered policy) must renumber both ids; otherwise a saved catalog points at
    rows that no longer exist.
    """
    group_ids: dict[str, int] = {}
    for position, clip in enumerate(clips):
        clip.clip_id = position
        group_key = clip.group_key
        if group_key not in group_ids:
            group_ids[group_key] = len(group_ids)
        clip.group_id = group_ids[group_key]


def select_representative_groups(
    clips: Sequence[SeedClip],
    target_groups: int,
    *,
    seed: int = 3407,
    stratify_key: str = "package",
    quantiles: int = 5,
) -> list[str]:
    """Pick whole take groups that cover the catalogue's shape.

    The plan's benchmark tiers must cover the long tail of clip lengths, the
    mirror frame deltas and the main packages; taking the first N groups in hash
    order covers none of that. Groups are therefore stratified by package and,
    inside each package, sampled from length quantiles, alternating between
    takes that have an official mirror and takes that do not.

    Returns the selected group keys (as built by :func:`build_group_key`).
    """
    if int(target_groups) <= 0:
        raise SeedCatalogError("target_groups must be positive")
    by_group: dict[str, list[SeedClip]] = {}
    for clip in clips:
        by_group.setdefault(clip.group_key, []).append(clip)
    if len(by_group) <= int(target_groups):
        return sorted(by_group)

    def group_length(members: Sequence[SeedClip]) -> int:
        return max(int(clip.nframes) for clip in members)

    def group_label(members: Sequence[SeedClip]) -> str:
        for member in members:
            if member.variant_id == VARIANT_ORIGINAL:
                return str(member.labels.get(stratify_key, "") or UNKNOWN_LABEL)
        return str(members[0].labels.get(stratify_key, "") or UNKNOWN_LABEL)

    def has_mirror(members: Sequence[SeedClip]) -> bool:
        return any(int(clip.variant_id) == VARIANT_OFFICIAL_MIRROR for clip in members)

    buckets: dict[str, dict[tuple[int, bool], list[str]]] = {}
    buckets_by_label: dict[str, list[str]] = {}
    for group_key in sorted(by_group):
        members = by_group[group_key]
        label = group_label(members)
        buckets_by_label.setdefault(label, []).append(group_key)
    for label, group_keys in buckets_by_label.items():
        lengths = np.asarray([group_length(by_group[key]) for key in group_keys], dtype=np.int64)
        edges = np.quantile(lengths, np.linspace(0.0, 1.0, int(quantiles) + 1)[1:-1]) if len(lengths) > 1 else []
        for group_key in group_keys:
            length = group_length(by_group[group_key])
            bucket_index = int(np.searchsorted(edges, length, side="right"))
            buckets.setdefault(label, {}).setdefault(
                (bucket_index, has_mirror(by_group[group_key])), []
            ).append(group_key)

    rng = np.random.default_rng(int(seed))
    for label in buckets:
        for key in buckets[label]:
            buckets[label][key] = sorted(
                buckets[label][key], key=lambda group_key: (_stable_u64(f"{seed}:{label}:{group_key}"), group_key)
            )
    selected: list[str] = []
    # Round-robin over (length-bucket, mirror-coverage, package) cells, ordered
    # so that one round visits every package before deepening into the next
    # length bucket. Ordering by package first would spend a small tier entirely
    # inside the alphabetically first package.
    cells = sorted(
        ((label, cell) for label, cell_buckets in buckets.items() for cell in cell_buckets),
        key=lambda item: (item[1][0], item[1][1], item[0]),
    )
    cursors = {cell: 0 for cell in cells}
    while len(selected) < int(target_groups):
        progressed = False
        for cell in cells:
            if len(selected) >= int(target_groups):
                break
            label, bucket_key = cell
            values = buckets[label][bucket_key]
            cursor = cursors[cell]
            if cursor >= len(values):
                continue
            selected.append(values[cursor])
            cursors[cell] = cursor + 1
            progressed = True
        if not progressed:
            break
    del rng
    while len(selected) < int(target_groups):
        remaining = [key for key in sorted(by_group) if key not in set(selected)]
        if not remaining:
            break
        selected.append(remaining[len(selected) % len(remaining)])
    return selected


def catalog_tier_report(clips: Sequence[SeedClip]) -> dict[str, Any]:
    """Shape of a (sub)catalogue: packages, length quantiles, mirror coverage."""
    if not clips:
        return {"clips": 0, "groups": 0}
    by_group: dict[str, list[SeedClip]] = {}
    for clip in clips:
        by_group.setdefault(clip.group_key, []).append(clip)
    lengths = np.asarray([max(int(clip.nframes) for clip in members) for members in by_group.values()], dtype=np.int64)
    packages: dict[str, int] = {}
    for members in by_group.values():
        label = str(members[0].labels.get("package", "") or UNKNOWN_LABEL)
        packages[label] = packages.get(label, 0) + 1
    with_mirror = sum(
        1 for members in by_group.values() if any(int(clip.variant_id) == VARIANT_OFFICIAL_MIRROR for clip in members)
    )
    return {
        "clips": len(clips),
        "groups": len(by_group),
        "frames": int(sum(clip.nframes for clip in clips)),
        "length_min": int(lengths.min()),
        "length_p50": int(np.quantile(lengths, 0.5)),
        "length_p95": int(np.quantile(lengths, 0.95)),
        "length_max": int(lengths.max()),
        "packages": packages,
        "groups_with_official_mirror": int(with_mirror),
        "groups_without_official_mirror": int(len(by_group) - with_mirror),
    }


def catalog_frame_audit(clips: Iterable[SeedClip]) -> dict[str, Any]:
    """Summarise the raw/target frame contract across a catalog."""
    clips = list(clips)
    if not clips:
        raise SeedCatalogError("Cannot audit an empty catalog")
    by_variant: dict[str, dict[str, int]] = {}
    for clip in clips:
        entry = by_variant.setdefault(clip.variant_name, {"clips": 0, "raw_frames": 0, "target_frames": 0})
        entry["clips"] += 1
        entry["raw_frames"] += int(clip.raw_frames)
        entry["target_frames"] += int(clip.nframes)
    odd_raw = sum(1 for clip in clips if clip.raw_frames % int(clip.source_fps // clip.target_fps) != 0)
    return {
        "clips": len(clips),
        "groups": len({clip.group_id for clip in clips}),
        "raw_frames": int(sum(clip.raw_frames for clip in clips)),
        "target_frames": int(sum(clip.nframes for clip in clips)),
        "target_hours": float(sum(clip.nframes for clip in clips)) / float(TARGET_FPS) / 3600.0,
        "raw_odd_frames": odd_raw,
        "by_variant": by_variant,
    }


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "FrameContract",
    "MIRROR_POLICIES",
    "SEED_DATASET",
    "SPLIT_POLICY",
    "SOURCE_FPS",
    "TARGET_FPS",
    "TEMPORAL_LABEL_RULE",
    "TEMPORAL_LABEL_VERSION",
    "TimeRange",
    "UNKNOWN_LABEL",
    "VARIANT_GENERATED_MIRROR",
    "VARIANT_NAMES",
    "VARIANT_OFFICIAL_MIRROR",
    "VARIANT_ORIGINAL",
    "SeedCatalog",
    "SeedCatalogError",
    "SeedClip",
    "assign_group_splits",
    "build_group_key",
    "catalog_frame_audit",
    "catalog_tier_report",
    "clip_label_value",
    "select_representative_groups",
    "discover_catalog",
    "load_temporal_labels",
    "read_parquet_rows",
    "refresh_catalog_counts",
    "renumber_clips",
    "seconds_to_frame",
    "time_range_to_frames",
]
