"""Resumable, bounded preprocessing infrastructure.

Plan §4.3: a one-shot job over ~142k BVH files cannot afford to start over
because one file failed halfway. Every unit of work therefore carries a stable
id, an input fingerprint and a preprocessing signature; success is committed
atomically to a journal, failures land in a report and stay retryable, and a
rerun reuses every unit whose fingerprint and signature still match.

The scheduler also bounds the pipeline: at most ``max_inflight`` tasks and at
most ``max_inflight_bytes`` of estimated output may be outstanding at once, and
results are consumed in completion order so one long clip no longer stalls the
whole run behind a FIFO queue.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

PREPROCESS_VERSION = "seed_preprocess_v4"
HEADER_PROBE_BYTES = 64 * 1024


def _sha256_file_prefix(path: Path, limit: int = HEADER_PROBE_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = int(limit)
        while remaining > 0:
            chunk = handle.read(min(65536, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileFingerprint:
    """Cheap but sensitive identity of one source file.

    Size, mtime and a digest of the file head are enough to notice a replaced
    or edited BVH without hashing hundreds of gigabytes of motion data. The
    header is where the skeleton, frame count and frame time live, which is
    exactly what preprocessing depends on.
    """

    size: int
    mtime_ns: int
    header_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {"size": int(self.size), "mtime_ns": int(self.mtime_ns), "header_sha256": self.header_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FileFingerprint":
        return cls(
            size=int(payload["size"]),
            mtime_ns=int(payload["mtime_ns"]),
            header_sha256=str(payload["header_sha256"]),
        )

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def fingerprint_file(path: str | Path) -> FileFingerprint:
    target = Path(path)
    stat = target.stat()
    return FileFingerprint(
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        header_sha256=_sha256_file_prefix(target),
    )


def preprocess_signature(
    *,
    prune_ends_and_fingers: bool,
    mirror_policy: str,
    target_fps: int,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Hash the preprocessing configuration that invalidates cached units.

    A change of skeleton pruning, mirror policy or target frame rate makes every
    previously written unit stale; ``extra`` carries task-specific knobs (packed
    shard size, feature version) so they participate in the same decision.
    """
    payload = {
        "preprocess_version": PREPROCESS_VERSION,
        "prune_ends_and_fingers": bool(prune_ends_and_fingers),
        "mirror_policy": str(mirror_policy),
        "target_fps": int(target_fps),
        "extra": dict(sorted((extra or {}).items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class WorkUnit:
    """One parseable unit of preprocessing work."""

    unit_id: str
    clip_id: int
    variant_id: int
    source: Path
    fingerprint: FileFingerprint
    target_frames: int
    start: int = 0
    stop: int = 0
    output: Path | None = None

    @property
    def nframes(self) -> int:
        return int(self.stop) - int(self.start)

    def estimated_bytes(self, motion_dim: int, bytes_per_value: int = 4) -> int:
        return int(self.nframes) * int(motion_dim) * int(bytes_per_value)


def build_work_unit(
    *,
    signature: str,
    clip_id: int,
    variant_id: int,
    source: str | Path,
    fingerprint: FileFingerprint,
    target_frames: int,
    start: int,
    stop: int,
    output: str | Path | None = None,
) -> WorkUnit:
    unit_id = hashlib.sha256(
        f"{signature}:{int(clip_id)}:{int(variant_id)}:{fingerprint.digest()}".encode("utf-8")
    ).hexdigest()[:32]
    return WorkUnit(
        unit_id=unit_id,
        clip_id=int(clip_id),
        variant_id=int(variant_id),
        source=Path(source),
        fingerprint=fingerprint,
        target_frames=int(target_frames),
        start=int(start),
        stop=int(stop),
        output=Path(output) if output is not None else None,
    )


@dataclass
class UnitResult:
    """Small descriptor a worker returns; large arrays stay on disk."""

    unit_id: str
    clip_id: int
    variant_id: int
    output: str
    sha256: str
    frames: int
    motion_dim: int
    skeleton_hash: str
    preprocess_version: str = PREPROCESS_VERSION
    bytes_written: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "clip_id": int(self.clip_id),
            "variant_id": int(self.variant_id),
            "output": self.output,
            "sha256": self.sha256,
            "frames": int(self.frames),
            "motion_dim": int(self.motion_dim),
            "skeleton_hash": self.skeleton_hash,
            "preprocess_version": self.preprocess_version,
            "bytes_written": int(self.bytes_written),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitResult":
        return cls(
            unit_id=str(payload["unit_id"]),
            clip_id=int(payload["clip_id"]),
            variant_id=int(payload["variant_id"]),
            output=str(payload["output"]),
            sha256=str(payload["sha256"]),
            frames=int(payload["frames"]),
            motion_dim=int(payload["motion_dim"]),
            skeleton_hash=str(payload["skeleton_hash"]),
            preprocess_version=str(payload.get("preprocess_version", PREPROCESS_VERSION)),
            bytes_written=int(payload.get("bytes_written", 0)),
            extra=dict(payload.get("extra", {})),
        )


class WorkJournal:
    """Append-only JSONL journal of committed and failed work units.

    Appends are flushed and fsynced before returning, so a crash mid-run can
    never leave a half-written record that a resume would trust. Every commit
    carries the output checksum, which the resume path re-verifies before it
    skips a unit.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._committed: dict[str, UnitResult] = {}
        self._failures: dict[str, dict[str, Any]] = {}
        self._signature: str | None = None
        self._records_read = 0
        self.load()

    def load(self) -> None:
        self._committed.clear()
        self._failures.clear()
        self._signature = None
        self._records_read = 0
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    # A torn tail can only come from a hard crash between write
                    # and fsync; everything before it is still authoritative.
                    if line_number == self._last_line_number():
                        break
                    raise
                self._records_read += 1
                kind = str(record.get("status", "committed"))
                if self._signature is None and record.get("signature") is not None:
                    self._signature = str(record["signature"])
                if kind == "committed":
                    result = UnitResult.from_dict(record)
                    self._committed[result.unit_id] = result
                else:
                    self._failures[str(record.get("unit_id", ""))] = dict(record)

    def _last_line_number(self) -> int:
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    @property
    def signature(self) -> str | None:
        return self._signature

    @property
    def committed(self) -> Mapping[str, UnitResult]:
        return dict(self._committed)

    @property
    def failures(self) -> Mapping[str, Mapping[str, Any]]:
        return {key: dict(value) for key, value in self._failures.items()}

    def _append(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("preprocess_version", PREPROCESS_VERSION)
        line = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def commit(self, result: UnitResult, *, signature: str) -> None:
        record = result.as_dict()
        record["status"] = "committed"
        record["signature"] = str(signature)
        self._append(record)
        self._committed[result.unit_id] = result
        self._signature = str(signature)
        self._records_read += 1

    def record_failure(self, unit_id: str, *, signature: str, error: str, source: str = "") -> None:
        record = {
            "unit_id": str(unit_id),
            "status": "failed",
            "signature": str(signature),
            "error": str(error)[:2000],
            "source": str(source),
        }
        self._append(record)
        self._failures[str(unit_id)] = record
        self._signature = str(signature)
        self._records_read += 1

    def commit_staging(self, staging: Path) -> None:
        """Atomically append a staging journal produced by another process."""
        if not staging.exists():
            return
        with staging.open("r", encoding="utf-8") as handle:
            payload = handle.read()
        if payload:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        self.load()

    def report(self) -> dict[str, Any]:
        return {
            "journal": str(self.path),
            "records": int(self._records_read),
            "committed": len(self._committed),
            "failed": len(self._failures),
            "signature": self._signature,
        }


def verify_completed_units(
    journal: WorkJournal,
    units: Sequence[WorkUnit],
    *,
    signature: str,
    store_root: Path,
    verify_checksums: bool = True,
) -> tuple[list[WorkUnit], dict[str, UnitResult]]:
    """Split units into pending work and reusable results.

    A unit is reused only when its signature, its input fingerprint and the
    on-disk output checksum all still match. Anything else — a rebuilt dataset,
    a changed preprocessing configuration, a truncated shard — is treated as
    pending so the run rebuilds it instead of silently publishing stale bytes.
    """
    completed: dict[str, UnitResult] = {}
    pending: list[WorkUnit] = []
    if journal.signature is not None and journal.signature != str(signature):
        return list(units), {}
    recorded = journal.committed
    for unit in units:
        result = recorded.get(unit.unit_id)
        if result is None or result.preprocess_version != PREPROCESS_VERSION:
            pending.append(unit)
            continue
        path = store_root / result.output
        if not path.exists():
            pending.append(unit)
            continue
        if verify_checksums and _sha256_file(path) != result.sha256:
            pending.append(unit)
            continue
        completed[unit.unit_id] = result
    return pending, completed


def _sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return _sha256_file(Path(path))


def bounded_map(
    executor: Executor,
    tasks: Iterable[Any],
    call: Callable[[Any], Any],
    *,
    max_inflight: int,
    max_inflight_bytes: int = 0,
    task_bytes: Callable[[Any], int] | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Yield ``(task, result)`` in completion order under bounded in-flight work.

    ``max_inflight_bytes`` bounds the estimated output of the outstanding
    tasks, so a pool of workers on long clips cannot balloon temporary storage.
    """
    if int(max_inflight) <= 0:
        raise ValueError("max_inflight must be positive")
    if int(max_inflight_bytes) < 0:
        raise ValueError("max_inflight_bytes must be non-negative")
    size_of = task_bytes or (lambda task: 0)
    limit_bytes = int(max_inflight_bytes)
    pending: dict[Future, Any] = {}
    inflight_bytes = 0
    iterator = iter(tasks)
    held: Any = None
    exhausted = False

    def admits(estimate: int, has_pending: bool) -> bool:
        if len(pending) >= int(max_inflight):
            return False
        if limit_bytes and has_pending and estimate + inflight_bytes > limit_bytes:
            return False
        return True

    def fill() -> None:
        nonlocal held, inflight_bytes, exhausted
        while True:
            if held is not None:
                estimate = int(size_of(held))
                if not admits(estimate, bool(pending)):
                    return
                pending[executor.submit(call, held)] = held
                inflight_bytes += estimate
                held = None
                continue
            if exhausted:
                return
            task = next(iterator, None)
            if task is None:
                exhausted = True
                return
            estimate = int(size_of(task))
            if not admits(estimate, bool(pending)):
                # Hold this task back until earlier work drains. With nothing
                # in flight the budget cannot be honoured, so admit it anyway
                # rather than deadlocking on a single over-budget task.
                held = task
                if pending:
                    return
                pending[executor.submit(call, task)] = task
                inflight_bytes += estimate
                held = None
                continue
            pending[executor.submit(call, task)] = task
            inflight_bytes += estimate

    fill()
    while pending:
        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        for future in done:
            task = pending.pop(future)
            inflight_bytes -= int(size_of(task))
            yield task, future.result()
        fill()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temp file and an atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def limit_thread_environment(threads: int = 1) -> dict[str, str]:
    """Environment overrides that keep worker processes single-threaded.

    ``workers x BLAS threads`` oversubscribes a 30 GiB host quickly; the
    preprocessing workers are I/O- and numpy-bound, not BLAS-bound.
    """
    value = str(max(1, int(threads)))
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
    }


def apply_thread_limits(threads: int = 1) -> None:
    for key, value in limit_thread_environment(threads).items():
        os.environ.setdefault(key, value)


def remove_tree(path: str | Path) -> None:
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif target.exists():
        target.unlink()


__all__ = [
    "PREPROCESS_VERSION",
    "FileFingerprint",
    "UnitResult",
    "WorkJournal",
    "WorkUnit",
    "apply_thread_limits",
    "atomic_write_bytes",
    "atomic_write_json",
    "bounded_map",
    "build_work_unit",
    "fingerprint_file",
    "limit_thread_environment",
    "preprocess_signature",
    "remove_tree",
    "sha256_file",
    "verify_completed_units",
]
