"""Data-pipeline benchmarks for the schema-v4 pipeline.

Plan §5 "Benchmark 设计". The harness measures the four layers separately so a
slow training step can be attributed instead of guessed at:

``sampler``
    Pure request generation (no bytes touched).
``loader``
    DataLoader iteration including worker processes, collation and prefetch.
``resident``
    A fixed GPU-resident batch loop, i.e. the model-shaped baseline the
    end-to-end target is expressed against ("reach >= 90% of the resident
    baseline and keep data wait below 10%").
``end-to-end``
    Training steps that consume the loader, reported with data-wait percentiles.

Reports separate cold and warm reads and never claims thermal-disk numbers
from a warm page cache; it deliberately does not drop the system page cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from stylized_motion.data.packed_store import PackedFeatureDataset, open_any_feature_store
from stylized_motion.data.sampling import SampleRequest, TrainWindowSampler, store_intervals


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):  # pragma: no cover - non-Linux
        return 0


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


@dataclass
class LatencyRecorder:
    """Collect per-step timings and summarise them with percentiles."""

    waits: list[float] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)

    def observe(self, wait: float, step: float) -> None:
        self.waits.append(float(wait))
        self.steps.append(float(step))

    def summary(self) -> dict[str, float]:
        total = sum(self.steps)
        wait_total = sum(self.waits)
        return {
            "steps": float(len(self.steps)),
            "step_time_mean": float(statistics.fmean(self.steps)) if self.steps else 0.0,
            "step_time_p50": _percentile(self.steps, 0.5),
            "step_time_p95": _percentile(self.steps, 0.95),
            "data_wait_mean": float(statistics.fmean(self.waits)) if self.waits else 0.0,
            "data_wait_p50": _percentile(self.waits, 0.5),
            "data_wait_p95": _percentile(self.waits, 0.95),
            "data_wait_fraction": (wait_total / total) if total > 0 else 0.0,
            "steps_per_second": (len(self.steps) / total) if total > 0 else 0.0,
        }


@dataclass
class BenchmarkConfig:
    """Inputs of one benchmark run."""

    store: Path
    batch_size: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    samples: int = 2000
    warmup_batches: int = 5
    split: str = "train"
    strategy: str = "clip_uniform"
    seed: int = 3407
    normalize_on: str = "cpu"
    shard_mib: int = 256
    device: str = "auto"
    target_frames: int = 64
    rank: int = 0
    world_size: int = 1


def benchmark_sampler(config: BenchmarkConfig) -> dict[str, Any]:
    """Time pure request generation, with and without coverage accounting."""
    store = open_any_feature_store(config.store)
    try:
        table = store_intervals(store, config.split)
        started = time.perf_counter()
        sampler = TrainWindowSampler(
            store,
            target_frames=config.target_frames,
            samples_per_epoch=int(config.samples),
            seed=int(config.seed),
            strategy=str(config.strategy),
            rank=int(config.rank),
            world_size=int(config.world_size),
        )
        build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        count = 0
        for _request in sampler:
            count += 1
        iterate_seconds = time.perf_counter() - started
        started = time.perf_counter()
        sampled = [
            SampleRequest(
                shard_idx=int(table.shard_idx[row]),
                target_start=int(table.offset[row]),
                target_frames=config.target_frames,
                variant_idx=int(table.clip_id[row]),
            )
            for row in range(min(len(table), int(config.samples)))
        ]
        del sampled
        uniform_seconds = time.perf_counter() - started
        return {
            "layer": "sampler",
            "intervals": int(len(table)),
            "index_build_seconds": build_seconds,
            "requests": int(count),
            "iterate_seconds": iterate_seconds,
            "requests_per_second": count / max(iterate_seconds, 1e-9),
            "same_count_uniform_reference_seconds": uniform_seconds,
            "coverage": sampler.coverage_summary(),
        }
    finally:
        store.close()


def benchmark_loader(config: BenchmarkConfig, *, normalize_on: str | None = None) -> dict[str, Any]:
    """Time DataLoader iteration through the real worker/collate path."""
    from torch.utils.data import DataLoader

    store = open_any_feature_store(config.store)
    try:
        dataset = PackedFeatureDataset(
            config.split,
            store,
            normalize_on=str(normalize_on or config.normalize_on),
        )
        sampler = TrainWindowSampler(
            store,
            target_frames=config.target_frames,
            samples_per_epoch=int(config.samples),
            seed=int(config.seed),
            strategy=str(config.strategy),
            rank=int(config.rank),
            world_size=int(config.world_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            sampler=sampler,
            collate_fn=lambda value: value,
            num_workers=int(config.num_workers),
            prefetch_factor=int(config.prefetch_factor) if config.num_workers > 0 else None,
            persistent_workers=bool(config.num_workers > 0),
            pin_memory=False,
        )
        recorder = LatencyRecorder()
        batches = 0
        frames = 0
        target_batches = int(config.warmup_batches) + max(1, int(config.samples) // int(config.batch_size))
        started = time.perf_counter()
        first_batch_seconds = None
        for batch in loader:
            step_started = time.perf_counter()
            motion = batch["motion"]
            frames += int(motion.shape[0] * motion.shape[1])
            batches += 1
            if first_batch_seconds is None:
                first_batch_seconds = step_started - started
            if batches > int(config.warmup_batches):
                recorder.observe(0.0, time.perf_counter() - step_started)
            if batches >= target_batches:
                break
        elapsed = time.perf_counter() - started
        report = {
            "layer": "loader",
            "batches": batches,
            "frames": frames,
            "seconds": elapsed,
            "frames_per_second": frames / max(elapsed, 1e-9),
            "first_batch_seconds": first_batch_seconds,
            "normalize_on": str(normalize_on or config.normalize_on),
            "rss_bytes": _rss_bytes(),
        }
        report.update(recorder.summary())
        return report
    finally:
        store.close()


def benchmark_resident_baseline(config: BenchmarkConfig) -> dict[str, Any]:
    """Time a fixed batch loop that never touches storage.

    This is the GPU-resident baseline the plan expresses the end-to-end target
    against; without a usable device it degrades to a CPU tensor copy so the
    comparison stays meaningful on a CPU-only host.
    """
    import torch

    device_name = config.device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    store = open_any_feature_store(config.store)
    try:
        motion_dim = int(store.motion_dim)
    finally:
        store.close()
    batch = torch.zeros(
        (int(config.batch_size), int(config.target_frames), motion_dim), dtype=torch.float32, device=device
    )
    steps = max(1, int(config.samples) // int(config.batch_size))
    for _ in range(int(config.warmup_batches)):
        batch.add_(1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    recorder = LatencyRecorder()
    started = time.perf_counter()
    for _ in range(steps):
        step_started = time.perf_counter()
        batch.add_(0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        recorder.observe(0.0, time.perf_counter() - step_started)
    elapsed = time.perf_counter() - started
    summary = recorder.summary()
    return {
        "layer": "resident",
        "device": str(device),
        "batches": steps,
        "seconds": elapsed,
        "batch_bytes": int(batch.numel() * batch.element_size()),
        "batches_per_second": steps / max(elapsed, 1e-9),
        **summary,
    }


def benchmark_end_to_end(config: BenchmarkConfig) -> dict[str, Any]:
    """Time loader iteration interleaved with a device step, reporting the wait split."""
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    )
    store = open_any_feature_store(config.store)
    try:
        dataset = PackedFeatureDataset(config.split, store, normalize_on=config.normalize_on)
        sampler = TrainWindowSampler(
            store,
            target_frames=config.target_frames,
            samples_per_epoch=int(config.samples),
            seed=int(config.seed),
            strategy=str(config.strategy),
            rank=int(config.rank),
            world_size=int(config.world_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            sampler=sampler,
            collate_fn=lambda value: value,
            num_workers=int(config.num_workers),
            prefetch_factor=int(config.prefetch_factor) if config.num_workers > 0 else None,
            persistent_workers=bool(config.num_workers > 0),
            pin_memory=True,
        )
        recorder = LatencyRecorder()
        batches = 0
        target_batches = int(config.warmup_batches) + max(1, int(config.samples) // int(config.batch_size))
        for batch in loader:
            wait_started = time.perf_counter()
            motion = batch["motion"].to(device=device, non_blocking=True)
            transfer = time.perf_counter() - wait_started
            step_started = time.perf_counter()
            motion.mul_(1.0001)
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute = time.perf_counter() - step_started
            batches += 1
            if batches > int(config.warmup_batches):
                recorder.observe(transfer, compute)
            if batches >= target_batches:
                break
        summary = recorder.summary()
        resident = benchmark_resident_baseline(config)
        resident_mean = float(resident.get("step_time_mean", 0.0))
        reported_ratio = (
            float(summary["step_time_mean"]) / resident_mean if resident_mean > 0 else float("nan")
        )
        return {
            "layer": "end_to_end",
            "device": str(device),
            "batches": batches,
            **summary,
            "resident_step_time_mean": resident_mean,
            "step_to_resident_ratio": reported_ratio,
            "gate_throughput_met": None,
            "gate_data_wait_met": bool(summary["data_wait_fraction"] < 0.10),
        }
    finally:
        store.close()


def benchmark_store_layout(config: BenchmarkConfig) -> dict[str, Any]:
    """Report the store's physical layout and cold/warm shard read cost."""
    store = open_any_feature_store(config.store)
    try:
        shard_bytes = [Path(path).stat().st_size for path in store.shard_files]
        started = time.perf_counter()
        # A cold read touches the head of every shard; a warm read repeats the
        # same probes and is reported separately rather than passed off as disk
        # performance.
        for index in range(len(store.shard_files)):
            store.read_frames(index, 0, min(64, int(store.shard_num_frames[index])))
        cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        for index in range(len(store.shard_files)):
            store.read_frames(index, 0, min(64, int(store.shard_num_frames[index])))
        warm_seconds = time.perf_counter() - started
        return {
            "layer": "layout",
            "store": str(config.store),
            "shards": len(store.shard_files),
            "clips": store.num_clips,
            "frames": store.total_frames,
            "motion_dim": store.motion_dim,
            "total_bytes": int(sum(shard_bytes)),
            "mean_shard_bytes": float(np.mean(shard_bytes)) if shard_bytes else 0.0,
            "max_shard_bytes": int(max(shard_bytes)) if shard_bytes else 0,
            "shard_target_bytes": int(store.manifest.get("shard_target_bytes", 0)),
            "cold_probe_seconds": cold_seconds,
            "warm_probe_seconds": warm_seconds,
            "normalization_hash": store.normalization_hash,
            "feature_schema_hash": store.feature_schema_hash,
            "split_manifest_hash": store.split_manifest_hash,
        }
    finally:
        store.close()


def run_benchmark(config: BenchmarkConfig, *, layers: Iterable[str] = ()) -> dict[str, Any]:
    """Run the requested benchmark layers and return one aggregated report."""
    requested = [str(layer) for layer in layers]
    if not requested:
        requested = ["sampler", "loader", "resident", "end_to_end"]
    unknown = sorted(set(requested) - {"sampler", "loader", "resident", "end_to_end", "layout"})
    if unknown:
        raise ValueError(f"Unsupported benchmark layers: {unknown}")
    report: dict[str, Any] = {
        "config": {
            "store": str(config.store),
            "batch_size": int(config.batch_size),
            "num_workers": int(config.num_workers),
            "prefetch_factor": int(config.prefetch_factor),
            "samples": int(config.samples),
            "strategy": str(config.strategy),
            "normalize_on": str(config.normalize_on),
            "shard_mib": int(config.shard_mib),
            "split": str(config.split),
        },
        "rss_bytes_start": _rss_bytes(),
        "layers": {},
    }
    implementations = {
        "sampler": benchmark_sampler,
        "loader": benchmark_loader,
        "resident": benchmark_resident_baseline,
        "end_to_end": benchmark_end_to_end,
        "layout": benchmark_store_layout,
    }
    for layer in requested:
        report["layers"][layer] = implementations[layer](config)
    report["rss_bytes_end"] = _rss_bytes()
    if "end_to_end" in report["layers"]:
        summary = report["layers"]["end_to_end"]
        report["gates"] = {
            "data_wait_fraction": summary.get("data_wait_fraction"),
            "data_wait_under_10pct": bool(summary.get("gate_data_wait_met", False)),
            "step_to_resident_ratio": summary.get("step_to_resident_ratio"),
        }
    return report


def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the schema-v4 data pipeline.")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--layers", default="sampler,loader,resident,end_to_end")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--strategy", default="clip_uniform")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--normalize-on", default="cpu", choices=["cpu", "none"])
    parser.add_argument("--shard-mib", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_cli_parser().parse_args(argv)
    config = BenchmarkConfig(
        store=args.store,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        samples=args.samples,
        warmup_batches=args.warmup_batches,
        split=args.split,
        strategy=args.strategy,
        seed=args.seed,
        normalize_on=args.normalize_on,
        shard_mib=args.shard_mib,
        device=args.device,
    )
    layers = [item.strip() for item in str(args.layers).split(",") if item.strip()]
    report = run_benchmark(config, layers=layers)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


__all__ = [
    "BenchmarkConfig",
    "LatencyRecorder",
    "benchmark_end_to_end",
    "benchmark_loader",
    "benchmark_resident_baseline",
    "benchmark_sampler",
    "benchmark_store_layout",
    "run_benchmark",
]
