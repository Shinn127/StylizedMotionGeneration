"""Canonical DataLoader assembly for all data contracts."""

from __future__ import annotations

from dataclasses import dataclass
import random
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .feature_data import FeatureDataset, FeatureStore
from .sampling import FixedWindowSampler, SampleRequest, TrainWindowSampler
from .token_data import TokenDataset, TokenStore
from .trajectory_data import ConditionalTokenDataset, TrajectoryStore


DataKind = Literal["representation", "generator", "conditional_generator"]


class _EmptySampler(Sampler[SampleRequest]):
    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    samplers: dict[str, Sampler[SampleRequest]]
    prefetch_bytes: int = 0

    @property
    def loaders(self) -> dict[str, DataLoader]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def __getitem__(self, split: str) -> DataLoader:
        return self.loaders[split]

    @property
    def train_loader(self) -> DataLoader:
        return self.train

    @property
    def val_loader(self) -> DataLoader:
        return self.val

    @property
    def test_loader(self) -> DataLoader:
        return self.test

    def set_epoch(self, epoch: int) -> None:
        for sampler in self.samplers.values():
            setter = getattr(sampler, "set_epoch", None)
            if setter is not None:
                setter(int(epoch))


def _identity(value: Any) -> Any:
    return value


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    seed = int(info.seed if info is not None else worker_id)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _build_sampler(
    store: Any,
    split: str,
    sampling: Mapping[str, object],
    *,
    kind: DataKind,
    rank: int,
    world_size: int,
) -> Sampler[SampleRequest]:
    target_frames = int(sampling.get("target_frames", 64))
    required_frames = target_frames + 1 if kind != "representation" else target_frames
    if split == "train":
        return TrainWindowSampler(
            store,
            target_frames=target_frames,
            required_frames=required_frames,
            samples_per_epoch=int(sampling.get("samples_per_epoch", 100000)),
            seed=int(sampling.get("seed", 3407)),
            mirror_probability=float(sampling.get("mirror_probability", 0.5)),
            balance_key=sampling.get("balance_key"),
            rank=rank,
            world_size=world_size,
        )
    try:
        return FixedWindowSampler(
            store,
            split,
            target_frames=target_frames,
            required_frames=required_frames,
            stride=int(sampling.get("stride", 64)),
            include_tail=bool(sampling.get("include_tail", False)),
            rank=rank,
            world_size=world_size,
        )
    except ValueError as error:
        if "contains no ranges" in str(error) or "no valid" in str(error):
            return _EmptySampler()
        raise


def _validate_sampling_contract(sampling: Mapping[str, object], kind: DataKind) -> int:
    strategy = str(sampling.get("strategy", "clip_uniform"))
    if strategy == "frame_uniform":
        strategy = "clip_uniform"
    if kind == "representation" and strategy != "clip_uniform":
        raise ValueError(f"Unsupported sampling strategy: {strategy!r}")
    if kind != "representation" and strategy not in {"clip_uniform", "group_balanced"}:
        raise ValueError(f"Unsupported sampling strategy: {strategy!r}")
    target_frames = int(sampling.get("target_frames", 64))
    if target_frames != 64:
        raise ValueError("Canonical data loaders require sampling.target_frames=64")
    balance_key = sampling.get("balance_key")
    if balance_key is not None and balance_key not in {"style", "action"}:
        raise ValueError("sampling.balance_key must be null, style, or action")
    if kind == "representation" and balance_key is not None:
        raise ValueError("FSQ sampling does not support balance_key; use clip_uniform sampling")
    if strategy == "group_balanced" and balance_key is None:
        raise ValueError("group_balanced sampling requires sampling.balance_key")
    return target_frames


def _batch_bytes(kind: DataKind, batch_size: int, target_frames: int, motion_dim: int, trajectory_dim: int) -> int:
    if kind == "representation":
        return int(batch_size * (target_frames * motion_dim * 4 + target_frames))
    value = int(batch_size * (target_frames + 1) * 40)
    if kind == "conditional_generator":
        value += int(batch_size * target_frames * trajectory_dim * 4 + batch_size * target_frames)
    return value


def build_data_loaders(
    kind: DataKind,
    store: FeatureStore | TokenStore,
    *,
    trajectory_store: TrajectoryStore | None = None,
    sampling_config: Mapping[str, object],
    loader_config: Mapping[str, object],
    rank: int = 0,
    world_size: int = 1,
) -> DataLoaders:
    if not isinstance(sampling_config, Mapping) or not isinstance(loader_config, Mapping):
        raise TypeError("sampling_config and loader_config must be mappings")
    if kind not in {"representation", "generator", "conditional_generator"}:
        raise ValueError(f"Unsupported data kind: {kind!r}")
    if kind == "representation" and not isinstance(store, FeatureStore):
        raise TypeError("representation loaders require a FeatureStore")
    if kind != "representation" and not isinstance(store, TokenStore):
        raise TypeError("generator loaders require a TokenStore")
    if kind == "conditional_generator" and trajectory_store is None:
        raise ValueError("conditional_generator loaders require trajectory_store")
    if trajectory_store is not None and not isinstance(trajectory_store, TrajectoryStore):
        raise TypeError("trajectory_store must be a TrajectoryStore")
    if kind != "conditional_generator" and trajectory_store is not None:
        raise ValueError("trajectory_store is only valid for conditional_generator loaders")
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError("invalid rank/world_size")
    batch_size = int(loader_config.get("batch_size", 128))
    num_workers = int(loader_config.get("num_workers", 4))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("loader batch_size must be positive and num_workers non-negative")
    target_frames = _validate_sampling_contract(sampling_config, kind)
    trajectory_dim = int(trajectory_store.trajectory_dim) if trajectory_store is not None else 0
    estimated = _batch_bytes(kind, batch_size, target_frames, int(getattr(store, "motion_dim", 230)), trajectory_dim)
    prefetch_factor = int(loader_config.get("prefetch_factor", 2))
    limit_mb = loader_config.get("prefetch_memory_limit_mb", 512)
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if limit_mb is not None:
        limit_bytes = int(float(limit_mb) * 1024 * 1024)
        estimate = estimated * num_workers * prefetch_factor
        if estimate > limit_bytes and not bool(loader_config.get("allow_prefetch_over_budget", False)):
            raise ValueError(
                f"Estimated prefetch memory {estimate / 1024**2:.1f} MiB exceeds "
                f"prefetch_memory_limit_mb={float(limit_mb):.1f}"
            )
    pin_memory_value = loader_config.get("pin_memory", "auto")
    if pin_memory_value == "auto":
        pin_memory = torch.cuda.is_available()
    elif isinstance(pin_memory_value, bool):
        pin_memory = pin_memory_value
    else:
        raise ValueError("loader.pin_memory must be true, false, or auto")
    max_open_shards = int(loader_config.get("max_open_shards", 32))
    if max_open_shards <= 0:
        raise ValueError("loader.max_open_shards must be positive")
    store.max_open_shards = max_open_shards
    if trajectory_store is not None:
        trajectory_store.max_open_shards = max_open_shards
    samplers: dict[str, Sampler[SampleRequest]] = {
        split: _build_sampler(store, split, sampling_config, kind=kind, rank=rank, world_size=world_size)
        for split in ("train", "val", "test")
    }
    datasets: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        if kind == "representation":
            datasets[split] = FeatureDataset(split, store, max_open_shards=max_open_shards)
        elif kind == "generator":
            datasets[split] = TokenDataset(split, store, sequence_frames=65, max_open_shards=max_open_shards)
        else:
            assert trajectory_store is not None
            datasets[split] = ConditionalTokenDataset(
                split,
                store,
                trajectory_store,
                max_open_shards=max_open_shards,
            )
    common: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "sampler": None,
        "collate_fn": _identity,
        "pin_memory": pin_memory,
        "worker_init_fn": _worker_init,
    }
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        kwargs = dict(common)
        kwargs["sampler"] = samplers[split]
        kwargs["drop_last"] = bool(loader_config.get("drop_last_train", True)) if split == "train" else False
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
            kwargs["prefetch_factor"] = prefetch_factor
        loaders[split] = DataLoader(datasets[split], **kwargs)
    return DataLoaders(
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        samplers=samplers,
        prefetch_bytes=estimated * num_workers * prefetch_factor,
    )


__all__ = ["DataLoaders", "build_data_loaders"]
