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
from .packed_store import PackedFeatureDataset, PackedFeatureStore
from .sampling import FixedWindowSampler, SampleRequest, TrainWindowSampler, sampling_contract
from .token_data import TokenDataset, TokenStore
from .trajectory_data import ConditionalTokenDataset, TrajectoryStore


DataKind = Literal["representation", "generator", "conditional_generator"]


@dataclass
class DataLoaders:
    """Loaders for one run, with the validation split split into two roles.

    ``val`` is the *monitor* loader: bounded by ``sampling.eval_limit`` so it can
    run every epoch or every N steps cheaply and deterministically. ``full_val``
    is the unbounded validation sweep and is ``None`` when no limit was
    configured (then ``val`` already is the full split). ``test`` is never
    limited, because it is evaluated once.
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader
    samplers: dict[str, Sampler[SampleRequest]]
    prefetch_bytes: int = 0
    full_val: DataLoader | None = None

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
    def full_val_loader(self) -> DataLoader | None:
        return self.full_val

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
    limited: bool = True,
) -> Sampler[SampleRequest]:
    """Build the train sampler or one of the evaluation samplers.

    ``limited=False`` ignores ``sampling.eval_limit`` and walks the whole split;
    that is what the periodic full validation and the test split use.
    """
    contract = sampling_contract(sampling, kind)
    if split == "train":
        return TrainWindowSampler(
            store,
            target_frames=contract["target_frames"],
            required_frames=contract["required_frames"],
            samples_per_epoch=contract["samples_per_epoch"],
            seed=contract["seed"],
            mirror_probability=contract["mirror_probability"],
            balance_key=contract["balance_key"],
            strategy=contract["strategy"],
            balance_mix=contract["balance_mix"],
            balance_max_ratio=contract["balance_max_ratio"],
            tail=contract["tail"],
            track_coverage=bool(sampling.get("track_coverage", True)),
            rank=rank,
            world_size=world_size,
        )
    limit = contract["eval_limit"] if limited else None
    return FixedWindowSampler(
        store,
        split,
        target_frames=contract["target_frames"],
        required_frames=contract["required_frames"],
        stride=contract["stride"],
        include_tail=contract["include_tail"],
        limit=None if limit is None else int(limit) * int(world_size),
        rank=rank,
        world_size=world_size,
    )


def _validate_sampling_contract(sampling: Mapping[str, object], kind: DataKind) -> int:
    return int(sampling_contract(sampling, kind)["target_frames"])


def _batch_bytes(kind: DataKind, batch_size: int, target_frames: int, motion_dim: int, trajectory_dim: int) -> int:
    if kind == "representation":
        return int(batch_size * (target_frames * motion_dim * 4 + target_frames))
    value = int(batch_size * (target_frames + 1) * 40)
    if kind == "conditional_generator":
        value += int(batch_size * target_frames * trajectory_dim * 4 + batch_size * target_frames)
    return value


def build_data_loaders(
    kind: DataKind,
    store: FeatureStore | PackedFeatureStore | TokenStore,
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
    if kind == "representation" and not isinstance(store, (FeatureStore, PackedFeatureStore)):
        raise TypeError("representation loaders require a FeatureStore or a PackedFeatureStore")
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
    eval_limit = sampling_contract(sampling_config, kind)["eval_limit"]
    # A bounded evaluation limit means a second, unbounded validation loader will
    # exist; its prefetch buffers live in the same process and are budgeted too.
    loader_count = 4 if eval_limit is not None else 3
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
    # The test split is evaluated once, so it is always unbounded; validation is
    # only limited for the cheap monitoring sweep, which gets its own bounded
    # sampler while the periodic full sweep reuses the unbounded one.
    samplers["test"] = _build_sampler(
        store, "test", sampling_config, kind=kind, rank=rank, world_size=world_size, limited=False
    )
    full_val_sampler = None
    if eval_limit is not None:
        full_val_sampler = _build_sampler(
            store, "val", sampling_config, kind=kind, rank=rank, world_size=world_size, limited=False
        )
    normalize_on = str(loader_config.get("normalize_on", "cpu"))
    if normalize_on not in {"cpu", "none"}:
        raise ValueError("loader.normalize_on must be 'cpu' or 'none'")

    def _dataset(split: str) -> Any:
        if kind == "representation":
            if isinstance(store, PackedFeatureStore):
                return PackedFeatureDataset(
                    split, store, normalize_on=normalize_on, max_open_shards=max_open_shards
                )
            if normalize_on == "none":
                raise ValueError(
                    "loader.normalize_on='none' requires a packed store; v3 stores are pre-normalized"
                )
            return FeatureDataset(split, store, max_open_shards=max_open_shards)
        if kind == "generator":
            return TokenDataset(split, store, sequence_frames=65, max_open_shards=max_open_shards)
        assert trajectory_store is not None
        return ConditionalTokenDataset(split, store, trajectory_store, max_open_shards=max_open_shards)

    datasets: dict[str, Any] = {split: _dataset(split) for split in ("train", "val", "test")}
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
    full_val_loader = None
    if full_val_sampler is not None:
        kwargs = dict(common)
        kwargs["sampler"] = full_val_sampler
        kwargs["drop_last"] = False
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
            kwargs["prefetch_factor"] = prefetch_factor
        full_val_loader = DataLoader(_dataset("val"), **kwargs)
        samplers["full_val"] = full_val_sampler
    del loader_count  # documented above; the budget check already counted it
    return DataLoaders(
        train=loaders["train"],
        val=loaders["val"],
        test=loaders["test"],
        samplers=samplers,
        prefetch_bytes=estimated * num_workers * prefetch_factor,
        full_val=full_val_loader,
    )


__all__ = ["DataLoaders", "build_data_loaders"]
