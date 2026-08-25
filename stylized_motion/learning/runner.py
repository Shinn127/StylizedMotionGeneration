"""One train/validate/test lifecycle for all canonical FSQ representations."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as distributed
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stylized_motion.data import FeatureStore, build_data_loaders, open_feature_store
from stylized_motion.learning.checkpoint import CheckpointManager
from stylized_motion.learning.gradient_probe import compute_gradient_probe
from stylized_motion.learning.losses import compute_motion_reconstruction_losses
from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.representation import (
    FLAT_FSQ_FAMILY,
    LEGACY_MODEL_FAMILY,
    REPRESENTATION_FAMILIES,
    RepresentationProtocol,
    build_representation,
    checkpoint_metadata,
    load_representation_checkpoint,
)


Batch = Mapping[str, Any]
MetricFn = Callable[[Mapping[str, Any], Batch], torch.Tensor | float]
LOSS_COMPONENTS = (
    "recon",
    "delta",
    "root_pos",
    "root_rot",
    "joint",
    "contact",
    "foot_slide",
    "foot_height",
    "base_recon",
    "part_edit_transfer",
    "part_edit_preserve",
)
_WEIGHTED_MOTION_COMPONENTS = {
    "recon": None,
    "delta": "delta_weight",
    "root_pos": "root_pos_weight",
    "root_rot": "root_rot_weight",
    "joint": "joint_weight",
    "contact": "contact_weight",
    "foot_slide": "foot_slide_weight",
    "foot_height": "foot_height_weight",
}


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Batch, device: torch.device) -> dict[str, Any]:
    """Transfer each tensor exactly once; token uint8 remains uint8 on device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _matches_requested_device(actual: torch.device, requested: torch.device) -> bool:
    """Treat an unindexed CUDA request as the current CUDA device."""
    return actual.type == requested.type and (requested.index is None or actual.index == requested.index)


def load_experiment_config(path: str | Path) -> dict[str, object]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Experiment config must contain a mapping: {path}")
    required = {"representation", "data", "training", "evaluation", "sampling", "loader"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Experiment config is missing required sections: {missing}")
    representation = value["representation"]
    if not isinstance(representation, Mapping):
        raise ValueError("representation must be a mapping")
    family = representation.get("family")
    variant = representation.get("variant")
    if family not in REPRESENTATION_FAMILIES or not isinstance(variant, str):
        raise ValueError("representation.family/variant are invalid")
    data = value["data"]
    if not isinstance(data, Mapping) or int(data.get("required_data_schema_version", 0)) != 3:
        raise ValueError("representation workflows require data.required_data_schema_version=3")
    resolved = json.loads(json.dumps(value))
    model_config = representation.get("config")
    if isinstance(model_config, str):
        model_path = Path(model_config)
        if not model_path.is_absolute():
            model_path = path.parent / model_path
            if not model_path.exists():
                model_path = Path(model_config)
        model_value = yaml.safe_load(model_path.read_text(encoding="utf-8")) or {}
        if not isinstance(model_value, Mapping):
            raise ValueError(f"representation.config must point to a mapping: {model_path}")
        resolved["representation"]["config"] = dict(model_value)
    return resolved


def _config_family(config: Mapping[str, object]) -> str:
    value = config["representation"]
    assert isinstance(value, Mapping)
    family = value["family"]
    if family not in REPRESENTATION_FAMILIES:
        raise ValueError(f"Unsupported representation family: {family!r}")
    return str(family)


def build_dataloaders(config: Mapping[str, object], store: FeatureStore) -> tuple[dict[str, DataLoader], dict[str, object]]:
    data = config["data"]
    training = config["training"]
    assert isinstance(data, Mapping) and isinstance(training, Mapping)
    sampling = config.get("sampling", {})
    loader = config.get("loader", {})
    if not isinstance(sampling, Mapping) or not isinstance(loader, Mapping):
        raise ValueError("sampling and loader must be mappings")
    assembled = build_data_loaders(
        "representation",
        store,
        sampling_config=sampling,
        loader_config=loader,
    )
    return assembled.loaders, {"prefetch_bytes": assembled.prefetch_bytes}


def _foot_indices(store: FeatureStore) -> tuple[int, int] | None:
    try:
        return store.names.index("LeftToeBase"), store.names.index("RightToeBase")
    except ValueError:
        return None


def _joint_weights_from_feature_weights(
    feature_weights: torch.Tensor,
    num_joints: int,
) -> torch.Tensor:
    """Project the stored per-feature weights to non-root joint weights."""
    if num_joints <= 1:
        raise ValueError(f"num_joints must be greater than one, got {num_joints}")
    weights = feature_weights.reshape(-1)
    rotation_start = 9
    rotation_stop = rotation_start + (num_joints - 1) * 6
    if weights.numel() < rotation_stop:
        raise ValueError(
            f"Feature weights have length {weights.numel()}, too short for "
            f"{num_joints} joints"
        )
    rotation_weights = weights[rotation_start:rotation_stop].reshape(num_joints - 1, 6)
    non_root = rotation_weights.mean(dim=-1)
    return torch.cat((weights.new_zeros(1), non_root), dim=0)


def build_loss_context(config: Mapping[str, object], store: FeatureStore, device: torch.device) -> dict[str, Any]:
    training = config["training"]
    evaluation = config["evaluation"]
    assert isinstance(training, Mapping) and isinstance(evaluation, Mapping)
    stats = store.stats
    feature_weights = torch.from_numpy(store.model_feature_weights()).to(device)
    context: dict[str, Any] = {
        "feature_weights": feature_weights,
        "joint_weights": _joint_weights_from_feature_weights(feature_weights, len(store.names)),
        "feature_offset": torch.from_numpy(stats.offset.astype(np.float32)).to(device),
        "feature_scale": torch.from_numpy(stats.scale.astype(np.float32)).to(device),
        "ref_pos": torch.from_numpy(stats.ref_pos.astype(np.float32)).to(device),
        "parents": tuple(int(value) for value in store.parents.tolist()),
        "foot_indices": _foot_indices(store),
        "root_dt": float(evaluation.get("root_dt", 1.0 / 60.0)),
        "delta_weight": float(training.get("delta_weight", 3.0)),
        "root_pos_weight": float(training.get("root_pos_weight", 0.1)),
        "root_rot_weight": float(training.get("root_rot_weight", 0.1)),
        "joint_weight": float(training.get("joint_weight", 0.5)),
        "contact_weight": float(training.get("contact_weight", 0.1)),
        "foot_slide_weight": float(training.get("foot_slide_weight", 0.1)),
        "foot_height_weight": float(training.get("foot_height_weight", 0.1)),
        "contact_temperature": float(training.get("contact_temperature", 10.0)),
        "reuse_weight": float(training.get("reuse_weight", 0.01)),
        "base_reuse_weight": float(training.get("base_reuse_weight", 0.0025)),
        "base_reuse_threshold": float(training.get("base_reuse_threshold", 1.0)),
        "latent_energy_weight": float(training.get("latent_energy_weight", 0.01)),
        "base_recon_weight": float(training.get("base_recon_weight", 0.1)),
        "edit_weight": float(training.get("edit_weight", 0.25)),
        "edit_preserve_weight": float(training.get("edit_preserve_weight", 1.0)),
    }
    if context["joint_weight"] > 0.0 and context["foot_indices"] is None:
        raise ValueError("joint/foot losses require LeftToeBase and RightToeBase in the feature schema")
    return context


def build_loss_fn(representation: RepresentationProtocol, context: Mapping[str, Any], config: Mapping[str, object]):
    def compute(output: Mapping[str, Any], batch: Batch) -> dict[str, torch.Tensor]:
        motion = batch["motion"]
        if not isinstance(motion, torch.Tensor):
            raise TypeError("FeatureDataset batches must contain tensor motion")
        if motion.ndim != 3 or motion.shape[1] != 64:
            raise ValueError(
                "Canonical representation training requires motion with shape [B,64,motion_dim]"
            )
        loss_mask = batch.get("loss_mask")
        if not isinstance(loss_mask, torch.Tensor) or loss_mask.shape != motion.shape[:2]:
            raise ValueError("Canonical representation batches require loss_mask with shape [B,64]")
        if batch.get("_all_frames_valid") is not True and not bool(loss_mask.to(dtype=torch.bool).all()):
            raise ValueError("Canonical 64-frame representation batches require all loss_mask values to be true")
        loss_values = compute_motion_reconstruction_losses(
            batch_motion=motion,
            output=dict(output),
            feature_weights=context["feature_weights"],
            feature_offset=context["feature_offset"],
            feature_scale=context["feature_scale"],
            delta_weight=context["delta_weight"],
            commit_weight=0.0,
            root_pos_weight=context["root_pos_weight"],
            root_rot_weight=context["root_rot_weight"],
            root_dt=context["root_dt"],
            joint_weight=context["joint_weight"],
            joint_weights=context["joint_weights"],
            contact_weight=context["contact_weight"],
            foot_slide_weight=context["foot_slide_weight"],
            foot_height_weight=context["foot_height_weight"],
            contact_temperature=context["contact_temperature"],
            ref_pos=context["ref_pos"],
            parents=context["parents"],
            foot_indices=context["foot_indices"],
            loss_mask=loss_mask,
        )
        rep_batch = dict(context)
        rep_batch["motion"] = motion
        rep_batch["loss_mask"] = loss_mask
        specific = representation.compute_representation_losses(dict(output), rep_batch)
        result = {
            "loss": loss_values.loss + sum(specific.values(), motion.new_zeros(())),
            "recon": loss_values.recon,
            "delta": loss_values.delta,
            "root_pos": loss_values.root_pos,
            "root_rot": loss_values.root_rot,
            "joint": loss_values.joint,
            "contact": loss_values.contact,
            "foot_slide": loss_values.foot_slide,
            "foot_height": loss_values.foot_height,
        }
        result.update(specific)
        return result

    return compute


def _effective_loss_components(
    values: Mapping[str, torch.Tensor],
    context: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return scalar loss contributions exactly as included in ``values['loss']``."""
    components: dict[str, torch.Tensor] = {}
    for name, weight_key in _WEIGHTED_MOTION_COMPONENTS.items():
        value = values.get(name)
        if not isinstance(value, torch.Tensor):
            continue
        if weight_key is None:
            components[name] = value
        else:
            components[name] = value * float(context[weight_key])
    for name in ("base_recon", "part_edit_transfer", "part_edit_preserve"):
        value = values.get(name)
        if isinstance(value, torch.Tensor):
            # V2 representation losses already apply their configured scalar.
            components[name] = value
    return components


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Runner metrics must be scalar")
        return float(value.detach().cpu())
    return float(value)


class _DeviceMetricAccumulator:
    """Accumulate scalar metrics on-device and transfer them once per epoch."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.sums: dict[str, torch.Tensor] = {}
        self.weights: dict[str, int] = {}

    def update(self, values: Mapping[str, Any], weight: int) -> None:
        if weight <= 0:
            return
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"Runner metrics must be scalar: {name}")
                tensor = value.detach().float()
            else:
                tensor = torch.as_tensor(float(value), dtype=torch.float32, device=self.device)
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
            weighted = tensor * float(weight)
            self.sums[name] = weighted if name not in self.sums else self.sums[name] + weighted
            self.weights[name] = self.weights.get(name, 0) + int(weight)

    def finalize(self) -> dict[str, float]:
        if not self.sums:
            return {}
        names = sorted(self.sums)
        sums = torch.stack([self.sums[name] for name in names])
        weights = torch.tensor(
            [float(self.weights[name]) for name in names],
            dtype=sums.dtype,
            device=sums.device,
        )
        packed = torch.cat((sums, weights), dim=0)
        reduce_device = self.device if self.device.type in {"cpu", "cuda"} else torch.device("cpu")
        if packed.device != reduce_device:
            packed = packed.to(reduce_device)
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(packed, op=distributed.ReduceOp.SUM)
        packed = packed.cpu()
        return {
            name: float(packed[index] / max(float(packed[len(names) + index]), 1.0))
            for index, name in enumerate(names)
        }


def _reduce_counts(device: torch.device, *counts: int) -> tuple[float, ...]:
    values = torch.tensor(
        [float(count) for count in counts],
        dtype=torch.float64,
        device=device if device.type in {"cpu", "cuda"} else torch.device("cpu"),
    )
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
    return tuple(float(value) for value in values.cpu().tolist())


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model


def _is_main_process() -> bool:
    return not distributed.is_available() or not distributed.is_initialized() or distributed.get_rank() == 0


class RepresentationRunner:
    def __init__(
        self,
        representation: nn.Module,
        *,
        family: str,
        train_loader: DataLoader | None,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        loss_fn: Callable[[Mapping[str, Any], Batch], Mapping[str, torch.Tensor]],
        metric_suite: Mapping[str, MetricFn] | None,
        checkpoint_manager: CheckpointManager,
        config: Mapping[str, object],
        feature_schema: Mapping[str, object],
        feature_stats: Mapping[str, object],
        device: torch.device,
        epochs: int,
        grad_clip_norm: float | None = None,
        precision: str = "fp32",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        writer: SummaryWriter | None = None,
        gradient_probe_path: Path | None = None,
    ) -> None:
        self.representation = representation
        self.family = family
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.metric_suite = dict(metric_suite or {})
        self.checkpoint_manager = checkpoint_manager
        self.config = dict(config)
        self.feature_schema = dict(feature_schema)
        self.feature_stats = dict(feature_stats)
        self.device = device
        self.epochs = int(epochs)
        self.grad_clip_norm = grad_clip_norm
        self.precision = precision
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        training = self.config.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("training config must be a mapping")
        self.gradient_probe_enabled = bool(training.get("gradient_probe_enabled", False))
        self.gradient_probe_interval = int(training.get("gradient_probe_interval", 503))
        if self.gradient_probe_enabled and self.gradient_probe_interval <= 0:
            raise ValueError("training.gradient_probe_interval must be positive")
        self.gradient_probe_file = None
        if self.gradient_probe_enabled and _is_main_process() and gradient_probe_path is not None:
            gradient_probe_path.parent.mkdir(parents=True, exist_ok=True)
            self.gradient_probe_file = gradient_probe_path.open("a", encoding="utf-8")
        self.global_step = 0
        self.best_val: float | None = None
        evaluation = self.config.get("evaluation", {})
        if not isinstance(evaluation, Mapping):
            raise ValueError("evaluation config must be a mapping")
        self.metrics_interval = int(evaluation.get("metrics_interval", 100))
        if self.metrics_interval <= 0:
            raise ValueError("evaluation.metrics_interval must be positive")
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("training.precision must be fp32 or amp")
        if self.precision == "amp" and device.type != "cuda":
            raise ValueError("AMP is only enabled for CUDA in the canonical runner")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "amp")

    def _write_gradient_probe(
        self,
        epoch: int,
        step: int,
        output: Mapping[str, Any],
        probe: Mapping[str, Any],
    ) -> None:
        if self.gradient_probe_file is None:
            return
        record = {
            "epoch": int(epoch),
            "step": int(step),
            "edit_part": output.get("edit_part"),
            **probe,
        }
        self.gradient_probe_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.gradient_probe_file.flush()
        if self.writer is None:
            return
        self.writer.add_scalar("gradient/train/total_norm", probe["total_norm"], step)
        self.writer.add_scalar("gradient/train/component_norm_sum", probe["component_norm_sum"], step)
        self.writer.add_scalar("gradient/train/loss_recompose_error", probe["loss_recompose_error"], step)
        components = probe.get("components", {})
        if isinstance(components, Mapping):
            for name, metrics in components.items():
                if not isinstance(metrics, Mapping):
                    continue
                for metric_name in ("value", "norm", "share", "projection", "cosine"):
                    if metric_name in metrics:
                        self.writer.add_scalar(
                            f"gradient/train/{name}/{metric_name}",
                            float(metrics[metric_name]),
                            step,
                        )

    def close(self) -> None:
        if self.gradient_probe_file is not None:
            self.gradient_probe_file.close()
            self.gradient_probe_file = None

    @property
    def protocol(self) -> RepresentationProtocol:
        return _unwrap(self.representation)  # type: ignore[return-value]

    @contextlib.contextmanager
    def _autocast(self):
        if self.precision == "amp":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    def _forward(self, batch: Batch, *, collect_metrics: bool, compact_output: bool) -> dict[str, Any]:
        motion = batch["motion"]
        if not isinstance(motion, torch.Tensor) or not _matches_requested_device(motion.device, self.device):
            raise ValueError("Runner expects a device batch produced by move_batch_to_device()")
        if motion.ndim != 3 or motion.shape[1] != 64:
            raise ValueError("Canonical representation runner requires motion with shape [B,64,motion_dim]")
        with self._autocast():
            if self.family == "latent_residual_fsq_v2":
                kwargs: dict[str, Any] = {
                    "collect_metrics": collect_metrics,
                    "compact_output": compact_output,
                    "decode_base": True,
                }
                if self.representation.training and motion.shape[0] > 1:
                    part_index = self.global_step % len(PART_NAMES)
                    kwargs["edit_part"] = PART_NAMES[part_index]
                    kwargs["donor_permutation"] = torch.roll(
                        torch.arange(motion.shape[0], device=motion.device), shifts=1
                    )
                return self.representation(motion, **kwargs)
            return self.representation(
                motion,
                collect_metrics=collect_metrics,
                compact_output=compact_output,
            )

    def _metric_values(
        self,
        output: Mapping[str, Any],
        batch: Batch,
        loss_values: Mapping[str, torch.Tensor],
        *,
        collect_metrics: bool,
    ) -> dict[str, Any]:
        result = dict(loss_values)
        if not collect_metrics:
            return result
        metrics = output.get("representation_metrics", {})
        if isinstance(metrics, Mapping):
            for name, value in metrics.items():
                if isinstance(value, (torch.Tensor, float, int)):
                    result[f"representation/{name}"] = value
        for name, callback in self.metric_suite.items():
            result[name] = callback(output, batch)
        return result

    @staticmethod
    def _validate_cpu_batch(batch: Batch) -> int:
        motion = batch.get("motion")
        loss_mask = batch.get("loss_mask")
        if not isinstance(motion, torch.Tensor):
            raise ValueError("Canonical representation batches require a motion tensor")
        if loss_mask is None:
            return int(motion.shape[0] * motion.shape[1])
        if not isinstance(loss_mask, torch.Tensor):
            raise ValueError("Canonical representation loss_mask must be a tensor")
        if loss_mask.shape != motion.shape[:2]:
            raise ValueError("Canonical representation batches require loss_mask with shape [B,64]")
        if not bool(loss_mask.to(dtype=torch.bool).all()):
            raise ValueError("Canonical 64-frame representation batches require all loss_mask values to be true")
        return int(loss_mask.numel())

    def evaluate(self, split: str) -> dict[str, Any]:
        if split not in {"val", "test"}:
            raise ValueError("evaluate split must be val or test")
        loader = self.val_loader if split == "val" else self.test_loader
        if loader is None:
            raise ValueError(f"No loader configured for {split}")
        self.representation.eval()
        accumulator = _DeviceMetricAccumulator(self.device)
        sample_count = 0
        valid_frame_count = 0
        data_wait = 0.0
        started = time.perf_counter()
        iterator = iter(loader)
        with torch.inference_mode():
            while True:
                wait_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                data_wait += time.perf_counter() - wait_started
                valid_frames = self._validate_cpu_batch(batch)
                batch = dict(batch)
                batch["_all_frames_valid"] = True
                batch["_valid_frames"] = valid_frames
                device_batch = move_batch_to_device(batch, self.device)
                output = self._forward(
                    device_batch,
                    collect_metrics=True,
                    compact_output=not bool(self.metric_suite),
                )
                values = self.loss_fn(output, device_batch)
                record = self._metric_values(
                    output, device_batch, values, collect_metrics=True
                )
                accumulator.update(record, valid_frames)
                sample_count += int(device_batch["motion"].shape[0])
                valid_frame_count += valid_frames
        metrics = accumulator.finalize()
        elapsed = max(time.perf_counter() - started, 1e-8)
        reduced_count, reduced_samples = _reduce_counts(
            self.device, valid_frame_count, sample_count
        )
        step_time = max(elapsed - data_wait, 0.0)
        return {
            "mode": split,
            "metrics": metrics,
            "valid_frames": reduced_count,
            "samples": reduced_samples,
            "data_wait_seconds": data_wait,
            "step_time_seconds": step_time,
            "target_frames_per_second": reduced_count / elapsed,
            "samples_per_second": reduced_samples / max(elapsed, 1e-8),
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None or self.optimizer is None:
            raise ValueError("Training requires train_loader and optimizer")
        self.representation.train()
        accumulator = _DeviceMetricAccumulator(self.device)
        count = 0
        sample_count = 0
        data_wait = 0.0
        started = time.perf_counter()
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        iterator = iter(self.train_loader)
        while True:
            wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait += time.perf_counter() - wait_started
            valid_frames = self._validate_cpu_batch(batch)
            batch = dict(batch)
            batch["_all_frames_valid"] = True
            batch["_valid_frames"] = valid_frames
            collect_metrics = self.global_step % self.metrics_interval == 0
            self.optimizer.zero_grad(set_to_none=True)
            device_batch = move_batch_to_device(batch, self.device)
            output = self._forward(
                device_batch,
                collect_metrics=collect_metrics,
                compact_output=not bool(self.metric_suite),
            )
            values = self.loss_fn(output, device_batch)
            loss = values["loss"]
            probe_step = self.global_step + 1
            should_probe = (
                self.gradient_probe_enabled
                and _is_main_process()
                and probe_step % self.gradient_probe_interval == 0
            )
            if should_probe:
                components = _effective_loss_components(values, self.config["training"])
                probe = compute_gradient_probe(
                    components,
                    loss,
                    tuple(self.representation.parameters()),
                )
                self._write_gradient_probe(epoch, probe_step, output, probe)
            if self.precision == "amp":
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.representation.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.representation.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            record = self._metric_values(
                output,
                device_batch,
                values,
                collect_metrics=collect_metrics,
            )
            accumulator.update(record, valid_frames)
            self.global_step += 1
            count += valid_frames
            sample_count += int(device_batch["motion"].shape[0])
            if self.writer is not None and collect_metrics:
                self.writer.add_scalar(
                    "train/step_loss",
                    float(values["loss"].detach().cpu()),
                    self.global_step,
                )
        if self.scheduler is not None:
            self.scheduler.step()
        result = accumulator.finalize()
        reduced_count, reduced_samples = _reduce_counts(self.device, count, sample_count)
        elapsed = max(time.perf_counter() - started, 1e-8)
        result["valid_frames"] = reduced_count
        result["samples"] = reduced_samples
        result["data_wait_seconds"] = data_wait
        result["step_time_seconds"] = max(elapsed - data_wait, 0.0)
        result["target_frames_per_second"] = reduced_count / elapsed
        result["samples_per_second"] = reduced_samples / elapsed
        return result

    def checkpoint_payload(self, epoch: int, metrics: Mapping[str, Any]) -> dict[str, object]:
        protocol = self.protocol
        metadata = checkpoint_metadata(self.config, protocol, self.feature_schema)
        feature_stats = dict(self.feature_stats)
        feature_stats["motion_dim"] = int(protocol.motion_dim)
        payload: dict[str, object] = {
            "schema_version": 2,
            "model_family": LEGACY_MODEL_FAMILY[self.family],
            "model_config": dict(getattr(protocol, "config", {})),
            "model": protocol.state_dict(),
            "config": self.config,
            "representation": metadata,
            "feature_schema": self.feature_schema,
            "feature_stats": feature_stats,
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "metrics": dict(metrics),
        }
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler"] = self.scheduler.state_dict()
        return payload

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        for epoch in range(1, self.epochs + 1):
            if _is_main_process():
                print(f"Epoch {epoch}/{self.epochs} started", flush=True)
            train = self.train_epoch(epoch)
            val_result = self.evaluate("val") if self.val_loader is not None else {}
            val = val_result.get("metrics", {})
            record = {"epoch": epoch, "train": train, "val": val}
            history.append(record)
            val_loss = float(val.get("loss", train.get("loss", float("inf"))))
            if _is_main_process():
                train_loss = float(train.get("loss", float("nan")))
                train_recon = float(train.get("recon", float("nan")))
                train_samples_per_second = float(train.get("samples_per_second", float("nan")))
                val_loss_text = f"{float(val['loss']):.6f}" if "loss" in val else "n/a"
                val_recon_text = f"{float(val['recon']):.6f}" if "recon" in val else "n/a"
                val_samples_per_second = (
                    f"{float(val_result.get('samples_per_second', float('nan'))):.2f}"
                    if val_result
                    else "n/a"
                )
                print(
                    f"Epoch {epoch}/{self.epochs} complete | "
                    f"train_loss={train_loss:.6f} | train_recon={train_recon:.6f} | "
                    f"train_samples/s={train_samples_per_second:.2f} | "
                    f"val_loss={val_loss_text} | val_recon={val_recon_text} | "
                    f"val_samples/s={val_samples_per_second}",
                    flush=True,
                )
            is_best = self.best_val is None or val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
            payload = self.checkpoint_payload(epoch, record)
            if _is_main_process():
                self.checkpoint_manager.save(payload, "last.pt")
                if is_best:
                    self.checkpoint_manager.save(payload, "best.pt")
            if self.writer is not None and _is_main_process():
                self.writer.add_scalar("epoch/train_loss", train.get("loss", 0.0), epoch)
                for name in LOSS_COMPONENTS:
                    if name in train:
                        self.writer.add_scalar(f"epoch/train_{name}", train[name], epoch)
                if val:
                    self.writer.add_scalar("epoch/val_loss", val.get("loss", 0.0), epoch)
                    for name in LOSS_COMPONENTS:
                        if name in val:
                            self.writer.add_scalar(f"epoch/val_{name}", val[name], epoch)
        return {"mode": "train", "global_step": self.global_step, "history": history}

    def run(self, mode: str, split: str | None = None) -> dict[str, Any]:
        if mode == "train":
            return self.fit()
        if mode == "validate":
            return self.evaluate(split or "val")
        if mode == "test":
            return self.evaluate(split or "test")
        raise ValueError("mode must be train, validate or test")


def _feature_stats_payload(store: FeatureStore) -> dict[str, object]:
    return {
        "offset": store.stats.offset.astype(np.float32),
        "scale": store.stats.scale.astype(np.float32),
        "dist": store.stats.dist.astype(np.float32),
        "weights": store.stats.weights.astype(np.float32),
        "ref_pos": store.stats.ref_pos.astype(np.float32),
        "names": list(store.names),
        "parents": [int(value) for value in store.parents.tolist()],
        "joint_subset": store.joint_subset,
    }


def _family_cli(value: str) -> str:
    mapping = {
        "flat-fsq": "flat_fsq",
        "part-fsq": "part_fsq",
        "residual-part-fsq": "residual_part_fsq",
        "latent-residual-fsq": "latent_residual_fsq",
        "latent-residual-fsq-v2": "latent_residual_fsq_v2",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported --representation {value!r}")
    return mapping[value]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, validate, or test one canonical FSQ representation.")
    parser.add_argument("--workflow-mode", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--representation", choices=["flat-fsq", "part-fsq", "residual-part-fsq", "latent-residual-fsq", "latent-residual-fsq-v2"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    family = _family_cli(args.representation)
    if _config_family(config) != family:
        raise ValueError("--representation does not match representation.family in --config")
    training = config["training"]
    data = config["data"]
    assert isinstance(training, Mapping) and isinstance(data, Mapping)
    set_seed(int(training.get("seed", 3407)), bool(training.get("deterministic", False)))
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_world_size > 1 and not distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        distributed.init_process_group(backend=backend, init_method="env://")
    if distributed.is_initialized():
        if torch.cuda.is_available() and args.device in {"auto", "cuda"}:
            torch.cuda.set_device(local_rank)
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    device = choose_device(args.device)
    feature_database = data.get("fsq_window_index", data.get("feature_database"))
    if feature_database is None:
        raise ValueError("data.fsq_window_index is required for FSQ training")
    store = open_feature_store(feature_database)
    feature_schema = store.feature_schema()
    sampling = config.get("sampling", {})
    loader_config = config.get("loader", {})
    if not isinstance(sampling, Mapping) or not isinstance(loader_config, Mapping):
        raise ValueError("sampling and loader must be mappings")
    assembled_loaders = build_data_loaders(
        "representation",
        store,
        sampling_config=sampling,
        loader_config=loader_config,
        rank=rank,
        world_size=world_size,
    )
    loaders = assembled_loaders.loaders
    if args.workflow_mode in {"validate", "test"} and args.checkpoint is None:
        raise ValueError("--checkpoint is required for validate/test")
    if args.checkpoint is not None:
        checkpoint, representation = load_representation_checkpoint(args.checkpoint, device, feature_schema=feature_schema)
    else:
        representation = build_representation(config, feature_store=store, feature_schema=feature_schema).to(device)
        checkpoint = None
    if representation.family != family:
        raise ValueError("Checkpoint/config representation family mismatch")
    data_parallel = bool(training.get("data_parallel", False))
    if world_size > 1:
        representation = nn.parallel.DistributedDataParallel(
            representation,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    elif data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        representation = nn.DataParallel(representation)
    context = build_loss_context(config, store, device)
    loss_fn = build_loss_fn(_unwrap(representation), context, config)
    optimizer = None
    scheduler = None
    if args.workflow_mode == "train":
        optimizer = torch.optim.AdamW(representation.parameters(), lr=float(training.get("lr", 2e-4)), weight_decay=float(training.get("weight_decay", 0.0)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(training.get("epochs", 100))))
        if checkpoint is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
    output = args.output or Path(training.get("output_dir", f"outputs/{family}_40x9"))
    writer = SummaryWriter(output / "tensorboard") if args.workflow_mode == "train" and _is_main_process() else None
    runner = RepresentationRunner(
        representation,
        family=family,
        train_loader=loaders["train"] if args.workflow_mode == "train" else None,
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        loss_fn=loss_fn,
        metric_suite={},
        checkpoint_manager=CheckpointManager(output),
        config=config,
        feature_schema=feature_schema,
        feature_stats=_feature_stats_payload(store),
        device=device,
        epochs=int(training.get("epochs", 100)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        precision=str(training.get("precision", "fp32")),
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        gradient_probe_path=output / "gradient_probe.jsonl",
    )
    try:
        result = runner.run(args.workflow_mode, split=args.split)
    finally:
        runner.close()
        if writer is not None:
            writer.close()
        store.close()
    if _is_main_process():
        print(json.dumps(result, indent=2, default=str))


__all__ = ["RepresentationRunner", "build_cli_parser", "build_representation", "load_experiment_config", "main"]


if __name__ == "__main__":
    main()
