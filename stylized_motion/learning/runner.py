"""One train/validate/test lifecycle for all canonical FSQ representations."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stylized_motion.data.feature_dataset import FeatureDataset, FeatureStore, build_feature_store
from stylized_motion.learning.checkpoint import CheckpointManager
from stylized_motion.learning.losses import compute_motion_reconstruction_losses
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


def load_experiment_config(path: str | Path) -> dict[str, object]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Experiment config must contain a mapping: {path}")
    required = {"representation", "data", "training", "evaluation"}
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


def build_dataloaders(config: Mapping[str, object], store: FeatureStore) -> tuple[dict[str, DataLoader], dict[str, FeatureDataset]]:
    data = config["data"]
    training = config["training"]
    assert isinstance(data, Mapping) and isinstance(training, Mapping)
    window_size = int(data.get("window_size", 64))
    if window_size != 64 or store.window_size != 64:
        raise ValueError("Canonical FSQ training requires a 64-frame feature window")
    batch_size = int(training.get("batch_size", 512))
    num_workers = int(training.get("num_workers", 0))
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(training.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    datasets = {
        "train": FeatureDataset(str(data.get("split_train", "train")), store),
        "val": FeatureDataset(str(data.get("split_val", "val")), store),
        "test": FeatureDataset(str(data.get("split_test", "test")), store),
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, **kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **kwargs),
    }
    return loaders, datasets


def _foot_indices(store: FeatureStore) -> tuple[int, int] | None:
    try:
        return store.names.index("LeftToeBase"), store.names.index("RightToeBase")
    except ValueError:
        return None


def build_loss_context(config: Mapping[str, object], store: FeatureStore, device: torch.device) -> dict[str, Any]:
    training = config["training"]
    evaluation = config["evaluation"]
    assert isinstance(training, Mapping) and isinstance(evaluation, Mapping)
    stats = store.stats
    context: dict[str, Any] = {
        "feature_weights": torch.from_numpy(store.model_feature_weights()).to(device),
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
    }
    if context["joint_weight"] > 0.0 and context["foot_indices"] is None:
        raise ValueError("joint/foot losses require LeftToeBase and RightToeBase in the feature schema")
    return context


def build_loss_fn(representation: RepresentationProtocol, context: Mapping[str, Any], config: Mapping[str, object]):
    def compute(output: Mapping[str, Any], batch: Batch) -> dict[str, torch.Tensor]:
        motion = batch["motion"]
        if not isinstance(motion, torch.Tensor):
            raise TypeError("FeatureDataset batches must contain tensor motion")
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
            contact_weight=context["contact_weight"],
            foot_slide_weight=context["foot_slide_weight"],
            foot_height_weight=context["foot_height_weight"],
            contact_temperature=context["contact_temperature"],
            ref_pos=context["ref_pos"],
            parents=context["parents"],
            foot_indices=context["foot_indices"],
        )
        rep_batch = dict(context)
        rep_batch["motion"] = motion
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


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Runner metrics must be scalar")
        return float(value.detach().cpu())
    return float(value)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


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
        self.global_step = 0
        self.best_val: float | None = None
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("training.precision must be fp32 or amp")
        if self.precision == "amp" and device.type != "cuda":
            raise ValueError("AMP is only enabled for CUDA in the canonical runner")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "amp")

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

    def _forward(self, batch: Batch) -> dict[str, Any]:
        motion = batch["motion"].to(self.device, non_blocking=True)
        with self._autocast():
            return self.representation(motion, collect_metrics=True)

    def _record(self, output: Mapping[str, Any], batch: Batch, loss_values: Mapping[str, torch.Tensor]) -> dict[str, float]:
        result = {name: _scalar(value) for name, value in loss_values.items()}
        metrics = output.get("representation_metrics", {})
        if isinstance(metrics, Mapping):
            for name, value in metrics.items():
                if isinstance(value, (torch.Tensor, float, int)):
                    result[f"representation/{name}"] = _scalar(value)
        for name, callback in self.metric_suite.items():
            result[name] = _scalar(callback(output, batch))
        return result

    @staticmethod
    def _average(total: dict[str, float], count: int) -> dict[str, float]:
        return {name: value / max(count, 1) for name, value in total.items()}

    def evaluate(self, split: str) -> dict[str, Any]:
        if split not in {"val", "test"}:
            raise ValueError("evaluate split must be val or test")
        loader = self.val_loader if split == "val" else self.test_loader
        if loader is None:
            raise ValueError(f"No loader configured for {split}")
        self.representation.eval()
        total: dict[str, float] = {}
        count = 0
        with torch.inference_mode():
            for batch in loader:
                output = self._forward(batch)
                values = self.loss_fn(output, {"motion": batch["motion"].to(self.device)})
                record = self._record(output, batch, values)
                for name, value in record.items():
                    total[name] = total.get(name, 0.0) + value
                count += int(batch["motion"].shape[0])
        return {"mode": split, "metrics": self._average(total, count)}

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None or self.optimizer is None:
            raise ValueError("Training requires train_loader and optimizer")
        self.representation.train()
        total: dict[str, float] = {}
        count = 0
        started = time.perf_counter()
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            output = self._forward(batch)
            values = self.loss_fn(output, {"motion": batch["motion"].to(self.device)})
            loss = values["loss"]
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
            self.global_step += 1
            record = self._record(output, batch, values)
            for name, value in record.items():
                total[name] = total.get(name, 0.0) + value
            count += int(batch["motion"].shape[0])
            if self.writer is not None:
                self.writer.add_scalar("train/step_loss", record["loss"], self.global_step)
        if self.scheduler is not None:
            self.scheduler.step()
        result = self._average(total, count)
        result["samples_per_second"] = count / max(time.perf_counter() - started, 1e-8)
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
            train = self.train_epoch(epoch)
            val = self.evaluate("val")["metrics"] if self.val_loader is not None else {}
            record = {"epoch": epoch, "train": train, "val": val}
            history.append(record)
            val_loss = float(val.get("loss", train.get("loss", float("inf"))))
            is_best = self.best_val is None or val_loss < self.best_val
            if is_best:
                self.best_val = val_loss
            payload = self.checkpoint_payload(epoch, record)
            self.checkpoint_manager.save(payload, "last.pt")
            if is_best:
                self.checkpoint_manager.save(payload, "best.pt")
            if self.writer is not None:
                self.writer.add_scalar("epoch/train_loss", train.get("loss", 0.0), epoch)
                if val:
                    self.writer.add_scalar("epoch/val_loss", val.get("loss", 0.0), epoch)
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
    }
    if value not in mapping:
        raise ValueError(f"Unsupported --representation {value!r}")
    return mapping[value]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, validate, or test one canonical FSQ representation.")
    parser.add_argument("--workflow-mode", choices=["train", "validate", "test"], required=True)
    parser.add_argument("--representation", choices=["flat-fsq", "part-fsq", "residual-part-fsq", "latent-residual-fsq"], required=True)
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
    device = choose_device(args.device)
    feature_database = data.get("feature_database")
    if feature_database is None:
        raise ValueError("data.feature_database is required")
    store = build_feature_store(feature_database)
    feature_schema = store.feature_schema()
    loaders, datasets = build_dataloaders(config, store)
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
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
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
    writer = SummaryWriter(output / "tensorboard") if args.workflow_mode == "train" else None
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
    )
    try:
        result = runner.run(args.workflow_mode, split=args.split)
    finally:
        if writer is not None:
            writer.close()
    print(json.dumps(result, indent=2, default=str))


__all__ = ["RepresentationRunner", "build_cli_parser", "build_representation", "load_experiment_config", "main"]


if __name__ == "__main__":
    main()
