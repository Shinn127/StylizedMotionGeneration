"""Training loop for the style-free base transport.

Kept separate from the CLI so the acceptance criteria of Phase 2 — small-batch
overfit, masked cross-entropy, full-mask generation — are testable without a
dataset or a GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .contract import masked_cross_entropy
from .layout_adapter import LayoutAdapter
from .masking import MaskBatch, MaskGenerator
from .transport import MotionTransportTransformer


@dataclass
class TrainerConfig:
    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    steps_per_epoch: int | None = None
    max_steps: int | None = None
    val_every_steps: int = 0
    seed: int = 3407
    precision: str = "fp32"
    log_every_steps: int = 50
    output_dir: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "TrainerConfig":
        payload = dict(value or {})
        known = {field_name for field_name in cls().__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown training options {unknown}")
        config = cls(**payload)  # type: ignore[arg-type]
        if config.epochs <= 0 or config.lr <= 0:
            raise ValueError("training.epochs and training.lr must be positive")
        if config.precision not in {"fp32", "amp"}:
            raise ValueError("training.precision must be fp32 or amp")
        for name in ("steps_per_epoch", "max_steps"):
            value_ = getattr(config, name)
            if value_ is not None and int(value_) <= 0:
                raise ValueError(f"training.{name} must be positive")
        return config


@dataclass
class TransportMetrics:
    loss: float
    accuracy: float
    supervised_tokens: int
    hidden_fraction: float
    kind: str
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "supervised_tokens": self.supervised_tokens,
            "hidden_fraction": self.hidden_fraction,
            "mask_kind": self.kind,
            **self.extra,
        }


class TransportTrainer:
    """Optimizes the masked-token objective of one transport model."""

    def __init__(
        self,
        model: MotionTransportTransformer,
        *,
        adapter: LayoutAdapter,
        optimizer: torch.optim.Optimizer | None = None,
        mask_generator: MaskGenerator | None = None,
        device: torch.device | str = "cpu",
        config: TrainerConfig | None = None,
        writer: Any | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.device = torch.device(device)
        self.model.to(self.device)
        self.config = config or TrainerConfig()
        self.mask_generator = mask_generator or MaskGenerator()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.writer = writer
        self.global_step = 0
        self.amp = self.config.precision == "amp" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self._rng = torch.Generator(device="cpu").manual_seed(int(self.config.seed))

    # -- one step ---------------------------------------------------------
    def sample_mask(self, batch: int, frames: int) -> MaskBatch:
        return self.mask_generator.sample(
            batch, frames, adapter=self.adapter, generator=self._rng, device=self.device
        )

    def loss(
        self,
        tokens: torch.Tensor,
        mask: MaskBatch,
        *,
        valid_mask: torch.Tensor | None = None,
        content_condition: Any | None = None,
        extra_coordinate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransportMetrics]:
        output = self.model(
            tokens, mask.visible_mask, content_condition=content_condition, valid_mask=valid_mask
        )
        supervision = mask.supervision_mask.to(self.device)
        if extra_coordinate_mask is not None:
            supervision = supervision & extra_coordinate_mask.to(self.device).bool()
        loss = masked_cross_entropy(
            output.logits, tokens, valid_mask=valid_mask, coordinate_mask=supervision
        )
        with torch.no_grad():
            predicted = output.logits.argmax(dim=-1)
            correct = (predicted == tokens) & supervision
            if valid_mask is not None:
                correct = correct & valid_mask.to(self.device).bool().unsqueeze(-1)
            supervised = int(supervision.sum())
            accuracy = float(correct.sum()) / max(supervised, 1)
        return loss, TransportMetrics(
            loss=float(loss.detach()),
            accuracy=accuracy,
            supervised_tokens=supervised,
            hidden_fraction=float(mask.supervision_mask.float().mean()),
            kind=mask.kind,
        )

    def train_step(
        self,
        tokens: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        content_condition: Any | None = None,
        mask: MaskBatch | None = None,
    ) -> TransportMetrics:
        tokens = self.adapter.token_spec().validate_tokens(tokens).to(self.device)
        if mask is None:
            mask = self.sample_mask(tokens.shape[0], tokens.shape[1])
        if content_condition is not None:
            content_condition = self._to_device(content_condition)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(
            tokens, mask, valid_mask=valid_mask, content_condition=content_condition
        )
        if self.amp:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
        self.global_step += 1
        if self.writer is not None:
            self.writer.add_scalar("transport/train_loss", metrics.loss, self.global_step)
            self.writer.add_scalar("transport/train_accuracy", metrics.accuracy, self.global_step)
            self.writer.add_scalar("transport/hidden_fraction", metrics.hidden_fraction, self.global_step)
        return metrics

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Any]) -> dict[str, Any]:
        """Averages the objective over an iterable of ``(tokens, mask, valid)``."""
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for item in batches:
            tokens, mask, valid, condition = self._unpack(item)
            loss, metrics = self.loss(
                tokens, mask, valid_mask=valid, content_condition=condition
            )
            weight = max(metrics.supervised_tokens, 1)
            for key, value in (
                ("loss", float(loss.detach()) * weight),
                ("accuracy", metrics.accuracy * weight),
            ):
                totals[key] = totals.get(key, 0.0) + value
            count += weight
        if count == 0:
            return {"loss": float("nan"), "accuracy": float("nan"), "supervised_tokens": 0}
        return {
            "loss": totals["loss"] / count,
            "accuracy": totals["accuracy"] / count,
            "supervised_tokens": count,
        }

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return value

    def _unpack(self, item: Any) -> tuple[torch.Tensor, MaskBatch, Any, Any]:
        """Accepts a token tensor, a ``(tokens, valid)`` pair, or a dict."""
        if isinstance(item, torch.Tensor):
            tokens, valid, condition = item, None, None
        elif isinstance(item, Mapping):
            tokens = item["tokens"]
            valid = item.get("valid_mask")
            condition = item.get("content_condition")
        else:
            tokens, valid = item[0], item[1]
            condition = item[2] if len(item) > 2 else None
        tokens = self.adapter.token_spec().validate_tokens(tokens).to(self.device)
        mask = self.sample_mask(tokens.shape[0], tokens.shape[1])
        return tokens, mask, self._to_device(valid), self._to_device(condition)

    # -- epochs -----------------------------------------------------------
    def fit(
        self,
        train_batches: Callable[[int], Iterable[Any]],
        *,
        epochs: int | None = None,
        val_batches: Callable[[int], Iterable[Any]] | None = None,
        on_epoch_end: Callable[[int, Mapping[str, float], "TransportTrainer"], None] | None = None,
        log: Callable[[str], None] | None = print,
    ) -> dict[str, Any]:
        epochs = int(epochs or self.config.epochs)
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            totals: dict[str, float] = {}
            steps = 0
            for item in train_batches(epoch):
                if self.config.steps_per_epoch is not None and steps >= self.config.steps_per_epoch:
                    break
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break
                tokens, _, valid, condition = self._unpack(item)
                metrics = self.train_step(
                    tokens, valid_mask=valid, content_condition=condition
                )
                steps += 1
                for key, value in (
                    ("loss", metrics.loss),
                    ("accuracy", metrics.accuracy),
                    ("hidden_fraction", metrics.hidden_fraction),
                ):
                    totals[key] = totals.get(key, 0.0) + float(value)
                if (
                    log is not None
                    and self.config.log_every_steps
                    and self.global_step % self.config.log_every_steps == 0
                ):
                    log(
                        f"step {self.global_step}: loss={metrics.loss:.4f} "
                        f"acc={metrics.accuracy:.4f} mask={metrics.kind}"
                    )
            epoch_metrics = {
                key: value / max(steps, 1) for key, value in totals.items()
            }
            epoch_metrics["steps"] = float(steps)
            epoch_metrics["seconds"] = time.perf_counter() - started
            if val_batches is not None:
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in self.evaluate(val_batches(epoch)).items()}
                )
            history.append({"epoch": epoch, **epoch_metrics})
            if log is not None:
                log(
                    f"epoch {epoch}/{epochs}: train_loss={epoch_metrics.get('loss', float('nan')):.4f} "
                    f"val_loss={epoch_metrics.get('val_loss', float('nan')):.4f}"
                )
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_metrics, self)
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
        return {"history": history, "global_step": self.global_step}


class OperatorTrainer:
    """Optimizes the reference-conditioned operator objective (Phase 4).

    The transport and (optionally) the style encoder are frozen upstream, so the
    gradient that reaches the operator is the style signal plus the masked
    target-token likelihood — nothing else.
    """

    def __init__(
        self,
        model: Any,
        *,
        adapter: LayoutAdapter,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
        config: TrainerConfig | None = None,
        content_weight: float = 0.0,
        writer: Any | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.device = torch.device(device)
        self.model.to(self.device)
        self.config = config or TrainerConfig()
        self.content_weight = float(content_weight)
        self.writer = writer
        parameters = list(model.trainable_parameters())
        if not parameters:
            raise ValueError("The operator model exposes no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            parameters, lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.global_step = 0
        self.amp = self.config.precision == "amp" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _to_device(self, batch: Any) -> Any:
        from .model import OperatorBatch

        if isinstance(batch, OperatorBatch):
            payload = {}
            for name in batch.__dataclass_fields__:
                value = getattr(batch, name)
                payload[name] = value.to(self.device) if isinstance(value, torch.Tensor) else value
            return OperatorBatch(**payload)
        return batch

    def train_step(self, batch: Any) -> dict[str, float]:
        batch = self._to_device(batch)
        self.model.train()
        if self.model.freeze_transport:
            self.model.transport.eval()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.model.loss(batch, content_weight=self.content_weight)
        if self.amp:
            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), self.config.grad_clip_norm)
            self.optimizer.step()
        self.global_step += 1
        record = {"loss": float(loss.detach())}
        record.update({name: float(value) for name, value in metrics.items() if torch.is_tensor(value)})
        if self.writer is not None:
            for name, value in record.items():
                self.writer.add_scalar(f"operator/train/{name}", value, self.global_step)
        return record

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Any]) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in batches:
            loss, metrics = self.model.loss(self._to_device(batch), content_weight=self.content_weight)
            weight = float(metrics.get("supervision_fraction", 1.0))
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * weight
            totals["nll"] = totals.get("nll", 0.0) + float(metrics["nll"]) * weight
            count += 1
        if count == 0:
            return {"loss": float("nan"), "nll": float("nan"), "batches": 0}
        return {
            "loss": totals["loss"] / count,
            "nll": totals["nll"] / count,
            "batches": float(count),
        }

    def fit(
        self,
        train_batches: Callable[[int], Iterable[Any]],
        *,
        epochs: int | None = None,
        val_batches: Callable[[int], Iterable[Any]] | None = None,
        on_epoch_end: Callable[[int, Mapping[str, float], "OperatorTrainer"], None] | None = None,
        log: Callable[[str], None] | None = print,
    ) -> dict[str, Any]:
        epochs = int(epochs or self.config.epochs)
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            totals: dict[str, float] = {}
            steps = 0
            for batch in train_batches(epoch):
                if self.config.steps_per_epoch is not None and steps >= self.config.steps_per_epoch:
                    break
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break
                record = self.train_step(batch)
                steps += 1
                for key, value in record.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                if log is not None and self.config.log_every_steps and self.global_step % self.config.log_every_steps == 0:
                    log(f"step {self.global_step}: loss={record['loss']:.4f} nll={record.get('nll', float('nan')):.4f}")
            epoch_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
            epoch_metrics["steps"] = float(steps)
            epoch_metrics["seconds"] = time.perf_counter() - started
            if val_batches is not None:
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in self.evaluate(val_batches(epoch)).items()}
                )
            history.append({"epoch": epoch, **epoch_metrics})
            if log is not None:
                log(
                    f"epoch {epoch}/{epochs}: train_loss={epoch_metrics.get('loss', float('nan')):.4f} "
                    f"val_nll={epoch_metrics.get('val_nll', float('nan')):.4f}"
                )
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_metrics, self)
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
        return {"history": history, "global_step": self.global_step}


__all__ = [
    "OperatorTrainer",
    "TrainerConfig",
    "TransportMetrics",
    "TransportTrainer",
]
