"""Training loop for the style-free base transport.

Kept separate from the CLI so the acceptance criteria of Phase 2 — small-batch
overfit, masked cross-entropy, full-mask generation — are testable without a
dataset or a GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .contract import masked_cross_entropy
from .layout_adapter import LayoutAdapter
from .masking import MaskBatch, MaskGenerator
from .transport import MotionTransportTransformer


def planned_step_budget(config: "TrainerConfig", *, epochs: int | None = None) -> int | None:
    """The optimizer steps a run states it will take, or ``None`` when unpinned.

    ``max_steps`` wins; otherwise ``epochs x steps_per_epoch``.  A run with
    neither has no budget: its length would be whatever the loader happens to
    serve, which is how a "40 epoch" recipe turned out to mean a different number
    of steps on every machine.
    """
    if config.max_steps is not None:
        return int(config.max_steps)
    if config.steps_per_epoch is not None:
        return int(epochs if epochs is not None else config.epochs) * int(config.steps_per_epoch)
    return None


def resolve_budget(config: "TrainerConfig", *, where: str = "run") -> dict[str, Any]:
    """The run's explicit optimizer-step budget, or a refusal.

    Training requires one of the two explicit forms, and the achieved count is
    checked against it afterwards (``fit`` returns ``steps_shortfall``).
    """
    planned = planned_step_budget(config)
    if planned is None:
        raise ValueError(
            f"{where}: training.max_steps (or training.steps_per_epoch) must be an explicit "
            "positive integer before training: the optimizer-step budget may not be derived "
            "from how many batches the loader happens to serve"
        )
    return {
        "planned_steps": int(planned),
        "source": "training.max_steps"
        if config.max_steps is not None
        else "training.epochs x training.steps_per_epoch",
        "epochs": int(config.epochs),
        "steps_per_epoch": None if config.steps_per_epoch is None else int(config.steps_per_epoch),
        "max_steps": None if config.max_steps is None else int(config.max_steps),
    }


#: Files that mean "a run already wrote here".  A second run must not overwrite
#: them silently; the protocol/README a dry run leaves behind are not runs.
RUN_ARTIFACTS = (
    "best.pt",
    "last.pt",
    "init.pt",
    "overfit_last.pt",
    "train_summary.json",
    "history.jsonl",
)


def refuse_existing_output(output: Any, *, allow: bool, where: str) -> list[str]:
    """Refuses to start when ``output`` already holds a run's artifacts.

    The exit code is numeric so an automation gets a status, not a string.
    """
    from pathlib import Path

    output = Path(output)
    existing = [str(output / name) for name in RUN_ARTIFACTS if (output / name).exists()]
    if existing and not allow:
        import sys

        print(
            f"{where}: {output} already holds a run ({existing}). Nothing overwrites an existing "
            "run: pick a new directory, or pass --allow-existing-output for a deliberate scratch "
            "rerun.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    return existing


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
    #: Learning rate of the style encoder when it is trainable (N05b's low-LR
    #: fine-tuning ablation).  ``None`` keeps the historical behaviour: one group
    #: at ``lr`` for everything.
    style_encoder_lr: float | None = None
    #: Weight of the auxiliary seen-style cross-entropy on the descriptor (the
    #: "one auxiliary CE term" ablation).  ``0.0`` keeps it off; the term trains a
    #: linear head on ``style_embedding`` against the reference's own style id and
    #: is added to the loss inside :meth:`OperatorTrainer.train_step`.
    aux_style_ce_weight: float = 0.0
    #: How many classes the auxiliary head reads (the reference styles).
    aux_style_classes: int = 3

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> TrainerConfig:
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
    nll_sum: float = 0.0
    correct_tokens: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "nll_sum": self.nll_sum,
            "accuracy": self.accuracy,
            "correct_tokens": self.correct_tokens,
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
        nll_sum = masked_cross_entropy(
            output.logits,
            tokens,
            valid_mask=valid_mask,
            coordinate_mask=supervision,
            reduction="sum",
        )
        with torch.no_grad():
            supervised_mask = supervision.clone()
            if valid_mask is not None:
                supervised_mask = supervised_mask & valid_mask.to(self.device).bool().unsqueeze(-1)
            predicted = output.logits.argmax(dim=-1)
            correct_tokens = int(((predicted == tokens) & supervised_mask).sum())
            supervised = int(supervised_mask.sum())
        return loss, TransportMetrics(
            loss=float(loss.detach()),
            nll_sum=float(nll_sum.detach()),
            correct_tokens=correct_tokens,
            accuracy=correct_tokens / supervised if supervised else 0.0,
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
        """Evaluates on fixed masks, aggregating by supervised token count.

        Total NLL over total supervised tokens, never an average of batch means:
        a batch-level mean over a 5% supervision fraction used to report 0.1 for a
        per-token NLL of 2.  An all-empty input reports ``None`` plus a count.
        """
        self.model.eval()
        nll_sum = 0.0
        correct = 0
        supervised = 0
        batches_seen = 0
        for item in batches:
            tokens, mask, valid, condition = self._unpack(item)
            _, metrics = self.loss(tokens, mask, valid_mask=valid, content_condition=condition)
            nll_sum += metrics.nll_sum
            correct += metrics.correct_tokens
            supervised += metrics.supervised_tokens
            batches_seen += 1
        if supervised == 0:
            # An all-empty supervision set reports a count, never a flattering loss.
            return {"loss": None, "accuracy": None, "supervised_tokens": 0, "batches": batches_seen}
        return {
            "loss": nll_sum / supervised,
            "accuracy": correct / supervised,
            "nll_sum": nll_sum,
            "correct_tokens": correct,
            "supervised_tokens": supervised,
            "batches": batches_seen,
        }

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return value

    def _unpack(self, item: Any) -> tuple[torch.Tensor, MaskBatch, Any, Any]:
        """Accepts a token tensor, a ``(tokens, valid)`` pair, or a dict.

        A mapping may carry an explicit ``mask`` or ``visible_mask``: a frozen
        validation protocol scores a *stated* mask, so re-sampling one here would
        make the number move between epochs for no reason.
        """
        explicit: MaskBatch | None = None
        if isinstance(item, torch.Tensor):
            tokens, valid, condition = item, None, None
        elif isinstance(item, Mapping):
            tokens = item["tokens"]
            valid = item.get("valid_mask")
            condition = item.get("content_condition")
            candidate = item.get("mask")
            if candidate is not None:
                explicit = candidate
            elif item.get("visible_mask") is not None:
                explicit = MaskBatch(
                    visible_mask=item["visible_mask"].to(self.device).bool(),
                    kind=str(item.get("kind", "explicit")),
                    config=dict(item.get("mask_config") or {}),
                )
        else:
            tokens, valid = item[0], item[1]
            condition = item[2] if len(item) > 2 else None
        tokens = self.adapter.token_spec().validate_tokens(tokens).to(self.device)
        mask = explicit if explicit is not None else self.sample_mask(tokens.shape[0], tokens.shape[1])
        if explicit is not None and tuple(mask.visible_mask.shape) != tuple(tokens.shape):
            raise ValueError(
                f"An explicit mask must match the tokens {tuple(tokens.shape)}, got "
                f"{tuple(mask.visible_mask.shape)}"
            )
        return tokens, mask, self._to_device(valid), self._to_device(condition)

    # -- epochs -----------------------------------------------------------
    def fit(
        self,
        train_batches: Callable[[int], Iterable[Any]],
        *,
        epochs: int | None = None,
        val_batches: Callable[[int], Iterable[Any]] | None = None,
        monitor_batches: Callable[[int], Iterable[Any]] | None = None,
        on_epoch_end: Callable[[int, Mapping[str, float], TransportTrainer], None] | None = None,
        max_seconds: float | None = None,
        log: Callable[[str], None] | None = print,
    ) -> dict[str, Any]:
        """Runs epochs of masked-token training.

        ``val_batches`` is the held-out protocol and may select a best
        checkpoint.  ``monitor_batches`` is a *training* diagnostic (an overfit
        run watching its own frozen windows): its numbers are recorded as
        ``monitor_*`` and are never named ``val_*``, so a training number cannot
        be mistaken for a generalization number.

        ``max_seconds`` is an independent wall-clock cap: when it runs out the
        loop stops and the result says ``interrupted``, so a run that was cut
        short never claims to have finished its budget.
        """
        epochs = int(epochs or self.config.epochs)
        wall_deadline = None if max_seconds is None else time.perf_counter() + float(max_seconds)
        interrupted = False
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            totals: dict[str, float] = {}
            counts = {"nll_sum": 0.0, "supervised_tokens": 0, "correct_tokens": 0}
            steps = 0
            for item in train_batches(epoch):
                if wall_deadline is not None and time.perf_counter() >= wall_deadline:
                    interrupted = True
                    break
                if self.config.steps_per_epoch is not None and steps >= self.config.steps_per_epoch:
                    break
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break
                tokens, _, valid, condition = self._unpack(item)
                metrics = self.train_step(
                    tokens, valid_mask=valid, content_condition=condition
                )
                steps += 1
                counts["nll_sum"] += metrics.nll_sum
                counts["supervised_tokens"] += metrics.supervised_tokens
                counts["correct_tokens"] += metrics.correct_tokens
                for key, value in (
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
            # The epoch loss is total NLL over total supervised tokens, so a step
            # with few supervised tokens cannot weigh as much as a dense one.
            epoch_metrics["loss"] = (
                counts["nll_sum"] / counts["supervised_tokens"]
                if counts["supervised_tokens"]
                else None
            )
            epoch_metrics.update({f"{name}_total": value for name, value in counts.items()})
            epoch_metrics["steps"] = float(steps)
            epoch_metrics["seconds"] = time.perf_counter() - started
            if val_batches is not None:
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in self.evaluate(val_batches(epoch)).items()}
                )
            if monitor_batches is not None:
                epoch_metrics.update(
                    {
                        f"monitor_{key}": value
                        for key, value in self.evaluate(monitor_batches(epoch)).items()
                    }
                )
            epoch_metrics["optimizer_steps"] = int(self.global_step)
            epoch_metrics["global_step"] = int(self.global_step)
            if on_epoch_end is not None:
                # The callback may return the metrics it computed (the frozen
                # protocol's objective, its timings).  They are merged *before*
                # the history entry is written, so the log, the history and the
                # checkpoint carry the same numbers -- and the callback is the
                # only place validation runs.
                returned = on_epoch_end(epoch, epoch_metrics, self)
                if isinstance(returned, Mapping):
                    epoch_metrics.update(returned)
            history.append({"epoch": epoch, **epoch_metrics})
            if log is not None:
                train_loss = epoch_metrics.get("loss")
                val_loss = epoch_metrics.get("val_loss", epoch_metrics.get("val_objective"))
                # An epoch cut short before its first step has no loss at all, and
                # printing "nan" there would look like a measured value.
                log(
                    f"epoch {epoch}/{epochs}: "
                    f"train_loss={'n/a' if train_loss is None else f'{train_loss:.4f}'} "
                    f"val_loss={'n/a' if val_loss is None else f'{val_loss:.4f}'}"
                )
            if interrupted:
                break
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
        planned_steps = planned_step_budget(self.config, epochs=epochs)
        return {
            "history": history,
            "global_step": self.global_step,
            "planned_steps": planned_steps,
            "steps_shortfall": None
            if planned_steps is None
            else max(0, int(planned_steps) - int(self.global_step)),
            "interrupted": bool(interrupted),
            "interrupt_reason": "wall_time_cap" if interrupted else None,
        }


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
        self.style_encoder_lr = (
            None if self.config.style_encoder_lr is None else float(self.config.style_encoder_lr)
        )
        if optimizer is None and self.style_encoder_lr is not None and model.style_encoder is not None:
            encoder_ids = {id(parameter) for parameter in model.style_encoder.parameters()}
            encoder_group = [p for p in parameters if id(p) in encoder_ids]
            rest = [p for p in parameters if id(p) not in encoder_ids]
            groups = []
            if rest:
                groups.append({"params": rest, "lr": self.config.lr})
            if encoder_group:
                groups.append({"params": encoder_group, "lr": self.style_encoder_lr})
            self.optimizer = torch.optim.AdamW(
                groups, lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer or torch.optim.AdamW(
                parameters, lr=self.config.lr, weight_decay=self.config.weight_decay
            )
        # The auxiliary seen-style head: one linear readout on the descriptor, trained
        # only when ``aux_style_ce_weight`` is positive.  It exists so the ablation can
        # add *one* term instead of a contrastive/adversarial family.
        self.aux_style_ce_weight = float(self.config.aux_style_ce_weight)
        self.style_head: torch.nn.Module | None = None
        if self.aux_style_ce_weight > 0.0:
            output_dim = int(getattr(model.style_encoder, "output_dim", 0) or 0)
            if output_dim <= 0:
                raise ValueError(
                    "training.aux_style_ce_weight needs a style encoder with an output_dim"
                )
            self.style_head = torch.nn.Linear(output_dim, int(self.config.aux_style_classes)).to(
                self.device
            )
            self.optimizer.add_param_group(
                {"params": self.style_head.parameters(), "lr": self.config.lr}
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
        # N05b's "one auxiliary CE term": a linear readout on the descriptor against
        # the reference's own style id, added to the loss.  The term only exists when
        # the config asks for it and the batch carries reference style ids, so a
        # validation batch (which never does) is unaffected.
        if self.aux_style_ce_weight > 0.0 and self.style_head is not None:
            reference_style_ids = getattr(batch, "reference_style_ids", None)
            embedding = metrics.get("style_embedding")
            if reference_style_ids is None or embedding is None:
                raise ValueError(
                    "training.aux_style_ce_weight is set but the batch carries no reference_style_ids; "
                    "the paired source must fill them or the term cannot be computed"
                )
            aux = torch.nn.functional.cross_entropy(
                self.style_head(embedding), reference_style_ids.to(self.device)
            )
            loss = loss + self.aux_style_ce_weight * aux
            metrics["aux_style_ce"] = aux.detach()
        supervised = int(metrics.get("supervised_tokens", 0))
        if supervised == 0:
            # Nothing to predict in this batch: a zero step would report progress
            # that never happened, so it is skipped and counted instead.
            self.skipped_steps = int(getattr(self, "skipped_steps", 0)) + 1
            return {"loss": None, "supervised_tokens": 0.0, "skipped": 1.0}
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
        record = {
            "loss": float(loss.detach()),
            "supervised_tokens": float(supervised),
            "correct_tokens": float(metrics.get("correct_tokens", 0)),
            "skipped": 0.0,
        }
        record.update(
            {
                name: float(value)
                for name, value in metrics.items()
                if torch.is_tensor(value)
                and name not in {"supervised_tokens", "correct_tokens", "style_embedding"}
            }
        )
        if self.writer is not None:
            for name, value in record.items():
                self.writer.add_scalar(f"operator/train/{name}", value, self.global_step)
        return record

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Any]) -> dict[str, float]:
        """Evaluates on fixed masks, aggregated by supervised token count.

        ``loss`` is total NLL over total supervised tokens, never an average of
        batch means (a batch mean over a 5% supervision fraction reported 0.1 for a
        per-token NLL of 2).  An input with no supervision reports ``None`` plus
        counts, and the caller must not treat that as a best metric.
        """
        self.model.eval()
        nll_sum = 0.0
        correct = 0
        supervised_total = 0
        batches_seen = 0
        for batch in batches:
            _, metrics = self.model.loss(self._to_device(batch), content_weight=self.content_weight)
            nll_sum += float(metrics.get("nll_sum", 0.0))
            correct += int(metrics.get("correct_tokens", 0))
            supervised_total += int(metrics.get("supervised_tokens", 0))
            batches_seen += 1
        if supervised_total == 0:
            return {
                "loss": None,
                "nll": None,
                "accuracy": None,
                "supervised_tokens": 0,
                "batches": batches_seen,
            }
        return {
            "loss": nll_sum / supervised_total,
            "nll": nll_sum / supervised_total,
            "accuracy": correct / supervised_total,
            "nll_sum": nll_sum,
            "correct_tokens": correct,
            "supervised_tokens": supervised_total,
            "batches": batches_seen,
        }

    def fit(
        self,
        train_batches: Callable[[int], Iterable[Any]],
        *,
        epochs: int | None = None,
        val_batches: Callable[[int], Iterable[Any]] | None = None,
        monitor_batches: Callable[[int], Iterable[Any]] | None = None,
        on_epoch_end: Callable[[int, Mapping[str, float], OperatorTrainer], None] | None = None,
        max_seconds: float | None = None,
        log: Callable[[str], None] | None = print,
    ) -> dict[str, Any]:
        """Runs epochs of masked-token training.

        ``val_batches`` is the held-out protocol and may select a best checkpoint;
        ``monitor_batches`` is a training diagnostic recorded as ``monitor_*``.  A
        callback may return metrics to merge, and validation runs exactly once per
        epoch (in the callback), so the history and the checkpoint cannot disagree
        about what was measured.  ``max_seconds`` is an independent wall-clock cap:
        an interrupted run says so instead of claiming its budget was met.
        """
        epochs = int(epochs or self.config.epochs)
        wall_deadline = None if max_seconds is None else time.perf_counter() + float(max_seconds)
        interrupted = False
        history: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            totals: dict[str, float] = {}
            counts = {"nll_sum": 0.0, "supervised_tokens": 0.0, "correct_tokens": 0.0, "skipped": 0.0}
            steps = 0
            for batch in train_batches(epoch):
                if wall_deadline is not None and time.perf_counter() >= wall_deadline:
                    interrupted = True
                    break
                if self.config.steps_per_epoch is not None and steps >= self.config.steps_per_epoch:
                    break
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    break
                record = self.train_step(batch)
                steps += 1
                for key, value in record.items():
                    if value is None:
                        continue
                    if key in counts:
                        counts[key] += float(value)
                    else:
                        totals[key] = totals.get(key, 0.0) + float(value)
                if (
                    log is not None
                    and self.config.log_every_steps
                    and record["loss"] is not None
                    and self.global_step % self.config.log_every_steps == 0
                ):
                    log(
                        f"step {self.global_step}: loss={record['loss']:.4f} "
                        f"nll={record.get('nll', float('nan')):.4f}"
                    )
            epoch_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
            # Exact token-weighted epoch loss plus the counts it came from.
            epoch_metrics["loss"] = (
                counts["nll_sum"] / counts["supervised_tokens"]
                if counts["supervised_tokens"]
                else None
            )
            epoch_metrics["skipped_steps"] = counts["skipped"]
            epoch_metrics["supervised_tokens"] = counts["supervised_tokens"]
            epoch_metrics["correct_tokens"] = counts["correct_tokens"]
            epoch_metrics["steps"] = float(steps)
            epoch_metrics["seconds"] = time.perf_counter() - started
            if val_batches is not None:
                epoch_metrics.update(
                    {f"val_{key}": value for key, value in self.evaluate(val_batches(epoch)).items()}
                )
            if monitor_batches is not None:
                epoch_metrics.update(
                    {
                        f"monitor_{key}": value
                        for key, value in self.evaluate(monitor_batches(epoch)).items()
                    }
                )
            epoch_metrics["optimizer_steps"] = int(self.global_step)
            epoch_metrics["global_step"] = int(self.global_step)
            if on_epoch_end is not None:
                returned = on_epoch_end(epoch, epoch_metrics, self)
                if isinstance(returned, Mapping):
                    epoch_metrics.update(returned)
            history.append({"epoch": epoch, **epoch_metrics})
            if log is not None:
                train_loss = epoch_metrics.get("loss")
                val_nll = epoch_metrics.get("val_nll", epoch_metrics.get("val_objective"))
                skipped = int(epoch_metrics.get("skipped_steps", 0.0))
                log(
                    f"epoch {epoch}/{epochs}: "
                    f"train_loss={'n/a' if train_loss is None else f'{train_loss:.4f}'} "
                    f"val_nll={'n/a' if val_nll is None else f'{val_nll:.4f}'} "
                    f"skipped={skipped}"
                )
            if interrupted:
                break
            if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                break
        planned_steps = planned_step_budget(self.config, epochs=epochs)
        return {
            "history": history,
            "global_step": self.global_step,
            "planned_steps": planned_steps,
            "steps_shortfall": None
            if planned_steps is None
            else max(0, int(planned_steps) - int(self.global_step)),
            "interrupted": bool(interrupted),
            "interrupt_reason": "wall_time_cap" if interrupted else None,
        }


__all__ = [
    "OperatorTrainer",
    "RUN_ARTIFACTS",
    "TrainerConfig",
    "TransportMetrics",
    "TransportTrainer",
    "planned_step_budget",
    "refuse_existing_output",
    "resolve_budget",
]
