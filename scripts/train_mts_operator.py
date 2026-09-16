#!/usr/bin/env python
"""Train the reference-conditioned style operator (MTS-FSQ plan, Phase 4).

    python scripts/train_mts_operator.py \
      --config data/configs/mts_operator_style.yaml \
      --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
      --transport-checkpoint outputs/mts_transport/seed3407/best.pt \
      --operator birth_death \
      --output outputs/mts_operator/birth_death/seed3407

The tokenizer and the transport stay frozen; the style encoder and the operator
learn.  Reference clips come from the audited style split, so zero-shot styles
never enter parameter learning.  ``--overfit-pairs N`` freezes a handful of
pairs and trains on them repeatedly, which is the Phase 3 sanity check before a
full run.

Report the correct / wrong / random reference differences from
``scripts/evaluate_mts_operator.py``; this script only trains.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store, open_token_store  # noqa: E402
from stylized_motion.learning.mts_operator import (  # noqa: E402
    LayoutAdapter,
    MaskGenerator,
    MotionTransportTransformer,
    OperatorBatch,
    build_operator,
    load_mts_checkpoint,
    mts_checkpoint_payload,
    save_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator  # noqa: E402
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    StylePairSampler,
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.style_encoder import (  # noqa: E402
    GlobalStyleEncoder,
    StyleIDEncoder,
)
from stylized_motion.learning.mts_operator.training import OperatorTrainer, TrainerConfig  # noqa: E402
from stylized_motion.learning.mts_operator.windows import read_window_tokens, windows_by_clip  # noqa: E402
from stylized_motion.learning.representation import (  # noqa: E402
    NEF_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import choose_device, set_seed  # noqa: E402

REFERENCE_ENCODER_KEYS = (
    "dim",
    "depth",
    "heads",
    "dropout",
    "graph_depth",
    "temporal_mode",
    "output_dim",
    "pooling",
)
STYLE_ID_ENCODER_KEYS = ("num_styles", "output_dim", "dim")
# Options every family accepts, plus the kind-specific ones.  A config may
# carry both (so `--operator` can switch families), but only the chosen family's
# keys are passed on: a typo still fails because the *union* is validated.
COMMON_OPERATOR_KEYS = ("hidden_dim", "coordinate_dim")
OPERATOR_SPECIFIC_KEYS = {
    "logit_field": (),
    "arbitrary_kernel": ("identity_mix",),
    "birth_death": ("max_rate", "uniformization_tolerance", "max_terms"),
}
OPERATOR_KEYS = COMMON_OPERATOR_KEYS + tuple(
    key for keys in OPERATOR_SPECIFIC_KEYS.values() for key in keys
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MTS style operator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--transport-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-database", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--operator", choices=["logit_field", "arbitrary_kernel", "birth_death"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overfit-pairs", type=int, default=0,
        help="Freeze this many reference/target pairs and train on them repeatedly.",
    )
    return parser


def load_operator_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Operator config must be a mapping: {path}")
    required = {"tokenizer", "transport", "style_encoder", "operator", "data", "masking", "training"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Operator config is missing sections: {missing}")
    unknown = sorted(set(value) - required - {"sampling", "loader", "evaluation"})
    if unknown:
        raise ValueError(f"Unknown operator config sections: {unknown}")
    return dict(value)


class PairedBatchSource:
    """Builds reference-conditioned batches from audited style pairs."""

    def __init__(
        self,
        *,
        store: Any,
        sampler: StylePairSampler,
        windows_by_clip: Mapping[int, list[Any]],
        tokenizer: Any | None,
        feature_stats: Mapping[str, object] | None,
        adapter: LayoutAdapter,
        mask_generator: MaskGenerator,
        device: torch.device,
        batch_size: int,
        frames: int,
        history: int,
        mode: str = "same_style",
        stage: str = "train",
        strength: tuple[float, float] | None = None,
        seed: int = 3407,
    ) -> None:
        self.store = store
        self.sampler = sampler
        self.windows_by_clip = windows_by_clip
        self.tokenizer = tokenizer
        self.feature_stats = feature_stats
        self.adapter = adapter
        self.mask_generator = mask_generator
        self.device = device
        self.batch_size = int(batch_size)
        self.frames = int(frames)
        self.history = int(history)
        self.mode = str(mode)
        self.stage = str(stage)
        self.strength = strength
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._shards: dict[int, Any] = {}

    def _tokens(self, request: Any) -> torch.Tensor:
        return read_window_tokens(
            self.store,
            request,
            frames=self.frames,
            history=self.history,
            tokenizer=self.tokenizer,
            feature_stats=self.feature_stats,
            shards=self._shards,
        )

    def _pick_window(self, clip_id: int) -> Any | None:
        candidates = self.windows_by_clip.get(int(clip_id))
        if not candidates:
            return None
        index = int(self.rng.integers(len(candidates)))
        return candidates[index]

    def batch(self, *, size: int | None = None) -> OperatorBatch | None:
        size = self.batch_size if size is None else int(size)
        pairs = self.sampler.sample(
            count=size, mode=self.mode, stage=self.stage, generator=self.rng
        )
        if not pairs:
            return None
        targets, references = [], []
        for pair in pairs:
            target_window = self._pick_window(pair.target.clip_id)
            reference_window = self._pick_window(pair.reference.clip_id)
            if target_window is None or reference_window is None:
                continue
            targets.append(self._tokens(target_window))
            references.append(self._tokens(reference_window))
        if not targets:
            return None
        target_tokens = torch.stack(targets)
        reference_tokens = torch.stack(references)
        batch_size, frames = target_tokens.shape[:2]
        mask = self.mask_generator.sample(
            batch_size,
            frames,
            adapter=self.adapter,
            generator=self.generator,
            device=self.device,
        )
        strength = None
        if self.strength is not None:
            low, high = self.strength
            values = torch.from_numpy(
                self.rng.uniform(low, high, size=batch_size).astype(np.float32)
            )
            strength = values
        return OperatorBatch(
            target_tokens=target_tokens.to(self.device),
            reference_tokens=reference_tokens.to(self.device),
            visible_mask=mask.visible_mask,
            target_valid_mask=torch.ones(batch_size, frames, dtype=torch.bool, device=self.device),
            strength=1.0 if strength is None else strength.to(self.device),
            kind=mask.kind,
        )

    def batches(self, count: int) -> Iterable[OperatorBatch]:
        for _ in range(int(count)):
            batch = self.batch()
            if batch is not None:
                yield batch


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_operator_config(args.config)
    training = dict(config["training"])
    content_weight = float(training.pop("content_weight", 0.0))
    strength_range = training.pop("strength_range", None)
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.steps_per_epoch is not None:
        training["steps_per_epoch"] = args.steps_per_epoch
    if args.seed is not None:
        training["seed"] = args.seed
    trainer_config = TrainerConfig.from_mapping(training)
    set_seed(trainer_config.seed, deterministic=False)
    device = choose_device(args.device)

    tokenizer_path = args.tokenizer_checkpoint or config["tokenizer"].get("checkpoint")
    if tokenizer_path is None:
        raise ValueError("tokenizer.checkpoint or --tokenizer-checkpoint is required")
    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(Path(tokenizer_path), torch.device("cpu"))
    if tokenizer.family != NEF_FSQ_FAMILY:
        raise ValueError(f"The MTS operator requires a {NEF_FSQ_FAMILY!r} tokenizer")
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    layout = tokenizer.token_layout()
    if layout is None:
        raise ValueError("NEF tokenizer did not expose a token layout")
    adapter = LayoutAdapter(layout, num_levels=int(tokenizer.num_levels))
    tokenizer_metadata = tokenizer.representation_metadata()
    token_spec = adapter.token_spec(representation_id=tokenizer.representation_id)

    transport_path = args.transport_checkpoint or config["transport"].get("checkpoint")
    if transport_path is None:
        raise ValueError("transport.checkpoint or --transport-checkpoint is required")
    _, transport = load_mts_checkpoint(
        Path(transport_path),
        kind="transport",
        build_model=lambda stored: MotionTransportTransformer(adapter, **stored),
        device=device,
        token_spec=token_spec,
        tokenizer_metadata=tokenizer_metadata,
    )
    transport.eval()

    # Unset YAML keys are dropped, and each kind only accepts its own options,
    # so a config can carry both kinds' keys without confusing either one.
    style_config = {key: value for key, value in dict(config["style_encoder"]).items() if value is not None}
    kind = str(style_config.pop("kind", "reference"))
    allowed = REFERENCE_ENCODER_KEYS if kind == "reference" else STYLE_ID_ENCODER_KEYS
    unknown = sorted(set(style_config) - set(allowed) - {"num_styles"} if kind == "reference" else set(style_config) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown style_encoder options {unknown} for kind {kind!r}")
    if kind == "reference":
        style_encoder = GlobalStyleEncoder(adapter, **style_config)
    elif kind == "style_id":
        if "num_styles" not in style_config:
            raise ValueError("style_encoder.kind=style_id requires num_styles")
        style_encoder = StyleIDEncoder(
            num_styles=int(style_config["num_styles"]),
            output_dim=int(style_config.get("output_dim") or style_config.get("dim") or 256),
        )
    else:
        raise ValueError(f"Unknown style_encoder.kind {kind!r}; expected reference or style_id")

    operator_config = dict(config["operator"])
    # Pop before choosing: `args.operator or config.pop(...)` would skip the pop
    # whenever the CLI flag is set, leaving `name` in the constructor kwargs.
    configured_operator = str(operator_config.pop("name", "birth_death"))
    operator_name = args.operator or configured_operator
    operator_config.pop("strength", None)
    unknown = sorted(set(operator_config) - set(OPERATOR_KEYS))
    if unknown:
        raise ValueError(f"Unknown operator options {unknown}")
    accepted = set(COMMON_OPERATOR_KEYS) | set(OPERATOR_SPECIFIC_KEYS.get(operator_name, ()))
    ignored = sorted(set(operator_config) - accepted)
    if ignored:
        print(f"ignoring options that {operator_name!r} does not accept: {ignored}", flush=True)
    operator = build_operator(
        operator_name,
        num_levels=adapter.num_levels,
        stream_dim=int(transport.dim),
        **{key: value for key, value in operator_config.items() if key in accepted},
    )
    model = MtsStyleOperator(
        adapter,
        transport=transport,
        style_encoder=style_encoder,
        operator=operator,
        freeze_transport=bool(config["transport"].get("freeze", True)),
        freeze_style_encoder=False,
    )

    store_path = args.token_store or config["data"].get("token_store")
    feature_path = args.feature_database or config["data"].get("fsq_window_index")
    if store_path:
        store = open_token_store(Path(store_path))
    elif feature_path:
        store = open_any_feature_store(feature_path)
    else:
        raise ValueError("data.token_store or data.fsq_window_index is required")

    style_split_config = dict(config["data"].get("style_split") or {})
    records = clip_records_from_store(store)
    style_split = split_styles_by_performer(
        records,
        val_fraction=float(style_split_config.get("val_fraction", 0.2)),
        unseen_fraction=float(style_split_config.get("unseen_fraction", 0.2)),
        seed=trainer_config.seed,
    )
    sampler = StylePairSampler(records, style_split=style_split, seed=trainer_config.seed)
    pair_config = dict(config["data"].get("pairs") or {})
    loader_config = dict(config.get("loader") or {})
    frames = int(config["data"].get("frames", 64))

    def windows_for(split: str) -> dict[int, list[Any]]:
        return windows_by_clip(store, split, frames=frames)

    history = int(tokenizer.history_frames)
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    sources = {
        split: PairedBatchSource(
            store=store,
            sampler=sampler,
            windows_by_clip=windows_for(split),
            tokenizer=None if store_path else tokenizer,
            feature_stats=None if store_path else feature_stats,
            adapter=adapter,
            mask_generator=MaskGenerator(dict(config["masking"])),
            device=device,
            batch_size=int(loader_config.get("batch_size", 32)),
            frames=frames,
            history=history,
            mode=str(pair_config.get("mode", "same_style")),
            stage=split,
            strength=tuple(strength_range) if strength_range else None,
            seed=trainer_config.seed,
        )
        for split in ("train", "val")
    }

    frozen_pairs: list[OperatorBatch] | None = None
    if args.overfit_pairs > 0:
        batch = sources["train"].batch(size=int(args.overfit_pairs))
        if batch is None:
            raise ValueError("No audited style pairs were available to overfit on")
        frozen_pairs = [batch]
        print(f"overfit mode: {int(batch.target_tokens.shape[0])} frozen pairs", flush=True)

    def train_batches(epoch: int) -> Iterable[OperatorBatch]:
        if frozen_pairs is not None:
            return frozen_pairs * max(1, int(loader_config.get("batch_size", 32)) // len(frozen_pairs))
        return sources["train"].batches(int(loader_config.get("batch_size", 32)))

    def val_batches(epoch: int) -> Iterable[OperatorBatch]:
        if frozen_pairs is not None:
            return frozen_pairs
        batch = sources["val"].batch()
        return [] if batch is None else [batch]

    trainer = OperatorTrainer(
        model,
        adapter=adapter,
        device=device,
        config=trainer_config,
        content_weight=content_weight,
    )
    output = Path(args.output or training.get("output_dir", "outputs/mts_operator/run"))
    output.mkdir(parents=True, exist_ok=True)
    model_config = model.describe()
    best_nll = float("inf")

    def on_epoch_end(epoch: int, metrics: Mapping[str, float], active: OperatorTrainer) -> None:
        nonlocal best_nll
        payload = mts_checkpoint_payload(
            kind="operator",
            model=model,
            model_config=model_config,
            token_spec=token_spec,
            tokenizer_metadata=tokenizer_metadata,
            metrics={
                "train_loss": metrics.get("loss"),
                "val_nll": metrics.get("val_nll"),
                "operator": operator_name,
                "style_encoder_kind": kind,
                "content_weight": content_weight,
                # Recorded so evaluation restores the exact frozen upstream.
                "transport_checkpoint": str(transport_path),
                "tokenizer_checkpoint": str(tokenizer_path),
            },
            epoch=epoch,
            global_step=active.global_step,
            optimizer=active.optimizer,
            extra={"style_split": style_split.as_dict(), "strength_range": strength_range},
        )
        save_mts_checkpoint(output / "last.pt", payload)
        val_nll = float(metrics.get("val_nll", metrics.get("loss", float("inf"))))
        if val_nll < best_nll:
            best_nll = val_nll
            save_mts_checkpoint(output / "best.pt", payload)

    result = trainer.fit(
        train_batches,
        epochs=trainer_config.epochs,
        val_batches=val_batches if frozen_pairs is None else None,
        on_epoch_end=on_epoch_end,
    )
    summary = {
        "output": str(output),
        "operator": operator_name,
        "style_encoder": kind,
        "global_step": result["global_step"],
        "last_epoch": result["history"][-1] if result["history"] else {},
        "best_val_nll": None if best_nll == float("inf") else best_nll,
        "train_styles": len(style_split.train_styles),
        "unseen_styles": len(style_split.test_unseen_styles),
        "token_spec_hash": token_spec.fingerprint(),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.trainable_parameters())
        ),
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
