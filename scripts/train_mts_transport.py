#!/usr/bin/env python
"""Train the style-free base transport (MTS-FSQ plan, Phase 2).

    python scripts/train_mts_transport.py \
      --config data/configs/mts_operator_transport.yaml \
      --tokenizer-checkpoint outputs/nef_fsq_40x9/best.pt \
      --output outputs/mts_transport/seed3407

Tokens come from a precomputed token store when the config sets one, otherwise
windows are read from the feature store and encoded by the frozen tokenizer in
the training loop.  The tokenizer is never updated; the checkpoint records its
fingerprint so the operator stages can refuse a mismatched alphabet.

Use ``--overfit-clips N`` to freeze a handful of encoded windows and train on
them repeatedly: that is the Phase 2 exit criterion before any full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import build_data_loaders, open_any_feature_store, open_token_store  # noqa: E402
from stylized_motion.learning.mts_operator import LayoutAdapter  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    load_mts_checkpoint,
    mts_checkpoint_payload,
    save_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.masking import MaskGenerator  # noqa: E402
from stylized_motion.learning.mts_operator.training import TrainerConfig, TransportTrainer  # noqa: E402
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer  # noqa: E402
from stylized_motion.learning.representation import (  # noqa: E402
    NEF_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import (  # noqa: E402
    apply_batch_normalization,
    choose_device,
    move_batch_to_device,
    set_seed,
)

TRANSPORT_KEYS = (
    "dim",
    "depth",
    "heads",
    "dropout",
    "graph_mode",
    "graph_depth",
    "temporal_mode",
    "content_dim",
    "content_classes",
    "feedforward_multiplier",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MTS base transport on NEF tokens.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-database", type=Path, default=None, help="Overrides data.fsq_window_index.")
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Resume a transport run.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overfit-clips",
        type=int,
        default=0,
        help="Freeze this many encoded windows and train on them repeatedly (0 = off).",
    )
    return parser


def load_transport_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Transport config must be a mapping: {path}")
    required = {"tokenizer", "data", "transport", "masking", "training"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Transport config is missing sections: {missing}")
    unknown = sorted(set(value) - required - {"sampling", "loader", "evaluation"})
    if unknown:
        raise ValueError(f"Unknown transport config sections: {unknown}")
    return dict(value)


class TokenSource:
    """Yields ``[B, T, 40]`` long token batches for one split.

    Two backends: a precomputed token store (fast path) and the feature store
    with the frozen tokenizer encoding windows on the fly.
    """

    def __init__(
        self,
        *,
        loader: torch.utils.data.DataLoader,
        tokenizer: Any | None,
        device: torch.device,
        frames: int | None = None,
    ) -> None:
        self.loader = loader
        self.tokenizer = tokenizer
        self.device = device
        self.frames = frames

    def __iter__(self) -> Iterable[torch.Tensor]:
        for batch in self.loader:
            if self.tokenizer is None:
                tokens = batch["indices"]
                if not isinstance(tokens, torch.Tensor):
                    raise TypeError("Token batches must carry an 'indices' tensor")
                yield tokens.to(self.device).long()
                continue
            batch = apply_batch_normalization(move_batch_to_device(batch, self.device), self.device)
            motion = batch["motion"]
            if not isinstance(motion, torch.Tensor):
                raise TypeError("Feature batches must carry a motion tensor")
            with torch.no_grad():
                tokens = self.tokenizer.encode_indices(motion)
            if self.frames is not None:
                tokens = tokens[:, : self.frames]
            yield tokens

    def freeze(self, *, clips: int, split: str = "train") -> list[torch.Tensor]:
        """Encodes ``clips`` windows once and holds them in memory."""
        frozen: list[torch.Tensor] = []
        collected = 0
        for tokens in self:
            frozen.append(tokens.detach())
            collected += int(tokens.shape[0])
            if collected >= clips:
                break
        if not frozen:
            raise ValueError(f"Split {split!r} produced no batches to overfit on")
        return frozen

    def __len__(self) -> int:
        return len(self.loader)


def build_sources(
    config: Mapping[str, Any],
    *,
    tokenizer: Any | None,
    tokenizer_motion_dim: int | None,
    device: torch.device,
    token_store_path: Path | None,
) -> dict[str, TokenSource]:
    data = config["data"]
    sampling = dict(config.get("sampling") or {})
    loader_config = dict(config.get("loader") or {})
    sampling.setdefault("strategy", "clip_uniform")
    sampling.setdefault("target_frames", 64)
    sampling.setdefault("samples_per_epoch", 100000)
    sampling.setdefault("seed", 3407)
    loader_config.setdefault("batch_size", 256)
    loader_config.setdefault("num_workers", 0)
    if token_store_path is not None:
        store = open_token_store(token_store_path)
        assembled = build_data_loaders(
            "generator", store, sampling_config=sampling, loader_config=loader_config
        )
        return {
            split: TokenSource(loader=assembled.loaders[split], tokenizer=None, device=device)
            for split in ("train", "val")
        }
    if tokenizer is None:
        raise ValueError("Training without a token store requires the frozen tokenizer")
    feature_database = data.get("fsq_window_index", data.get("feature_database"))
    if feature_database is None:
        raise ValueError("data.fsq_window_index is required when no token store is configured")
    store = open_any_feature_store(feature_database)
    if tokenizer_motion_dim is not None and int(store.motion_dim) != int(tokenizer_motion_dim):
        raise ValueError(
            f"Feature store motion_dim {store.motion_dim} does not match the tokenizer's "
            f"{tokenizer_motion_dim}"
        )
    assembled = build_data_loaders(
        "representation", store, sampling_config=sampling, loader_config=loader_config
    )
    return {
        split: TokenSource(loader=assembled.loaders[split], tokenizer=tokenizer, device=device)
        for split in ("train", "val")
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_transport_config(args.config)
    if args.feature_database is not None:
        config["data"] = {**config["data"], "fsq_window_index": str(args.feature_database)}
    training = dict(config["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        training["steps_per_epoch"] = args.steps_per_epoch
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.seed is not None:
        training["seed"] = args.seed
    trainer_config = TrainerConfig.from_mapping(training)
    set_seed(trainer_config.seed, deterministic=False)
    device = choose_device(args.device)

    tokenizer_section = dict(config["tokenizer"])
    tokenizer_path = (
        args.tokenizer_checkpoint
        or tokenizer_section.get("checkpoint")
    )
    if tokenizer_path is None:
        raise ValueError("tokenizer.checkpoint or --tokenizer-checkpoint is required")
    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(
        Path(tokenizer_path), torch.device("cpu")
    )
    if tokenizer.family != NEF_FSQ_FAMILY:
        raise ValueError(
            f"The MTS transport requires a {NEF_FSQ_FAMILY!r} tokenizer, got {tokenizer.family!r}"
        )
    if bool(tokenizer_section.get("freeze", True)) is False:
        raise ValueError("The MTS transport never updates its tokenizer; set tokenizer.freeze: true")
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    layout = tokenizer.token_layout()
    if layout is None:
        raise ValueError("NEF tokenizer did not expose a token layout")
    adapter = LayoutAdapter(layout, num_levels=int(tokenizer.num_levels))

    token_store_path = args.token_store or config["data"].get("token_store")
    sources = build_sources(
        config,
        tokenizer=tokenizer,
        tokenizer_motion_dim=int(tokenizer.motion_dim),
        device=device,
        token_store_path=Path(token_store_path) if token_store_path else None,
    )

    model_options = {key: config["transport"][key] for key in TRANSPORT_KEYS if key in config["transport"]}
    unknown = sorted(set(config["transport"]) - set(TRANSPORT_KEYS))
    if unknown:
        raise ValueError(f"Unknown transport options {unknown}")
    model_config = {"dim": 256, "depth": 8, "heads": 8, "dropout": 0.1, "graph_mode": "local_relational"}
    model_config.update(model_options)
    resume_metrics: Mapping[str, object] = {}
    if args.checkpoint is not None:
        tokenizer_metadata = tokenizer.representation_metadata()
        checkpoint, model = load_mts_checkpoint(
            args.checkpoint,
            kind="transport",
            build_model=lambda stored: MotionTransportTransformer(adapter, **stored),
            device=device,
            token_spec=adapter.token_spec(representation_id=tokenizer.representation_id),
            tokenizer_metadata=tokenizer_metadata,
        )
        model_config = dict(checkpoint["metadata"]["model_config"])
        resume_metrics = checkpoint.get("metrics", {})
        print(f"resumed transport from {args.checkpoint}", flush=True)
    else:
        model = MotionTransportTransformer(adapter, **model_config).to(device)

    mask_generator = MaskGenerator(dict(config["masking"]))
    trainer = TransportTrainer(
        model,
        adapter=adapter,
        mask_generator=mask_generator,
        device=device,
        config=trainer_config,
    )
    output = Path(args.output or trainer_config.output_dir or "outputs/mts_transport/run")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_metadata = tokenizer.representation_metadata()
    token_spec = adapter.token_spec(representation_id=tokenizer.representation_id)

    frozen_batches: list[torch.Tensor] | None = None
    if args.overfit_clips > 0:
        frozen_batches = sources["train"].freeze(clips=int(args.overfit_clips))
        print(
            f"overfit mode: {sum(int(batch.shape[0]) for batch in frozen_batches)} frozen windows",
            flush=True,
        )

    def train_batches(epoch: int) -> Iterable[Any]:
        if frozen_batches is not None:
            return frozen_batches
        return sources["train"]

    def val_batches(epoch: int) -> Iterable[Any]:
        if frozen_batches is not None:
            return frozen_batches
        return sources["val"]

    best_loss = float(resume_metrics.get("val_loss", float("inf")))
    saved_best = args.checkpoint is not None

    def on_epoch_end(epoch: int, metrics: Mapping[str, float], active: TransportTrainer) -> None:
        nonlocal best_loss, saved_best
        payload = mts_checkpoint_payload(
            kind="transport",
            model=active.model,
            model_config=model_config,
            token_spec=token_spec,
            tokenizer_metadata=tokenizer_metadata,
            metrics={
                "train_loss": metrics.get("loss"),
                "val_loss": metrics.get("val_loss"),
                "train_accuracy": metrics.get("accuracy"),
            },
            epoch=epoch,
            global_step=active.global_step,
            optimizer=active.optimizer,
            extra={"masking": mask_generator.config.as_dict()},
        )
        save_mts_checkpoint(output / "last.pt", payload)
        val_loss = float(metrics.get("val_loss", metrics.get("loss", float("inf"))))
        if val_loss < best_loss:
            best_loss = val_loss
            saved_best = True
            save_mts_checkpoint(output / "best.pt", payload)

    result = trainer.fit(
        train_batches,
        epochs=trainer_config.epochs,
        val_batches=val_batches,
        on_epoch_end=on_epoch_end,
    )
    summary = {
        "output": str(output),
        "global_step": result["global_step"],
        "history": result["history"][-1] if result["history"] else {},
        "best_val_loss": best_loss if saved_best else None,
        "token_spec_hash": token_spec.fingerprint(),
        "layout_hash": adapter.layout_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
