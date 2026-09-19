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
import hashlib
import itertools
import json
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import build_data_loaders, open_any_feature_store  # noqa: E402
from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator import LayoutAdapter, MASK_KINDS  # noqa: E402
from stylized_motion.learning.mts_operator.content_schema import (  # noqa: E402
    apply_content_schema,
    content_schema_from_config,
)
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    build_provenance,
    load_mts_checkpoint,
    mts_checkpoint_payload,
    require_token_store_binding,
    save_mts_checkpoint,
    store_identity_block,
    training_exposure,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.eval_protocol import (  # noqa: E402
    ValidationBatchBuilder,
    ValidationProtocol,
    validation_evidence,
)
from stylized_motion.learning.mts_operator.masking import MaskGenerator  # noqa: E402
from stylized_motion.learning.mts_operator.training import (  # noqa: E402
    RUN_ARTIFACTS,
    TrainerConfig,
    TransportTrainer,
    planned_step_budget,
    refuse_existing_output,
    resolve_budget,
)
from stylized_motion.learning.mts_operator.pairs import clip_records_from_store  # noqa: E402
from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer  # noqa: E402
from stylized_motion.learning.mts_operator.windows import ContentVocabulary  # noqa: E402
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource as WindowTokenSource,
    windows_by_clip,
)
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
    "token_embed_dim",
    "position_encoding",
    "content_vocabulary",
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

#: The frozen-validation recipe's own fields.  The row count is a property of the
#: recipe, not of ``loader.batch_size``.
EVALUATION_KEYS = frozenset({"protocol_id", "validation_kinds", "validation_rows_per_kind"})
#: Legacy names that were accepted (and, for the transport, ignored).  They are
#: refused with the replacement name instead of being quietly dropped.
LEGACY_EVALUATION_KEYS = {
    "validation_batches_per_kind": (
        "evaluation.validation_rows_per_kind (an explicit row count per mask kind, "
        "independent of loader.batch_size)"
    ),
    "validation_batches": "evaluation.validation_rows_per_kind",
    "val_rows": "evaluation.validation_rows_per_kind",
}

#: The monitor kinds a diagnostic run scores.  ``full_generation`` is deliberately
#: absent: the plan's E02 judges learnability on the kinds that leave visible
#: context, not on fitting whole motions token by token.
MONITOR_KINDS: tuple[str, ...] = (
    "random_coordinate",
    "stream",
    "temporal_span",
    "spatiotemporal_block",
)


def _row(item: Any, index: int) -> Any:
    """One window out of a batch mapping, carried with everything attached to it."""
    if not isinstance(item, Mapping):
        return item[index : index + 1]
    payload: dict[str, Any] = {}
    for name, value in item.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > index:
            payload[name] = value[index : index + 1]
        elif isinstance(value, list) and len(value) > index:
            payload[name] = [value[index]]
        else:
            payload[name] = value
    return payload


def fix_monitor_masks(
    frozen_batches: Sequence[Mapping[str, Any]],
    *,
    mask_generator: MaskGenerator,
    adapter: Any,
    spec: Any,
    seed: int,
    kinds: Sequence[str] = MONITOR_KINDS,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """States one mask per frozen window and records what it states.

    Returns the monitor items (the window plus an explicit ``visible_mask``) and
    the manifest that names each window's clip, start, action, mask kind, mask seed
    and hidden fraction.  The kind is assigned round-robin over ``kinds`` so eight
    windows cover four context tasks twice instead of sampling a mixture and
    hoping; the mask itself comes from that window's own seed, so it is identical
    at B=1 and at B=8 and across epochs.
    """
    kinds = tuple(str(kind) for kind in kinds)
    if not kinds:
        raise ValueError("monitor kinds must not be empty")
    items: list[Mapping[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    index = 0
    for batch in frozen_batches:
        for row in range(int(batch["tokens"].shape[0])):
            kind = kinds[index % len(kinds)]
            mask_seed = int(seed) + index
            generator = torch.Generator(device="cpu").manual_seed(mask_seed)
            window = _row(batch, row)
            frames = int(window["tokens"].shape[1])
            mask = mask_generator.sample_kind(
                kind, 1, frames, adapter=adapter, spec=spec, generator=generator, device=torch.device("cpu")
            )
            hidden = int((~mask.visible_mask).sum())
            if hidden <= 0:
                raise ValueError(
                    f"The monitor mask for window {index} ({kind}) hides nothing; an empty "
                    "supervision set would report no number at all"
                )
            metadata = (window.get("sample_metadata") or [{}])[0]
            items.append(
                {
                    **window,
                    "visible_mask": mask.visible_mask,
                    "kind": kind,
                    "mask_config": mask_generator.config.as_dict(),
                }
            )
            manifest.append(
                {
                    "window": index,
                    "clip_id": metadata.get("variant_idx", metadata.get("clip_id")),
                    "target_start": metadata.get("target_start"),
                    "action": metadata.get("action"),
                    "frames": frames,
                    "mask_kind": kind,
                    "mask_seed": mask_seed,
                    "hidden_tokens": hidden,
                    "hidden_fraction": hidden / max(frames * int(mask.visible_mask.shape[-1]), 1),
                }
            )
            index += 1
    return items, manifest


def resolve_validation_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """The transport's frozen-validation recipe: protocol id, kinds, rows per kind.

    ``validation_batches_per_kind`` used to be accepted by this entry point and
    never read (the protocol was hard-wired to one batch of ``--val-rows`` rows),
    so a run could claim a budget it did not have.  It is now an explicit error,
    and the protocol's size comes from the recipe alone -- changing
    ``loader.batch_size`` must not change what is scored.
    """
    section = dict(config.get("evaluation") or {})
    legacy = sorted(set(section) & set(LEGACY_EVALUATION_KEYS))
    if legacy:
        raise ValueError(
            f"evaluation.{legacy[0]} is a legacy field: replace it with "
            f"{LEGACY_EVALUATION_KEYS[legacy[0]]}"
        )
    unknown = sorted(set(section) - EVALUATION_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown evaluation fields {unknown}; expected {sorted(EVALUATION_KEYS)}"
        )
    protocol_id = str(section.get("protocol_id") or "").strip()
    if not protocol_id:
        raise ValueError(
            "evaluation.protocol_id is required: a run must name the frozen protocol it "
            "selects its best checkpoint on, so a profile protocol and a full protocol "
            "cannot be confused"
        )
    kinds = tuple(str(kind) for kind in (section.get("validation_kinds") or ()))
    if not kinds:
        raise ValueError(
            "evaluation.validation_kinds must list every mask kind the protocol scores "
            f"(the revision-2 recipes list all of {list(MASK_KINDS)})"
        )
    unknown_kinds = sorted(set(kinds) - set(MASK_KINDS))
    if unknown_kinds:
        raise ValueError(
            f"evaluation.validation_kinds has unknown entries {unknown_kinds}; "
            f"expected a subset of {list(MASK_KINDS)}"
        )
    rows = section.get("validation_rows_per_kind")
    if rows is None:
        raise ValueError(
            "evaluation.validation_rows_per_kind is required (a positive row count per mask "
            "kind; it does not depend on loader.batch_size)"
        )
    rows = int(rows)
    if rows <= 0:
        raise ValueError("evaluation.validation_rows_per_kind must be positive")
    return {"protocol_id": protocol_id, "kinds": kinds, "rows_per_kind": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MTS base transport on NEF tokens.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--feature-database", type=Path, default=None, help="Overrides data.fsq_window_index.")
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Start a NEW run from these weights (same architecture revision); optimizer, step "
        "counter, best metric and RNG all start from zero. This is not a resume.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Removed: exact resume is not implemented. Use --warm-start to begin a new run "
        "from these weights.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Overrides loader.batch_size. The temporal encoder folds the 13 streams "
        "into the batch axis, so the effective batch is 13x this value.",
    )
    parser.add_argument("--dim", type=int, default=None, help="Overrides transport.dim.")
    parser.add_argument(
        "--val-rows",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overfit-clips",
        type=int,
        default=0,
        help="Freeze this many encoded windows and train on them repeatedly (0 = off).",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Independent wall-clock cap; an interrupted run records `interrupted` and is "
        "never reported as completed.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Deliberately write into a directory that already holds a run's artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config, bind the artifacts and build the frozen validation, "
        "then stop without stepping the optimizer.",
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
    """Yields token batches for one split, with the action condition when asked.

    Two backends: a precomputed token store (fast path) and the feature store
    with the frozen tokenizer encoding windows on the fly.  With an
    ``action_id`` vocabulary the batch becomes a mapping that also carries the
    ``content_condition``, taken from the *target* clip's own action label.
    """

    def __init__(
        self,
        *,
        loader: torch.utils.data.DataLoader,
        tokenizer: Any | None,
        device: torch.device,
        frames: int | None = None,
        content_vocabulary: ContentVocabulary | None = None,
        content_schema: Any | None = None,
    ) -> None:
        self.loader = loader
        self.tokenizer = tokenizer
        self.device = device
        self.frames = frames
        self.content = content_vocabulary or ContentVocabulary()
        # The loader's metadata carries the store's *raw* action label; the
        # vocabulary holds canonical ones.  Without this map the two disagree the
        # moment a schema renames a label, and the run would either crash or (worse)
        # borrow an id.
        self.content_schema = content_schema

    def _tokens_for(self, batch: Mapping[str, Any]) -> torch.Tensor:
        if self.tokenizer is None:
            # v3 token stores yield "indices", the packed store yields "tokens".
            tokens = batch.get("indices", batch.get("tokens"))
            if not isinstance(tokens, torch.Tensor):
                raise TypeError("Token batches must carry an 'indices' or 'tokens' tensor")
            return tokens.to(self.device).long()
        batch = apply_batch_normalization(move_batch_to_device(batch, self.device), self.device)
        motion = batch["motion"]
        if not isinstance(motion, torch.Tensor):
            raise TypeError("Feature batches must carry a motion tensor")
        with torch.no_grad():
            tokens = self.tokenizer.encode_indices(motion)
        if self.frames is not None:
            tokens = tokens[:, : self.frames]
        return tokens

    def condition_for(self, batch: Mapping[str, Any], count: int) -> Any:
        if self.content.unconditional:
            return None
        labels = batch.get("metadata")
        if not labels:
            raise ValueError(
                "data.content.kind=action_id needs the loader to return clip metadata; "
                "set loader.return_metadata: true (or disable the action condition)"
            )
        if len(labels) != int(count):
            raise ValueError(
                f"Metadata has {len(labels)} entries for a batch of {int(count)} windows"
            )
        actions = [entry.get("action", "") for entry in labels]
        if self.content_schema is not None:
            actions = [self.content_schema.canonical(action) for action in actions]
        return self.content.vector(actions).to(self.device)

    def __iter__(self) -> Iterable[Any]:
        """One fixed mapping per batch: tokens, valid_mask, condition, metadata.

        An unconditional run used to yield a bare tensor, so the batch lost its
        valid mask and its provenance -- the two fields the validation protocol
        states explicitly.  The canonical loader only serves windows that lie
        inside their clip (the sampler never requests an overrun), so every frame
        of a training window exists; the mask is still carried, and the frozen
        validation path derives it from the clip geometry instead of assuming it.
        """
        for batch in self.loader:
            tokens = self._tokens_for(batch)
            count, frames = int(tokens.shape[0]), int(tokens.shape[1])
            metadata = list(batch.get("metadata") or [])
            for entry in metadata:
                if "frames" not in entry and "target_frames" in entry:
                    entry["frames"] = int(entry["target_frames"])
            yield {
                "tokens": tokens,
                "valid_mask": torch.ones((count, frames), dtype=torch.bool),
                "content_condition": self.condition_for(batch, count),
                "sample_metadata": metadata,
            }

    def freeze(self, *, clips: int, split: str = "train") -> list[Any]:
        """Holds exactly ``clips`` windows in memory, never a whole extra batch.

        ``--overfit-clips 8`` used to keep whatever batch crossed the threshold
        (128 windows at the default batch size), which is a different experiment
        than the one that was asked for.
        """
        frozen: list[Any] = []
        collected = 0
        for item in self:
            tokens = item["tokens"]
            take = int(tokens.shape[0])
            if collected + take > int(clips):
                keep = int(clips) - collected
                if keep <= 0:
                    break
                item = self._truncate(item, keep)
                take = keep
            frozen.append(item)
            collected += take
            if collected >= int(clips):
                break
        if not frozen:
            raise ValueError(f"Split {split!r} produced no batches to overfit on")
        if collected < int(clips):
            raise ValueError(
                f"Split {split!r} only produced {collected} windows, fewer than the "
                f"{int(clips)} requested"
            )
        return frozen

    def _truncate(self, item: Any, keep: int) -> Any:
        """Keeps the first ``keep`` windows of a batch and everything attached to them."""
        if not isinstance(item, Mapping):
            return item[:keep]
        payload = dict(item)
        for name, value in list(payload.items()):
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] > keep:
                payload[name] = value[:keep]
            elif isinstance(value, list) and len(value) > keep:
                payload[name] = value[:keep]
        return payload

    def __len__(self) -> int:
        return len(self.loader)


def build_sources(
    config: Mapping[str, Any],
    *,
    tokenizer: Any | None,
    tokenizer_motion_dim: int | None,
    device: torch.device,
    token_store_path: Path | None,
    tokenizer_checkpoint_path: Path | None = None,
) -> tuple[dict[str, TokenSource], "ContentVocabulary", Any, dict[int, dict[str, str]], dict[str, Any]]:
    """Builds the per-split token sources, the frozen action vocabulary and the store.

    The vocabulary comes from the *training* split's clip labels and is sorted;
    clip metadata is requested from the loader only when an action condition is
    actually configured, so an unconditional run pays nothing.  The store is
    returned because the frozen validation protocol reads windows by clip id
    through :mod:`windows`, not through the training loaders.
    """
    data = config["data"]
    sampling = dict(config.get("sampling") or {})
    loader_config = dict(config.get("loader") or {})
    content_config = dict(data.get("content") or {})
    content_kind = str(content_config.get("kind", "none"))
    if content_kind not in {"none", "action_id"}:
        raise ValueError(f"Unknown data.content.kind {content_kind!r}; expected none or action_id")
    sampling.setdefault("strategy", "clip_uniform")
    sampling.setdefault("target_frames", 64)
    sampling.setdefault("samples_per_epoch", 100000)
    sampling.setdefault("seed", 3407)
    loader_config.setdefault("batch_size", 256)
    loader_config.setdefault("num_workers", 0)
    if content_kind == "action_id":
        loader_config["return_metadata"] = True
    store = None
    store_kind = None
    if token_store_path is not None:
        store = open_any_token_store(token_store_path)
        store_kind = "token"
    elif tokenizer is None:
        raise ValueError("Training without a token store requires the frozen tokenizer")
    else:
        feature_database = data.get("fsq_window_index", data.get("feature_database"))
        if feature_database is None:
            raise ValueError("data.fsq_window_index is required when no token store is configured")
        store = open_any_feature_store(feature_database)
        if tokenizer_motion_dim is not None and int(store.motion_dim) != int(tokenizer_motion_dim):
            raise ValueError(
                f"Feature store motion_dim {store.motion_dim} does not match the tokenizer's "
                f"{tokenizer_motion_dim}"
            )
    if store_kind == "token":
        require_token_store_binding(
            store, tokenizer_checkpoint=tokenizer_checkpoint_path,
            where="transport token store",
        )
    validate_store_binding(store, store_kind=store_kind or "feature")
    dataset = data.get("dataset")
    records = clip_records_from_store(store, dataset=None if dataset is None else str(dataset))
    # N04: the content map is a property of the run, not of the store.  A recipe
    # without one keeps the raw labels; a recipe with one gets canonical labels and
    # an audit report that travels into the dry-run and the checkpoint.
    content_schema, content_schema_report = content_schema_from_config(content_config, root=REPO_ROOT)
    records, content_mapping = apply_content_schema(
        records, content_schema, strict=bool(content_schema_report.get("strict"))
    )
    content_schema_report = {**content_schema_report, "mapping": content_mapping}
    if content_mapping.get("applied"):
        print(
            f"content schema v{content_mapping['version']}: {content_mapping['classes_before']} raw "
            f"labels -> {content_mapping['classes_after']} canonical classes "
            f"({content_mapping['renamed_rows']} rows renamed)",
            flush=True,
        )
    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise ValueError(
            "The store has no train-split records, so the action vocabulary would have to be "
            "built from validation/test data; fix the split before training"
        )
    vocabulary = ContentVocabulary.build(train_records, kind=content_kind)
    labels = {
        int(record.clip_id): {
            "content": str(record.content),
            "style": str(record.style),
            "actor": str(record.performer or ""),
        }
        for record in records
    }
    kind = "generator" if token_store_path is not None else "representation"
    assembled = build_data_loaders(kind, store, sampling_config=sampling, loader_config=loader_config)
    # The canonical v3 token loader hands over 65 tokens per window; every other
    # MTS stage (operator, evaluator, generator) reads exactly ``data.frames``
    # from the same window, so the transport must see the same tokens or its
    # frozen validation would measure a different window than it trained on.
    window_frames = int(data.get("frames", 64))
    if window_frames <= 0:
        raise ValueError("data.frames must be positive")
    sources = {
        split: TokenSource(
            loader=assembled.loaders[split],
            tokenizer=None if token_store_path is not None else tokenizer,
            device=device,
            content_vocabulary=vocabulary,
            content_schema=content_schema,
            frames=window_frames,
        )
        for split in ("train", "val")
    }
    if vocabulary.unconditional:
        print(
            "content condition: unconditional (data.content.kind=none); the transport is NOT "
            "conditioned on the action label",
            flush=True,
        )
    else:
        print(
            f"content condition: action_id with {len(vocabulary.classes)} classes from the "
            f"training split: {list(vocabulary.classes)}",
            flush=True,
        )
    return sources, vocabulary, store, labels, content_schema_report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.val_rows is not None:
        raise SystemExit(
            "--val-rows was removed: the frozen protocol's row count lives in the recipe "
            "(evaluation.validation_rows_per_kind) so a profile protocol and a full protocol "
            "cannot share a size by accident"
        )
    if args.checkpoint is not None:
        raise SystemExit(
            "--checkpoint used to pretend to resume a transport run; exact resume "
            "(optimizer/sampler/scaler state) is not implemented. Use --warm-start to start a "
            "new run from those weights."
        )
    config = load_transport_config(args.config)
    if args.feature_database is not None:
        config["data"] = {**config["data"], "fsq_window_index": str(args.feature_database)}
    if args.batch_size is not None:
        config["loader"] = {**dict(config.get("loader") or {}), "batch_size": int(args.batch_size)}
    if args.dim is not None:
        config["transport"] = {**dict(config["transport"]), "dim": int(args.dim)}
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
    validation_recipe = resolve_validation_recipe(config)
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
    # The tokenizer follows the compute device: batches (and therefore the
    # motion handed to encode_indices) already live there, and encoding on the
    # GPU is an order of magnitude faster than shuttling tensors back to CPU.
    tokenizer = tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    layout = tokenizer.token_layout()
    if layout is None:
        raise ValueError("NEF tokenizer did not expose a token layout")
    adapter = LayoutAdapter(layout, num_levels=int(tokenizer.num_levels))

    token_store_path = args.token_store or config["data"].get("token_store")
    # The loader block is resolved here so the training source and the validation
    # batch builder cannot disagree about the batch size (the validation path
    # used to read a name that only existed inside build_sources).
    loader_config = dict(config.get("loader") or {})
    loader_config.setdefault("batch_size", 256)
    loader_config.setdefault("num_workers", 0)
    config = {**config, "loader": loader_config}
    sources, content_vocabulary, store, labels, content_schema_report = build_sources(
        config,
        tokenizer=tokenizer,
        tokenizer_motion_dim=int(tokenizer.motion_dim),
        device=device,
        token_store_path=Path(token_store_path) if token_store_path else None,
        tokenizer_checkpoint_path=Path(tokenizer_path) if tokenizer_path else None,
    )
    # The store's identity is resolved here, before the protocol: the frozen
    # protocol carries it, so the protocol hash names the data it was built from
    # (a path is not an identity).
    store_identity = store_identity_block(
        store,
        store_kind="token" if token_store_path else "feature",
        store_path=token_store_path or config["data"].get("fsq_window_index"),
    )
    # The frozen protocol addresses windows by (clip, start); the training loaders
    # address them by batch order, so validation needs the window reader.  Reading
    # through the loaders here used to be an AttributeError on the first epoch.
    #
    # The pool is the *val* split.  A validation set built from training windows
    # cannot choose a generalizing checkpoint, and a run that reports it as
    # validation is wrong about what it measured.  An empty val split is an error,
    # never a fallback to train.
    validation_frames = int(config["data"].get("frames", 64))
    val_windows = windows_by_clip(store, "val", frames=validation_frames)
    if not val_windows:
        raise ValueError(
            f"The store's val split has no clip with a full {validation_frames}-frame window, "
            "so the frozen validation protocol cannot be built; the transport's validation is "
            "never taken from the train split"
        )
    window_source = WindowTokenSource(
        store=store,
        windows_by_clip=val_windows,
        adapter=adapter,
        frames=validation_frames,
        history=int(tokenizer.history_frames),
        tokenizer=None if token_store_path else tokenizer,
        feature_stats=None if token_store_path else tokenizer_checkpoint.get("feature_stats"),
    )

    model_options = {key: config["transport"][key] for key in TRANSPORT_KEYS if key in config["transport"]}
    unknown = sorted(set(config["transport"]) - set(TRANSPORT_KEYS))
    if unknown:
        raise ValueError(f"Unknown transport options {unknown}")
    model_config = {"dim": 256, "depth": 8, "heads": 8, "dropout": 0.1, "graph_mode": "local_relational"}
    model_config.update(model_options)
    if not content_vocabulary.unconditional:
        # Recorded with the model so the operator and the evaluators inherit the
        # exact map instead of rebuilding one from whatever split is loaded.
        model_config["content_vocabulary"] = content_vocabulary.as_dict()
    resume_metrics: Mapping[str, object] = {}
    warm_start_source = None
    model = MotionTransportTransformer(adapter, **model_config).to(device)
    if args.warm_start is not None:
        # Weights only: a fresh optimizer, step counter and best metric.
        _, warm_model = load_mts_checkpoint(
            args.warm_start,
            kind="transport",
            build_model=lambda stored: MotionTransportTransformer(adapter, **stored),
            device=device,
            token_spec=adapter.token_spec(representation_id=tokenizer.representation_id),
            tokenizer_metadata=tokenizer.representation_metadata(),
            tokenizer_checkpoint=Path(tokenizer_path),
        )
        stored_vocabulary = getattr(warm_model, "content_vocabulary", None)
        if stored_vocabulary is not None and stored_vocabulary != content_vocabulary:
            raise ValueError(
                "The warm-start transport was trained with a different action vocabulary "
                f"({stored_vocabulary.as_dict()}) than this run built "
                f"({content_vocabulary.as_dict()})"
            )
        model.load_state_dict(warm_model.state_dict())
        warm_start_source = str(args.warm_start)
        print(
            f"warm start from {warm_start_source}: weights loaded, optimizer/step/best reset",
            flush=True,
        )

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
    refuse_existing_output(output, allow=bool(args.allow_existing_output), where="transport run")
    existing = [name for name in RUN_ARTIFACTS if (output / name).exists()]
    tokenizer_metadata = tokenizer.representation_metadata()
    token_spec = adapter.token_spec(representation_id=tokenizer.representation_id)

    frozen_batches: list[Mapping[str, Any]] | None = None
    monitor_items: list[Mapping[str, Any]] = []
    monitor_manifest: list[dict[str, Any]] = []
    frozen_window_count = 0
    if args.overfit_clips > 0:
        frozen_batches = sources["train"].freeze(clips=int(args.overfit_clips))
        frozen_window_count = sum(int(item["tokens"].shape[0]) for item in frozen_batches)
        # The monitor scores a *stated* mask, one per window, fixed for the whole
        # run: re-sampling it every epoch would move the number for a reason that
        # has nothing to do with learning, and the plan's diagnostic is explicitly
        # "same mask at step 20/100/200".  Training keeps sampling from the
        # mixture, exactly as the profile recipe does.
        monitor_items, monitor_manifest = fix_monitor_masks(
            frozen_batches,
            mask_generator=mask_generator,
            adapter=adapter,
            spec=token_spec,
            seed=int(trainer_config.seed),
        )
        print(
            f"overfit mode: {frozen_window_count} frozen windows, "
            f"monitor kinds={sorted({row['mask_kind'] for row in monitor_manifest})} "
            "(training monitor only, never a validation-best)",
            flush=True,
        )
        if not args.dry_run:
            budget_for_frozen = resolve_budget(trainer_config)
            (output / "overfit_frozen.json").write_text(
                json.dumps(
                    {
                        "windows": frozen_window_count,
                        "clips_requested": int(args.overfit_clips),
                        "batches": len(frozen_batches),
                        "repeated_to_steps": budget_for_frozen["planned_steps"],
                        "planned_steps_source": budget_for_frozen["source"],
                        "masking": mask_generator.config.as_dict(),
                        "content_condition": content_vocabulary.as_dict(),
                        "seed": int(trainer_config.seed),
                        "frozen_batch_count": len(frozen_batches),
                        "windows_per_batch": [
                            int(item["tokens"].shape[0]) for item in frozen_batches
                        ],
                        # Which windows, which mask: without this the diagnostic's
                        # curve cannot be reproduced or checked for input leakage.
                        "monitor": monitor_manifest,
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )

    def train_batches(epoch: int) -> Iterable[Any]:
        if frozen_batches is not None:
            # The frozen windows are repeated until the stated step budget is met.
            # Returning the list once let an epoch end after a single batch, so a
            # "5 step" overfit run quietly took one step.
            return itertools.cycle(frozen_batches)
        return sources["train"]

    # The validation set is frozen once: no pairs (a base model is not a style
    # operator), one mask per row, the same windows every epoch.  Its rows are
    # verified against the store's own split table while it is built, and the
    # store's identity is part of the protocol, not a side note.
    validation_protocol = None
    validation_builder = None
    if frozen_batches is None:
        validation_protocol = ValidationProtocol.from_target_windows(
            window_source,
            protocol_id=validation_recipe["protocol_id"],
            kinds=validation_recipe["kinds"],
            rows_per_kind=validation_recipe["rows_per_kind"],
            frames=validation_frames,
            seed=trainer_config.seed,
            split="val",
            mask_generator=mask_generator,
            labels=labels,
            content_vocabulary=content_vocabulary,
            store_identity=store_identity,
        )
        validation_builder = ValidationBatchBuilder(
            token_source=window_source,
            mask_generator=mask_generator,
            adapter=adapter,
            device=device,
            content_vocabulary=content_vocabulary,
        )
        validation_protocol.write(output / "validation_protocol.json")
        print(
            f"frozen validation protocol {validation_protocol.protocol_id!r}: "
            f"{len(validation_protocol.samples)} rows "
            f"({validation_recipe['rows_per_kind']}/kind), "
            f"kinds={list(validation_protocol.kinds)}, "
            f"splits={sorted({sample.split for sample in validation_protocol.samples})}, "
            f"hash={validation_protocol.fingerprint()[:16]}…",
            flush=True,
        )

    def monitor_batches(epoch: int) -> Iterable[Any]:
        """The frozen *training* windows under one stated mask, named as such.

        Overfit runs are monitored on their own frozen clips; sending them
        through ``val_batches`` used to label a training number ``val_loss`` and
        made an overfit checkpoint look validated.  The masks come from
        :func:`fix_monitor_masks` (one per window, fixed for the run), so the
        curve across monitor points moves because of learning, not because a new
        mask was drawn.
        """
        return monitor_items or []

    provenance = build_provenance(
        store_identity=store_identity,
        action_vocabulary=content_vocabulary.as_dict(),
        resolved_config=config,
        seed=trainer_config.seed,
        training_protocol_id=validation_recipe["protocol_id"],
        content_schema=content_schema_report,
    )
    provenance["validation_protocol_id"] = validation_recipe["protocol_id"]
    provenance["training_exposure"] = training_exposure(
        actions=list(content_vocabulary.classes) or None,
    )
    if args.warm_start is not None:
        # A warm start is a new run: the optimizer, the step counter, the best
        # metric and the RNG all start at zero, so its step count is not a
        # continuation of the checkpoint it started from.
        provenance["warm_start"] = str(args.warm_start)
        provenance["warm_start_is_resume"] = False

    if args.dry_run:
        # Everything below this line is training.  The dry run resolves the config,
        # binds the artifacts and builds the frozen protocol, then stops without
        # touching the optimizer -- so "the run is ready" is a checked claim.
        preview_steps = planned_step_budget(trainer_config)
        budget_preview = {
            "planned_steps": preview_steps,
            "source": None
            if preview_steps is None
            else ("training.max_steps" if trainer_config.max_steps is not None
                  else "training.epochs x training.steps_per_epoch"),
            "note": None
            if preview_steps is not None
            else "the recipe pins no step budget: training would refuse to start until "
            "training.max_steps or training.steps_per_epoch is set",
        }
        dry_run_report = {
            "config": str(args.config),
            "resolved_transport": model_config,
            "store": str(token_store_path or config["data"].get("fsq_window_index") or ""),
            "store_identity": provenance["store_identity"],
            "token_spec_hash": token_spec.fingerprint(),
            "layout_hash": adapter.layout_hash,
            "content_condition": content_vocabulary.as_dict(),
            "content_schema": content_schema_report,
            "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
            "token_embed_dim": int(model.token_embed_dim),
            "architecture_revision": model.config().get("architecture_revision"),
            "window_frames": validation_frames,
            "loader": dict(loader_config),
            "training": {
                "epochs": trainer_config.epochs,
                "steps_per_epoch": trainer_config.steps_per_epoch,
                "max_steps": trainer_config.max_steps,
                "lr": trainer_config.lr,
                "precision": trainer_config.precision,
                "seed": trainer_config.seed,
            },
            "validation": {
                "protocol_id": validation_protocol.protocol_id if validation_protocol else None,
                "rows": len(validation_protocol.samples) if validation_protocol else 0,
                "rows_per_kind": validation_recipe["rows_per_kind"],
                "kinds": list(validation_protocol.kinds) if validation_protocol else [],
                "splits": sorted({sample.split for sample in validation_protocol.samples})
                if validation_protocol
                else [],
                "protocol_hash": validation_protocol.fingerprint() if validation_protocol else None,
            },
            "budget": budget_preview,
            "overfit_clips": int(args.overfit_clips),
            # The frozen monitor's own size: an overfit run is only interpretable if
            # the windows it repeated are named in its artifacts.
            "overfit_windows": None
            if frozen_batches is None
            else int(sum(int(item["tokens"].shape[0]) for item in frozen_batches)),
            "monitor": None
            if not monitor_manifest
            else {
                "windows": len(monitor_manifest),
                "kinds": sorted({row["mask_kind"] for row in monitor_manifest}),
                "mask_config": mask_generator.config.as_dict(),
                "rows": monitor_manifest,
            },
            "output": str(output),
            "existing_artifacts": existing,
            "dry_run": True,
        }
        # The report is written as well as printed: a dry run is the artifact that
        # says what a later run would do, and extracting it from a console log is
        # neither durable nor exact.
        output.mkdir(parents=True, exist_ok=True)
        (output / "dry_run.json").write_text(
            json.dumps(dry_run_report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(dry_run_report, indent=2, default=str), flush=True)
        return

    # A training run states its step budget and is checked against it afterwards.
    budget = resolve_budget(trainer_config)
    print(
        f"step budget: {budget['planned_steps']} optimizer steps ({budget['source']})",
        flush=True,
    )
    best_loss = float("inf")
    saved_best = False

    def on_epoch_end(
        epoch: int, metrics: Mapping[str, float], active: TransportTrainer
    ) -> dict[str, Any]:
        """Runs the frozen protocol once and records the same numbers everywhere.

        The callback is the single validation pass per epoch: the number it prints
        is the number the history entry gets and the number the checkpoint stores,
        so "best by objective" and "the reported val_objective" cannot diverge.
        """
        nonlocal best_loss, saved_best
        validation_report = None
        evidence: dict[str, Any] = {}
        val_started = time.perf_counter()
        if validation_protocol is not None:
            validation_report = validation_builder.evaluate(
                active, validation_protocol, batch_size=int(loader_config["batch_size"]),
                transport=True,
            )
            evidence = validation_evidence(
                validation_protocol,
                validation_report,
                seconds={"val_seconds": time.perf_counter() - val_started},
            )
            readable = (
                "n/a"
                if validation_report["objective"] is None
                else f"{float(validation_report['objective']):.4f}"
            )
            print(
                f"epoch {epoch}: validation objective={readable} "
                f"per_kind={ {k: round(v, 4) for k, v in validation_report['objectives_per_kind'].items()} } "
                f"counts={validation_report['counts']} missing={validation_report['missing_kinds']} "
                f"invalid={validation_report['invalid_kinds']}",
                flush=True,
            )
        checkpoint_started = time.perf_counter()
        payload = mts_checkpoint_payload(
            kind="transport",
            model=active.model,
            model_config=model_config,
            token_spec=token_spec,
            tokenizer_metadata=tokenizer_metadata,
            provenance=provenance,
            tokenizer_checkpoint=Path(tokenizer_path),
            metrics={
                "train_loss": metrics.get("loss"),
                "val_loss": evidence.get("val_objective"),
                "train_accuracy": metrics.get("accuracy"),
                "supervised_tokens": metrics.get("supervised_tokens_total"),
                "optimizer_steps": int(active.global_step),
                **evidence,
            },
            epoch=epoch,
            global_step=active.global_step,
            optimizer=active.optimizer,
            extra={"masking": mask_generator.config.as_dict()},
        )
        if validation_report is None:
            # Overfit mode has no validation at all: its checkpoint is written
            # under its own name and is never claimed to be a validation-best.
            save_mts_checkpoint(output / "overfit_last.pt", payload)
            # The aggregate monitor number can move because of one window, so the
            # diagnostic records every window separately at every monitor point:
            # that is what makes "the trend is not one window" checkable instead of
            # asserted.  Each item is scored under its own stated mask.
            per_window: list[dict[str, Any]] = []
            for index, item in enumerate(monitor_items):
                metrics_one = active.evaluate([item])
                manifest_row = monitor_manifest[index]
                per_window.append(
                    {
                        "window": index,
                        "clip_id": manifest_row.get("clip_id"),
                        "target_start": manifest_row.get("target_start"),
                        "action": manifest_row.get("action"),
                        "mask_kind": manifest_row.get("mask_kind"),
                        "mask_seed": manifest_row.get("mask_seed"),
                        "nll": metrics_one["loss"],
                        "accuracy": metrics_one["accuracy"],
                        "supervised_tokens": metrics_one["supervised_tokens"],
                    }
                )
            evidence["monitor_per_window"] = per_window
            readable = "n/a" if metrics.get("monitor_loss") is None else f"{float(metrics['monitor_loss']):.4f}"
            print(
                f"epoch {epoch}: overfit monitor nll={readable} "
                f"per_window={[None if row['nll'] is None else round(float(row['nll']), 4) for row in per_window]} "
                "(no validation-best is claimed)",
                flush=True,
            )
        else:
            save_mts_checkpoint(output / "last.pt", payload)
            if not validation_report["usable"] or validation_report["objective"] is None:
                # Never fall back to the train loss: an unusable validation makes
                # this epoch ineligible for best.pt instead of making best easy.
                print(
                    f"epoch {epoch}: validation unusable "
                    f"(missing={validation_report['missing_kinds']}, "
                    f"invalid={validation_report['invalid_kinds']}); best.pt untouched"
                )
            else:
                objective = float(validation_report["objective"])
                # A checkpoint with zero optimizer steps is not a validation-best.
                if objective < best_loss and active.global_step > 0:
                    best_loss = objective
                    saved_best = True
                    save_mts_checkpoint(output / "best.pt", payload)
        if evidence:
            evidence["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        evidence["train_seconds"] = float(metrics.get("seconds", 0.0))
        evidence["total_seconds"] = (
            evidence["train_seconds"]
            + float(evidence.get("val_seconds", 0.0))
            + float(evidence.get("checkpoint_seconds", 0.0))
        )
        return evidence

    # Step 0 is a reference the plan asks for: the initialisation this recipe
    # actually produced, saved before the first step, so "what did the budget buy"
    # is measured against the run's own starting weights instead of a
    # reconstruction from the seed.  No optimizer state: it is a reference model,
    # not a resume point.
    if frozen_batches is None and not args.dry_run:
        save_mts_checkpoint(
            output / "init.pt",
            mts_checkpoint_payload(
                kind="transport",
                model=model,
                model_config=model_config,
                token_spec=token_spec,
                tokenizer_metadata=tokenizer_metadata,
                provenance=provenance,
                tokenizer_checkpoint=Path(tokenizer_path),
                metrics={"optimizer_steps": 0, "val_objective": None},
                epoch=0,
                global_step=0,
            ),
        )
        print(f"step 0: initialisation saved to {output / 'init.pt'} (no optimizer state)", flush=True)

    result = trainer.fit(
        train_batches,
        epochs=trainer_config.epochs,
        # Validation runs in the callback exactly once per epoch; passing it here
        # as well would score the protocol twice and let the two numbers differ.
        monitor_batches=monitor_batches if frozen_batches is not None else None,
        on_epoch_end=on_epoch_end,
        max_seconds=args.max_wall_seconds,
    )
    history_path = output / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for entry in result["history"]:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    shortfall = result["steps_shortfall"]
    interrupted = bool(result.get("interrupted"))
    completed = shortfall in (None, 0) and not interrupted
    summary = {
        "output": str(output),
        "global_step": result["global_step"],
        "optimizer_steps": int(result["global_step"]),
        "planned_steps": result["planned_steps"],
        "steps_shortfall": shortfall,
        "completed": bool(completed),
        "interrupted": interrupted,
        "interrupt_reason": result.get("interrupt_reason"),
        "budget_source": budget["source"],
        "content_condition": {
            "kind": content_vocabulary.kind,
            "classes": list(content_vocabulary.classes),
            "unconditional": content_vocabulary.unconditional,
        },
        "epochs_run": len(result["history"]),
        "last_epoch": result["history"][-1] if result["history"] else {},
        "history_file": history_path.name,
        "supervised_tokens_total": sum(
            float(entry.get("supervised_tokens_total", 0.0) or 0.0) for entry in result["history"]
        ),
        "best_val_loss": best_loss if saved_best else None,
        "validation_protocol_id": None if validation_protocol is None else validation_protocol.protocol_id,
        "validation_protocol_hash": None if validation_protocol is None else validation_protocol.fingerprint(),
        "overfit_windows": frozen_window_count or None,
        "resume": {
            "supported": False,
            "warm_start_from": str(args.warm_start) if args.warm_start else None,
            "note": "a warm start begins a new run: optimizer, step counter, best metric and "
            "RNG start at zero",
        },
        "token_spec_hash": token_spec.fingerprint(),
        "layout_hash": adapter.layout_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "token_embed_dim": int(model.token_embed_dim),
        "architecture_revision": int(model.config()["architecture_revision"]),
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    if not completed:
        reason = (
            "the wall-clock cap stopped it"
            if interrupted
            else f"it is {shortfall} steps short of the stated {result['planned_steps']}"
        )
        print(
            f"the run took {result['global_step']} optimizer steps and is not a completed run: "
            f"{reason} ({budget['source']})",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
