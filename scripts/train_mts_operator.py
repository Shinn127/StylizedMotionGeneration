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
import itertools
import json
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator import (  # noqa: E402
    MASK_KINDS,
    LayoutAdapter,
    MaskGenerator,
    MotionTransportTransformer,
    OperatorBatch,
    build_operator,
    load_mts_checkpoint,
    mts_checkpoint_payload,
    save_mts_checkpoint,
)
from stylized_motion.learning.mts_operator.content_schema import (  # noqa: E402
    apply_content_schema,
    content_schema_from_config,
)
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    build_provenance,
    file_sha256,
    load_operator_bundle,
    store_identity_block,
    training_exposure,
)
from stylized_motion.learning.mts_operator.eval_protocol import (  # noqa: E402
    ValidationBatchBuilder,
    ValidationProtocol,
    validation_evidence,
)
from stylized_motion.learning.mts_operator.model import MtsStyleOperator  # noqa: E402
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    StylePairSampler,
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.style_encoder import (  # noqa: E402
    ConstantStyleEncoder,
    GlobalStyleEncoder,
    StyleIDEncoder,
)
from stylized_motion.learning.mts_operator.training import (  # noqa: E402
    RUN_ARTIFACTS,
    OperatorTrainer,
    TrainerConfig,
    planned_step_budget,
    refuse_existing_output,
    resolve_budget,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    ContentVocabulary,
    PairedBatchSource,
    TokenSource,
    windows_by_clip,
)
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
    "position_encoding",
    "output_dim",
    "pooling",
    # N05b: a reference encoder trained by the supervised seen-style task can be
    # loaded here instead of starting from a fresh init.  The path is not a
    # constructor argument; it is popped before construction and the weights are
    # restored with the tokenizer binding checked.
    "checkpoint",
)
STYLE_ID_ENCODER_KEYS = ("num_styles", "output_dim", "dim")
#: The no-reference control takes a width and nothing else: no reference, no id.
CONSTANT_ENCODER_KEYS = ("output_dim", "dim")
# Options every family accepts, plus the kind-specific ones.  A config may
# carry both (so `--operator` can switch families), but only the chosen family's
# keys are passed on: a typo still fails because the *union* is validated.
COMMON_OPERATOR_KEYS = ("hidden_dim", "coordinate_dim")
OPERATOR_SPECIFIC_KEYS = {
    "logit_field": (),
    "arbitrary_kernel": ("identity_mix",),
    "birth_death": ("max_rate", "uniformization_tolerance", "max_terms", "level_order"),
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
    parser.add_argument("--batch-size", type=int, default=None, help="Overrides loader.batch_size.")
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="Overrides operator.hidden_dim and the style encoder width."
    )
    parser.add_argument(
        "--style-encoder-kind",
        choices=["reference", "style_id", "constant"],
        default=None,
        help="reference (Phase 4) vs style_id (Phase 3 sandbox) vs constant (the "
        "no-reference control, which reads no style input at all); overrides "
        "style_encoder.kind.",
    )
    parser.add_argument("--num-styles", type=int, default=None, help="Required for style_id encoders.")
    parser.add_argument(
        "--shuffled-adjacency",
        type=int,
        default=None,
        metavar="SEED",
        help="Geometry control: permute which FSQ levels count as neighbours (birth_death only).",
    )
    parser.add_argument(
        "--overfit-pairs", type=int, default=0,
        help="Freeze this many reference/target pairs and train on them repeatedly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the run (model, parameters, maps, identities, budget) without "
        "training and without writing best.pt.",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Start a NEW run from these weights (same architecture revision): optimizer, step "
        "counter, best metric and RNG all start from zero. This is not a resume.",
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
    return parser


def _load_reference_encoder(
    path: Path, encoder: Any, *, adapter: Any, tokenizer_checkpoint: Path
) -> dict[str, Any]:
    """Restores the supervised encoder's weights into a fresh module.

    Refuses when the checkpoint's classes or width disagree with this run: an
    encoder trained for three styles cannot be dropped into a four-style index, and
    a silent partial load would leave the operator reading noise.
    """
    from stylized_motion.learning.mts_operator.checkpoint import require_tokenizer_checkpoint

    if not path.exists():
        raise FileNotFoundError(f"style_encoder.checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "style_encoder":
        raise ValueError(
            f"{path} is a {checkpoint.get('kind')!r} checkpoint; style_encoder.checkpoint needs the "
            "'style_encoder' kind the supervised task writes"
        )
    require_tokenizer_checkpoint(
        checkpoint, tokenizer_checkpoint=tokenizer_checkpoint, where="reference encoder checkpoint"
    )
    stored = dict(checkpoint["metadata"]["model_config"]["style_encoder"])
    stored.pop("kind", None)
    for key, value in stored.items():
        if key == "checkpoint":
            continue
        current = getattr(encoder, key, None)
        if current is not None and key in {"output_dim", "dim"} and int(current) != int(value):
            raise ValueError(
                f"The stored encoder has {key}={value} but this recipe builds {current}; the "
                "weights would not fit"
            )
    state = {key.removeprefix("."): value for key, value in checkpoint["model"].items()}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Loading {path} left missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}; "
            "the stored encoder is not this architecture"
        )
    encoder.eval()
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "step": int(checkpoint.get("global_step", 0)),
        "classes": list(checkpoint["metadata"]["model_config"].get("classes") or []),
        "metrics": dict(checkpoint.get("metrics") or {}),
        "provenance_training_protocol": (checkpoint.get("provenance") or {}).get("training_protocol_id"),
    }


def _trainable_style_index(sampler: Any, *, mode: str) -> dict[str, int]:
    """Style ids for styles that are trainable *and* can form a legal pair."""
    eligible = sampler.eligible_targets(stage="train")
    styles: set[str] = set()
    for target in eligible:
        if sampler.pairs_for(target, mode=mode, count=1, stage="train"):
            styles.add(str(target.style))
    if not styles:
        raise ValueError(
            "No style can form a legal training pair, so a style-ID run has nothing to "
            "condition on; check data.pairs.mode and the held-out styles"
        )
    return {style: index for index, style in enumerate(sorted(styles))}


def apply_warm_start(
    model: Any,
    path: Path,
    *,
    adapter: LayoutAdapter,
    tokenizer_identity: Mapping[str, object] | None,
    tokenizer_checkpoint: Path | None = None,
    device: torch.device | str,
) -> str:
    """Loads same-revision weights into ``model`` and nothing else.

    Weights only: no optimizer state, no step counter, no best metric, no RNG.  The
    caller builds a fresh trainer afterwards, so the run starts at step zero with a
    fresh best -- this is a *warm start*, not a resume, and exact resume is not
    implemented (the saved optimizer block is provenance, not a restored state).
    """
    _, warm_model = load_operator_bundle(
        path,
        adapter=adapter,
        tokenizer_identity=tokenizer_identity,
        tokenizer_checkpoint=tokenizer_checkpoint,
        device=device,
    )
    stored = warm_model.state_dict()
    current = model.state_dict()
    missing = sorted(set(current) - set(stored))
    mismatched = sorted(
        key
        for key, value in current.items()
        if key in stored and tuple(value.shape) != tuple(stored[key].shape)
    )
    if missing or mismatched:
        raise ValueError(
            "Warm-start weights do not fit this architecture "
            f"(missing={missing[:3]}, shape mismatch={mismatched[:3]}); the stored revision "
            "differs, so retrain from scratch instead of partially loading"
        )
    model.load_state_dict(stored)
    return str(path)


def resolve_operator_config(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    """Applies CLI overrides to a freshly loaded config, in one stated order.

    YAML first, then the CLI, then (in ``main``) the data/model derivations and
    the validation.  An override that only reached the parser would be silently
    ignored, which is how the first sandbox runs trained reference encoders while
    the log claimed style ids.
    """
    resolved = {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in config.items()
    }
    # Each override merges into the *current* resolved section, so two flags that
    # touch the same section accumulate instead of the later one silently reverting
    # the earlier one (``--hidden-dim`` followed by ``--style-encoder-kind style_id``
    # used to reset the width back to the YAML value).
    if args.feature_database is not None:
        resolved["data"] = {**resolved["data"], "fsq_window_index": str(args.feature_database)}
    if args.batch_size is not None:
        resolved["loader"] = {
            **dict(resolved.get("loader") or {}),
            "batch_size": int(args.batch_size),
        }
    if args.hidden_dim is not None:
        resolved["operator"] = {**resolved["operator"], "hidden_dim": int(args.hidden_dim)}
        resolved["style_encoder"] = {
            **resolved["style_encoder"],
            "dim": int(args.hidden_dim),
            "output_dim": int(args.hidden_dim),
        }
    if args.style_encoder_kind is not None:
        encoder_overrides: dict[str, object] = {"kind": args.style_encoder_kind}
        if args.style_encoder_kind == "style_id":
            if args.num_styles is None:
                raise ValueError("--style-encoder-kind style_id requires --num-styles")
            encoder_overrides["num_styles"] = int(args.num_styles)
            encoder_overrides["output_dim"] = int(
                args.hidden_dim or resolved["style_encoder"].get("output_dim") or 256
            )
        elif args.style_encoder_kind == "constant":
            if args.num_styles is not None:
                raise ValueError(
                    "--style-encoder-kind constant takes no num_styles: the control must not "
                    "know how many styles exist"
                )
            encoder_overrides["output_dim"] = int(
                args.hidden_dim or resolved["style_encoder"].get("output_dim") or 256
            )
        resolved["style_encoder"] = {**resolved["style_encoder"], **encoder_overrides}
    return resolved


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


#: Fields a revision-2 run does not honour yet.  A config that sets one to a
#: non-default value is refused instead of being printed as "ignored": a knob
#: that looks like it does something is worse than no knob.
UNSUPPORTED_TRAINING_FIELDS: dict[str, object] = {
    "val_every_steps": 0,
    "precision": "fp32",
    "amp": False,
}


def validate_revision2_training(training: Mapping[str, Any], *, where: str) -> None:
    for field, allowed in UNSUPPORTED_TRAINING_FIELDS.items():
        if field not in training:
            continue
        value = training[field]
        if value in (None, allowed):
            continue
        raise ValueError(
            f"{where}: {field}={value!r} is not supported in revision 2 (only {allowed!r}); "
            "validation runs at epoch end and the first revision-2 round is fp32 only"
        )
    unknown = sorted(set(training) - set(TRAINING_FIELDS))
    if unknown:
        raise ValueError(f"{where}: unknown training fields {unknown}")


TRAINING_FIELDS = frozenset(
    {
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "precision",
        "amp",
        "seed",
        "log_every_steps",
        "steps_per_epoch",
        "max_steps",
        "output_dir",
        "content_weight",
        "strength_range",
        "val_every_steps",
        "checkpoint_every_steps",
        # N05b ablations: a separate encoder learning rate, and the single auxiliary
        # seen-style CE term.  Both default to off.
        "style_encoder_lr",
        "aux_style_ce_weight",
        "aux_style_classes",
    }
)


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
    validate_revision2_training(training, where=str(args.config))
    evaluation_config = dict(config.get("evaluation") or {})
    validation_kinds = tuple(evaluation_config.get("validation_kinds") or MASK_KINDS)
    unknown_eval = sorted(
        set(evaluation_config)
        - {"validation_batches_per_kind", "validation_kinds", "protocol_id", "seed"}
    )
    if unknown_eval:
        raise ValueError(f"Unknown evaluation fields {unknown_eval}")
    # The validation rows are frozen once for the *experiment*, not per training
    # seed: two seeds of the same recipe must score the same rows, or the seed axis
    # silently changes the measurement as well as the model.  A recipe without
    # ``evaluation.seed`` keeps the historical behaviour (the training seed).
    protocol_seed = evaluation_config.get("seed")
    protocol_seed = int(training.get("seed", 3407)) if protocol_seed is None else int(protocol_seed)
    # The frozen protocol's name: profile recipes and full recipes must not share
    # an id, and the checkpoint records which protocol selected its best epoch.
    protocol_id = str(evaluation_config.get("protocol_id") or "").strip()
    if not protocol_id:
        raise ValueError(
            "evaluation.protocol_id is required: the run must name the frozen protocol it "
            "selects its best checkpoint on"
        )
    batches_per_kind = int(evaluation_config.get("validation_batches_per_kind", 1))
    if batches_per_kind <= 0:
        raise ValueError("evaluation.validation_batches_per_kind must be positive")
    unknown_kinds = sorted(set(validation_kinds) - set(MASK_KINDS))
    if unknown_kinds:
        raise ValueError(
            f"evaluation.validation_kinds has unknown entries {unknown_kinds}; "
            f"expected a subset of {list(MASK_KINDS)}"
        )
    reference_frames = config["data"].get("reference_frames")
    frames_config = int(config["data"].get("frames", 64))
    if reference_frames is not None and int(reference_frames) != frames_config:
        raise ValueError(
            "data.reference_frames must equal data.frames in revision 2: reference windows "
            f"of a different length are not implemented (got {int(reference_frames)} vs "
            f"{frames_config})"
        )
    trainer_config = TrainerConfig.from_mapping(training)
    set_seed(trainer_config.seed, deterministic=False)
    device = choose_device(args.device)
    config = resolve_operator_config(config, args)

    tokenizer_path = args.tokenizer_checkpoint or config["tokenizer"].get("checkpoint")
    if tokenizer_path is None:
        raise ValueError("tokenizer.checkpoint or --tokenizer-checkpoint is required")
    tokenizer_checkpoint, tokenizer = load_representation_checkpoint(Path(tokenizer_path), torch.device("cpu"))
    if tokenizer.family != NEF_FSQ_FAMILY:
        raise ValueError(f"The MTS operator requires a {NEF_FSQ_FAMILY!r} tokenizer")
    tokenizer = tokenizer.to(device).eval()
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
        # The upstream transport must have been trained with *this* tokenizer
        # file: the same architecture with different weights would silently
        # condition the operator on a different token space.
        tokenizer_checkpoint=Path(tokenizer_path),
    )
    transport.eval()


    # Unset YAML keys are dropped, and each kind only accepts its own options,
    # so a config can carry both kinds' keys without confusing either one.
    # A config may carry both kinds' keys (so `--style-encoder-kind` can switch
    # families); validate against the union to catch typos, then pass only the
    # chosen kind's arguments.
    style_config = {
        key: value for key, value in dict(config["style_encoder"]).items() if value is not None
    }
    kind = str(style_config.pop("kind", "reference"))
    union = (
        set(REFERENCE_ENCODER_KEYS) | set(STYLE_ID_ENCODER_KEYS) | set(CONSTANT_ENCODER_KEYS) | {"freeze"}
    )
    unknown = sorted(set(style_config) - union)
    if unknown:
        raise ValueError(f"Unknown style_encoder options {unknown}")
    if kind == "constant":
        # A config may carry both reference/style-id key sets so the kind can be
        # switched by flag; the *constant* control is different: it owns no encoder
        # to configure, and a config that looks like it has one would misreport the
        # control's parameter count.
        stray = sorted(set(style_config) - set(CONSTANT_ENCODER_KEYS))
        if stray:
            raise ValueError(
                f"style_encoder.kind=constant must not carry {stray}: the no-reference control "
                "reads no style input and has only a width"
            )
    encoder_checkpoint = None
    if kind == "reference":
        style_encoder = GlobalStyleEncoder(
            adapter,
            **{
                key: value
                for key, value in style_config.items()
                if key in REFERENCE_ENCODER_KEYS and key != "checkpoint"
            },
        )
        encoder_source = style_config.get("checkpoint")
        if encoder_source:
            # A pretrained encoder is a *different experiment* from a fresh init:
            # the file, its SHA and the classes it was trained on travel with the
            # checkpoint, and a mismatch in width refuses to load.
            encoder_checkpoint = _load_reference_encoder(
                Path(str(encoder_source)), style_encoder, adapter=adapter,
                tokenizer_checkpoint=args.tokenizer_checkpoint or Path(config["tokenizer"]["checkpoint"]),
            )
    elif kind == "style_id":
        if "num_styles" not in style_config:
            raise ValueError("style_encoder.kind=style_id requires num_styles")
        style_encoder = StyleIDEncoder(
            num_styles=int(style_config["num_styles"]),
            output_dim=int(style_config.get("output_dim") or style_config.get("dim") or 256),
        )
    elif kind == "constant":
        # The no-reference control: a learned descriptor that reads no style input.
        style_encoder = ConstantStyleEncoder(
            output_dim=int(style_config.get("output_dim") or style_config.get("dim") or 256)
        )
    else:
        raise ValueError(
            f"Unknown style_encoder.kind {kind!r}; expected reference, style_id or constant"
        )
    print(f"style encoder: {kind} -> {style_encoder.config()}", flush=True)

    operator_config = dict(config["operator"])
    # Pop before choosing: `args.operator or config.pop(...)` would skip the pop
    # whenever the CLI flag is set, leaving `name` in the constructor kwargs.
    configured_operator = str(operator_config.pop("name", "birth_death"))
    operator_name = args.operator or configured_operator
    operator_config.pop("strength", None)
    unknown = sorted(set(operator_config) - set(OPERATOR_KEYS))
    if unknown:
        raise ValueError(f"Unknown operator options {unknown}")
    if args.shuffled_adjacency is not None:
        if operator_name != "birth_death":
            raise ValueError("--shuffled-adjacency only applies to the birth_death operator")
        import numpy as _np

        order = [
            int(value)
            for value in _np.random.default_rng(int(args.shuffled_adjacency)).permutation(adapter.num_levels)
        ]
        operator_config["level_order"] = order
        print(f"shuffled adjacency (seed {args.shuffled_adjacency}): {order}", flush=True)
    accepted = set(COMMON_OPERATOR_KEYS) | set(OPERATOR_SPECIFIC_KEYS.get(operator_name, ()))
    unsupported = sorted(set(operator_config) - accepted)
    if unsupported:
        raise ValueError(
            f"operator options {unsupported} are not implemented by {operator_name!r}; remove "
            "them instead of relying on them being ignored"
        )
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
        freeze_style_encoder=bool(style_config.get("freeze", False)),
    )
    warm_start_source = None
    if args.warm_start is not None:
        warm_start_source = apply_warm_start(
            model,
            args.warm_start,
            adapter=adapter,
            tokenizer_identity=tokenizer.representation_metadata(),
            tokenizer_checkpoint=Path(tokenizer_path),
            device=device,
        )
        print(
            f"warm start from {warm_start_source}: weights loaded, optimizer/step/best reset",
            flush=True,
        )

    store_path = args.token_store or config["data"].get("token_store")
    feature_path = args.feature_database or config["data"].get("fsq_window_index")
    if store_path:
        store = open_any_token_store(Path(store_path))
    elif feature_path:
        store = open_any_feature_store(feature_path)
    else:
        raise ValueError("data.token_store or data.fsq_window_index is required")

    style_split_config = dict(config["data"].get("style_split") or {})
    dataset = config["data"].get("dataset")
    records = clip_records_from_store(store, dataset=None if dataset is None else str(dataset))
    # N04: the same content map the transport was trained with has to be applied here,
    # or the operator would build ids for labels the transport never learned.
    _content_config = dict(config["data"].get("content") or {})
    content_schema, content_schema_report = content_schema_from_config(_content_config, root=REPO_ROOT)
    records, content_mapping = apply_content_schema(
        records, content_schema, strict=bool(content_schema_report.get("strict"))
    )
    content_schema_report = {**content_schema_report, "mapping": content_mapping}
    if content_mapping.get("applied"):
        print(
            f"content schema v{content_mapping['version']}: {content_mapping['classes_before']} raw "
            f"labels -> {content_mapping['classes_after']} canonical classes",
            flush=True,
        )
    style_split = split_styles_by_performer(
        records,
        val_fraction=float(style_split_config.get("val_fraction", 0.2)),
        unseen_fraction=float(style_split_config.get("unseen_fraction", 0.2)),
        seed=trainer_config.seed,
    )
    pair_config = dict(config["data"].get("pairs") or {})
    held_out_styles = tuple(str(style) for style in (pair_config.get("held_out_styles") or ()))
    if held_out_styles:
        print(f"held-out styles (never used for operator training): {list(held_out_styles)}", flush=True)
    frames = int(config["data"].get("frames", 64))
    target_sampling = str(pair_config.get("target_sampling", "clip_uniform"))
    content_config = dict(config["data"].get("content") or {})
    content_kind = str(content_config.get("kind", "none"))
    if content_kind not in {"none", "action_id"}:
        raise ValueError(f"Unknown data.content.kind {content_kind!r}; expected none or action_id")
    sampler = StylePairSampler(
        records,
        style_split=style_split,
        seed=trainer_config.seed,
        held_out_styles=held_out_styles,
        window_frames=frames,
        target_sampling=target_sampling,
    )
    print(
        f"pair sampling: mode={pair_config.get('mode', 'same_style')} "
        f"target_sampling={target_sampling} window_frames={frames} "
        f"held_out={list(held_out_styles)}",
        flush=True,
    )
    pair_report = sampler.pair_report(
        count=256,
        mode=str(pair_config.get("mode", "same_style")),
        stage="train",
    )
    # The vocabulary is frozen from the *training* split, sorted, and carried in
    # the checkpoint; nothing here looks at the evaluation split.
    content_vocabulary = ContentVocabulary.build(
        sampler.eligible_targets(stage="train"), kind=content_kind
    )
    # The style-id sandbox may only name styles the operator can actually learn
    # from: a held-out, test-only or evidence-free style would get an embedding with
    # no training signal and then be reported as if it meant something.
    style_index = None
    aux_only_style_index = False
    if kind == "style_id":
        style_index = _trainable_style_index(
            sampler, mode=str(pair_config.get("mode", "same_style"))
        )
    elif float(training.get("aux_style_ce_weight") or 0.0) > 0.0:
        # A reference arm with the auxiliary CE still needs a style->id map: it
        # labels the *reference* for the auxiliary head, not the encoder.  The map is
        # the same pairable set the style-ID arms use, and it is recorded as
        # aux-only so a reader cannot mistake it for a conditioning id.
        style_index = _trainable_style_index(
            sampler, mode=str(pair_config.get("mode", "same_style"))
        )
        aux_only_style_index = True
        declared_styles = int(config["style_encoder"].get("num_styles") or 0)
        if declared_styles and declared_styles != len(style_index):
            raise ValueError(
                f"style_encoder.num_styles={declared_styles} but {len(style_index)} styles can "
                f"actually be trained: {sorted(style_index)}"
            )
        print(
            f"style ids ({len(style_index)} trainable, pairable styles): {style_index}",
            flush=True,
        )
    for key in ("style_encoder_lr", "aux_style_ce_weight"):
        value = training.get(key)
        if value is not None and float(value) <= 0.0:
            raise ValueError(f"training.{key} must be positive when given (got {value!r})")
    if float(training.get("aux_style_ce_weight") or 0.0) > 0.0 and not style_index:
        raise ValueError(
            "training.aux_style_ce_weight needs a style index for the reference's style id; the "
            "auxiliary term is defined on the paired reference's own style"
        )
    if float(training.get("aux_style_ce_weight") or 0.0) > 0.0:
        training["aux_style_classes"] = int(len(style_index))
    transport_vocabulary = getattr(transport, "content_vocabulary", None)
    if transport_vocabulary is not None:
        # The operator inherits the frozen transport's map: it may use a *subset* of
        # the transport's actions, but never re-index one.
        if transport_vocabulary.kind != content_vocabulary.kind:
            raise ValueError(
                f"The frozen transport conditions on {transport_vocabulary.kind!r} but this "
                f"run asks for {content_vocabulary.kind!r}"
            )
        unknown = sorted(set(content_vocabulary.classes) - set(transport_vocabulary.classes))
        if unknown:
            raise ValueError(
                f"Actions {unknown} are missing from the frozen transport's vocabulary "
                f"({list(transport_vocabulary.classes)}); the operator cannot introduce ids "
                "the transport never learned"
            )
        content_vocabulary = transport_vocabulary
    elif not content_vocabulary.unconditional:
        raise ValueError(
            "data.content.kind=action_id needs a transport trained with a content "
            "conditioner, but the frozen transport has none"
        )
    if content_vocabulary.unconditional:
        print(
            "content condition: unconditional (data.content.kind=none); the model is NOT "
            "conditioned on the action label",
            flush=True,
        )
    else:
        print(
            f"content condition: action_id with {len(content_vocabulary.classes)} classes "
            f"from the training split: {list(content_vocabulary.classes)}",
            flush=True,
        )
    print(
        "pair coverage: "
        f"styles={len(pair_report['pairs_per_style'])}/{len(sampler.style_split.train_styles)} "
        f"actions={len(pair_report['pairs_per_action'])} "
        f"actors={sorted(pair_report['pairs_per_actor'])} "
        f"rejections={pair_report['rejections']} "
        f"targets_skipped={pair_report['targets_skipped']}",
        flush=True,
    )
    loader_config = dict(config.get("loader") or {})

    def windows_for(split: str) -> dict[int, list[Any]]:
        return windows_by_clip(store, split, frames=frames)

    history = int(tokenizer.history_frames)
    feature_stats = tokenizer_checkpoint.get("feature_stats")
    sources = {
        split: PairedBatchSource(
            token_source=TokenSource(
                store=store,
                windows_by_clip=windows_for(split),
                tokenizer=None if store_path else tokenizer,
                feature_stats=None if store_path else feature_stats,
                adapter=adapter,
                frames=frames,
                history=history,
                rng=np.random.default_rng(trainer_config.seed),
            ),
            sampler=sampler,
            mask_generator=MaskGenerator(dict(config["masking"])),
            device=device,
            batch_size=int(loader_config.get("batch_size", 32)),
            mode=str(pair_config.get("mode", "same_style")),
            stage=split,
            strength=tuple(strength_range) if strength_range else None,
            seed=trainer_config.seed,
            style_index=style_index,
            content_vocabulary=content_vocabulary,
            target_sampling=target_sampling,
        )
        for split in ("train", "val")
    }

    # Artifact binding: the store identities, the action/style maps, the resolved
    # config and the code identity all travel with the checkpoint.
    store_identity = store_identity_block(
        store,
        store_kind="token" if store_path else "feature",
        store_path=store_path or feature_path,
    )
    provenance = build_provenance(
        store_identity=store_identity,
        action_vocabulary=content_vocabulary.as_dict(),
        style_index=style_index,
        resolved_config=config,
        seed=trainer_config.seed,
        upstream_transport_sha256=file_sha256(transport_path),
        training_protocol_id="mts-operator-revision2",
        content_schema=content_schema_report,
    )
    provenance["warm_start"] = warm_start_source
    if encoder_checkpoint is not None:
        provenance["pretrained_style_encoder"] = encoder_checkpoint
    provenance["validation_protocol_id"] = protocol_id
    # What the operator could actually learn from, recorded with the checkpoint:
    # an evaluation may *claim* a style was held out, and this is the fact the
    # claim is checked against.
    provenance["training_exposure"] = training_exposure(
        trainable_styles=sorted(style_index) if style_index else sorted(style_split.train_styles),
        held_out_styles=sorted(held_out_styles) or None,
        actions=list(content_vocabulary.classes) or None,
        pairs=pair_report["pairs_per_style"],
    )

    # The run directory and the frozen protocol are resolved *before* the dry-run
    # return: the protocol is a no-training artifact, and a preflight needs it to
    # answer "is this ready to train" without the "train first to validate" cycle.
    output = Path(args.output or training.get("output_dir", "outputs/mts_operator/run"))
    output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        refuse_existing_output(
            output, allow=bool(args.allow_existing_output), where="operator run"
        )

    validation_protocol = None
    validation_builder = None
    if int(args.overfit_pairs) == 0:
        protocol_samples = int(evaluation_config.get("validation_batches_per_kind", 1))
        validation_protocol = ValidationProtocol.build(
            sampler,
            protocol_id=protocol_id,
            token_source=sources["val"].tokens,
            kinds=tuple(evaluation_config.get("validation_kinds") or MASK_KINDS),
            batches_per_kind=protocol_samples,
            batch_size=int(loader_config.get("batch_size", 32)),
            frames=frames,
            seed=protocol_seed,
            split="val",
            stage="val",
            target_sampling=target_sampling,
            # The model can only be conditioned on the *transport's* frozen action
            # vocabulary: a val row with another label cannot be scored, so those
            # rows are excluded and counted instead of borrowing an id.
            content_vocabulary=content_vocabulary,
            # The rows have to state the mask they were frozen with: a row that says
            # "stream" without the ratios does not describe the same experiment, and
            # the preflight compares the recorded block with the recipe's.
            mask_generator=MaskGenerator(dict(config["masking"])),
        )
        validation_builder = ValidationBatchBuilder(
            token_source=sources["val"].tokens,
            mask_generator=MaskGenerator(dict(config["masking"])),
            adapter=adapter,
            device=device,
            content_vocabulary=content_vocabulary,
            # A style-ID validation batch carries the style ids the encoder expects,
            # taken from the checkpoint's own map -- without them the encoder has
            # nothing to condition on and validation cannot run at all.
            style_index=style_index,
            encoder_kind=kind,
        )
        validation_path = validation_protocol.write(output / "validation_protocol.json")
        print(
            f"frozen validation protocol {protocol_id!r}: "
            f"{len(validation_protocol.samples)} items, "
            f"{protocol_samples} batches/kind, kinds={list(validation_protocol.kinds)}, "
            f"splits={sorted({sample.split for sample in validation_protocol.samples})}, "
            f"written to {validation_path}",
            flush=True,
        )

    if args.dry_run:
        resolved = {
            "config": str(args.config),
            "operator": operator_name,
            "operator_config": {key: value for key, value in operator_config.items() if key in accepted},
            "style_encoder": kind,
            "transport_parameters": int(sum(p.numel() for p in transport.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in model.trainable_parameters())),
            "token_embed_dim": int(getattr(transport, "token_embed_dim", 0)),
            "architecture_revision": transport.config().get("architecture_revision"),
            "style_index": style_index,
            "style_encoder_checkpoint": encoder_checkpoint,
            "action_vocabulary": content_vocabulary.as_dict(),
            "content_schema": content_schema_report,
            "store_identity": store_identity,
            "mask_mixture": dict(config["masking"]),
            "training": {
                "epochs": trainer_config.epochs,
                "steps_per_epoch": trainer_config.steps_per_epoch,
                "max_steps": trainer_config.max_steps,
                "batch_size": int(loader_config.get("batch_size", 32)),
                "lr": trainer_config.lr,
                "precision": trainer_config.precision,
                "seed": trainer_config.seed,
            },
            "evaluation": {
                "protocol_id": protocol_id,
                "seed": protocol_seed,
                "validation_batches_per_kind": batches_per_kind,
                "validation_kinds": list(validation_kinds),
            },
            "validation": {
                "protocol_id": None if validation_protocol is None else validation_protocol.protocol_id,
                "items": 0 if validation_protocol is None else len(validation_protocol.samples),
                "kinds": [] if validation_protocol is None else list(validation_protocol.kinds),
                "splits": []
                if validation_protocol is None
                else sorted({sample.split for sample in validation_protocol.samples}),
                "protocol_hash": None
                if validation_protocol is None
                else validation_protocol.fingerprint(),
                "overfit_pairs": int(args.overfit_pairs),
            },
            "target_sampling": target_sampling,
            "pair_report": pair_report,
            "output": str(output),
            "warm_start": warm_start_source,
            "dry_run": True,
        }
        (output / "dry_run.json").write_text(
            json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(resolved, indent=2, default=str))
        print("dry run: nothing was trained and no checkpoint was written", flush=True)
        return

    frozen_pairs: list[OperatorBatch] | None = None
    if args.overfit_pairs > 0:
        batch = sources["train"].batch(size=int(args.overfit_pairs))
        if batch is None:
            raise ValueError("No audited style pairs were available to overfit on")
        frozen_pairs = [batch]
        print(
            f"overfit mode: {int(batch.target_tokens.shape[0])} frozen pairs "
            "(training monitor only, never a validation-best)",
            flush=True,
        )

    budget = resolve_budget(trainer_config, where="operator run")
    print(
        f"step budget: {budget['planned_steps']} optimizer steps ({budget['source']})",
        flush=True,
    )
    if frozen_pairs is not None:
        (output / "overfit_frozen.json").write_text(
            json.dumps(
                {
                    "pairs": int(frozen_pairs[0].target_tokens.shape[0]),
                    "repeated_to_steps": int(budget["planned_steps"]),
                    "planned_steps_source": budget["source"],
                    "masking": dict(config["masking"]),
                    "content_condition": content_vocabulary.as_dict(),
                    "seed": int(trainer_config.seed),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    def train_batches(epoch: int) -> Iterable[OperatorBatch]:
        if frozen_pairs is not None:
            # Repeat the frozen pairs until the stated budget is met, instead of
            # letting an epoch end after the single frozen batch.
            return itertools.cycle(frozen_pairs)
        # One *batch* per training step: passing the batch size here trained only
        # `batch_size` steps per epoch, which silently capped long runs at a few
        # hundred steps.
        steps = int(trainer_config.steps_per_epoch or trainer_config.max_steps or 1000)
        return sources["train"].batches(steps)

    def monitor_batches(epoch: int) -> Iterable[OperatorBatch]:
        return frozen_pairs or []

    trainer = OperatorTrainer(
        model,
        adapter=adapter,
        device=device,
        config=trainer_config,
        content_weight=content_weight,
    )
    model_config = model.describe()
    best_nll = float("inf")

    def on_epoch_end(
        epoch: int, metrics: Mapping[str, float], active: OperatorTrainer
    ) -> dict[str, Any]:
        """Runs the frozen protocol once and records the same numbers everywhere."""
        nonlocal best_nll
        validation_report = None
        evidence: dict[str, Any] = {}
        val_started = time.perf_counter()
        if validation_protocol is not None:
            # Fixed weights over the frozen kinds, never an average of batch means.
            validation_report = validation_builder.evaluate(
                active, validation_protocol, batch_size=int(loader_config.get("batch_size", 32))
            )
            evidence = validation_evidence(
                validation_protocol,
                validation_report,
                seconds={"val_seconds": time.perf_counter() - val_started},
            )
            objective = validation_report["objective"]
            readable = "n/a" if objective is None else f"{float(objective):.4f}"
            per_kind = {
                kind: round(value, 4)
                for kind, value in validation_report["objectives_per_kind"].items()
            }
            print(
                f"epoch {epoch}: validation objective={readable} "
                f"per_kind={per_kind} counts={validation_report['counts']} "
                f"missing={validation_report['missing_kinds']}",
                flush=True,
            )
        checkpoint_started = time.perf_counter()
        payload = mts_checkpoint_payload(
            kind="operator",
            model=model,
            model_config=model_config,
            token_spec=token_spec,
            tokenizer_metadata=tokenizer_metadata,
            provenance=provenance,
            tokenizer_checkpoint=Path(tokenizer_path),
            metrics={
                "train_loss": metrics.get("loss"),
                "val_nll": evidence.get("val_objective"),
                "val_objective": evidence.get("val_objective"),
                "operator": operator_name,
                "style_encoder_kind": kind,
                "content_weight": content_weight,
                # Recorded so evaluation restores the exact frozen upstream.
                "transport_checkpoint": str(transport_path),
                "tokenizer_checkpoint": str(tokenizer_path),
                "supervised_tokens": metrics.get("supervised_tokens"),
                "skipped_steps": metrics.get("skipped_steps"),
                "optimizer_steps": int(active.global_step),
                **evidence,
            },
            epoch=epoch,
            global_step=active.global_step,
            optimizer=active.optimizer,
            extra={"style_split": style_split.as_dict(), "strength_range": strength_range},
        )
        if validation_report is None:
            # Overfit mode has no validation at all: the checkpoint is written
            # under its own name and is never claimed to be a validation-best.
            save_mts_checkpoint(output / "overfit_last.pt", payload)
            print(f"epoch {epoch}: overfit mode, no validation-best is claimed")
        else:
            save_mts_checkpoint(output / "last.pt", payload)
            objective = validation_report["objective"]
            if objective is None or not validation_report["usable"]:
                # Never fall back to the train loss: an unusable protocol makes
                # this epoch ineligible for best.pt instead of making best easy.
                print(
                    f"epoch {epoch}: the frozen validation protocol supervised no token "
                    f"(missing={validation_report['missing_kinds']}); best.pt untouched"
                )
            else:
                objective = float(objective)
                # A checkpoint with zero optimizer steps is not a validation-best.
                if objective < best_nll and active.global_step > 0:
                    best_nll = objective
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

    result = trainer.fit(
        train_batches,
        epochs=trainer_config.epochs,
        # Validation runs in the callback exactly once per epoch.
        monitor_batches=monitor_batches if frozen_pairs is not None else None,
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
        "operator": operator_name,
        "style_encoder": kind,
        "target_sampling": target_sampling,
        "pair_report": pair_report,
        "validation_protocol": None if validation_protocol is None else validation_protocol.describe(),
        "protocol_id": protocol_id,
        "protocol_hash": None if validation_protocol is None else validation_protocol.fingerprint(),
        "warm_start": warm_start_source,
        "resolved_training": dict(trainer_config.__dict__),
        "content_condition": {
            "kind": content_vocabulary.kind,
            "classes": list(content_vocabulary.classes),
            "unconditional": content_vocabulary.unconditional,
        },
        "global_step": result["global_step"],
        "optimizer_steps": int(result["global_step"]),
        "planned_steps": result["planned_steps"],
        "steps_shortfall": shortfall,
        "completed": bool(completed),
        "interrupted": interrupted,
        "interrupt_reason": result.get("interrupt_reason"),
        "budget_source": budget["source"],
        "epochs_run": len(result["history"]),
        "last_epoch": result["history"][-1] if result["history"] else {},
        "history_file": history_path.name,
        "supervised_tokens_total": sum(
            float(entry.get("supervised_tokens", 0.0) or 0.0) for entry in result["history"]
        ),
        "best_val_nll": None if best_nll == float("inf") else best_nll,
        "overfit_pairs": None
        if frozen_pairs is None
        else int(frozen_pairs[0].target_tokens.shape[0]),
        "resume": {
            "supported": False,
            "warm_start_from": warm_start_source,
            "note": "a warm start begins a new run: optimizer, step counter, best metric and "
            "RNG start at zero",
        },
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
