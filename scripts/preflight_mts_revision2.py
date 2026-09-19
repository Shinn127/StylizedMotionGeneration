#!/usr/bin/env python
"""Revision-2 preflight: can this data/checkpoint combination be trained on?

The question "is the artifact reusable" must be answered *before* a training run,
and it must not be answered by "the files exist".  This script reports one entry
per check with a status, a reason and the evidence it looked at:

* ``data`` level needs only a tokenizer, a store and a recipe -- it answers
  whether the tokens belong to this tokenizer, whether the splits are isolated,
  whether the label tables can carry the style/action evidence the run needs, and
  whether the configured windows are actually readable (a bounded number of them).
* ``experiment`` level adds the upstream bindings (transport/operator checkpoints),
  the frozen protocol and the budget, and is ``blocked`` -- never "passed" -- when
  no revision-2 model exists yet.

Nothing here writes a model or starts training.  The output is ``preflight.json``,
and the exit status is non-zero when a check the caller asked for failed.

    python scripts/preflight_mts_revision2.py \
      --config data/configs/mts_revision2_style.yaml \
      --output outputs/mts_revision2_closure/C10/preflight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.learning.mts_operator import MASK_KINDS  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    checkpoint_action_vocabulary,
    checkpoint_style_index,
    code_identity,
    file_sha256,
    require_token_store_binding,
    source_digest,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    StylePairSampler,
    build_pair_audit,
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource,
    windows_by_clip,
)
from stylized_motion.learning.nef_probe import KinematicContext  # noqa: E402
from stylized_motion.learning.representation import (  # noqa: E402
    NEF_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import choose_device  # noqa: E402

PREFLIGHT_VERSION = 2
#: Windows this script may read.  A preflight is a look, not a pass over the data.
DEFAULT_MAX_WINDOWS = 8

#: What the caller is preparing.  The stage decides which checks are *required*:
#: a transport_train preflight must not demand a trained transport, and an
#: operator_train preflight must not demand the operator it is about to train.
STAGES = ("data", "transport_train", "operator_train", "evaluate")
#: The old ``--level`` values, mapped explicitly rather than reinterpreted.
LEVEL_TO_STAGE = {"data": "data", "experiment": "operator_train"}
#: Required checks per stage.  Anything else is diagnostic: it may be blocked or
#: not_applicable without making the stage unready.
REQUIRED_CHECKS: dict[str, tuple[str, ...]] = {
    "data": (
        "tokenizer",
        "store_identity",
        "split_isolation",
        "label_tables",
        "pair_matrix",
        "windows",
        "mask_distribution",
        "budget",
    ),
    "transport_train": (
        "tokenizer",
        "store_identity",
        "split_isolation",
        "label_tables",
        "pair_matrix",
        "windows",
        "mask_distribution",
        "budget",
        "transport_recipe",
        "validation_protocol",
    ),
    "operator_train": (
        "tokenizer",
        "store_identity",
        "split_isolation",
        "label_tables",
        "pair_matrix",
        "windows",
        "mask_distribution",
        "budget",
        "transport",
        "action_vocabulary",
        "validation_protocol",
    ),
    "evaluate": (
        "tokenizer",
        "store_identity",
        "checkpoint_bindings",
        "evaluate_model",
        "manifest_leakage",
    ),
}
#: Checks that only exist from a given stage on.  ``transport`` is the run itself
#: at the transport_train stage, and an operator checkpoint is the run itself at
#: the operator_train stage, so both are deferred to the stage that needs them.
STAGE_LEVEL = {
    "data": "data",
    "transport_train": "transport",
    "operator_train": "operator",
    "evaluate": "evaluate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight a revision-2 MTS run (data / transport_train / operator_train / evaluate)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=list(STAGES),
        default=None,
        help="data (default): data and recipe only. transport_train: + recipe, frozen protocol, "
        "budget/output path. operator_train: + bound transport, action map, frozen protocol. "
        "evaluate: + the checkpoint, its bindings and a fixed eval manifest.",
    )
    parser.add_argument(
        "--level",
        choices=["data", "experiment"],
        default=None,
        help="Deprecated alias: --level data == --stage data, --level experiment == "
        "--stage operator_train.",
    )
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=None)
    parser.add_argument("--token-store", type=Path, default=None)
    parser.add_argument("--feature-database", type=Path, default=None)
    parser.add_argument("--transport-checkpoint", type=Path, default=None)
    parser.add_argument("--operator-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--validation-protocol",
        type=Path,
        default=None,
        help="The frozen validation_protocol.json the run will score (written by --dry-run).",
    )
    parser.add_argument(
        "--eval-manifest", type=Path, default=None, help="Manifest to audit for leakage."
    )
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def resolve_stage(args: Any) -> tuple[str, str | None]:
    """The requested stage, with the deprecated level mapped explicitly."""
    if args.stage is not None and args.level is not None:
        mapped = LEVEL_TO_STAGE[args.level]
        if mapped != args.stage:
            raise SystemExit(
                f"--level {args.level} maps to --stage {mapped}, which contradicts "
                f"--stage {args.stage}; pass one of them"
            )
    if args.stage is not None:
        return args.stage, None
    if args.level is not None:
        return LEVEL_TO_STAGE[args.level], args.level
    return "data", None


def check(
    identifier: str,
    status: str,
    reason: str,
    *,
    evidence: Mapping[str, Any] | None = None,
    level: str = "data",
) -> dict[str, Any]:
    if status not in {"pass", "fail", "blocked", "not_applicable"}:
        raise ValueError(f"Unknown preflight status {status!r}")
    return {
        "id": identifier,
        "level": level,
        "status": status,
        "reason": reason,
        "evidence": dict(evidence or {}),
    }


def safe(identifier: str, fn: Any, *args: Any, level: str = "data", **kwargs: Any) -> dict[str, Any]:
    """Runs one check, turning an exception into a failing check with its reason."""
    try:
        return fn(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the message is the evidence
        return check(
            identifier,
            "fail",
            f"{type(error).__name__}: {error}",
            level=level,
        )


def _repo_relative(path: Path) -> str | None:
    """The path relative to the repository, or ``None`` when it lies outside it.

    Used by the budget check: the "new runs live under outputs/mts_revision2/"
    rule protects the repository's historical artifacts, and a run directory
    outside the repository (a scratch or test directory) cannot overwrite them.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return None


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    return dict(value)


def check_tokenizer(path: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any]:
    if path is None or not Path(path).exists():
        return (
            check("tokenizer", "fail", f"tokenizer checkpoint {path} does not exist"),
            None,
            None,
        )
    checkpoint, model = load_representation_checkpoint(Path(path), torch.device("cpu"))
    if model.family != NEF_FSQ_FAMILY:
        return (
            check(
                "tokenizer",
                "fail",
                f"the checkpoint holds a {model.family!r} model, not {NEF_FSQ_FAMILY!r}",
            ),
            checkpoint,
            model,
        )
    metadata = model.representation_metadata()
    frozen = bool(config.get("tokenizer", {}).get("freeze", True))
    status = "pass" if frozen else "fail"
    reason = (
        f"{metadata['representation_id']} ({metadata['variant']}), motion_dim="
        f"{int(model.motion_dim)}, RF={int(model.receptive_field)}, "
        f"lookahead={int(model.lookahead_frames)}, history={int(model.history_frames)}"
        if frozen
        else "tokenizer.freeze must stay true: revision 2 never updates the tokenizer"
    )
    return (
        check(
            "tokenizer",
            status,
            reason,
            evidence={
                "path": str(path),
                "sha256": file_sha256(path),
                "representation_id": metadata["representation_id"],
                "motion_dim": int(model.motion_dim),
                "receptive_field": int(model.receptive_field),
                "lookahead_frames": int(model.lookahead_frames),
                "history_frames": int(model.history_frames),
                "num_levels": int(model.num_levels),
                "num_coordinates": int(model.num_coordinates),
            },
        ),
        checkpoint,
        model,
    )


def open_store(token_store: Path | None, feature_database: Path | None, kind: str):
    if kind == "token":
        if not token_store:
            raise ValueError("data.token_store is required for a token run")
        return open_any_token_store(token_store), "token"
    if not feature_database:
        raise ValueError("data.fsq_window_index is required when no token store is configured")
    return open_any_feature_store(feature_database), "feature"


def store_schema_version(store: Any) -> int:
    """The store's data-schema version, from the attribute or its manifest."""
    value = getattr(store, "data_schema_version", None)
    if value is None and isinstance(getattr(store, "manifest", None), Mapping):
        value = store.manifest.get("data_schema_version")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def align_actor_table(store: Any, feature_store: Any | None) -> dict[str, str]:
    """Performer labels for ``store``'s clips, aligned by source identity.

    A token store does not have to carry the actor column, and two stores of the
    same catalogue are only comparable through their source names -- row numbers
    are an alignment *claim*, not evidence.  The feature store is the authority:
    its ``clip_performer_id``/``performer_names`` are attached to the names both
    stores share, and names the other store does not have are reported instead of
    being dropped silently.
    """
    names = [str(value) for value in (getattr(store, "manifest", {}) or {}).get("clip_names", [])]
    if not names:
        return {}
    direct_names = getattr(store, "source_performer_names", None)
    if direct_names:
        ids = getattr(store, "clip_performer_id", None)
        if ids is not None:
            return {
                names[row]: str(direct_names[int(ids[row])])
                for row in range(min(len(names), len(ids)))
                if 0 <= int(ids[row]) < len(direct_names)
            }
    if feature_store is None:
        return {}
    performer_names = getattr(feature_store, "source_performer_names", None)
    performer_ids = getattr(feature_store, "clip_performer_id", None)
    feature_names = [
        str(value) for value in (getattr(feature_store, "manifest", {}) or {}).get("clip_names", [])
    ]
    if not performer_names or performer_ids is None or not feature_names:
        return {}
    by_name = {
        feature_names[row]: str(performer_names[int(performer_ids[row])])
        for row in range(min(len(feature_names), len(performer_ids)))
        if 0 <= int(performer_ids[row]) < len(performer_names)
    }
    return {name: by_name[name] for name in names if name in by_name}


def check_store_identity(
    store: Any,
    *,
    store_kind: str,
    tokenizer_path: Path | None,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "kind": store_kind,
        "motion_dim": int(getattr(store, "motion_dim", 0) or 0),
    }
    problems: list[str] = []
    if store_kind == "token":
        try:
            observed = require_token_store_binding(
                store, tokenizer_checkpoint=tokenizer_path, where="preflight token store"
            )
            evidence.update(observed)
        except ValueError as error:
            return check("store_identity", "fail", str(error), evidence=evidence)
    try:
        observed = validate_store_binding(store, store_kind=store_kind)
        evidence.update({key: value for key, value in observed.items() if value})
    except ValueError as error:
        problems.append(str(error))
    if tokenizer is not None:
        if int(getattr(store, "motion_dim", 0) or 0) != int(tokenizer.motion_dim):
            problems.append(
                f"store motion_dim {int(store.motion_dim)} does not match the tokenizer's "
                f"{int(tokenizer.motion_dim)}"
            )
        store_representation = getattr(store, "representation_id", None)
        if store_kind == "token" and store_representation != tokenizer.representation_id:
            problems.append(
                f"store representation_id {store_representation!r} is not the tokenizer's "
                f"{tokenizer.representation_id!r}"
            )
        required_schema = config.get("data", {}).get("required_data_schema_version")
        schema = store_schema_version(store)
        evidence["data_schema_version"] = schema
        if required_schema is not None and schema != int(required_schema):
            problems.append(
                f"store schema {schema} is not the configured {int(required_schema)}"
            )
    if problems:
        return check("store_identity", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "store_identity",
        "pass",
        "the store reports the identities this run declares (tokenizer SHA included)"
        if store_kind == "token"
        else "the store reports its schema/normalization/split identities",
        evidence=evidence,
    )


def check_splits(
    records: Sequence[Any], *, actor_table: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Split isolation by actor, take group and mirror family.

    ``actor_table`` (clip name -> performer) is used when the store itself carries
    no actor column: the labels come from the aligned catalogue, not from row
    numbers, and the record is reported as exposure-*reported* only when every
    clip of a split really has a label.
    """
    by_split: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    for record in records:
        by_split.setdefault(record.split, []).append(record)
    actor_table = {str(key): str(value) for key, value in (actor_table or {}).items()}
    fallback_names = [
        str(value)
        for value in (getattr(records[0], "clip_name", "") if records else "",)
        if value
    ]

    def actors(split: str) -> set[str]:
        found = {record.performer for record in by_split[split] if record.performer}
        if found or not actor_table:
            return found
        return {
            actor_table[str(getattr(record, "name", ""))] for record in by_split[split]
            if str(getattr(record, "name", "")) in actor_table
        }

    def names(kind: str, split: str) -> set[str]:
        if kind == "actor":
            return actors(split)
        if kind == "take":
            return {
                str(record.source_group)
                for record in by_split[split]
                if int(record.source_group) >= 0
            }
        return {
            f"{record.source_group}:{int(record.variant)}"
            for record in by_split[split]
            if int(record.source_group) >= 0
        }

    counts = {split: len(items) for split, items in by_split.items()}
    labelled_actors = [record for record in records if record.performer]
    actors_known = bool(labelled_actors) or bool(actor_table)
    evidence: dict[str, Any] = {
        "clips": counts,
        "actor_labels_in_store": len(labelled_actors),
        "actor_labels_aligned_from_catalogue": len(actor_table),
        "actor_source": "store"
        if labelled_actors
        else ("aligned_catalogue_by_clip_name" if actor_table else "unavailable"),
        "test_actor_exposure": "unknown" if not actors_known else "reported",
    }
    problems: list[str] = []
    for kind in ("actor", "take", "mirror"):
        train = names(kind, "train")
        val = names(kind, "val")
        test = names(kind, "test")
        if kind == "actor" and not actors_known:
            evidence[f"{kind}_overlap_train_test"] = None
            continue
        evidence[f"{kind}_train"] = len(train)
        evidence[f"{kind}_val"] = len(val)
        evidence[f"{kind}_test"] = len(test)
        evidence[f"{kind}_overlap_train_test"] = len(train & test)
        if train and test and train & test and kind != "mirror":
            problems.append(
                f"{len(train & test)} {kind}(s) appear in both train and test, so a "
                "held-out claim on that axis is not supported"
            )
    if not counts.get("test"):
        problems.append("the test split is empty")
    if not counts.get("train"):
        problems.append("the train split is empty")
    if problems:
        return check("split_isolation", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "split_isolation",
        "pass",
        "train/val/test are disjoint on every axis the store can report"
        + ("" if actors_known else " (no actor table: exposure_unknown)"),
        evidence=evidence,
    )


def check_labels(store: Any, records: Sequence[Any]) -> dict[str, Any]:
    styles = Counter(record.style for record in records)
    actions = Counter(record.content for record in records)
    empty_style = sum(1 for record in records if not record.style)
    empty_content = sum(1 for record in records if not record.content)
    evidence = {
        "styles": len(styles),
        "actions": len(actions),
        "style_names": dict(sorted(styles.items())[:10]),
        "action_names": sorted(actions)[:20],
        "empty_style_labels": empty_style,
        "empty_action_labels": empty_content,
        "distinct": bool(styles) and bool(actions) and len(styles) != 1,
        "child_tokens": "style labels and action labels are separate columns",
    }
    problems: list[str] = []
    if empty_style or empty_content:
        problems.append(
            f"{empty_style} clips have no style label and {empty_content} no action label"
        )
    if len(styles) == 1:
        problems.append("only one style is present, so no style axis exists")
    if not actions:
        problems.append("no action labels are present")
    if problems:
        return check("label_tables", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "label_tables",
        "pass",
        f"{len(styles)} styles and {len(actions)} actions with no missing labels",
        evidence=evidence,
    )


def check_pair_matrix(
    records: Sequence[Any], config: Mapping[str, Any], *, seed: int, frames: int
) -> dict[str, Any]:
    pairs_config = dict(config.get("data", {}).get("pairs") or {})
    style_split_config = dict(config.get("data", {}).get("style_split") or {})
    held_out = tuple(str(value) for value in (pairs_config.get("held_out_styles") or ()))
    mode = str(pairs_config.get("mode", "same_style"))
    target_sampling = str(pairs_config.get("target_sampling", "clip_uniform"))
    style_split = split_styles_by_performer(
        records,
        val_fraction=float(style_split_config.get("val_fraction", 0.2)),
        unseen_fraction=float(style_split_config.get("unseen_fraction", 0.2)),
        seed=seed,
    )
    sampler = StylePairSampler(
        records,
        style_split=style_split,
        seed=seed,
        held_out_styles=held_out,
        window_frames=frames,
        target_sampling=target_sampling,
    )
    audit = build_pair_audit(
        records,
        style_split=style_split,
        sample_pairs=256,
        seed=seed,
        held_out_styles=held_out,
        target_sampling=target_sampling,
        window_frames=frames,
    )
    evidence = {
        "train_styles": list(style_split.train_styles),
        "val_styles": list(style_split.val_styles),
        "test_unseen_styles": list(style_split.test_unseen_styles),
        "held_out_styles": sorted(held_out),
        "mode": mode,
        "target_sampling": target_sampling,
        "pairs_per_style": audit["pair_report"]["pairs_per_style"],
        "pairs_per_action": audit["pair_report"]["pairs_per_action"],
        "rejections": audit["pair_report"]["rejections"],
        "excluded_clips": audit["pair_report"]["excluded_clips"],
        "same_clip_leakage": audit["same_clip_leakage"],
        "warnings": audit["warnings"],
        "single_content_styles": [
            style
            for style, count in audit["content_diversity"].items()
            if int(count) <= 1
        ],
    }
    problems: list[str] = []
    trained_styles = [
        style for style in style_split.train_styles if audit["pair_report"]["pairs_per_style"].get(style)
    ]
    if not trained_styles:
        problems.append(
            "no train style produced a legal "
            f"{mode} pair; the operator would have nothing to learn the style axis from"
        )
    if audit["same_clip_leakage"]:
        problems.append(f"{audit['same_clip_leakage']} sampled pairs leak a clip or take")
    if not style_split.val_styles:
        problems.append("no validation styles were held out; model selection has no style split")
    if problems:
        return check("pair_matrix", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "pair_matrix",
        "pass",
        f"{len(trained_styles)} of {len(style_split.train_styles)} train styles form legal pairs",
        evidence=evidence,
    )


def check_windows(
    store: Any,
    *,
    store_kind: str,
    split: str,
    frames: int,
    max_windows: int,
    tokenizer: Any,
    checkpoint: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """Reads a bounded number of real windows and checks they are usable."""
    cap = int(max_windows)
    if cap < 1:
        return check("windows", "fail", f"max_windows must be positive, got {cap}")
    try:
        grouped = windows_by_clip(store, split, frames=frames)
    except Exception as error:  # noqa: BLE001 - the reason is the evidence
        return check("windows", "fail", f"could not enumerate windows: {error}")
    clip_ids = sorted(grouped)[:cap]
    if not clip_ids:
        return check("windows", "fail", f"split {split!r} has no {frames}-frame window")
    source = TokenSource(
        store=store,
        windows_by_clip={clip: grouped[clip] for clip in clip_ids},
        adapter=None,
        frames=frames,
        history=int(getattr(tokenizer, "history_frames", 0) or 0),
        tokenizer=None if store_kind == "token" else tokenizer,
        feature_stats=None
        if store_kind == "token"
        else (checkpoint or {}).get("feature_stats"),
        rng=np.random.default_rng(int(seed)),
    )
    read = 0
    starts: list[int] = []
    shapes: set[tuple[int, int]] = set()
    for clip in clip_ids:
        sample = source.window(int(clip))
        if sample is None:
            continue
        read += 1
        starts.append(int(sample.metadata.get("target_start", -1)))
        shapes.add(tuple(int(value) for value in sample.tokens.shape))
        if sample.tokens.numel() and int(sample.tokens.max()) >= 8:
            pass
    evidence = {
        "split": split,
        "clips_available": len(grouped),
        "clips_read": read,
        "max_windows": cap,
        "frames": frames,
        "token_shapes": sorted(shapes),
        "target_starts": starts,
        "non_zero_starts": sum(1 for value in starts if value > 0),
        "cap_respected": read <= cap,
    }
    if not read:
        return check("windows", "fail", "no window could be read", evidence=evidence)
    if not evidence["cap_respected"]:
        return check(
            "windows", "fail", f"read {read} windows, above the {cap} the caller allowed",
            evidence=evidence,
        )
    if shapes != {(frames, int(getattr(store, "num_coordinates", 40) or 40))}:
        return check(
            "windows",
            "fail",
            f"window shapes {sorted(shapes)} are not {(frames, 40)}",
            evidence=evidence,
        )
    return check(
        "windows",
        "pass",
        f"{read} windows read (cap {cap}), shapes {sorted(shapes)}",
        evidence=evidence,
    )


def check_manifest_leakage(path: Path | None, *, required: bool = False) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return check(
            "manifest_leakage",
            "blocked" if required else "not_applicable",
            "the evaluate stage needs a fixed --eval-manifest to audit"
            if required
            else "no --eval-manifest given; leakage is audited when a manifest exists",
            level="evaluate" if required else "data",
        )
    from stylized_motion.learning.mts_operator.eval_protocol import (
        read_eval_manifest,
        store_split_of_clip,
    )

    metadata, rows = read_eval_manifest(path)
    same_take = 0
    overlap = 0
    missing_negatives = Counter()
    for row in rows:
        target = row.target
        for role, reference in (("correct", row.correct), ("wrong", row.wrong), ("random", row.random)):
            if reference is None:
                missing_negatives[row.reasons.get(role, "unavailable")] += 1
                continue
            if int(reference.take) >= 0 and int(reference.take) == int(target.take):
                same_take += 1
            if (
                int(reference.clip_id) == int(target.clip_id)
                and int(reference.start) == int(target.start)
            ):
                overlap += 1
    evidence = {
        "manifest": str(path),
        "rows": len(rows),
        "metadata": metadata,
        "same_take_pairs": same_take,
        "identical_crop_pairs": overlap,
        "unavailable_references": dict(missing_negatives),
        "per_row_mask_kind": len({row.mask_kind for row in rows}),
    }
    if same_take or overlap:
        return check(
            "manifest_leakage",
            "fail",
            f"{same_take} same-take and {overlap} identical-crop reference pairs in the manifest",
            evidence=evidence,
        )
    return check(
        "manifest_leakage",
        "pass",
        "no reference shares its target's take or crop",
        evidence=evidence,
    )


def check_action_vocabulary(
    transport: Any | None,
    config: Mapping[str, Any],
    records: Sequence[Any],
    *,
    level: str = "operator",
) -> dict[str, Any]:
    content_block = dict(config.get("data", {}).get("content", {}) or {})
    content_kind = str(content_block.get("kind", "none"))
    # A recipe with a content schema trains on *canonical* labels; comparing the raw
    # store labels against a canonical transport vocabulary would call a correctly
    # mapped run a mismatch (and, worse, would pass a run whose map was never applied).
    from stylized_motion.learning.mts_operator.content_schema import (
        apply_content_schema,
        content_schema_from_config,
    )

    schema, schema_report = content_schema_from_config(content_block, root=REPO_ROOT)
    mapped_records, mapping = apply_content_schema(
        records, schema, strict=False  # the preflight reports; it never re-labels the run
    )
    train_actions = sorted({record.content for record in mapped_records if record.split == "train"})
    transport_vocabulary = getattr(transport, "content_vocabulary", None)
    evidence: dict[str, Any] = {
        "configured_kind": content_kind,
        "train_actions": train_actions,
        "content_schema": {
            "path": schema_report.get("source_path"),
            "sha256": schema_report.get("source_sha256"),
            "version": None if schema is None else int(schema.version),
            "applied": bool(mapping.get("applied")),
            "renamed_rows": int(mapping.get("renamed_rows", 0)),
            "undeclared_on_train": mapping.get("undeclared", {}),
        },
        "transport_kind": None if transport_vocabulary is None else transport_vocabulary.kind,
        "transport_actions": None
        if transport_vocabulary is None
        else list(transport_vocabulary.classes),
    }
    if content_kind == "none":
        return check(
            "action_vocabulary",
            "pass",
            "content.kind=none: the run is unconditional and needs no action vocabulary",
            evidence=evidence,
        )
    if transport_vocabulary is None:
        return check(
            "action_vocabulary",
            "blocked",
            "content.kind=action_id needs a transport trained with a content conditioner; "
            "no revision-2 transport is bound yet",
            evidence=evidence,
        )
    unknown = sorted(set(train_actions) - set(transport_vocabulary.classes))
    evidence["unknown_actions"] = unknown
    if unknown:
        return check(
            "action_vocabulary",
            "fail",
            f"actions {unknown} are not in the transport's vocabulary; the operator cannot "
            "introduce ids the transport never learned",
            evidence=evidence,
        )
    return check(
        "action_vocabulary",
        "pass",
        f"{len(train_actions)} train actions are all inside the transport's vocabulary",
        evidence=evidence,
    )


def check_exposure(
    *,
    operator_path: Path | None,
    transport_path: Path | None,
) -> dict[str, Any]:
    """Held-out style exposure of the operator and of its upstream, reported apart.

    A held-out style is only held out for the component that never saw it: the
    tokenizer/transport may have been trained on the same catalogue.  Reporting one
    number for both would let an operator claim an axis its upstream already used.
    """
    evidence: dict[str, Any] = {}
    if operator_path is not None and Path(operator_path).exists():
        payload = torch.load(Path(operator_path), map_location="cpu", weights_only=False)
        exposure = (payload.get("provenance") or {}).get("training_exposure")
        evidence["operator_exposure"] = exposure
        if exposure is None:
            evidence["operator_exposure_status"] = "exposure_unknown"
        else:
            evidence["operator_exposure_status"] = (
                "exposure_unknown" if exposure.get("exposure_unknown") else "reported"
            )
    else:
        evidence["operator_exposure"] = None
        evidence["operator_exposure_status"] = "no operator checkpoint"
    if transport_path is not None and Path(transport_path).exists():
        payload = torch.load(Path(transport_path), map_location="cpu", weights_only=False)
        evidence["transport_exposure"] = (payload.get("provenance") or {}).get(
            "training_exposure"
        )
        evidence["tokenizer_exposure"] = (
            "the tokenizer sees the whole catalogue: its exposure is the store's split, "
            "reported by split_isolation"
        )
    else:
        evidence["transport_exposure"] = None
    if evidence["operator_exposure"] is None:
        return check(
            "exposure",
            "not_applicable",
            "no operator checkpoint yet, so its held-out style exposure cannot be reported "
            "(diagnostic: it never blocks a stage)",
            evidence=evidence,
            level="operator",
        )
    return check(
        "exposure",
        "pass",
        "operator exposure is reported separately from the upstream tokenizer/transport",
        evidence=evidence,
        level="operator",
    )


def check_validation_protocol(
    protocol_path: Path | None,
    config: Mapping[str, Any],
    *,
    stage: str,
    store: Any | None,
    seed: int,
) -> dict[str, Any]:
    """Reads the *actual* frozen protocol file and checks it against the recipe.

    The old check was called with ``None`` unconditionally, so it always reported
    "blocked" and verified nothing.  The file is an input here
    (``--validation-protocol``); when the stage requires it and it is missing, the
    check is ``blocked`` and names the cycle-free command that produces it
    (``--dry-run`` freezes the protocol without training).

    A file that exists is not evidence by itself: every row's clip is looked up in
    the store's own split table, the row counts and kinds are compared with the
    recipe, and the rows' mask configuration is compared with the recipe's masking
    block.
    """
    required = "validation_protocol" in REQUIRED_CHECKS[stage]
    evaluation = dict(config.get("evaluation") or {})
    loader = dict(config.get("loader") or {})
    evidence: dict[str, Any] = {
        "stage": stage,
        "required": required,
        "configured_protocol_id": evaluation.get("protocol_id"),
        "configured_kinds": evaluation.get("validation_kinds"),
        "configured_rows_per_kind": evaluation.get("validation_rows_per_kind"),
        "configured_batches_per_kind": evaluation.get("validation_batches_per_kind"),
        "configured_batch_size": loader.get("batch_size"),
        "seed": int(seed),
    }
    freeze_hint = (
        "freeze it first (no training): python scripts/train_mts_transport.py "
        "--config <recipe> --dry-run --output <run_dir>   # writes validation_protocol.json"
    )
    if protocol_path is None:
        return check(
            "validation_protocol",
            "blocked" if required else "not_applicable",
            f"no --validation-protocol given; {freeze_hint}",
            evidence=evidence,
        )
    path = Path(protocol_path)
    evidence["path"] = str(path)
    if not path.exists():
        return check(
            "validation_protocol",
            "blocked",
            f"{path} does not exist yet; {freeze_hint}",
            evidence=evidence,
        )
    from stylized_motion.learning.mts_operator.eval_protocol import store_split_of_clip

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") or []
    evidence["version"] = payload.get("version")
    evidence["protocol_id"] = payload.get("protocol_id")
    evidence["samples"] = payload.get("samples")
    evidence["samples_per_kind"] = payload.get("samples_per_kind")
    evidence["splits"] = payload.get("splits")
    evidence["selection"] = payload.get("selection")
    weights = payload.get("weights") or {}
    problems: list[str] = []
    if int(payload.get("version", 0)) != 2:
        problems.append(f"protocol version {payload.get('version')} is not revision 2")
    if not str(payload.get("protocol_id") or "").strip():
        problems.append("the protocol records no protocol_id")
    configured_id = str(evaluation.get("protocol_id") or "").strip()
    if configured_id and str(payload.get("protocol_id") or "") != configured_id:
        problems.append(
            f"the protocol's id {payload.get('protocol_id')!r} is not the recipe's "
            f"{configured_id!r}"
        )
    if not items:
        problems.append("the protocol has no rows")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        problems.append(f"protocol weights sum to {sum(weights.values())}, not 1")
    row_kinds = Counter(str(item.get("kind", "")) for item in items)
    evidence["row_kinds"] = dict(row_kinds)
    configured_kinds = [str(kind) for kind in (evaluation.get("validation_kinds") or ())]
    if configured_kinds and set(row_kinds) != set(configured_kinds):
        problems.append(
            f"the protocol scores {sorted(row_kinds)} but the recipe declares {sorted(set(configured_kinds))}"
        )
    if not configured_kinds and set(row_kinds) != set(MASK_KINDS):
        problems.append(
            f"the recipe declares no validation_kinds and the protocol does not cover all of "
            f"{list(MASK_KINDS)}: {sorted(row_kinds)}"
        )
    # Every row is looked up in the store's own split table.  A protocol that claims
    # "val" while scoring a train clip is refused here rather than discovered after
    # a training run has already reported numbers on it.
    rows_verified = 0
    wrong_split: list[dict[str, Any]] = []
    if store is not None and items:
        for item in items:
            clip = int(item.get("target_clip", -1))
            claimed = str(item.get("split", ""))
            observed = store_split_of_clip(store, clip)
            rows_verified += 1
            if observed != claimed:
                wrong_split.append({"clip": clip, "claimed": claimed, "observed": observed})
        evidence["rows_verified_against_store"] = rows_verified
        evidence["rows_with_wrong_store_split"] = len(wrong_split)
        evidence["wrong_split_examples"] = wrong_split[:5]
        if wrong_split:
            problems.append(
                f"{len(wrong_split)} of {rows_verified} protocol rows are not from the split they "
                f"claim (examples: {wrong_split[:3]})"
            )
    # The rows' mask configuration must be the recipe's, not just a matching kind: a
    # "stream" row with other ratios is a different experiment under the same name.
    # Compared in the canonical ``MaskConfig`` form: the recipe states the mixture
    # flat and a row records it with the nested mixture, and those are the same
    # configuration -- comparing the raw JSON would reject a correct protocol.
    from stylized_motion.learning.mts_operator.masking import MaskConfig

    recipe_masking = dict(config.get("masking") or {})
    if recipe_masking:
        try:
            canonical_recipe = MaskConfig.from_mapping(recipe_masking).as_dict()
        except (TypeError, ValueError) as error:
            problems.append(f"the recipe's masking block is invalid: {error}")
            canonical_recipe = None
        row_forms = set()
        for item in items:
            try:
                row_forms.add(
                    json.dumps(MaskConfig.from_mapping(item.get("mask_config") or {}).as_dict(), sort_keys=True)
                )
            except (TypeError, ValueError) as error:
                problems.append(f"protocol row {item.get('sample_id')} has an invalid mask_config: {error}")
                break
        if canonical_recipe is not None and row_forms != {json.dumps(canonical_recipe, sort_keys=True)}:
            problems.append(
                "the protocol rows' mask_config does not match the recipe's masking block "
                f"(rows={sorted(row_forms)[:2]}, recipe={json.dumps(canonical_recipe, sort_keys=True)})"
            )
    rows_per_kind = evaluation.get("validation_rows_per_kind")
    if rows_per_kind is not None:
        rows_per_kind = int(rows_per_kind)
        wrong = sorted(kind for kind, count in row_kinds.items() if count != rows_per_kind)
        if wrong:
            return check(
                "validation_protocol",
                "fail",
                f"kinds {wrong} do not have the recipe's {rows_per_kind} rows per kind "
                f"(counts={row_kinds})",
                evidence=evidence,
            )
    else:
        batches = evaluation.get("validation_batches_per_kind")
        batch_size = loader.get("batch_size")
        if batches is not None and batch_size is not None:
            expected = int(batches) * int(batch_size)
            # A shortfall is allowed only when the protocol's own recorded exclusions
            # account for it, kind by kind: dropping rows because the frozen action
            # vocabulary cannot condition on them is a documented decision, while a
            # protocol that is simply smaller than its recipe is a defect.
            excluded_per_kind = {
                str(kind): int(count)
                for kind, count in (
                    (payload.get("selection") or {}).get("content_vocabulary_filter", {}).get(
                        "excluded_per_kind", {}
                    )
                    or {}
                ).items()
            }
            unexplained = {}
            for kind, count in row_kinds.items():
                missing = expected - int(count)
                if missing <= 0:
                    continue
                if excluded_per_kind.get(kind, 0) != missing:
                    unexplained[kind] = {"missing": missing, "excluded": excluded_per_kind.get(kind, 0)}
            evidence["expected_rows_per_kind"] = expected
            evidence["excluded_per_kind"] = excluded_per_kind
            if unexplained:
                return check(
                    "validation_protocol",
                    "fail",
                    f"kinds {sorted(unexplained)} do not have the recipe's {expected} rows and the "
                    f"protocol's recorded exclusions do not account for the difference: {unexplained}",
                    evidence=evidence,
                )
            if any(int(count) != expected for count in row_kinds.values()):
                evidence["shortfall_explained_by_exclusions"] = True
    if problems:
        return check("validation_protocol", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "validation_protocol",
        "pass",
        f"frozen protocol {payload.get('protocol_id')!r} with {len(items)} rows over "
        f"{sorted(row_kinds)} from the val split"
        + (
            "; the shortfall against the recipe is fully explained by the protocol's own "
            "content-vocabulary exclusions"
            if evidence.get("shortfall_explained_by_exclusions")
            else ""
        ),
        evidence=evidence,
    )


def check_transport_recipe(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """The transport recipe's own admission rules, at the transport_train stage.

    The main transport must be action-conditioned: an unconditional upstream makes
    the operator's ``action_id`` path unusable, and the earlier recipes disagreed
    about this.  A debug recipe that really wants no condition can still be
    preflighted with ``--stage data``.
    """
    evaluation = dict(config.get("evaluation") or {})
    legacy = sorted(
        set(evaluation) & {"validation_batches_per_kind", "validation_batches", "val_rows"}
    )
    content_kind = str(config.get("data", {}).get("content", {}).get("kind", "none"))
    evidence: dict[str, Any] = {
        "config": str(config_path),
        "evaluation": dict(evaluation),
        "content_kind": content_kind,
        "training": dict(config.get("training") or {}),
    }
    problems: list[str] = []
    if legacy:
        problems.append(
            f"evaluation.{legacy[0]} is a legacy field; the transport recipe uses "
            "evaluation.validation_rows_per_kind"
        )
    if not str(evaluation.get("protocol_id") or "").strip():
        problems.append("evaluation.protocol_id is required")
    if not evaluation.get("validation_kinds"):
        problems.append("evaluation.validation_kinds must list the protocol's mask kinds")
    if evaluation.get("validation_rows_per_kind") is None:
        problems.append(
            "evaluation.validation_rows_per_kind is required (a positive row count per kind)"
        )
    elif int(evaluation["validation_rows_per_kind"]) <= 0:
        problems.append("evaluation.validation_rows_per_kind must be positive")
    if content_kind != "action_id":
        problems.append(
            f"data.content.kind is {content_kind!r}: the main transport must be action-conditioned "
            "(kind=action_id) so the operator can inherit its frozen action map; an unconditional "
            "run is a debug recipe and cannot pass the transport_train stage"
        )
    if problems:
        return check("transport_recipe", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "transport_recipe",
        "pass",
        f"recipe {config_path.name}: action-conditioned, protocol "
        f"{evaluation['protocol_id']!r} with {int(evaluation['validation_rows_per_kind'])} rows per kind",
        evidence=evidence,
    )


def check_evaluate_model(
    *,
    operator_path: Path | None,
    transport_path: Path | None,
    tokenizer_path: Path | None,
    adapter: Any | None,
    tokenizer_metadata: Mapping[str, Any] | None,
    device: str,
) -> dict[str, Any]:
    """Loads the artifact an evaluation would load, with its binding enforced.

    "The file exists" is not the question: the evaluation stage needs a checkpoint
    that rebuilds against this tokenizer *file* and this layout, which is exactly
    what the loaders check.
    """
    evidence: dict[str, Any] = {
        "operator_checkpoint": None if operator_path is None else str(operator_path),
        "transport_checkpoint": None if transport_path is None else str(transport_path),
    }
    if adapter is None:
        return check(
            "evaluate_model",
            "blocked",
            "no tokenizer/layout is available, so no checkpoint can be rebuilt",
            evidence=evidence,
            level="evaluate",
        )
    if operator_path is not None and Path(operator_path).exists():
        from stylized_motion.learning.mts_operator.checkpoint import load_operator_bundle

        try:
            payload, model = load_operator_bundle(
                Path(operator_path),
                adapter=adapter,
                tokenizer_identity=tokenizer_metadata,
                tokenizer_checkpoint=tokenizer_path,
                device=device,
            )
        except Exception as error:  # noqa: BLE001 - the message is the evidence
            return check("evaluate_model", "fail", str(error), evidence=evidence, level="evaluate")
        evidence["kind"] = "operator"
        evidence["global_step"] = int(payload.get("global_step", 0))
        evidence["trainable_parameters"] = int(
            sum(p.numel() for p in model.trainable_parameters())
        )
        return check(
            "evaluate_model",
            "pass",
            "the operator bundle rebuilds against this tokenizer file",
            evidence=evidence,
            level="evaluate",
        )
    if transport_path is not None and Path(transport_path).exists():
        from stylized_motion.learning.mts_operator.checkpoint import load_mts_checkpoint
        from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer

        try:
            payload, model = load_mts_checkpoint(
                Path(transport_path),
                kind="transport",
                build_model=lambda stored: MotionTransportTransformer(adapter, **stored),
                device=device,
                token_spec=adapter.token_spec(
                    representation_id=str((tokenizer_metadata or {}).get("representation_id", ""))
                ),
                tokenizer_metadata=tokenizer_metadata,
                tokenizer_checkpoint=tokenizer_path,
            )
        except Exception as error:  # noqa: BLE001 - the message is the evidence
            return check("evaluate_model", "fail", str(error), evidence=evidence, level="evaluate")
        evidence["kind"] = "transport"
        evidence["global_step"] = int(payload.get("global_step", 0))
        return check(
            "evaluate_model",
            "pass",
            "the transport rebuilds against this tokenizer file",
            evidence=evidence,
            level="evaluate",
        )
    if operator_path is None and transport_path is None:
        return check(
            "evaluate_model",
            "blocked",
            "the evaluate stage needs --operator-checkpoint or --transport-checkpoint",
            evidence=evidence,
            level="evaluate",
        )
    return check(
        "evaluate_model",
        "blocked",
        f"the given checkpoint does not exist yet: "
        f"{operator_path or transport_path}",
        evidence=evidence,
        level="evaluate",
    )


def check_mask_distribution(
    config: Mapping[str, Any], *, tokenizer: Any, frames: int, seed: int, draws: int = 500
) -> dict[str, Any]:
    """Samples the configured masking mixture: the realised kinds and hidden fraction.

    The mixture in the recipe is a *declaration*; what a training step actually sees
    is a draw from it.  Sampling it here catches a mixture that cannot be drawn (and
    gives the supervised-token budget a real fraction instead of a guess).
    """
    from stylized_motion.learning.mts_operator import LayoutAdapter
    from stylized_motion.learning.mts_operator.masking import MaskConfig, MaskGenerator

    masking = dict(config.get("masking") or {})
    if not masking:
        return check("mask_distribution", "fail", "the recipe declares no masking mixture")
    if tokenizer is None:
        return check(
            "mask_distribution",
            "skipped",
            "no tokenizer loaded, so a stream mask cannot be drawn",
        )
    parsed = MaskConfig.from_mapping(masking)
    adapter = LayoutAdapter(tokenizer.token_layout(), num_levels=int(tokenizer.num_levels))
    generator = MaskGenerator(parsed)
    torch_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    counts: Counter[str] = Counter()
    hidden_total = 0
    legal_total = 0
    for _ in range(int(draws)):
        mask = generator.sample(
            1, int(frames), adapter=adapter, generator=torch_generator, device=torch.device("cpu")
        )
        counts[mask.kind] += 1
        hidden_total += int(mask.supervision_mask.sum())
        legal_total += int(mask.visible_mask.numel())
    realised = {kind: counts[kind] / float(draws) for kind in sorted(counts)}
    declared = {kind: float(value) for kind, value in parsed.normalized_mixture().items()}
    evidence = {
        "draws": int(draws),
        "frames": int(frames),
        "declared": declared,
        "realised": realised,
        "hidden_fraction": hidden_total / max(legal_total, 1),
        "coordinates": int(adapter.num_coordinates),
    }
    missing = sorted(kind for kind, weight in declared.items() if weight > 0 and kind not in counts)
    if missing:
        return check(
            "mask_distribution",
            "fail",
            f"the declared mixture contains {missing} but no draw produced them",
            evidence=evidence,
        )
    return check(
        "mask_distribution",
        "pass",
        f"sampled {draws} masks; realised kinds {sorted(counts)} with "
        f"hidden fraction {evidence['hidden_fraction']:.3f}",
        evidence=evidence,
    )


def check_budget(
    config: Mapping[str, Any],
    *,
    model_parameters: int | None,
    output: Path | None,
    seed: int,
    experiment: bool = False,
    hidden_fraction: float | None = None,
) -> dict[str, Any]:
    training = dict(config.get("training") or {})
    data = dict(config.get("data") or {})
    loader = dict(config.get("loader") or {})
    evidence: dict[str, Any] = {
        "epochs": training.get("epochs"),
        "max_steps": training.get("max_steps"),
        "steps_per_epoch": training.get("steps_per_epoch"),
        "batch_size": loader.get("batch_size"),
        "frames": data.get("frames"),
        "masking": dict(config.get("masking") or {}),
        "seed": int(seed),
        "precision": training.get("precision"),
        "output_dir": str(training.get("output_dir", "")),
        "parameter_count": model_parameters,
        "code": code_identity(REPO_ROOT),
        "source_digest": source_digest(REPO_ROOT),
    }
    if hidden_fraction is not None:
        # Supervised tokens per step, from the *sampled* mask fraction: this is the
        # budget a training run is actually paid for, not the token count of a batch.
        batch = int(loader.get("batch_size", 0) or 0)
        frames = int(data.get("frames", 0) or 0)
        per_step = float(hidden_fraction) * float(batch * frames * 40)
        steps_per_epoch = int(training.get("steps_per_epoch") or 0)
        epochs = int(training.get("epochs") or 0)
        max_steps = training.get("max_steps")
        planned_steps = int(max_steps) if max_steps else steps_per_epoch * epochs
        evidence["supervised_tokens_per_step"] = per_step
        evidence["hidden_fraction_source"] = "sampled from the configured mask mixture"
        if planned_steps > 0:
            evidence["planned_steps"] = planned_steps
            evidence["supervised_token_budget"] = per_step * float(planned_steps)
        else:
            # The recipes leave steps-per-epoch unpinned (the run derives it), so a
            # budget number here would be invented.  Say so instead of printing 0.
            evidence["planned_steps"] = None
            evidence["supervised_token_budget"] = None
            evidence["supervised_token_budget_reason"] = (
                "steps_per_epoch and max_steps are unset in the recipe: the training "
                "script resolves the step count at start-up, so the token budget cannot "
                "be pinned from the config alone"
            )
    problems: list[str] = []
    if str(training.get("precision", "fp32")) != "fp32":
        problems.append("revision 2 is fp32 only")
    output_dir = Path(str(training.get("output_dir", "outputs/mts_revision2/run")))
    evidence["resolved_output_dir"] = str(output_dir)
    if experiment:
        # Where a run writes is an experiment-level question: a data preflight must
        # not fail because the recipe's output directory has not been created yet.
        repo_relative = _repo_relative(output_dir)
        evidence["repo_relative_output_dir"] = repo_relative
        if repo_relative is not None and not str(repo_relative).startswith("outputs/mts_revision2/"):
            # The rule protects the repository's historical artifacts; a run
            # directory outside the repository cannot overwrite them.
            problems.append(
                f"training.output_dir {output_dir} does not live under outputs/mts_revision2/: "
                "a new run must not overwrite a historical one"
            )
        for artifact in ("best.pt", "last.pt", "train_summary.json"):
            if (output_dir / artifact).exists():
                problems.append(
                    f"{output_dir / artifact} already exists; pick a new run directory "
                    "(nothing overwrites an existing run)"
                )
        writable = output_dir.parent
        while writable != writable.parent and not writable.exists():
            writable = writable.parent
        evidence["writable_root"] = str(writable)
        evidence["writable"] = bool(writable.exists() and writable.is_dir())
        if not evidence["writable"]:
            problems.append(f"{writable} does not exist, so the run cannot write its outputs")
    if output is not None:
        evidence["preflight_output"] = str(output)
    if problems:
        return check("budget", "fail", "; ".join(problems), evidence=evidence)
    return check(
        "budget",
        "pass",
        f"{training.get('epochs')} epoch(s), batch {loader.get('batch_size')}, "
        f"{data.get('frames')} frames, seed {seed}",
        evidence=evidence,
    )


def check_checkpoint_bindings(
    *,
    operator_path: Path | None,
    transport_path: Path | None,
    tokenizer_path: Path | None,
    model: Any | None,
    required: bool = False,
) -> dict[str, Any]:
    """The upstream binding of an existing revision-2 checkpoint.

    Two shapes are accepted, because the evaluate stage can score either artifact:
    an operator (whose recorded upstream transport SHA is checked) or a base
    transport (whose recorded tokenizer SHA is checked).  Either way the tokenizer
    file is compared, not its shape.
    """
    from stylized_motion.learning.mts_operator.checkpoint import (
        checkpoint_action_vocabulary as _actions,
        checkpoint_style_index as _styles,
        require_tokenizer_checkpoint,
    )

    evidence: dict[str, Any] = {}
    problems: list[str] = []
    if operator_path is not None and Path(operator_path).exists():
        payload = torch.load(Path(operator_path), map_location="cpu", weights_only=False)
        recorded = payload.get("tokenizer_checkpoint_sha256")
        provenance = payload.get("provenance") or {}
        evidence["operator"] = str(operator_path)
        evidence["operator_schema_version"] = payload.get("schema_version")
        evidence["recorded_tokenizer_sha256"] = recorded
        evidence["upstream_transport_sha256"] = provenance.get("upstream_transport_sha256")
        evidence["style_to_id"] = _styles(payload)
        evidence["action_to_id"] = _actions(payload)
        if int(payload.get("schema_version", 0)) != 2:
            problems.append(
                f"the operator checkpoint has schema {payload.get('schema_version')}, not 2"
            )
        if not recorded:
            problems.append(
                "the operator checkpoint records no tokenizer_checkpoint_sha256, so its "
                "tokenizer cannot be identified"
            )
        if tokenizer_path is not None and recorded and file_sha256(tokenizer_path) != recorded:
            problems.append(
                "the operator was trained with a different tokenizer than the one supplied"
            )
        if transport_path is not None and Path(transport_path).exists():
            if provenance.get("upstream_transport_sha256") not in (None, file_sha256(transport_path)):
                problems.append("the supplied transport is not the operator's frozen upstream")
    elif operator_path is not None:
        return check(
            "checkpoint_bindings",
            "blocked",
            f"{operator_path} does not exist yet; an artifact that is about to be produced is a "
            "missing dependency, not a failure",
            evidence=evidence,
            level="evaluate",
        )
    elif transport_path is not None and Path(transport_path).exists():
        payload = torch.load(Path(transport_path), map_location="cpu", weights_only=False)
        evidence["transport"] = str(transport_path)
        evidence["transport_schema_version"] = payload.get("schema_version")
        evidence["recorded_tokenizer_sha256"] = payload.get("tokenizer_checkpoint_sha256")
        if int(payload.get("schema_version", 0)) != 2:
            problems.append(
                f"the transport checkpoint has schema {payload.get('schema_version')}, not 2"
            )
        try:
            require_tokenizer_checkpoint(
                payload, tokenizer_checkpoint=tokenizer_path, where="preflight transport"
            )
        except ValueError as error:
            problems.append(str(error))
    elif transport_path is not None:
        return check(
            "checkpoint_bindings",
            "blocked",
            f"{transport_path} does not exist yet; an artifact that is about to be produced is a "
            "missing dependency, not a failure",
            evidence=evidence,
            level="evaluate",
        )
    else:
        return check(
            "checkpoint_bindings",
            "blocked" if required else "not_applicable",
            "the evaluate stage needs --operator-checkpoint (or --transport-checkpoint)"
            if required
            else "no checkpoint given: there is no trained artifact to bind yet",
            evidence=evidence,
            level="evaluate",
        )
    if model is not None and tokenizer_path is not None:
        evidence["architecture_revision"] = getattr(model, "architecture_revision", None)
    if problems:
        return check("checkpoint_bindings", "fail", "; ".join(problems), evidence=evidence,
                     level="evaluate")
    return check(
        "checkpoint_bindings",
        "pass",
        "the checkpoint's recorded upstreams match the supplied files",
        evidence=evidence,
        level="evaluate",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage, legacy_level = resolve_stage(args)
    config = load_config(args.config)
    tokenizer_path = (
        args.tokenizer_checkpoint
        or config.get("tokenizer", {}).get("checkpoint")
    )
    token_store_path = args.token_store or config.get("data", {}).get("token_store")
    feature_path = args.feature_database or config.get("data", {}).get("fsq_window_index")
    frames = int(config.get("data", {}).get("frames", 64))
    transport_path = args.transport_checkpoint or config.get("transport", {}).get("checkpoint")
    checks: list[dict[str, Any]] = []
    required = REQUIRED_CHECKS[stage]

    tokenizer_check, checkpoint, tokenizer = check_tokenizer(
        Path(tokenizer_path) if tokenizer_path else None, config
    )
    checks.append(tokenizer_check)

    store = None
    records: list[Any] = []
    actor_table: dict[str, str] = {}
    store_kind = "token" if token_store_path else "feature"
    try:
        store, store_kind = open_store(
            Path(token_store_path) if token_store_path else None,
            Path(feature_path) if feature_path else None,
            store_kind,
        )
    except Exception as error:  # noqa: BLE001 - reported as a failing check
        checks.append(check("store", "fail", str(error)))
    if store is not None:
        checks.append(
            check_store_identity(
                store,
                store_kind=store_kind,
                tokenizer_path=Path(tokenizer_path) if tokenizer_path else None,
                tokenizer=tokenizer,
                config=config,
            )
        )
        records = safe("records", clip_records_from_store, store)
        if isinstance(records, dict):
            checks.append(records)
            records = []
        if not any(getattr(record, "performer", "") for record in records) and feature_path:
            # The token store may carry no actor column; the catalogue is the
            # authority, and the join is by clip name, never by row number.
            try:
                catalogue = open_any_feature_store(Path(feature_path))
            except Exception:  # noqa: BLE001 - reported through the split check
                catalogue = None
            if catalogue is not None:
                try:
                    actor_table = align_actor_table(store, catalogue)
                finally:
                    catalogue.close()
        checks.append(safe("split_isolation", check_splits, records, actor_table=actor_table))
        checks.append(safe("label_tables", check_labels, store, records))
        checks.append(
            safe(
                "pair_matrix",
                check_pair_matrix,
                records,
                config,
                seed=int(args.seed),
                frames=frames,
            )
        )
        checks.append(
            safe(
                "windows",
                check_windows,
                store,
                store_kind=store_kind,
                split=str(config.get("data", {}).get("split", "train")),
                frames=frames,
                max_windows=int(args.max_windows),
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                seed=int(args.seed),
            ),
        )
        checks.append(
            safe(
                "mask_distribution",
                check_mask_distribution,
                config,
                tokenizer=tokenizer,
                frames=frames,
                seed=int(args.seed),
            )
        )

    hidden_fraction = None
    mask_check = next((item for item in checks if item["id"] == "mask_distribution"), None)
    if mask_check is not None and mask_check["status"] == "pass":
        hidden_fraction = float(mask_check["evidence"]["hidden_fraction"])

    # The frozen protocol is an input, not something this script invents: the
    # recipe's --dry-run writes it without training, which is what breaks the
    # "train first to validate" cycle.
    checks.append(
        safe(
            "validation_protocol",
            check_validation_protocol,
            args.validation_protocol,
            config,
            stage=stage,
            store=store,
            seed=int(args.seed),
            level=STAGE_LEVEL[stage],
        )
    )

    # The eval manifest: required at the evaluate stage, a diagnostic before it.
    checks.append(
        safe(
            "manifest_leakage",
            check_manifest_leakage,
            args.eval_manifest,
            required="manifest_leakage" in required,
        )
    )

    adapter = None
    if tokenizer is not None:
        from stylized_motion.learning.mts_operator import LayoutAdapter

        adapter = LayoutAdapter(tokenizer.token_layout(), num_levels=int(tokenizer.num_levels))

    transport = None
    if stage in {"operator_train", "evaluate"}:
        from stylized_motion.learning.mts_operator.checkpoint import load_mts_checkpoint
        from stylized_motion.learning.mts_operator.transport import MotionTransportTransformer

        if transport_path and Path(transport_path).exists() and adapter is not None:
            try:
                _, transport = load_mts_checkpoint(
                    Path(transport_path),
                    kind="transport",
                    build_model=lambda stored: MotionTransportTransformer(adapter, **stored),
                    device=args.device,
                    token_spec=adapter.token_spec(representation_id=tokenizer.representation_id),
                    tokenizer_metadata=tokenizer.representation_metadata(),
                    tokenizer_checkpoint=Path(tokenizer_path) if tokenizer_path else None,
                )
                checks.append(
                    check(
                        "transport",
                        "pass",
                        "the frozen transport loads against this tokenizer file",
                        evidence={
                            "path": str(transport_path),
                            "parameters": int(sum(p.numel() for p in transport.parameters())),
                            "architecture_revision": transport.config().get("architecture_revision"),
                            "recorded_tokenizer_sha256": None
                            if tokenizer_path is None
                            else file_sha256(tokenizer_path),
                        },
                        level="transport",
                    )
                )
            except Exception as error:  # noqa: BLE001 - reported as a failing check
                checks.append(check("transport", "fail", str(error), level="transport"))
        else:
            checks.append(
                check(
                    "transport",
                    "blocked",
                    "no revision-2 transport checkpoint exists yet: train the transport first",
                    evidence={"path": str(transport_path)},
                    level="transport",
                )
            )
        if records:
            checks.append(
                safe(
                    "action_vocabulary",
                    check_action_vocabulary,
                    transport,
                    config,
                    records,
                    level="operator",
                )
            )
    else:
        checks.append(
            check(
                "transport",
                "not_applicable",
                "the transport_train stage prepares the transport run itself, so it must not "
                "require a trained transport",
                evidence={"stage": stage},
                level="transport",
            )
        )
    checks.append(
        safe(
            "exposure",
            check_exposure,
            operator_path=args.operator_checkpoint,
            transport_path=Path(transport_path) if transport_path else None,
            level="operator",
        )
    )
    # The binding check is about an existing artifact: it runs at the stages that
    # consume one, or when the caller explicitly names a checkpoint.  At the data
    # stage a recipe's not-yet-trained checkpoint is not a defect.
    bindings_relevant = stage in {"operator_train", "evaluate"} or bool(
        args.operator_checkpoint or args.transport_checkpoint
    )
    if bindings_relevant:
        checks.append(
            safe(
                "checkpoint_bindings",
                check_checkpoint_bindings,
                operator_path=args.operator_checkpoint,
                transport_path=Path(transport_path) if transport_path else None,
                tokenizer_path=Path(tokenizer_path) if tokenizer_path else None,
                model=transport,
                required="checkpoint_bindings" in required,
            )
        )
    else:
        checks.append(
            check(
                "checkpoint_bindings",
                "not_applicable",
                f"the {stage} stage binds no MTS checkpoint; nothing to verify yet",
                evidence={"stage": stage, "configured_transport": str(transport_path or "")},
                level="evaluate",
            )
        )
    if stage == "evaluate":
        checks.append(
            safe(
                "evaluate_model",
                check_evaluate_model,
                operator_path=args.operator_checkpoint,
                transport_path=Path(transport_path) if transport_path else None,
                tokenizer_path=Path(tokenizer_path) if tokenizer_path else None,
                adapter=adapter,
                tokenizer_metadata=None if tokenizer is None else tokenizer.representation_metadata(),
                device=args.device,
            )
        )

    if stage == "transport_train":
        checks.append(
            safe(
                "transport_recipe",
                check_transport_recipe,
                config,
                args.config,
                level="transport",
            )
        )

    parameters = (
        int(sum(p.numel() for p in transport.parameters())) if transport is not None else None
    )
    checks.append(
        safe(
            "budget",
            check_budget,
            config,
            model_parameters=parameters,
            output=args.output,
            seed=int(args.seed),
            experiment=stage in {"transport_train", "operator_train"},
            hidden_fraction=hidden_fraction,
        )
    )

    if store is not None:
        store.close()
    by_id = {item["id"]: item for item in checks}
    missing_required = [identifier for identifier in required if identifier not in by_id]
    unready = [
        identifier
        for identifier in required
        if identifier in by_id and by_id[identifier]["status"] != "pass"
    ]
    failed = [item for item in checks if item["status"] == "fail"]
    blocked = [item for item in checks if item["status"] == "blocked"]
    not_applicable = [item for item in checks if item["status"] == "not_applicable"]
    # ``ready`` is the stage's admission decision: every required check passed.
    # ``ok`` stays as the diagnostic "nothing failed"; it is NOT the admission
    # decision, because a required check can be blocked while ok stays true.
    ready = not missing_required and not unready
    payload = {
        "kind": "mts_revision2_preflight",
        "preflight_version": PREFLIGHT_VERSION,
        "stage": stage,
        "level": legacy_level,
        "stage_level": STAGE_LEVEL[stage],
        "config": str(args.config),
        "tokenizer_checkpoint": str(tokenizer_path) if tokenizer_path else None,
        "store": str(token_store_path or feature_path or ""),
        "store_kind": store_kind,
        "frames": frames,
        "seed": int(args.seed),
        "max_windows": int(args.max_windows),
        "validation_protocol": None
        if args.validation_protocol is None
        else str(args.validation_protocol),
        "checks": checks,
        "required": list(required),
        "required_missing": missing_required,
        "required_not_passed": unready,
        "ready": bool(ready),
        "failed": [item["id"] for item in failed],
        "blocked": [item["id"] for item in blocked],
        "not_applicable": [item["id"] for item in not_applicable],
        "ok": not failed,
    }
    output = args.output
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "preflight.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    printable = {
        "stage": stage,
        "ready": payload["ready"],
        "ok": payload["ok"],
        "required_not_passed": payload["required_not_passed"],
        "failed": payload["failed"],
        "blocked": payload["blocked"],
        "not_applicable": payload["not_applicable"],
        "checks": [
            {"id": item["id"], "status": item["status"], "reason": item["reason"]}
            for item in checks
        ],
    }
    print(json.dumps(printable, indent=2, default=str), flush=True)
    if not ready or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
