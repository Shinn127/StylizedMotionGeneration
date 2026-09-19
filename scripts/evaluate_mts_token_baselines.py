#!/usr/bin/env python
"""E00.3: train-only token baselines for the revision-2 transport runs.

The transport is a conditional model; before spending GPU steps on it there has
to be a number saying what the *data alone* predicts, or a falling NLL says
nothing.  This script streams the token store, counts token levels on the **train
split only**, and evaluates those counts on the **val split**:

* global and per-coordinate unigram NLL (each token's level under the train
  distribution of its own coordinate);
* action-conditioned unigram NLL, for the frozen train action vocabulary only --
  val clips with another action label are excluded and counted, never given a
  borrowed id;
* per-coordinate entropy and level usage;
* clip/frame counts per split, plus which val clips are eligible.

Two weightings are reported side by side because they bound different things:
``frame_weighted`` (every train frame counts once -- the target distribution a
token-level objective sees) and ``clip_weighted`` (every clip counts once, the
approximation of ``clip_uniform`` sampling).  Nothing is smoothed silently:
``--smoothing`` is written into the report together with the number of
coordinate/level cells that had no train count at all.

Optionally (``--protocol``) it also scores a *frozen* protocol's hidden positions
per mask kind with the same unigram model, so a later run's per-kind numbers have
a data reference instead of only a step-0 reference.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/evaluate_mts_token_baselines.py \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --output outputs/.../token_baselines.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    file_sha256,
    require_token_store_binding,
)
from stylized_motion.learning.mts_operator.masking import MaskGenerator  # noqa: E402

SPLITS = ("train", "val", "test")
#: Frames streamed per counting chunk; bounds peak memory independent of shard size.
CHUNK_FRAMES = 65_536


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only token baselines for the MTS transport.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64, help="window length used for eligibility")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="additive (Laplace) count added to every coordinate/level cell",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="recipe that supplies the masking block for --protocol"
    )
    parser.add_argument(
        "--protocol", type=Path, default=None, help="frozen validation protocol JSON to score per mask kind"
    )
    parser.add_argument("--limit-train-clips", type=int, default=None, help="debug aid: count fewer train clips")
    parser.add_argument("--limit-val-clips", type=int, default=None, help="debug aid: score fewer val clips")
    parser.add_argument("--verbose", action="store_true", help="print progress per shard")
    return parser


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def stream_train_counts(
    store,
    *,
    num_coordinates: int,
    num_levels: int,
    num_actions: int,
    action_ids: np.ndarray,
    limit_clips: int | None,
    log=lambda message: None,
) -> dict[str, object]:
    """Count train-split tokens, streaming one shard at a time.

    Returns frame-weighted and clip-weighted counts over an
    ``(action block, coordinate, level)`` cube whose block 0 is the pooled slice
    and block ``action + 1`` is one action.
    """
    split = np.asarray(store.clip_split)
    clip_shard = np.asarray(store.clip_shard)
    offset = np.asarray(store.clip_offset)
    length = np.asarray(store.clip_length)
    train = np.flatnonzero(split == 0)
    if limit_clips is not None:
        train = train[: int(limit_clips)]
    per_shard: dict[int, list[int]] = {}
    for clip in train:
        per_shard.setdefault(int(clip_shard[clip]), []).append(int(clip))

    cell = num_coordinates * num_levels
    cells = (num_actions + 1) * cell
    frame_counts = np.zeros(cells, dtype=np.float64)
    clip_counts = np.zeros(cells, dtype=np.float64)
    coordinate_offsets = np.arange(num_coordinates, dtype=np.int64)[None, :] * num_levels
    frames_seen = 0
    for shard_index in sorted(per_shard):
        clips = per_shard[shard_index]
        total_frames = int(store.shard_num_frames[shard_index])
        shard = store.read_frames(shard_index, 0, total_frames)
        # Frame -> owning clip row, for this shard's train clips only.
        owner = np.full(total_frames, -1, dtype=np.int64)
        for clip in clips:
            owner[offset[clip] : offset[clip] + length[clip]] = clip
        frame_weight = (owner >= 0).astype(np.float64)
        clip_weight = np.zeros(total_frames, dtype=np.float64)
        owned = owner >= 0
        clip_weight[owned] = 1.0 / np.maximum(length[owner[owned]], 1)
        frames_seen += int(owned.sum())
        for start in range(0, total_frames, CHUNK_FRAMES):
            stop = min(total_frames, start + CHUNK_FRAMES)
            weights_f = frame_weight[start:stop]
            if not np.any(weights_f > 0.0):
                continue
            weights_c = clip_weight[start:stop]
            tokens = shard[start:stop].astype(np.int64)
            # One bin per (action, coordinate, level): the action block and the
            # coordinate/level pair are independent, so a single bincount per
            # weighting counts every action at once.
            block = (action_ids[owner[start:stop]] + 1)[:, None] * cell
            bins = block + coordinate_offsets + tokens
            frame_counts += np.bincount(
                bins.ravel(), weights=np.broadcast_to(weights_f[:, None], bins.shape).ravel(), minlength=cells
            )
            clip_counts += np.bincount(
                bins.ravel(), weights=np.broadcast_to(weights_c[:, None], bins.shape).ravel(), minlength=cells
            )
        log(f"counted shard {shard_index}: {frames_seen} train frames so far")
    frame_cube = frame_counts.reshape(num_actions + 1, num_coordinates, num_levels)
    clip_cube = clip_counts.reshape(num_actions + 1, num_coordinates, num_levels)
    # Block 0 is the pooled slice: the sum over actions, so a clip whose action is
    # unknown to the vocabulary still has a pooled model to be scored against.
    frame_cube[0] = frame_cube[1:].sum(axis=0)
    clip_cube[0] = clip_cube[1:].sum(axis=0)
    return {
        "frame_counts": frame_cube,
        "clip_counts": clip_cube,
        "train_clips": int(train.size),
        "train_frames": int(frames_seen),
    }


def normalize_counts(
    cube: np.ndarray, *, num_coordinates: int, num_levels: int, smoothing: float
) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float64).reshape(-1, num_coordinates, num_levels)
    totals = cube.sum(axis=-1, keepdims=True)
    return (cube + smoothing) / (totals + smoothing * num_levels)


def level_probabilities(cube: np.ndarray) -> np.ndarray:
    """Level distribution pooled over coordinates, normalized to sum to one.

    A *level* probability has to be a distribution over levels: summing the
    per-coordinate probabilities without renormalizing would produce values above
    one and a negative "NLL".
    """
    cube = np.asarray(cube, dtype=np.float64)
    return cube.sum(axis=1) / cube.shape[1]


def score_clip(
    tokens: np.ndarray,
    *,
    action: int,
    pooled: np.ndarray,
    pooled_levels: np.ndarray,
    per_action: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Unigram NLLs of one clip's tokens, kept as per-token sums for weighting."""
    tokens = np.asarray(tokens, dtype=np.int64)
    coordinates = np.arange(tokens.shape[1], dtype=np.int64)[None, :]
    result = {
        "global": -np.log(pooled_levels[tokens]),
        "per_coordinate": -np.log(pooled[coordinates, tokens]),
    }
    if per_action is not None:
        result["per_action_per_coordinate"] = -np.log(per_action[coordinates, tokens])
    return result


def empty_accumulator(num_coordinates: int) -> dict[str, object]:
    return {
        "tokens": 0,
        "action_tokens": 0,
        "clips": 0,
        "global": 0.0,
        "per_coordinate": 0.0,
        "per_action": 0.0,
        "correct": 0,
        "per_coordinate_sums": np.zeros(num_coordinates, dtype=np.float64),
    }


def accumulate(accumulator: dict[str, object], scores: dict[str, np.ndarray], weight: float) -> None:
    accumulator["tokens"] = int(accumulator["tokens"]) + int(scores["global"].size)
    accumulator["global"] = float(accumulator["global"]) + float(scores["global"].sum()) * weight
    accumulator["per_coordinate"] = float(accumulator["per_coordinate"]) + float(
        scores["per_coordinate"].sum()
    ) * weight
    accumulator["per_coordinate_sums"] = np.asarray(accumulator["per_coordinate_sums"]) + (
        scores["per_coordinate"].sum(axis=0) * weight
    )
    if "per_action_per_coordinate" in scores:
        accumulator["per_action"] = float(accumulator["per_action"]) + float(
            scores["per_action_per_coordinate"].sum()
        ) * weight
        accumulator["action_tokens"] = int(accumulator["action_tokens"]) + int(scores["global"].size)


def finish(accumulator: dict[str, object], *, divisor: int) -> dict[str, object]:
    """Turns an accumulator into averages over ``divisor``.

    For the frame-weighted pass the divisor is the token count; for the
    clip-weighted pass it is the number of clips, because there every clip
    already contributed total weight one.  Both averages therefore divide by the
    same quantity and stay on one scale.
    """
    divisor = max(int(divisor), 1)
    action_conditioned = (
        float(accumulator["per_action"]) / divisor if int(accumulator["action_tokens"]) > 0 else None
    )
    return {
        "tokens": int(accumulator["tokens"]),
        "global": float(accumulator["global"]) / divisor,
        "per_coordinate": float(accumulator["per_coordinate"]) / divisor,
        "action_conditioned_per_coordinate": action_conditioned,
        "per_coordinate_by_coordinate": (np.asarray(accumulator["per_coordinate_sums"]) / divisor).tolist(),
    }


def score_protocol(
    store,
    tokenizer,
    *,
    protocol_path: Path,
    config_path: Path,
    frames: int,
    probabilities_cube: np.ndarray,
    num_levels: int,
    action_ids: np.ndarray,
    train_action_ids: set[int],
) -> dict[str, object]:
    """Unigram NLL over a frozen protocol's hidden positions, per mask kind."""
    import torch
    import yaml

    from stylized_motion.learning.mts_operator.eval_protocol import (
        ValidationProtocol,
        ValidationSample,
    )
    from stylized_motion.learning.mts_operator.windows import adapter_from_tokenizer

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mask_generator = MaskGenerator(dict(document.get("masking") or {}))
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    samples = tuple(ValidationSample(**item) for item in payload["items"])
    protocol = ValidationProtocol(
        samples=samples,
        kinds=tuple(payload["kinds"]),
        weights={str(k): float(v) for k, v in payload["weights"].items()},
        version=int(payload["version"]),
        protocol_id=str(payload.get("protocol_id") or ""),
        selection=dict(payload.get("selection") or {}),
    )
    adapter = adapter_from_tokenizer(tokenizer, num_levels=num_levels)
    spec = adapter.token_spec()
    argmax_levels = probabilities_cube.argmax(axis=-1)

    per_kind: dict[str, dict[str, object]] = {}
    for kind in protocol.kinds:
        entries = protocol.batches_for_kind(kind)
        hidden_total = 0
        pooled_sum = 0.0
        action_sum = 0.0
        action_tokens = 0
        correct = 0
        hidden_fractions: list[float] = []
        for sample in entries:
            tokens = np.asarray(
                store.read_window(sample.target_clip, sample.target_start, int(frames)), dtype=np.int64
            )
            generator = torch.Generator(device="cpu").manual_seed(int(sample.seed))
            mask = mask_generator.sample_kind(
                kind,
                1,
                int(tokens.shape[0]),
                adapter=adapter,
                spec=spec,
                generator=generator,
                device=torch.device("cpu"),
            )
            hidden = (~mask.visible_mask).numpy()[0]
            hidden_fractions.append(float(hidden.mean()))
            if not hidden.any():
                continue
            coordinates = np.arange(tokens.shape[1], dtype=np.int64)[None, :]
            selected = tokens[hidden]
            selected_coordinates = np.broadcast_to(coordinates, tokens.shape)[hidden]
            pooled_sum += float((-np.log(probabilities_cube[0][selected_coordinates, selected])).sum())
            hidden_total += int(selected.size)
            correct += int((argmax_levels[0][selected_coordinates] == selected).sum())
            action = int(action_ids[int(sample.target_clip)])
            if action in train_action_ids:
                action_sum += float(
                    (-np.log(probabilities_cube[action + 1][selected_coordinates, selected])).sum()
                )
                action_tokens += int(selected.size)
        per_kind[kind] = {
            "rows": len(entries),
            "hidden_tokens": hidden_total,
            "mean_hidden_fraction": float(np.mean(hidden_fractions)) if hidden_fractions else None,
            "unigram_nll": pooled_sum / hidden_total if hidden_total else None,
            "unigram_accuracy": correct / hidden_total if hidden_total else None,
            "action_conditioned_unigram_nll": action_sum / action_tokens if action_tokens else None,
            "action_conditioned_tokens": action_tokens,
            "action_ineligible_tokens": hidden_total - action_tokens,
            "clips": sorted({int(sample.target_clip) for sample in entries}),
            "splits": sorted({sample.split for sample in entries}),
            "seeds": sorted({int(sample.seed) for sample in entries}),
        }
    weighted = 0.0
    used = 0.0
    for kind, entry in per_kind.items():
        if entry["unigram_nll"] is None:
            continue
        weight = float(protocol.weights.get(kind, 0.0))
        weighted += weight * float(entry["unigram_nll"])
        used += weight
    return {
        "protocol_path": relative(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.fingerprint(),
        "kinds": list(protocol.kinds),
        "weights": {str(k): float(v) for k, v in protocol.weights.items()},
        "rows": len(samples),
        "splits": sorted({sample.split for sample in samples}),
        "mask_config": mask_generator.config.as_dict(),
        "per_kind": per_kind,
        "hidden_tokens_total": sum(int(entry["hidden_tokens"]) for entry in per_kind.values()),
        "weighted_unigram_nll": weighted / used if used > 0 else None,
        "note": (
            "unigram_nll is the data-only reference for the hidden positions; the model must beat "
            "it on the same rows, and action_conditioned_unigram_nll is the stronger reference for "
            "the action-conditioned transport"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    token_store = resolve(args.token_store)
    tokenizer_checkpoint = resolve(args.tokenizer_checkpoint)
    log = (lambda message: print(message, flush=True)) if args.verbose else (lambda message: None)

    import torch

    from stylized_motion.learning.representation import (
        NEF_FSQ_FAMILY,
        load_representation_checkpoint,
    )

    _, tokenizer = load_representation_checkpoint(tokenizer_checkpoint, torch.device("cpu"))
    if tokenizer.family != NEF_FSQ_FAMILY:
        raise ValueError(f"expected {NEF_FSQ_FAMILY}, got {tokenizer.family!r}")

    store = open_any_token_store(token_store)
    try:
        binding = require_token_store_binding(
            store, tokenizer_checkpoint=tokenizer_checkpoint, where="token baseline"
        )
        num_coordinates = int(store.num_coordinates)
        num_levels = int(store.manifest["num_levels"])
        action_ids = np.asarray(store.clip_action_id, dtype=np.int64)
        action_names = tuple(store.source_action_names or ())
        style_ids = np.asarray(store.clip_style_id, dtype=np.int64)
        style_names = tuple(store.source_style_names or ())
        num_actions = len(action_names) if action_names else int(action_ids.max()) + 1
        if int(action_ids.max()) >= num_actions:
            raise ValueError(
                f"store has action id {int(action_ids.max())} but only {num_actions} action names"
            )
        split = np.asarray(store.clip_split)
        length = np.asarray(store.clip_length)

        report: dict[str, object] = {
            "kind": "mts_token_baselines",
            "smoothing": float(args.smoothing),
            "frames": int(args.frames),
            "token_store": {
                "path": relative(token_store),
                "checkpoint_sha256": str(store.manifest.get("checkpoint_sha256")),
                "feature_schema_hash": str(store.manifest.get("feature_schema_hash")),
                "normalization_hash": str(store.manifest.get("normalization_hash")),
                "split_manifest_hash": str(store.manifest.get("split_manifest_hash")),
                "representation_id": str(store.manifest.get("representation_id")),
                "num_coordinates": num_coordinates,
                "num_levels": num_levels,
                "binding": binding,
            },
            "tokenizer": {
                "path": relative(tokenizer_checkpoint),
                "sha256": file_sha256(tokenizer_checkpoint),
                "representation_id": tokenizer.representation_id,
                "history_frames": int(tokenizer.history_frames),
            },
        }

        train_action_ids = sorted({int(value) for value in action_ids[split == 0]})
        full_window = length >= int(args.frames)
        report["counts"] = {
            "clips_per_split": {
                name: int(np.count_nonzero(split == index)) for index, name in enumerate(SPLITS)
            },
            "frames_per_split": {
                name: int(length[split == index].sum()) for index, name in enumerate(SPLITS)
            },
            "full_window_clips_per_split": {
                name: int(np.count_nonzero((split == index) & full_window))
                for index, name in enumerate(SPLITS)
            },
            "train_action_vocabulary": {
                "size": len(train_action_ids),
                "actions": [
                    action_names[index] if action_names else str(index) for index in train_action_ids
                ],
            },
            "actions_per_split": {
                name: sorted(
                    {
                        action_names[int(value)] if action_names else str(int(value))
                        for value in action_ids[split == index]
                    }
                )
                for index, name in enumerate(SPLITS)
            },
            "styles_per_split": {
                name: sorted(
                    {
                        style_names[int(value)] if style_names else str(int(value))
                        for value in style_ids[split == index]
                    }
                )
                for index, name in enumerate(SPLITS)
            },
        }

        counts = stream_train_counts(
            store,
            num_coordinates=num_coordinates,
            num_levels=num_levels,
            num_actions=num_actions,
            action_ids=action_ids,
            limit_clips=args.limit_train_clips,
            log=log,
        )
        frame_cube = normalize_counts(
            counts["frame_counts"],
            num_coordinates=num_coordinates,
            num_levels=num_levels,
            smoothing=float(args.smoothing),
        )
        clip_cube = normalize_counts(
            counts["clip_counts"],
            num_coordinates=num_coordinates,
            num_levels=num_levels,
            smoothing=float(args.smoothing),
        )
        pooled_frame = frame_cube[0]
        pooled_clip = clip_cube[0]
        pooled_levels_frame = level_probabilities(frame_cube)[0]
        pooled_levels_clip = level_probabilities(clip_cube)[0]
        report["train_statistics"] = {
            "train_clips_counted": int(counts["train_clips"]),
            "train_frames_counted": int(counts["train_frames"]),
            "zero_coordinate_level_cells": int((np.asarray(counts["frame_counts"])[0] == 0).sum()),
            "level_usage_frame_weighted": pooled_levels_frame.tolist(),
            "level_usage_clip_weighted": pooled_levels_clip.tolist(),
            "per_coordinate_entropy_bits_frame_weighted": (
                -(pooled_frame * np.log2(pooled_frame)).sum(axis=1)
            ).tolist(),
            "per_coordinate_entropy_bits_clip_weighted": (
                -(pooled_clip * np.log2(pooled_clip)).sum(axis=1)
            ).tolist(),
            "per_coordinate_level_probabilities_frame_weighted": pooled_frame.tolist(),
            "per_coordinate_level_probabilities_clip_weighted": pooled_clip.tolist(),
            "coordinate_order": list(getattr(store, "coordinate_order", ()) or ()),
        }

        eligible = (split == 1) & full_window & np.isin(action_ids, train_action_ids)
        excluded = (split == 1) & full_window & ~np.isin(action_ids, train_action_ids)
        report["counts"]["val_eligibility"] = {
            "full_window_clips": int(np.count_nonzero((split == 1) & full_window)),
            "eligible_clips": int(np.count_nonzero(eligible)),
            "excluded_clips": int(np.count_nonzero(excluded)),
            "eligible_frames": int(length[eligible].sum()),
            "excluded_frames": int(length[excluded].sum()),
            "excluded_actions": sorted(
                {
                    action_names[int(value)] if action_names else str(int(value))
                    for value in action_ids[excluded]
                }
            ),
        }

        frame_accumulator = empty_accumulator(num_coordinates)
        clip_accumulator = empty_accumulator(num_coordinates)
        excluded_sums = empty_accumulator(num_coordinates)
        per_action: dict[int, dict[str, object]] = {}
        argmax_levels = pooled_frame.argmax(axis=1)
        coordinates = np.arange(num_coordinates, dtype=np.int64)[None, :]
        val_clips = np.flatnonzero(split == 1)
        if args.limit_val_clips is not None:
            val_clips = val_clips[: int(args.limit_val_clips)]
        for clip in val_clips:
            clip = int(clip)
            if not full_window[clip]:
                continue
            tokens = np.asarray(store.read_clip(clip), dtype=np.int64)
            action = int(action_ids[clip])
            if action in train_action_ids:
                scores = score_clip(
                    tokens,
                    action=action,
                    pooled=pooled_frame,
                    pooled_levels=pooled_levels_frame,
                    per_action=frame_cube[action + 1],
                )
                accumulate(frame_accumulator, scores, 1.0)
                clip_scores = score_clip(
                    tokens,
                    action=action,
                    pooled=pooled_clip,
                    pooled_levels=pooled_levels_clip,
                    per_action=clip_cube[action + 1],
                )
                weight = 1.0 / max(int(length[clip]) * num_coordinates, 1)
                accumulate(clip_accumulator, clip_scores, weight)
                frame_accumulator["clips"] = int(frame_accumulator["clips"]) + 1
                clip_accumulator["clips"] = int(clip_accumulator["clips"]) + 1
                correct = int(
                    (argmax_levels[coordinates] == tokens).sum()
                )
                frame_accumulator["correct"] = int(frame_accumulator["correct"]) + correct
                bucket = per_action.setdefault(
                    action, {"tokens": 0, "sum": 0.0, "action_sum": 0.0, "clips": 0}
                )
                bucket["tokens"] = int(bucket["tokens"]) + int(tokens.size)
                bucket["sum"] = float(bucket["sum"]) + float(scores["per_coordinate"].sum())
                bucket["action_sum"] = float(bucket["action_sum"]) + float(
                    scores["per_action_per_coordinate"].sum()
                )
                bucket["clips"] = int(bucket["clips"]) + 1
            else:
                scores = score_clip(
                    tokens, action=action, pooled=pooled_frame, pooled_levels=pooled_levels_frame, per_action=None
                )
                accumulate(excluded_sums, scores, 1.0)
                excluded_sums["clips"] = int(excluded_sums["clips"]) + 1
        report["val_nll"] = {
            "eligible": {
                "clips": int(frame_accumulator["clips"]),
                "frame_weighted": {
                    **finish(frame_accumulator, divisor=int(frame_accumulator["tokens"])),
                    "level_accuracy": int(frame_accumulator["correct"])
                    / max(int(frame_accumulator["tokens"]), 1),
                },
                "clip_weighted": finish(clip_accumulator, divisor=int(clip_accumulator["clips"])),
                "per_action": {
                    (action_names[action] if action_names else str(action)): {
                        "nll": float(values["sum"]) / max(int(values["tokens"]), 1),
                        "action_conditioned_nll": float(values["action_sum"])
                        / max(int(values["tokens"]), 1),
                        "tokens": int(values["tokens"]),
                        "clips": int(values["clips"]),
                    }
                    for action, values in sorted(per_action.items())
                },
            },
            "excluded_action_clips": {
                **finish(excluded_sums, divisor=int(excluded_sums["tokens"])),
                "note": "these val clips carry an action the transport cannot be conditioned on",
            },
            "note": (
                "computed only on full-window clips (length >= frames); 'eligible' means the clip's "
                "action is in the train vocabulary. clip_weighted gives every clip the same total "
                "weight, frame_weighted every frame"
            ),
        }

        if args.protocol is not None:
            if args.config is None:
                raise SystemExit("--protocol requires --config for the masking block")
            report["protocol_baseline"] = score_protocol(
                store,
                tokenizer,
                protocol_path=resolve(args.protocol),
                config_path=resolve(args.config),
                frames=int(args.frames),
                probabilities_cube=frame_cube,
                num_levels=num_levels,
                action_ids=action_ids,
                train_action_ids=set(train_action_ids),
            )

        report["seconds"] = time.perf_counter() - started
        report["limits"] = {
            "train_clips": None if args.limit_train_clips is None else int(args.limit_train_clips),
            "val_clips": None if args.limit_val_clips is None else int(args.limit_val_clips),
            "truncated": bool(args.limit_train_clips is not None or args.limit_val_clips is not None),
        }
        report["note"] = (
            "statistics come from the train split only; val is used for evaluation; this script "
            "writes no store, checkpoint or model"
        )
    finally:
        store.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {
        "train_frames": report["train_statistics"]["train_frames_counted"],
        "train_actions": report["counts"]["train_action_vocabulary"]["size"],
        "eligible_val_clips": report["counts"]["val_eligibility"]["eligible_clips"],
        "excluded_val_clips": report["counts"]["val_eligibility"]["excluded_clips"],
        "val_frame_weighted": report["val_nll"]["eligible"]["frame_weighted"],
        "seconds": round(float(report["seconds"]), 2),
    }
    if "protocol_baseline" in report:
        summary["protocol_weighted_unigram_nll"] = report["protocol_baseline"]["weighted_unigram_nll"]
        summary["protocol_hash"] = report["protocol_baseline"]["protocol_hash"]
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
