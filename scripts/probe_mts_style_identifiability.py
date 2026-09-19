#!/usr/bin/env python
"""N05b-1: can three seen styles be read out of the reference descriptor at all?

This is the plan's "is the data identifiable" gate, run before any encoder
training.  A *closed-form* ridge probe is fitted on the frozen descriptors -- no
optimizer, no gradient, the encoder stays byte-identical -- and evaluated on clips
from a different split, with:

* **balanced accuracy** on unseen takes, with a take-cluster bootstrap interval,
  and a chance reference that accounts for the class mix;
* **unseen-content accuracy**: for each style, the contents used for fitting and
  the contents used for evaluation are disjoint where the data allows, so "the
  probe read the style" cannot be "the probe read the content";
* **descriptor geometry**: same-style-different-content distance against
  different-style distance (the margin the plan asks for);
* **swap response against the numeric floor**: re-encoding the same reference vs
  encoding a real different reference of the same style and of another style;
* a **label-shuffled control**, which must land at chance, and a **fresh encoder**
  of the same architecture, which separates "the input path cannot carry style"
  from "training removed it".

    python scripts/probe_mts_style_identifiability.py \
      --checkpoint outputs/mts_revision2/operator_reference_logit_s3407_20260918_1906/best.pt \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --output outputs/mts_next_round_20260918/N05b/style_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.pairs import (  # noqa: E402
    clip_records_from_store,
    split_styles_by_performer,
)
from stylized_motion.learning.mts_operator.style_encoder import GlobalStyleEncoder  # noqa: E402
from stylized_motion.learning.mts_operator.summary import (  # noqa: E402
    ArmSpec,
    load_arm,
    write_summary,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource,
    adapter_from_tokenizer,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402

FRAMES = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Style identifiability probe (closed form, read-only).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--fit-per-style", type=int, default=400)
    parser.add_argument("--eval-per-style", type=int, default=200)
    parser.add_argument("--fresh-seed", type=int, default=3407)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Override the class list (default: every train style).  The pairable three are "
        "what the operator's style index holds.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def sample_descriptors(
    encoder: GlobalStyleEncoder,
    source: TokenSource,
    pool: dict[str, list[tuple[int, str, int]]],
    *,
    per_style: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], list[str], list[int], list[int]]:
    """Encode one window per clip, up to ``per_style`` clips per style."""
    features: list[np.ndarray] = []
    styles: list[str] = []
    contents: list[str] = []
    takes: list[int] = []
    clips: list[int] = []
    for style in sorted(pool):
        candidates = list(pool[style])
        rng.shuffle(candidates)
        taken = 0
        for clip, content, take in candidates:
            if taken >= int(per_style):
                break
            window = source.window_at(int(clip), _first_start(source, int(clip)))
            if window is None:
                continue
            with torch.inference_mode():
                descriptor = encoder(
                    window.tokens[None], valid_mask=window.valid_mask[None]
                )
            features.append(descriptor[0].float().numpy())
            styles.append(style)
            contents.append(content)
            takes.append(int(take))
            clips.append(int(clip))
            taken += 1
    return (
        np.asarray(features, dtype=np.float64),
        styles,
        contents,
        takes,
        clips,
    )


def _first_start(source: TokenSource, clip: int) -> int:
    entries = source.windows_by_clip[int(clip)]
    return int(min(int(getattr(request, "target_start", 0)) for request in entries))


def ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    train_groups: np.ndarray,
    lams: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    folds: int = 4,
) -> dict[str, Any]:
    """Closed-form ridge on one-hot targets, with a take-grouped lambda search."""
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    train_z = (train_x - mean) / scale
    eval_z = (eval_x - mean) / scale
    unique_groups = np.unique(train_groups)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(unique_groups))
    fold_of_group = {
        int(group): index % int(folds) for index, group in enumerate(unique_groups[shuffled])
    }
    fold_of_row = np.asarray([fold_of_group[int(group)] for group in train_groups])

    def fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        width = x.shape[1]
        gram = x.T @ x + float(lam) * np.eye(width, dtype=np.float64) * len(x) / max(len(x), 1)
        return np.linalg.solve(gram, x.T @ y)

    def accuracy(scores: np.ndarray, y: np.ndarray) -> float:
        predicted = scores.argmax(axis=1)
        truth = y.argmax(axis=1)
        recalls = [
            float((predicted[truth == klass] == klass).mean())
            for klass in range(y.shape[1])
            if bool((truth == klass).any())
        ]
        return float(np.mean(recalls)) if recalls else float("nan")

    best = {"lam": None, "accuracy": -1.0}
    for lam in lams:
        scores = np.zeros_like(train_y)
        for fold in range(int(folds)):
            holdout = fold_of_row == fold
            if not bool(holdout.any()) or bool((~holdout).all()):
                continue
            weights = fit(train_z[~holdout], train_y[~holdout], lam)
            scores[holdout] = train_z[holdout] @ weights
        value = accuracy(scores, train_y)
        if value > best["accuracy"]:
            best = {"lam": float(lam), "accuracy": float(value)}
    weights = fit(train_z, train_y, float(best["lam"]))
    train_scores = train_z @ weights
    eval_scores = eval_z @ weights
    return {
        "lambda": best["lam"],
        "cv_balanced_accuracy": best["accuracy"],
        "train_balanced_accuracy": accuracy(train_scores, train_y),
        "eval_scores": eval_scores,
    }


def balanced_accuracy(scores: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict[str, Any]:
    predicted = scores.argmax(axis=1)
    recalls = {}
    for index, klass in enumerate(classes):
        mask = labels == index
        recalls[klass] = float((predicted[mask] == index).mean()) if bool(mask.any()) else None
    values = [value for value in recalls.values() if value is not None]
    return {
        "balanced_accuracy": float(np.mean(values)) if values else None,
        "per_class_recall": recalls,
        "per_class_support": {klass: int((labels == index).sum()) for index, klass in enumerate(classes)},
    }


def cluster_bootstrap(
    scores: np.ndarray, labels: np.ndarray, takes: np.ndarray, *, draws: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, take in enumerate(takes):
        groups[int(take)].append(index)
    clusters = list(groups.values())
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[int(p)] for p in picked])
        predicted = scores[indices].argmax(axis=1)
        truth = labels[indices]
        recalls = [
            float((predicted[truth == klass] == klass).mean())
            for klass in np.unique(truth)
            if bool((truth == klass).any())
        ]
        values[draw] = float(np.mean(recalls)) if recalls else float("nan")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"draws": int(draws), "clusters": len(clusters), "ci95_low": float(lower), "ci95_high": float(upper)}


def geometry(
    descriptors: np.ndarray, labels: np.ndarray, contents: list[str], classes: list[str]
) -> dict[str, Any]:
    """Distance geometry, in cosine *and* relative L2.

    Cosine distance between near-identical vectors is quadratic in the difference
    (``1 - cos ~ |dx|^2 / 2``), so a descriptor whose spread is 1e-5 shows a cosine
    margin of 1e-11 while still carrying a readable direction.  The scale-free
    relative L2 distance (``|dx| / |x|``) is reported beside it so "the margin is
    zero" is not an artefact of the metric; the ridge readout above is the
    direction test.
    """
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True).clip(1e-12)
    normalised = descriptors / norm
    similarity = normalised @ normalised.T
    same_content: list[float] = []
    cross_content: list[float] = []
    between: list[float] = []
    rel_same_content: list[float] = []
    rel_cross_content: list[float] = []
    rel_between: list[float] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            cosine = float(1.0 - similarity[left, right])
            relative = float(
                np.linalg.norm(descriptors[left] - descriptors[right])
                / max(0.5 * (norm[left, 0] + norm[right, 0]), 1e-12)
            )
            if labels[left] == labels[right]:
                if contents[left] == contents[right]:
                    same_content.append(cosine)
                    rel_same_content.append(relative)
                else:
                    cross_content.append(cosine)
                    rel_cross_content.append(relative)
            else:
                between.append(cosine)
                rel_between.append(relative)

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "pairs_within_style_same_content": len(same_content),
        "pairs_within_style_cross_content": len(cross_content),
        "pairs_between_styles": len(between),
        "cosine_distance_within_same_content": mean(same_content),
        "cosine_distance_within_cross_content": mean(cross_content),
        "cosine_distance_between_styles": mean(between),
        "cosine_margin_between_minus_cross_content": None
        if mean(cross_content) is None or mean(between) is None
        else mean(between) - mean(cross_content),
        "relative_l2_within_cross_content": mean(rel_cross_content),
        "relative_l2_between_styles": mean(rel_between),
        "relative_l2_margin": None
        if mean(rel_cross_content) is None or mean(rel_between) is None
        else mean(rel_between) - mean(rel_cross_content),
        "relative_l2_ratio_between_over_cross": None
        if not mean(rel_cross_content)
        else mean(rel_between) / mean(rel_cross_content),
        "descriptor_norm_mean": float(norm.mean()),
        "classes": classes,
        "note": "cosine distance is quadratic near zero; the ridge readout is the direction test "
        "and the relative L2 margin is the scale-aware one",
    }


def swap_measurement(
    encoder: GlobalStyleEncoder,
    source: TokenSource,
    pools: dict[str, list[tuple[int, str, int]]],
    classes: list[str],
    *,
    pairs: int,
    seed: int,
) -> dict[str, Any]:
    """How far a *real* reference swap moves the descriptor, against its own floor.

    ``same_input`` re-encodes the identical window (the numeric floor: exactly zero
    for a deterministic eval), ``same_style_same_content`` and
    ``same_style_cross_content`` swap in a real clip of the same style, and
    ``different_style`` swaps in another style's clip.  A descriptor that ignores
    its input shows every one of them at the floor.
    """
    rng_random = np.random.default_rng(int(seed) + 1)

    def rows_for(style: str, count: int) -> list[tuple[int, str, int]]:
        candidates = list(pools.get(style) or [])
        rng_local = np.random.default_rng(int(seed) + 7)
        rng_local.shuffle(candidates)
        return candidates[:count]

    def encode(clip: int) -> np.ndarray:
        window = source.window_at(int(clip), _first_start(source, int(clip)))
        with torch.inference_mode():
            descriptor = encoder(window.tokens[None], valid_mask=window.valid_mask[None])
        return descriptor[0].float().numpy()

    def relative(left: np.ndarray, right: np.ndarray) -> float:
        scale = max(0.5 * (float(np.linalg.norm(left)) + float(np.linalg.norm(right))), 1e-12)
        return float(np.linalg.norm(left - right) / scale)

    buckets: dict[str, list[float]] = {
        "same_input": [],
        "same_style_same_content": [],
        "same_style_cross_content": [],
        "different_style": [],
    }
    for style in classes:
        anchors = rows_for(style, max(1, int(pairs)))
        others = [value for value in classes if value != style]
        for clip, content, _take in anchors:
            base = encode(int(clip))
            buckets["same_input"].append(relative(base, encode(int(clip))))
            same_content = [
                candidate
                for candidate in (pools.get(style) or [])
                if candidate[1] == content and int(candidate[0]) != int(clip)
            ]
            cross_content = [
                candidate
                for candidate in (pools.get(style) or [])
                if candidate[1] != content and int(candidate[0]) != int(clip)
            ]
            if same_content:
                candidate = same_content[int(rng_random.integers(len(same_content)))]
                buckets["same_style_same_content"].append(relative(base, encode(int(candidate[0]))))
            if cross_content:
                candidate = cross_content[int(rng_random.integers(len(cross_content)))]
                buckets["same_style_cross_content"].append(relative(base, encode(int(candidate[0]))))
            for other in others:
                candidates = pools.get(other) or []
                if not candidates:
                    continue
                candidate = candidates[int(rng_random.integers(len(candidates)))]
                buckets["different_style"].append(relative(base, encode(int(candidate[0]))))
    summary = {
        name: (float(np.mean(values)) if values else None) for name, values in buckets.items()
    }
    floor = summary["same_input"]
    summary["rows"] = {name: len(values) for name, values in buckets.items()}
    summary["ratio_swap_over_floor"] = (
        None if not floor or floor == 0.0 else summary["same_style_cross_content"] / floor
    )
    summary["note"] = (
        "same_input is the numeric floor (deterministic eval -> exactly 0); a real swap must sit "
        "far above it, and a descriptor that ignores its input shows every bucket at the floor"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    _, tokenizer = load_representation_checkpoint(args.tokenizer_checkpoint, torch.device("cpu"))
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    identity = tokenizer.representation_metadata()
    store = open_any_token_store(args.token_store)
    report: dict[str, Any] = {
        "kind": "mts_style_identifiability_probe",
        "checkpoint": str(args.checkpoint),
        "method": "closed-form ridge on frozen descriptors; no optimizer, no encoder change",
    }
    try:
        # Two checkpoint kinds carry a reference encoder: the operator bundles of the
        # reference arm, and the encoder checkpoints the supervised task writes.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("kind") == "style_encoder":
            from stylized_motion.learning.mts_operator.checkpoint import require_tokenizer_checkpoint

            require_tokenizer_checkpoint(
                checkpoint, tokenizer_checkpoint=args.tokenizer_checkpoint, where="reference encoder checkpoint"
            )
            encoder_config = dict(checkpoint["metadata"]["model_config"]["style_encoder"])
            encoder_config.pop("kind", None)
            trained_encoder = GlobalStyleEncoder(adapter, **encoder_config)
            trained_encoder.load_state_dict(
                {key.removeprefix("."): value for key, value in checkpoint["model"].items()}
            )
            trained_encoder.eval()
            provenance = dict(checkpoint.get("provenance") or {})
        else:
            arm = load_arm(
                ArmSpec("reference", args.checkpoint),
                adapter=adapter,
                tokenizer_identity=identity,
                tokenizer_checkpoint=args.tokenizer_checkpoint,
                device="cpu",
            )
            if arm.encoder_kind != "reference":
                raise SystemExit(f"This checkpoint's encoder is {arm.encoder_kind!r}, not a reference encoder")
            encoder_config = dict(arm.checkpoint["metadata"]["model_config"]["style_encoder"])
            encoder_config.pop("kind", None)
            trained_encoder = arm.model.style_encoder
            provenance = dict(arm.checkpoint.get("provenance") or {})
        report["checkpoint_kind"] = checkpoint.get("kind")
        report["recorded_metrics"] = dict(checkpoint.get("metrics") or {})
        report["provenance_training_protocol"] = provenance.get("training_protocol_id")
        torch.manual_seed(int(args.fresh_seed))
        fresh = GlobalStyleEncoder(adapter, **encoder_config).eval()

        records = clip_records_from_store(store)
        style_split = split_styles_by_performer(records, seed=int(args.fresh_seed))
        if args.classes:
            classes = sorted(str(value) for value in args.classes)
        else:
            classes = sorted(str(style) for style in style_split.train_styles)
        windows = windows_by_clip(store, "val", frames=FRAMES)
        train_windows = windows_by_clip(store, "train", frames=FRAMES)
        pools: dict[str, dict[str, list[tuple[int, str, int]]]] = {}
        for name, window_map in (("fit", train_windows), ("eval", windows)):
            split_name = "train" if name == "fit" else "val"
            pool: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
            for record in records:
                if str(record.split) != split_name:
                    continue
                if str(record.style) not in classes:
                    continue
                if int(record.clip_id) not in window_map:
                    continue
                pool[str(record.style)].append(
                    (int(record.clip_id), str(record.content), int(record.source_group))
                )
            pools[name] = pool
        report["pool"] = {
            name: {style: len(values) for style, values in pool.items()} for name, pool in pools.items()
        }
        report["classes"] = classes

        results: dict[str, Any] = {}
        eval_source_for_swap = TokenSource(
            store=store,
            windows_by_clip=windows,
            adapter=adapter,
            frames=FRAMES,
            history=int(tokenizer.history_frames),
            rng=np.random.default_rng(int(args.seed)),
        )
        for name, encoder in (("trained", trained_encoder), ("fresh", fresh)):
            source_args = dict(adapter=adapter, frames=FRAMES, history=int(tokenizer.history_frames))
            fit_source = TokenSource(store=store, windows_by_clip=train_windows, rng=np.random.default_rng(int(args.seed)), **source_args)
            eval_source = TokenSource(store=store, windows_by_clip=windows, rng=np.random.default_rng(int(args.seed) + 1), **source_args)
            fit_x, fit_styles, fit_contents, fit_takes, _ = sample_descriptors(
                encoder, fit_source, pools["fit"], per_style=int(args.fit_per_style), rng=np.random.default_rng(int(args.seed) + 2)
            )
            eval_x, eval_styles, eval_contents, eval_takes, _ = sample_descriptors(
                encoder, eval_source, pools["eval"], per_style=int(args.eval_per_style), rng=np.random.default_rng(int(args.seed) + 3)
            )
            index = {style: position for position, style in enumerate(classes)}
            fit_y = np.zeros((len(fit_styles), len(classes)))
            fit_y[np.arange(len(fit_styles)), [index[style] for style in fit_styles]] = 1.0
            eval_labels = np.asarray([index[style] for style in eval_styles])
            probe = ridge_probe(fit_x, fit_y, eval_x, train_groups=np.asarray(fit_takes))
            scores = probe.pop("eval_scores")
            entry = {
                "fit_rows": len(fit_styles),
                "eval_rows": len(eval_styles),
                "content_mix_fit": dict(Counter(fit_contents)),
                "content_mix_eval": dict(Counter(eval_contents)),
                **probe,
            }
            entry["eval"] = balanced_accuracy(scores, eval_labels, classes)
            entry["eval"]["bootstrap"] = cluster_bootstrap(
                scores, eval_labels, np.asarray(eval_takes), draws=int(args.bootstrap), seed=int(args.seed)
            )
            # Unseen-content slice: evaluation rows whose content never appeared for
            # that style in the fit set.
            seen = defaultdict(set)
            for content, style in zip(fit_contents, fit_styles):
                seen[style].add(content)
            unseen_rows = np.asarray(
                [content not in seen[style] for content, style in zip(eval_contents, eval_styles)]
            )
            if bool(unseen_rows.any()):
                entry["eval_unseen_content"] = balanced_accuracy(
                    scores[unseen_rows], eval_labels[unseen_rows], classes
                )
                entry["eval_unseen_content"]["rows"] = int(unseen_rows.sum())
            else:
                entry["eval_unseen_content"] = {"rows": 0, "reason": "no evaluation content was unseen for its style"}
            entry["geometry"] = geometry(eval_x, eval_labels, eval_contents, classes)
            # Controls: shuffled labels (must sit at chance) and shuffled features.
            shuffled_y = fit_y[np.random.default_rng(int(args.seed)).permutation(len(fit_y))]
            control = ridge_probe(fit_x, shuffled_y, eval_x, train_groups=np.asarray(fit_takes))
            control_scores = control.pop("eval_scores")
            entry["label_shuffled_control"] = balanced_accuracy(control_scores, eval_labels, classes)
            results[name] = entry
        report["encoders"] = results
        report["swap_response"] = {
            name: swap_measurement(
                encoder,
                eval_source_for_swap,
                pools["eval"],
                classes,
                pairs=8,
                seed=int(args.seed),
            )
            for name, encoder in (("trained", trained_encoder), ("fresh", fresh))
        }
        write_summary(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "trained": {
                        "eval_balanced_accuracy": results["trained"]["eval"]["balanced_accuracy"],
                        "ci95": [results["trained"]["eval"]["bootstrap"]["ci95_low"], results["trained"]["eval"]["bootstrap"]["ci95_high"]],
                        "unseen_content": results["trained"]["eval_unseen_content"].get("balanced_accuracy"),
                        "margin_cosine": results["trained"]["geometry"]["cosine_margin_between_minus_cross_content"],
                        "margin_relative_l2": results["trained"]["geometry"]["relative_l2_margin"],
                        "geometry": results["trained"]["geometry"],
                        "control": results["trained"]["label_shuffled_control"]["balanced_accuracy"],
                    },
                    "fresh": {
                        "eval_balanced_accuracy": results["fresh"]["eval"]["balanced_accuracy"],
                        "ci95": [results["fresh"]["eval"]["bootstrap"]["ci95_low"], results["fresh"]["eval"]["bootstrap"]["ci95_high"]],
                        "unseen_content": results["fresh"]["eval_unseen_content"].get("balanced_accuracy"),
                        "margin_cosine": results["fresh"]["geometry"]["cosine_margin_between_minus_cross_content"],
                        "margin_relative_l2": results["fresh"]["geometry"]["relative_l2_margin"],
                        "geometry": results["fresh"]["geometry"],
                        "control": results["fresh"]["label_shuffled_control"]["balanced_accuracy"],
                    },
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
