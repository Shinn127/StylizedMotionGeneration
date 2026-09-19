#!/usr/bin/env python
"""The independent style evaluator: fixed kinematics + a closed-form linear readout.

N03 asked for a style baseline built from *motion*, not from the operator's own
token NLL, and N06 cannot rank anything against "matched style fidelity" without
one.  This script is that baseline, in two parts:

1. **The ceiling.**  Fixed kinematic features (root speed, joint speed, contact,
   foot slide, ranges -- all in metres through the shared :class:`PhysicsContext`)
   are computed on *real* motion; a ridge classifier is fitted on train-split
   windows and evaluated on val windows, grouped by take.  If real motion does not
   separate the styles, the evaluator cannot certify anything downstream and the
   plan says to fix the labels/task first.
2. **The application.**  The frozen classifier is applied to *generated* windows:
   the source motion and each arm's edit on the locked cases.  The question it
   answers is "does an evaluator that never saw the operator's NLL read the target
   style off the generated motion" -- never "is the style good".

The classifier is closed-form (no optimizer steps); the encoder, the operator and
the tokenizer are read-only here, and the reference encoder is never the judge.

    python scripts/evaluate_mts_style_baseline.py \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --arm reference_lowlr=outputs/mts_revision2/operator_reference_lowlr_s3407/best.pt \
      --arm style_id=outputs/mts_revision2/operator_styleid_logit_canonical_s3407/best.pt \
      --arm constant=outputs/mts_revision2/operator_noref_logit_canonical_s3407/best.pt \
      --manifest outputs/mts_next_round_20260918/N02/benchmark_manifest.json \
      --output outputs/mts_next_round_20260918/N06/style_evaluator.json
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
from stylized_motion.learning.mts_operator.model import OperatorBatch  # noqa: E402
from stylized_motion.learning.mts_operator.physics_context import (  # noqa: E402
    PHYSICAL_METRIC_VERSION,
    PhysicsContext,
)
from stylized_motion.learning.mts_operator.summary import (  # noqa: E402
    ArmSpec,
    load_arm,
    write_summary,
)
from stylized_motion.learning.mts_operator.masking import MaskConfig, MaskGenerator  # noqa: E402
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource,
    adapter_from_tokenizer,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402

FRAMES = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent (non-NLL) style evaluator.")
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument(
        "--feature-store",
        type=Path,
        required=True,
        help="The raw-motion store the ceiling is fitted on (the token store holds tokens, not "
        "features; a real-motion readout must start from motion).",
    )
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="The style classes; default is the pairable set the arms condition on.",
    )
    parser.add_argument("--fit-per-style", type=int, default=400)
    parser.add_argument("--eval-per-style", type=int, default=250)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument(
        "--regions",
        nargs="*",
        default=["whole_body"],
        help="whole_body rewrites everything hidden; a named region is a hard support, so most of "
        "the window stays real motion and the judge reads in-distribution frames.",
    )
    parser.add_argument("--graph-radius", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=20260918)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def kinematic_features(state: Any, context: PhysicsContext) -> dict[str, float]:
    """Fixed motion statistics in metres; nothing here is learned."""
    positions = np.asarray(state.global_positions)
    root = np.asarray(state.root_positions)
    dt = float(context.dt)
    frames = positions.shape[0]
    root_speed = np.linalg.norm(np.diff(root, axis=0), axis=-1) / dt
    joint_speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / dt
    vertical = np.diff(root[:, 1]) / dt
    per_joint_range = positions.std(axis=0).reshape(-1)
    toe_indices = context.kinematic().toe_indices
    features = {
        "root_speed_mean": float(root_speed.mean()),
        "root_speed_std": float(root_speed.std()),
        "root_speed_max": float(root_speed.max()),
        "root_vertical_speed_std": float(vertical.std()),
        "root_height_mean": float(root[:, 1].mean()),
        "root_height_std": float(root[:, 1].std()),
        "root_lateral_std": float(root[:, 0].std()),
        "root_forward_range": float(root[:, 2].max() - root[:, 2].min()),
        "joint_speed_mean": float(joint_speed.mean()),
        "joint_speed_p95": float(np.percentile(joint_speed, 95)),
        "joint_speed_max": float(joint_speed.max()),
        "joint_range_std_mean": float(per_joint_range.mean()),
        "joint_range_std_max": float(per_joint_range.max()),
    }
    if toe_indices is not None and frames >= 2:
        toe = positions[:, list(toe_indices)]
        toe_speed = np.linalg.norm(np.diff(toe, axis=0)[..., (0, 2)], axis=-1) / dt
        features.update(
            {
                "toe_height_mean": float(toe[..., 1].mean()),
                "toe_height_std": float(toe[..., 1].std()),
                "toe_speed_mean": float(toe_speed.mean()),
                "toe_speed_p95": float(np.percentile(toe_speed, 95)),
                "foot_contact_rate": float((toe[..., 1] < 0.08).mean()),
            }
        )
    return features


FEATURE_NAMES: tuple[str, ...] = tuple(
    [
        "root_speed_mean",
        "root_speed_std",
        "root_speed_max",
        "root_vertical_speed_std",
        "root_height_mean",
        "root_height_std",
        "root_lateral_std",
        "root_forward_range",
        "joint_speed_mean",
        "joint_speed_p95",
        "joint_speed_max",
        "joint_range_std_mean",
        "joint_range_std_max",
        "toe_height_mean",
        "toe_height_std",
        "toe_speed_mean",
        "toe_speed_p95",
        "foot_contact_rate",
    ]
)


class RidgeReadout:
    """Closed-form ridge on one-hot targets; no optimizer anywhere."""

    def __init__(self, lam: float, mean: np.ndarray, scale: np.ndarray, weights: np.ndarray) -> None:
        self.lam = float(lam)
        self.mean = mean
        self.scale = scale
        self.weights = weights

    def scores(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scores(x).argmax(axis=1)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, lam: float) -> "RidgeReadout":
        mean = x.mean(axis=0, keepdims=True)
        scale = x.std(axis=0, keepdims=True)
        scale[scale < 1e-12] = 1.0
        z = (x - mean) / scale
        width = x.shape[1]
        gram = z.T @ z + float(lam) * np.eye(width, dtype=np.float64)
        return cls(lam=lam, mean=mean, scale=scale, weights=np.linalg.solve(gram, z.T @ y))


def cross_validated_lambda(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, lams: tuple[float, ...], folds: int = 4
) -> tuple[float, float]:
    """Grouped-CV balanced accuracy per lambda; returns the best (lam, accuracy)."""
    unique = np.unique(groups)
    order = np.random.default_rng(0).permutation(len(unique))
    fold_of_group = {int(g): i % int(folds) for i, g in enumerate(unique[order])}
    fold_of_row = np.asarray([fold_of_group[int(g)] for g in groups])
    best = (-1.0, float(lams[0]))
    for lam in lams:
        scores = np.zeros_like(y)
        for fold in range(int(folds)):
            holdout = fold_of_row == fold
            if not bool(holdout.any()) or bool((~holdout).all()):
                continue
            readout = RidgeReadout.fit(x[~holdout], y[~holdout], lam=lam)
            scores[holdout] = readout.scores(x[holdout])
        predicted = scores.argmax(axis=1)
        truth = y.argmax(axis=1)
        recalls = [
            float((predicted[truth == k] == k).mean())
            for k in range(y.shape[1])
            if bool((truth == k).any())
        ]
        value = float(np.mean(recalls)) if recalls else float("nan")
        if value > best[0]:
            best = (value, float(lam))
    return best[1], best[0]


def balanced_accuracy(predicted: np.ndarray, truth: np.ndarray, classes: list[str]) -> dict[str, Any]:
    recalls = {
        klass: float((predicted[truth == index] == index).mean()) if bool((truth == index).any()) else None
        for index, klass in enumerate(classes)
    }
    values = [value for value in recalls.values() if value is not None]
    confusion = {
        classes[truth_row]: {
            classes[pred_row]: int(((truth == truth_row) & (predicted == pred_row)).sum())
            for pred_row in range(len(classes))
        }
        for truth_row in range(len(classes))
    }
    return {
        "balanced_accuracy": float(np.mean(values)) if values else None,
        "per_class_recall": recalls,
        "confusion": confusion,
    }


def take_bootstrap(
    predicted: np.ndarray, truth: np.ndarray, takes: np.ndarray, *, draws: int, seed: int
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, take in enumerate(takes):
        groups[int(take)].append(index)
    clusters = list(groups.values())
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws))
    for draw in range(int(draws)):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[int(p)] for p in picked])
        pred, true = predicted[indices], truth[indices]
        recalls = [
            float((pred[true == k] == k).mean())
            for k in np.unique(true)
            if bool((true == k).any())
        ]
        values[draw] = float(np.mean(recalls)) if recalls else float("nan")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"draws": int(draws), "clusters": len(clusters), "ci95_low": float(lower), "ci95_high": float(upper)}


def encode_pool(
    store: Any,
    windows: dict[int, list[Any]],
    context: PhysicsContext,
    pool: dict[str, list[tuple[int, str, int]]],
    *,
    per_style: int,
    rng: np.random.Generator,
    frames: int = FRAMES,
) -> tuple[np.ndarray, list[str], list[str], list[int], list[int]]:
    """Real-motion windows straight from the feature store (raw features, not tokens)."""
    from stylized_motion.learning.nef_data import read_clip_window  # noqa: PLC0415

    features: list[dict[str, float]] = []
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
            starts = [int(getattr(request, "target_start", 0)) for request in windows.get(int(clip), [])]
            if not starts:
                continue
            raw, _ = read_clip_window(store, int(clip), min(starts), frames, history=0)
            mirror = bool(np.asarray(store.clip_mirror)[int(clip)])
            state = context.world_state(
                np.asarray(raw, dtype=np.float32), mirror=mirror, normalized=False, contact_threshold=None,
            )
            features.append(kinematic_features(state, context))
            styles.append(style)
            contents.append(content)
            takes.append(int(take))
            clips.append(int(clip))
            taken += 1
    matrix = np.asarray(
        [[row.get(name, float("nan")) for name in FEATURE_NAMES] for row in features],
        dtype=np.float64,
    )
    if bool(np.isnan(matrix).any()):
        raise SystemExit("A kinematic feature is missing; the feature list and extractor disagree")
    return matrix, styles, contents, takes, clips


def _first_start(source: TokenSource, clip: int) -> int:
    entries = source.windows_by_clip[int(clip)]
    return int(min(int(getattr(request, "target_start", 0)) for request in entries))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    _, tokenizer = load_representation_checkpoint(args.tokenizer_checkpoint, torch.device("cpu"))
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    identity = tokenizer.representation_metadata()
    checkpoint = torch.load(args.tokenizer_checkpoint, map_location="cpu", weights_only=False)
    context = PhysicsContext.from_feature_stats(checkpoint["feature_stats"])
    store = open_any_token_store(args.token_store)
    # The ceiling reads *motion* from the feature store; the generation part reads
    # tokens (whose row ids are what the locked N02 manifest names) and decodes them.
    from stylized_motion.data import open_any_feature_store  # noqa: PLC0415

    feature_store = open_any_feature_store(args.feature_store)
    report: dict[str, Any] = {
        "kind": "mts_independent_style_evaluator",
        "physical_metric_version": PHYSICAL_METRIC_VERSION,
        "features": list(FEATURE_NAMES),
        "method": "fixed kinematic features on real motion + closed-form ridge; no optimizer, the "
        "reference encoder is never the judge",
    }
    try:
        from stylized_motion.learning.mts_operator.pairs import (  # noqa: PLC0415
            StylePairSampler,
            clip_records_from_store,
            split_styles_by_performer,
        )

        records = clip_records_from_store(feature_store)
        style_split = split_styles_by_performer(records, seed=int(args.seed))
        if args.classes:
            classes = sorted(str(value) for value in args.classes)
        else:
            sampler = StylePairSampler(
                records, style_split=style_split, seed=int(args.seed), window_frames=FRAMES
            )
            classes = sorted(
                {
                    str(target.style)
                    for target in sampler.eligible_targets(stage="train")
                    if sampler.pairs_for(target, mode="same_style", count=1, stage="train")
                }
            )
        report["classes"] = classes
        windows = windows_by_clip(feature_store, "val", frames=FRAMES)
        train_windows = windows_by_clip(feature_store, "train", frames=FRAMES)

        def pools(split: str, window_map: dict[int, list[Any]]) -> dict[str, list[tuple[int, str, int]]]:
            pool: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
            for record in records:
                if str(record.split) != split or str(record.style) not in set(classes):
                    continue
                if int(record.clip_id) not in window_map:
                    continue
                pool[str(record.style)].append(
                    (int(record.clip_id), str(record.content), int(record.source_group))
                )
            return pool

        fit_x, fit_styles, fit_contents, fit_takes, _ = encode_pool(
            feature_store, train_windows, context, pools("train", train_windows),
            per_style=int(args.fit_per_style), rng=np.random.default_rng(int(args.seed) + 2),
        )
        eval_x, eval_styles, eval_contents, eval_takes, _ = encode_pool(
            feature_store, windows, context, pools("val", windows),
            per_style=int(args.eval_per_style), rng=np.random.default_rng(int(args.seed) + 3),
        )
        index = {style: position for position, style in enumerate(classes)}
        fit_y = np.zeros((len(fit_styles), len(classes)))
        fit_y[np.arange(len(fit_styles)), [index[style] for style in fit_styles]] = 1.0
        eval_labels = np.asarray([index[style] for style in eval_styles])
        lam, cv_accuracy = cross_validated_lambda(
            fit_x, fit_y, np.asarray(fit_takes), lams=(1e-2, 1e-1, 1.0, 10.0, 100.0)
        )
        readout = RidgeReadout.fit(fit_x, fit_y, lam=lam)
        predicted = readout.predict(eval_x)
        ceiling = balanced_accuracy(predicted, eval_labels, classes)
        ceiling.update(
            {
                "rows": len(eval_labels),
                "fit_rows": len(fit_styles),
                "lambda": lam,
                "cv_balanced_accuracy": cv_accuracy,
                "chance": 1.0 / len(classes),
                "bootstrap": take_bootstrap(
                    predicted, eval_labels, np.asarray(eval_takes),
                    draws=int(args.bootstrap), seed=int(args.seed),
                ),
                "seen_contents_per_style": {
                    style: sorted({content for content, s in zip(fit_contents, fit_styles) if s == style})
                    for style in classes
                },
                "per_style_content_mix_eval": {
                    style: dict(Counter(content for content, s in zip(eval_contents, eval_styles) if s == style))
                    for style in classes
                },
            }
        )
        report["ceiling"] = ceiling
        # The same readout on generated motion, for the arms that were given.
        arms_report: dict[str, Any] = {}
        if args.arm:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
            if manifest is None:
                raise SystemExit("--arm needs --manifest (the locked N02 cases)")
            mask_generator = MaskGenerator(MaskConfig(mixture={"full_generation": 1.0}))
            loaded = [
                load_arm(
                    ArmSpec.parse(text), adapter=adapter, tokenizer_identity=identity,
                    tokenizer_checkpoint=args.tokenizer_checkpoint, device="cpu",
                )
                for text in args.arm
            ]
            rows = manifest["rows"][: int(args.cases)]
            per_arm: dict[str, Counter] = {arm.spec.name: Counter() for arm in loaded}
            per_arm["source"] = Counter()
            detail: list[dict[str, Any]] = []
            token_source = TokenSource(
                store=store, windows_by_clip=windows_by_clip(store, "val", frames=FRAMES),
                adapter=adapter, frames=FRAMES, history=int(tokenizer.history_frames),
                rng=np.random.default_rng(int(args.seed)),
            )
            for case_index, case in enumerate(rows):
                query, reference = case["query"], case["reference"]
                window = token_source.window_at(int(query["clip"]), int(query["start"]))
                reference_window = token_source.window_at(int(reference["clip"]), int(reference["start"]))
                if window is None or reference_window is None:
                    continue
                tokens = window.tokens[None]
                mirror = bool(query.get("mirror", False))
                base_mask = mask_generator.sample_kind(
                    "full_generation", 1, FRAMES, adapter=adapter,
                    generator=torch.Generator(device="cpu").manual_seed(
                        int(manifest.get("eval_seed", 3407)) + case_index
                    ),
                    device=torch.device("cpu"),
                )
                # The source is read through the same pipeline as the generated
                # windows: tokens decoded by the frozen tokenizer, then denormalized.
                # Comparing it against raw features would measure the tokenizer, not
                # the arm.
                with torch.inference_mode():
                    decoded_source = tokenizer.decode_indices(tokens)[0]
                source_features = kinematic_features(
                    context.world_state(decoded_source.numpy(), mirror=mirror, normalized=True,
                                        contact_threshold=None),
                    context,
                )
                source_prediction = classes[
                    int(readout.predict(np.asarray([[source_features[name] for name in FEATURE_NAMES]]))[0])
                ]
                for region in args.regions:
                    if region == "whole_body":
                        hard_mask = None
                        visible = base_mask.visible_mask
                    else:
                        hard_mask = adapter.hard_mask(
                            [name for name in str(region).split(",") if name],
                            graph_radius=int(args.graph_radius), length=FRAMES,
                            device=torch.device("cpu"),
                        )
                        visible = base_mask.visible_mask & ~hard_mask.unsqueeze(0).expand_as(
                            base_mask.visible_mask
                        )
                    detail.append(
                        {
                            "arm": "source",
                            "case_index": int(case_index),
                            "region": region,
                            "predicted_style": source_prediction,
                            "true_style": case["source_style"],
                        }
                    )
                    per_arm["source"][source_prediction] += 1
                    for arm in loaded:
                        recorded = (arm.checkpoint.get("provenance") or {}).get("content_schema")
                        uses_schema = bool(isinstance(recorded, dict) and recorded.get("schema"))
                        content_label = str(
                            query["content_canonical"] if uses_schema else query["content_raw"]
                        )
                        content_condition = None
                        if not (arm.content_vocabulary is None or arm.content_vocabulary.unconditional):
                            content_condition = arm.content_vocabulary.vector([content_label]).to(tokens.device)
                        style_ids = None
                        if arm.encoder_kind == "style_id":
                            style_ids = torch.tensor(
                                [int(arm.style_index[str(query["style"])])], dtype=torch.long,
                                device=tokens.device,
                            )
                        batch = OperatorBatch(
                            target_tokens=tokens,
                            reference_tokens=reference_window.tokens[None],
                            style_ids=style_ids,
                            visible_mask=visible,
                            hard_mask=hard_mask,
                            content_condition=content_condition,
                            sample_metadata=[case],
                            strength=1.0,
                        )
                        with torch.inference_mode():
                            styled = arm.model.generate_edit(
                                batch, sample_id=int(case_index), step_id=1, steps=1
                            )
                        state = context.world_state(
                            tokenizer.decode_indices(styled)[0].numpy(), mirror=mirror,
                            normalized=True, contact_threshold=None,
                        )
                        features = kinematic_features(state, context)
                        prediction = classes[
                            int(
                                readout.predict(
                                    np.asarray([[features[name] for name in FEATURE_NAMES]])
                                )[0]
                            )
                        ]
                        per_arm[arm.spec.name][prediction] += 1
                        detail.append(
                            {
                                "arm": arm.spec.name,
                                "case_index": int(case_index),
                                "region": region,
                                "conditioning_map": "content_schema_v1" if uses_schema else "legacy_raw",
                                "predicted_style": prediction,
                                "target_style": case["target_style"],
                                "source_style": case["source_style"],
                            }
                        )
            for name, counts in per_arm.items():
                total = sum(counts.values())
                entry = {
                    "windows": total,
                    "predicted_style_distribution": dict(sorted(counts.items())),
                }
                if name != "source":
                    edit_rows = [row for row in detail if row["arm"] == name]
                    if edit_rows:
                        hits = sum(1 for row in edit_rows if row["predicted_style"] == row["target_style"])
                        entry["edit_target_accuracy"] = hits / len(edit_rows)
                        entry["edit_rows"] = len(edit_rows)
                arms_report[name] = entry
                for region in sorted({row.get("region", "") for row in detail}):
                    region_rows = [
                        row for row in detail if row["arm"] == name and row.get("region") == region
                    ]
                    if not region_rows:
                        continue
                    region_entry = {
                        "windows": len(region_rows),
                        "predicted_style_distribution": dict(
                            sorted(Counter(row["predicted_style"] for row in region_rows).items())
                        ),
                    }
                    if name != "source":
                        hits = sum(1 for row in region_rows if row["predicted_style"] == row["target_style"])
                        region_entry["edit_target_accuracy"] = hits / len(region_rows)
                    arms_report.setdefault(name, {}).setdefault("by_region", {})[region] = region_entry
            report["generated_detail_rows"] = detail
            # Score the edit subtask: does the generated window read as the *target* style?
            for arm in loaded:
                rows_for_arm = [
                    row for row in detail if row["arm"] == arm.spec.name and "target_style" in row
                ]
                if rows_for_arm:
                    hits = sum(1 for row in rows_for_arm if row["predicted_style"] == row["target_style"])
                    arms_report[arm.spec.name]["edit_target_accuracy"] = hits / len(rows_for_arm)
                    arms_report[arm.spec.name]["edit_rows"] = len(rows_for_arm)
            source_rows = [row for row in detail if row["arm"] == "source"]
            if source_rows:
                hits = sum(1 for row in source_rows if row["predicted_style"] == row["true_style"])
                arms_report["source"]["source_style_accuracy"] = hits / len(source_rows)
            report["generated"] = arms_report
        write_summary(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "ceiling": {
                        "balanced_accuracy": ceiling["balanced_accuracy"],
                        "ci95": [ceiling["bootstrap"]["ci95_low"], ceiling["bootstrap"]["ci95_high"]],
                        "chance": ceiling["chance"],
                        "lambda": ceiling["lambda"],
                    },
                    "generated": {
                        name: {
                            "distribution": value["predicted_style_distribution"],
                            **({"edit_target_accuracy": value["edit_target_accuracy"]} if "edit_target_accuracy" in value else {}),
                        }
                        for name, value in arms_report.items()
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
