#!/usr/bin/env python
"""The formal read-only summary over a frozen MTS validation protocol.

This is where the temporary E04/E05/E06 verification scripts converge: one entry
that loads any number of operator bundles, scores the *same* frozen rows under
each of them, and reports every number under the name of the aggregation it was
computed with.  It never trains, never writes a checkpoint, and never touches the
run directories it reads.

    python scripts/summarize_mts_operator.py \
      --arm style_id=outputs/mts_revision2/operator_styleid_logit_s3407_20260918_1814/best.pt \
      --arm constant=outputs/mts_revision2/operator_noref_logit_s3407_20260918_1814/best.pt \
      --protocol outputs/mts_revision2/operator_styleid_logit_s3407_20260918_1814/validation_protocol.json \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --output outputs/mts_next_round_20260918/N00/summary.json

Checks that must hold before any table is written:

* the arms' transports are *identical tensor by tensor* (a recorded SHA says what
  was loaded, not what the model contains);
* the action condition is present -- a conditioned arm scored without it is
  refused, and ``--omit-action-ablation`` (which reproduces the historical bug)
  relabels the protocol ``action_omitted`` so the number can never be compared
  with a conditioned one by accident;
* each arm's correct-condition ``protocol_weighted`` NLL reproduces the
  ``val_objective`` its own checkpoint recorded (fp32 rounding tolerance);
* the protocol rows are the store's own val rows, verified one by one;
* every value written is finite.

Differences are reported as ``nll_improvement_vs_<other>`` and are positive when
the first arm is better.  ``--max-rows`` makes the sample cap explicit; it is
checked, not a suggestion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    file_sha256,
    require_token_store_binding,
    validate_store_binding,
)
from stylized_motion.learning.mts_operator.eval_protocol import (  # noqa: E402
    ValidationProtocol,
    ValidationSample,
    store_split_of_clip,
)
from stylized_motion.learning.mts_operator.metrics import ordinal_level_mass  # noqa: E402
from stylized_motion.learning.mts_operator.summary import (  # noqa: E402
    SUMMARY_SCHEMA_VERSION,
    ArmSpec,
    LoadedArm,
    ReferencePool,
    check_status,
    condition_label,
    load_arm,
    objective_reproduction,
    paired_metric_comparison,
    score_arms_on_rows,
    summarize_metric,
    transport_state_mismatches,
    write_summary,
)
from stylized_motion.learning.mts_operator.windows import (  # noqa: E402
    TokenSource,
    adapter_from_tokenizer,
    windows_by_clip,
)
from stylized_motion.learning.representation import load_representation_checkpoint  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only multi-arm MTS operator summary.")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="One operator bundle. Repeat for several arms; order fixes the report order.",
    )
    parser.add_argument("--protocol", type=Path, required=True, help="A frozen validation_protocol.json.")
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Hard cap on the number of protocol rows scored; the frozen protocol is 213 rows.",
    )
    parser.add_argument(
        "--omit-action-ablation",
        action="store_true",
        help="Reproduce the historical missing-condition path; the protocol label becomes "
        "action_omitted and the numbers are never compared with conditioned ones.",
    )
    parser.add_argument("--reference-variants", action="store_true", help="Score a real wrong-style reference per row.")
    parser.add_argument("--strengths", nargs="*", type=float, default=[], help="Optional styled strength sweep.")
    parser.add_argument(
        "--ordinal-diagnostics",
        action="store_true",
        help="Per-arm adjacent-mass fraction and level displacement (fixed boundary handling).",
    )
    parser.add_argument(
        "--from-rows",
        type=Path,
        default=None,
        help="Re-aggregate an existing rows JSONL written by this entry instead of scoring "
        "the models again.  The rows file is the evidence; the aggregation is deterministic, "
        "and the output records which rows file and SHA it came from.",
    )
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--condition-label",
        default="action",
        help="Only for --from-rows: the condition the rows file was scored under.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser


def read_protocol(path: Path) -> ValidationProtocol:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ValidationProtocol(
        samples=tuple(ValidationSample(**item) for item in payload["items"]),
        kinds=tuple(payload["kinds"]),
        weights={str(k): float(v) for k, v in payload["weights"].items()},
        version=int(payload["version"]),
        protocol_id=str(payload.get("protocol_id") or ""),
        selection=dict(payload.get("selection") or {}),
    )


def verify_rows_against_store(protocol: ValidationProtocol, store: Any, *, split: str) -> dict[str, Any]:
    """Every protocol row must be the store's own row and in the claimed split."""
    wrong_split: list[dict[str, Any]] = []
    unknown: list[int] = []
    for sample in protocol.samples:
        try:
            actual = store_split_of_clip(store, int(sample.target_clip))
        except (IndexError, ValueError):
            unknown.append(int(sample.target_clip))
            continue
        if actual != str(sample.split):
            wrong_split.append(
                {"sample_id": int(sample.sample_id), "clip": int(sample.target_clip), "claimed": str(sample.split), "store": actual}
            )
        if str(sample.split) != str(split):
            wrong_split.append(
                {"sample_id": int(sample.sample_id), "clip": int(sample.target_clip), "claimed": str(sample.split), "requested": str(split)}
            )
    return {
        "rows": len(protocol.samples),
        "rows_verified_against_store": len(protocol.samples) - len(unknown),
        "rows_with_wrong_split": wrong_split,
        "rows_with_unknown_clip": unknown,
        "passed": not wrong_split and not unknown,
    }


def strength_sweep(
    arm: LoadedArm,
    samples: list[ValidationSample],
    *,
    token_source: Any,
    adapter: Any,
    device: torch.device,
    strengths: list[float],
    batch_size: int,
    omit_action: bool,
) -> list[dict[str, Any]]:
    """Per-strength token-weighted NLL under the arm's own condition, one row per point."""
    from stylized_motion.learning.mts_operator.eval_protocol import ValidationBatchBuilder
    from stylized_motion.learning.mts_operator.masking import MaskGenerator
    from stylized_motion.learning.mts_operator.summary import assert_condition_present

    rows: list[dict[str, Any]] = []
    for strength in strengths:
        builder = ValidationBatchBuilder(
            token_source=token_source,
            mask_generator=MaskGenerator(dict(samples[0].mask_config)),
            adapter=adapter,
            device=device,
            content_vocabulary=arm.content_vocabulary,
            style_index=arm.style_index,
            encoder_kind=arm.encoder_kind,
            strength=float(strength),
        )
        total = 0.0
        tokens = 0
        with torch.inference_mode():
            for start in range(0, len(samples), int(batch_size)):
                batch = builder.build_batch(samples[start : start + int(batch_size)])
                assert_condition_present(arm, batch, omit_action=omit_action)
                _, metrics = arm.model.loss(batch)
                total += float(metrics["nll_sum"])
                tokens += int(metrics["supervised_tokens"])
        rows.append(
            {
                "arm": arm.spec.name,
                "strength": float(strength),
                "aggregation": "micro",
                "nll": total / tokens if tokens else None,
                "supervised_tokens": int(tokens),
            }
        )
    return rows


def ordinal_diagnostics(
    arms: list[LoadedArm],
    samples: list[ValidationSample],
    *,
    token_source: Any,
    adapter: Any,
    device: torch.device,
    batch_size: int,
    omit_action: bool,
) -> dict[str, Any]:
    """Adjacent-level mass and displacement per arm, with the boundary handled once."""
    from stylized_motion.learning.mts_operator.eval_protocol import ValidationBatchBuilder
    from stylized_motion.learning.mts_operator.masking import MaskGenerator
    from stylized_motion.learning.mts_operator.summary import assert_condition_present

    collected: dict[str, list[dict[str, Any]]] = {arm.spec.name: [] for arm in arms}
    for arm in arms:
        builder = ValidationBatchBuilder(
            token_source=token_source,
            mask_generator=MaskGenerator(dict(samples[0].mask_config)),
            adapter=adapter,
            device=device,
            content_vocabulary=arm.content_vocabulary,
            style_index=arm.style_index,
            encoder_kind=arm.encoder_kind,
        )
        with torch.inference_mode():
            for start in range(0, len(samples), int(batch_size)):
                chunk = samples[start : start + int(batch_size)]
                batch = builder.build_batch(chunk)
                assert_condition_present(arm, batch, omit_action=omit_action)
                result = arm.model(batch)
                collected[arm.spec.name].append(
                    ordinal_level_mass(
                        result.probabilities,
                        result.base_probabilities,
                        supervision=batch.supervision_mask(arm.model.spec),
                        valid_mask=batch.target_valid_mask,
                    )
                )
    report: dict[str, Any] = {}
    for name, entries in collected.items():
        positions = sum(int(entry["supervised_positions"]) for entry in entries)
        report[name] = {
            "supervised_positions": positions,
            "adjacent_mass_fraction": _weighted(entries, "adjacent_mass_fraction"),
            "expected_level_displacement": _weighted(entries, "expected_level_displacement"),
            "tv_from_base": _weighted(entries, "tv_from_base"),
            "note": "adjacent mass = p(base mode) + the existing neighbours only, so it cannot exceed 1",
        }
    return report


def _weighted(entries: list[dict[str, Any]], key: str) -> float | None:
    total = 0.0
    weight = 0
    for entry in entries:
        if entry.get(key) is None:
            continue
        total += float(entry[key]) * int(entry["supervised_positions"])
        weight += int(entry["supervised_positions"])
    return None if weight == 0 else total / weight


def reaggregate_from_rows(args: argparse.Namespace, protocol: ValidationProtocol) -> int:
    """Re-aggregates an existing rows file: no model, no tokenizer, no store read."""
    rows = [
        json.loads(line)
        for line in Path(args.from_rows).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    weights = protocol.weights
    arms = [ArmSpec.parse(text).name for text in args.arm]
    aggregations: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for name in arms:
        if not any(f"nll_{name}_styled_sum" in row for row in rows):
            raise SystemExit(f"Arm {name!r} has no styled rows in {args.from_rows}")
        entry: dict[str, Any] = {
            "styled": summarize_metric(rows, f"nll_{name}_styled_sum", weights=weights),
            "base": summarize_metric(rows, f"nll_{name}_base_sum", weights=weights),
        }
        if any(f"nll_{name}_wrong_sum" in row for row in rows):
            entry["wrong"] = summarize_metric(rows, f"nll_{name}_wrong_sum", weights=weights)
            paired[name] = paired_metric_comparison(
                rows,
                f"nll_{name}_styled_sum",
                f"nll_{name}_wrong_sum",
                weights=weights,
                label_left="correct_style_input",
                label_right="wrong_style_input",
            )
        aggregations[name] = entry
    report = {
        "kind": "mts_operator_summary",
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "label": args.label,
        "read_only": True,
        "mode": "reaggregate_from_rows",
        "engine": {
            "from_rows": str(args.from_rows),
            "from_rows_sha256": file_sha256(args.from_rows),
            "rows": len(rows),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "protocol": {
            "path": str(args.protocol),
            "protocol_id": protocol.protocol_id,
            "fingerprint": protocol.fingerprint(),
            "rows": len(protocol.samples),
            "kinds": list(protocol.kinds),
            "weights": {kind: float(protocol.weights[kind]) for kind in protocol.kinds},
        },
        "arms": [{"name": name} for name in arms],
        "aggregations": {args.condition_label: aggregations},
        "paired_style_input_comparison": paired,
        "not_evaluated": {
            "motion_quality": "re-aggregation of an existing rows file; no motion was produced here",
            "style_fidelity": "NLL is not a perceptual style measurement",
            "generalization": "the rows are the development protocol, not an untouched test set",
        },
    }
    write_summary(args.output, report)
    print(json.dumps({"output": str(args.output), "mode": report["mode"], "rows": len(rows)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")

    specs = [ArmSpec.parse(text) for text in args.arm]
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"Arm names must be unique, got {names}")

    protocol = read_protocol(args.protocol)
    samples = list(protocol.samples)
    if args.max_rows is not None and len(samples) > int(args.max_rows):
        samples = samples[: int(args.max_rows)]

    if args.from_rows is not None:
        if not Path(args.from_rows).exists():
            raise SystemExit(f"--from-rows {args.from_rows} does not exist")
        return reaggregate_from_rows(args, protocol)

    for spec in specs:
        if not spec.path.exists():
            raise SystemExit(f"Arm {spec.name!r} checkpoint does not exist: {spec.path}")
    tokenizer_path = args.tokenizer_checkpoint
    _, tokenizer = load_representation_checkpoint(tokenizer_path, torch.device("cpu"))
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    tokenizer_identity = tokenizer.representation_metadata()

    store = open_any_token_store(args.token_store)
    report: dict[str, Any] = {
        "kind": "mts_operator_summary",
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "label": args.label,
        "read_only": True,
        "engine": {
            "token_store": str(args.token_store),
            "tokenizer_checkpoint": str(tokenizer_path),
            "tokenizer_sha256": file_sha256(tokenizer_path),
            "frames": int(args.frames),
            "batch_size": int(args.batch_size),
            "max_rows": args.max_rows,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "protocol": {
            "path": str(args.protocol),
            "protocol_id": protocol.protocol_id,
            "fingerprint": protocol.fingerprint(),
            "rows": len(protocol.samples),
            "rows_scored": len(samples),
            "kinds": list(protocol.kinds),
            "weights": {kind: float(protocol.weights[kind]) for kind in protocol.kinds},
            "splits": sorted({str(sample.split) for sample in protocol.samples}),
            "mask_config": dict(protocol.samples[0].mask_config),
        },
    }
    try:
        store_identity = require_token_store_binding(
            store, tokenizer_checkpoint=tokenizer_path, where="summary token store"
        )
        report["engine"]["store_identity"] = {
            **{k: str(v) for k, v in store_identity.items()},
            **{k: v for k, v in validate_store_binding(store, store_kind="token").items()},
        }
        split_rows = {str(sample.split) for sample in samples}
        if split_rows != {"val"}:
            raise SystemExit(f"The protocol rows must all be held out; got splits {sorted(split_rows)}")
        report["checks"] = {}
        verification = verify_rows_against_store(protocol, store, split="val")
        report["checks"]["rows_verified_against_store"] = {
            "requirement": "every protocol row is the store's own val row",
            "passed": bool(verification["passed"]),
            "evidence": verification,
        }
        if not verification["passed"]:
            raise SystemExit("The protocol rows are not the store's val rows; refusing to score")

        arms = [
            load_arm(
                spec,
                adapter=adapter,
                tokenizer_identity=tokenizer_identity,
                tokenizer_checkpoint=tokenizer_path,
                device=device,
            )
            for spec in specs
        ]
        report["arms"] = [arm.describe() for arm in arms]
        baseline = arms[0]
        mismatches = {
            arm.spec.name: transport_state_mismatches(baseline.model, arm.model)
            for arm in arms[1:]
        }
        report["checks"]["same_upstream_transport"] = {
            "requirement": "every arm carries the same frozen transport weights as the first arm",
            "passed": all(not names_ for names_ in mismatches.values()),
            "evidence": {
                "reference_arm": baseline.spec.name,
                "differing_tensor_names": mismatches,
                "transport_digests": {arm.spec.name: arm.transport_digest for arm in arms},
            },
        }
        if not report["checks"]["same_upstream_transport"]["passed"]:
            raise SystemExit("The arms do not share one transport; a comparison would mix two upstreams")

        condition = condition_label(baseline, omit_action=bool(args.omit_action_ablation))
        for arm in arms[1:]:
            if condition_label(arm, omit_action=bool(args.omit_action_ablation)) != condition:
                raise SystemExit(
                    f"Arm {arm.spec.name!r} conditions on {condition_label(arm, omit_action=bool(args.omit_action_ablation))!r} "
                    f"while {baseline.spec.name!r} conditions on {condition!r}"
                )
        report["condition"] = {
            "label": condition,
            "omit_action_ablation": bool(args.omit_action_ablation),
            "note": "action_omitted reproduces the historical missing-condition path and is not comparable "
            "with action-conditioned numbers",
        }

        clips = sorted({int(sample.target_clip) for sample in samples} | {int(sample.reference_clip) for sample in samples})
        windows = windows_by_clip(store, "val", frames=int(args.frames))
        missing = [clip for clip in clips if clip not in windows]
        if missing:
            raise SystemExit(f"The protocol references clips without a window: {missing[:5]}")
        source = TokenSource(
            store=store,
            windows_by_clip=windows,
            adapter=adapter,
            frames=int(args.frames),
            history=int(tokenizer.history_frames),
            rng=np.random.default_rng(3407),
        )
        pool = (
            ReferencePool.from_store(store, windows) if args.reference_variants else None
        )
        rows, comparability = score_arms_on_rows(
            arms,
            samples,
            token_source=source,
            adapter=adapter,
            device=device,
            batch_size=int(args.batch_size),
            omit_action=bool(args.omit_action_ablation),
            reference_pool=pool,
            wrong_reference=bool(args.reference_variants),
        )
        report["comparability"] = comparability
        report["checks"]["batches_match_across_arms"] = {
            "requirement": "every arm is scored on the same targets, masks, references and condition",
            "passed": not comparability["batch_digest_mismatches"],
            "evidence": {
                "batches": len(range(0, len(samples), int(args.batch_size))),
                "mismatches": comparability["batch_digest_mismatches"][:3],
                "batches_with_a_condition": comparability["batches_with_a_condition"],
            },
        }

        aggregations: dict[str, Any] = {}
        paired: dict[str, Any] = {}
        for arm in arms:
            name = arm.spec.name
            entry: dict[str, Any] = {
                "styled": summarize_metric(
                    rows, f"nll_{name}_styled_sum", weights=protocol.weights, max_rows=args.max_rows
                ),
                "base": summarize_metric(rows, f"nll_{name}_base_sum", weights=protocol.weights),
            }
            if any(f"nll_{name}_wrong_sum" in row for row in rows):
                entry["wrong"] = summarize_metric(
                    rows, f"nll_{name}_wrong_sum", weights=protocol.weights
                )
                # Row-matched: the only comparison that says whether *swapping the
                # style input* changes the score of the same row.
                paired[name] = paired_metric_comparison(
                    rows,
                    f"nll_{name}_styled_sum",
                    f"nll_{name}_wrong_sum",
                    weights=protocol.weights,
                    label_left="correct_style_input",
                    label_right="wrong_style_input",
                )
            aggregations.setdefault(condition, {})[name] = entry
        report["aggregations"] = aggregations
        report["paired_style_input_comparison"] = paired

        # Every arm must reproduce its own recorded validation objective under the
        # condition it was trained with; that is the counterexample the E04-E06
        # path failed and the number a reader should be able to recompute.
        reproduction: dict[str, Any] = {}
        for arm in arms:
            reproduction[arm.spec.name] = objective_reproduction(
                arm.recorded_val_objective,
                aggregations[condition][arm.spec.name]["styled"]["protocol_weighted"]["nll"],
            )
        complete_protocol = len(samples) == len(protocol.samples) and not args.omit_action_ablation
        report["checks"]["objective_reproduction"] = {
            "requirement": "correct-condition scoring of the complete protocol reproduces each "
            "checkpoint's own val_objective (the E04-E06 path did not)",
            "passed": all(entry["passed"] is True for entry in reproduction.values())
            if complete_protocol
            else None,
            "not_applicable_reason": None
            if complete_protocol
            else "partial rows or the omitted-action ablation: the recorded objective is over all "
            "rows under the trained condition",
            "evidence": reproduction,
        }

        # The frozen base distribution is a property of the transport alone: if two
        # arms disagree about it, one of them is not running the checkpoint it says.
        base_bounds = {}
        for arm in arms:
            values = [float(row[f"nll_{arm.spec.name}_base_sum"]) for row in rows]
            base_bounds[arm.spec.name] = {"min": min(values), "max": max(values), "sum": sum(values)}
        spread = max(v["sum"] for v in base_bounds.values()) - min(v["sum"] for v in base_bounds.values())
        report["checks"]["same_base_distribution"] = {
            "requirement": "the arms agree on the frozen base distribution row by row",
            "passed": bool(spread <= 1e-3),
            "evidence": {"per_arm": base_bounds, "sum_spread": float(spread)},
        }
        if not report["checks"]["same_base_distribution"]["passed"]:
            raise SystemExit("The arms disagree about the base distribution; stopping the 同-base comparison")

        if args.strengths:
            sweep: list[dict[str, Any]] = []
            for arm in arms:
                sweep.extend(
                    strength_sweep(
                        arm,
                        samples,
                        token_source=source,
                        adapter=adapter,
                        device=device,
                        strengths=[float(value) for value in args.strengths],
                        batch_size=int(args.batch_size),
                        omit_action=bool(args.omit_action_ablation),
                    )
                )
            report["strength_response"] = sweep
        if args.ordinal_diagnostics:
            report["ordinal"] = ordinal_diagnostics(
                arms,
                samples,
                token_source=source,
                adapter=adapter,
                device=device,
                batch_size=int(args.batch_size),
                omit_action=bool(args.omit_action_ablation),
            )

        report["not_evaluated"] = {
            "motion_quality": "no decoded motion is produced by this entry; see the physics/animation stage",
            "style_fidelity": "NLL is not a perceptual style measurement; an independent evaluator is required",
            "generalization": "the rows are the development protocol, not an untouched test set",
        }
        report["acceptance"] = check_status(report["checks"])
    finally:
        store.close()

    rows_path = args.rows_output or Path(args.output).with_suffix(".rows.jsonl")
    with Path(rows_path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str, allow_nan=False) + "\n")
    report["rows_file"] = str(rows_path)
    write_summary(args.output, report)
    reproduction = report["checks"].get("objective_reproduction") or {}
    if reproduction.get("passed") is False and not args.omit_action_ablation:
        failed = {
            name: entry
            for name, entry in reproduction["evidence"].items()
            if entry["passed"] is not True
        }
        print(
            "REFUSED: the correct-condition scoring does not reproduce the checkpoint's own "
            f"val_objective: {json.dumps(failed, default=str)}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "condition": report["condition"]["label"],
                "checks": {name: entry["passed"] for name, entry in report["checks"].items()},
                "protocol_weighted": {
                    arm: {
                        metric: values["protocol_weighted"]["nll"]
                        for metric, values in entry.items()
                    }
                    for arm, entry in report["aggregations"][report["condition"]["label"]].items()
                },
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
