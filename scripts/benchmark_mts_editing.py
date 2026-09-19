#!/usr/bin/env python
"""N03: the real editing benchmark on the locked cases, with no training.

Every arm draws from the *same* cells: one source window, one mask, and one set of
common random numbers per (case, draw, step), so a difference between two arms is
a difference between their distributions, not between two draws.

For each case, arm, region, draw and strength the script records

* the token change overall / inside the edit support / outside it (leakage);
* the *style-input* effect at fixed tokens: the operator's own distribution with
  the correct style id (or reference) against the same batch with a wrong one,
  same CRN -- for the constant arm this is impossible by construction and is
  reported as null with the reason;
* the distance from the frozen base transport's distribution and the NLL of the
  source tokens under the styled distribution (a likelihood proxy, never a style
  measurement);
* decoded physics through the shared :class:`PhysicsContext` (denormalized root
  path/speed, contact rate, foot slide, FK change) for source -> base,
  base -> styled and source -> styled, kept apart;
* a fixed-camera flipbook: frames of the source, the base draw and the styled draw
  in one 2D projection with identical axis limits per case.

    python scripts/benchmark_mts_editing.py \
      --manifest outputs/mts_next_round_20260918/N02/benchmark_manifest.json \
      --arm style_id=outputs/.../best.pt --arm constant=outputs/.../best.pt \
      --token-store data/processed/seed_soma_pruned_v4_ah_tokens \
      --tokenizer-checkpoint outputs/nef_fsq_soma_packed_40x9_ah/best.pt \
      --steps 1 4 --strengths 0.0 0.5 1.0 1.5 --draws 2 \
      --output outputs/mts_next_round_20260918/N03
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data.packed_token import open_any_token_store  # noqa: E402
from stylized_motion.learning.mts_operator.checkpoint import (  # noqa: E402
    require_token_store_binding,
)
from stylized_motion.learning.mts_operator.masking import MaskConfig, MaskGenerator  # noqa: E402
from stylized_motion.learning.mts_operator.metrics import comparison_physics  # noqa: E402
from stylized_motion.learning.mts_operator.model import OperatorBatch  # noqa: E402
from stylized_motion.learning.mts_operator.physics_context import (  # noqa: E402
    PHYSICAL_METRIC_VERSION,
    PhysicsContext,
)
from stylized_motion.learning.mts_operator.sampling import CommonRandomNumbers  # noqa: E402
from stylized_motion.learning.mts_operator.summary import (  # noqa: E402
    ArmSpec,
    ReferencePool,
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
    parser = argparse.ArgumentParser(description="N03 editing benchmark (read-only, no training).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--token-store", type=Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", nargs="*", type=int, default=[1, 4])
    parser.add_argument("--strengths", nargs="*", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument(
        "--regions",
        nargs="*",
        default=["whole_body", "left_arm"],
        help="whole_body uses the mask's own visible set; a named region is a hard support.",
    )
    parser.add_argument("--graph-radius", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=12, help="Sample cap: the first N manifest rows.")
    parser.add_argument("--eval-seed", type=int, default=20260918)
    parser.add_argument("--mask-kind", default="full_generation")
    parser.add_argument("--flipbook-frames", type=int, default=8)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--save-motions",
        action="store_true",
        help="Also write the decoded source/base/styled windows of the figure rows as "
        "motions/caseNNN_<arm>_<region>.npz, so they can be rendered as characters "
        "(scripts/render_mts_generation.py) instead of only stick-figure flipbooks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "mts_content_style_benchmark":
        raise SystemExit(f"{path} is not an N02 benchmark manifest")
    return payload


def supervised_nll(probabilities: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor) -> float:
    """Mean NLL of ``tokens`` over ``positions`` under ``probabilities``."""
    selection = positions.unsqueeze(0).to(probabilities.device)
    log_probability = probabilities.clamp_min(1e-12).log().gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    weights = selection.to(log_probability.dtype)
    total = float((log_probability * weights).sum())
    count = float(weights.sum())
    return float(-total / count) if count > 0 else float("nan")


def _feature_contact_rate(motion: torch.Tensor, context: PhysicsContext, *, scale: np.ndarray) -> float:
    """Fraction of toe channels whose denormalized contact feature is closed.

    The contact *feature* is the label the tokenizer is trained to reproduce; the
    geometric toe-height gate in ``physics_metrics`` is a different measurement and
    both are reported, because a gate over zero frames says nothing.
    """
    values = motion[0, :, -2:].detach().cpu().numpy()
    offset = np.asarray(context.stats.offset, dtype=np.float32)[-2:]
    raw = values * np.asarray(scale, dtype=np.float32)[-2:] + offset
    return float((np.clip(raw, 0.0, 1.0) > 0.5).mean())


def style_counterfactual(
    arm: Any,
    batch: OperatorBatch,
    *,
    crn: CommonRandomNumbers,
    sample_id: int,
    steps: int,
    wrong_reference_tokens: torch.Tensor | None = None,
    wrong_reference_style: str | None = None,
) -> dict[str, Any]:
    """The same batch with a wrong style input, drawn under the same CRN.

    The style-ID arm is re-run under every other id; the reference arm gets a real
    different-style clip's tokens (chosen by the pool's leak rules, never a roll and
    never the target itself); the constant arm has no style input and says so
    instead of pretending a swap happened.
    """
    if arm.encoder_kind == "style_id":
        if arm.style_index is None or batch.style_ids is None:
            return {"available": False, "reason": "no style index on this arm"}
        variants = []
        for style_id in sorted(set(int(value) for value in arm.style_index.values())):
            if int(style_id) == int(batch.style_ids[0]):
                continue
            swapped = dataclasses.replace(batch, style_ids=torch.full_like(batch.style_ids, style_id))
            with torch.inference_mode():
                result = arm.model(swapped)
                drawn = arm.model.generate_edit(
                    swapped, crn=crn, sample_id=sample_id, step_id=int(steps), steps=int(steps)
                )
            variants.append({"label": f"style_id_{style_id}", "result": result, "tokens": drawn})
        return {"available": bool(variants), "variants": variants, "reason": None}
    if arm.encoder_kind == "reference":
        if wrong_reference_tokens is None:
            return {"available": False, "reason": "no legal different-style reference clip"}
        swapped = dataclasses.replace(batch, reference_tokens=wrong_reference_tokens)
        with torch.inference_mode():
            result = arm.model(swapped)
            drawn = arm.model.generate_edit(
                swapped, crn=crn, sample_id=sample_id, step_id=int(steps), steps=int(steps)
            )
        return {
            "available": True,
            "variants": [
                {"label": f"reference_{wrong_reference_style}", "result": result, "tokens": drawn}
            ],
            "reason": None,
        }
    return {"available": False, "reason": "this arm has no style input (constant control)"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest)
    rows = manifest["rows"][: int(args.max_cases)]
    checkpoint, tokenizer = load_representation_checkpoint(args.tokenizer_checkpoint, torch.device("cpu"))
    tokenizer = tokenizer.eval()
    adapter = adapter_from_tokenizer(tokenizer, num_levels=int(tokenizer.num_levels))
    tokenizer_identity = tokenizer.representation_metadata()
    context = PhysicsContext.from_feature_stats(checkpoint["feature_stats"])
    tokenizer_scale = np.asarray(checkpoint["feature_stats"]["scale"], dtype=np.float32)

    store = open_any_token_store(args.token_store)
    report: dict[str, Any] = {
        "kind": "mts_editing_benchmark",
        "physical_metric_version": PHYSICAL_METRIC_VERSION,
        "manifest": str(args.manifest),
        "manifest_fingerprint": manifest.get("fingerprint_sha256"),
        "manifest_schema": manifest.get("schema"),
        "eval_seed": int(args.eval_seed),
        "steps": [int(value) for value in args.steps],
        "strengths": [float(value) for value in args.strengths],
        "draws": int(args.draws),
        "regions": list(args.regions),
        "cases": len(rows),
        "context": context.describe(),
        "arms": {},
        "rows": [],
        "not_evaluated": {
            "perceptual_style_judgement": "the flipbooks are for a human; no perceptual score is computed here",
            "unseen_style": "no trained checkpoint exists for the zero-shot axis",
            "full_generation_quality": "the whole-body rows use the protocol's own mask, not a free-running rollout",
        },
    }
    try:
        require_token_store_binding(
            store, tokenizer_checkpoint=args.tokenizer_checkpoint, where="N03 token store"
        )
        windows = windows_by_clip(store, "val", frames=FRAMES)
        source = TokenSource(
            store=store,
            windows_by_clip=windows,
            adapter=adapter,
            frames=FRAMES,
            history=int(tokenizer.history_frames),
            rng=np.random.default_rng(int(args.eval_seed)),
        )
        arms = [
            load_arm(
                ArmSpec.parse(text),
                adapter=adapter,
                tokenizer_identity=tokenizer_identity,
                tokenizer_checkpoint=args.tokenizer_checkpoint,
                device=device,
            )
            for text in args.arm
        ]
        report["arms"] = [arm.describe() for arm in arms]
        pool = ReferencePool.from_store(store, windows)
        crn = CommonRandomNumbers(seed=int(args.eval_seed))
        mask_generator = MaskGenerator(MaskConfig(mixture={str(args.mask_kind): 1.0}))
        figures: list[dict[str, Any]] = []

        for case_index, case in enumerate(rows):
            query = case["query"]
            reference = case["reference"]
            query_window = source.window_at(int(query["clip"]), int(query["start"]))
            reference_window = source.window_at(int(reference["clip"]), int(reference["start"]))
            mirror = bool(query.get("mirror", False))
            if query_window is None or reference_window is None:
                report["rows"].append(
                    {
                        "case_index": int(case_index),
                        "subtask": case["subtask"],
                        "status": "unavailable",
                        "reason": "a case window is missing from the token source",
                    }
                )
                continue
            target_tokens = query_window.tokens[None].to(device)
            reference_tokens = reference_window.tokens[None].to(device)
            mask = mask_generator.sample_kind(
                str(args.mask_kind),
                1,
                FRAMES,
                adapter=adapter,
                generator=torch.Generator(device="cpu").manual_seed(int(manifest.get("eval_seed", 3407)) + case_index),
                device=torch.device("cpu"),
            )
            for region in args.regions:
                if region == "whole_body":
                    hard_mask = None
                    visible = mask.visible_mask.to(device)
                else:
                    hard_mask = adapter.hard_mask(
                        [name for name in str(region).split(",") if name],
                        graph_radius=int(args.graph_radius),
                        length=FRAMES,
                        device=device,
                    ).to(device)
                    visible = mask.visible_mask.to(device) & ~hard_mask.unsqueeze(0).expand_as(
                        mask.visible_mask.to(device)
                    )
                # The positions the operator may rewrite: everything hidden, and
                # inside the hard support when there is one.
                edit_positions = (~visible.bool()) & (
                    torch.ones_like(visible, dtype=torch.bool)
                    if hard_mask is None
                    else hard_mask.unsqueeze(0).expand_as(visible)
                )
                for arm in arms:
                    vocabulary = arm.content_vocabulary
                    # Which label the condition is built from is a property of the
                    # *checkpoint*: a round-1 arm was trained on raw action labels, a
                    # canonical arm on the v1 map.  The manifest carries both, so the
                    # probe serves either without inventing a label.
                    recorded_schema = (arm.checkpoint.get("provenance") or {}).get("content_schema")
                    uses_schema = bool(isinstance(recorded_schema, dict) and recorded_schema.get("schema"))
                    label = str(query["content_canonical"] if uses_schema else query["content_raw"])
                    if vocabulary is None or vocabulary.unconditional:
                        content_condition = None
                    else:
                        try:
                            content_condition = vocabulary.vector([label]).to(device)
                        except ValueError as error:
                            # An unknown content label is reported, never borrowed.
                            report["rows"].append(
                                {
                                    "case_index": int(case_index),
                                    "arm": arm.spec.name,
                                    "region": region,
                                    "status": "unavailable",
                                    "reason": str(error).splitlines()[0],
                                }
                            )
                            continue
                    style_ids = None
                    if arm.encoder_kind == "style_id":
                        if arm.style_index is None or str(query["style"]) not in (arm.style_index or {}):
                            report["rows"].append(
                                {
                                    "case_index": int(case_index),
                                    "arm": arm.spec.name,
                                    "region": region,
                                    "status": "unavailable",
                                    "reason": f"style {query['style']!r} is not in this arm's style index",
                                }
                            )
                            continue
                        style_ids = torch.tensor(
                            [int(arm.style_index[str(query["style"])])], dtype=torch.long, device=device
                        )
                    batch = OperatorBatch(
                        target_tokens=target_tokens,
                        reference_tokens=reference_tokens,
                        style_ids=style_ids,
                        visible_mask=visible,
                        hard_mask=hard_mask,
                        content_condition=content_condition,
                        sample_metadata=[case],
                        strength=1.0,
                    )
                    kinematic = context.kinematic(mirror=mirror)
                    for draw in range(int(args.draws)):
                        cell = int(case_index) * 100 + int(draw)
                        for strength in [float(value) for value in args.strengths]:
                            strength_batch = dataclasses.replace(batch, strength=float(strength))
                            with torch.inference_mode():
                                strength_result = arm.model(strength_batch)
                                styled_probs = strength_result.probabilities
                                base_probs = strength_result.base_probabilities
                            wrong_tokens = None
                            wrong_style = None
                            if arm.encoder_kind == "reference":
                                pick = pool.pick(
                                    int(query["clip"]), exclude=(int(reference["clip"]),)
                                )
                                if pick is not None:
                                    candidate_windows = source.windows_by_clip.get(
                                        int(pick["clip_id"])
                                    )
                                    wrong_window = (
                                        None
                                        if not candidate_windows
                                        else source.window_at(
                                            int(pick["clip_id"]),
                                            int(
                                                min(
                                                    int(getattr(request, "target_start", 0))
                                                    for request in candidate_windows
                                                )
                                            ),
                                        )
                                    )
                                    if wrong_window is not None:
                                        wrong_tokens = wrong_window.tokens[None].to(device)
                                        wrong_style = str(pick["style"])
                            # The style-input *distribution* effect: one pair of
                            # forwards per strength, independent of the draw schedule.
                            with torch.inference_mode():
                                style_effect = style_counterfactual(
                                    arm,
                                    strength_batch,
                                    crn=crn,
                                    sample_id=cell,
                                    steps=1,
                                    wrong_reference_tokens=wrong_tokens,
                                    wrong_reference_style=wrong_style,
                                )
                            for steps in [int(value) for value in args.steps]:
                                # The wrong-input *draw* is measured once (at one shot):
                                # a second full schedule per strength would triple the
                                # rollouts without adding evidence.
                                if int(steps) == 1:
                                    counterfactual = style_effect
                                else:
                                    counterfactual = {"available": False, "reason": "wrong-input draw is measured at steps=1"}
                                with torch.inference_mode():
                                    # The base draw shares the styled draw's CRN cell
                                    # *and its schedule*, so "which tokens changed" is a
                                    # coupled statistic: two independent draws of the same
                                    # distribution differ on ~half the tokens by
                                    # construction, and a 4-step draw against a 1-step base
                                    # would differ by the schedule alone (measured: 0.49 at
                                    # every strength, including the identity anchor).
                                    base_draw = arm.model.generate_edit(
                                        strength_batch,
                                        crn=crn,
                                        sample_id=cell,
                                        step_id=int(steps),
                                        steps=int(steps),
                                        use_base=True,
                                    )
                                    styled_draw = arm.model.generate_edit(
                                        strength_batch,
                                        crn=crn,
                                        sample_id=cell,
                                        step_id=int(steps),
                                        steps=int(steps),
                                        use_base=False,
                                    )
                                entry: dict[str, Any] = {
                                    "case_index": int(case_index),
                                    "subtask": case["subtask"],
                                    "arm": arm.spec.name,
                                    "region": region,
                                    "draw": int(draw),
                                    "strength": float(strength),
                                    "steps": int(steps),
                                    "query_clip": int(query["clip"]),
                                    "reference_clip": int(reference["clip"]),
                                    "source_style": case["source_style"],
                                    "target_style": case["target_style"],
                                    "source_content_raw": str(query["content_raw"]),
                                    "source_content_canonical": str(query["content_canonical"]),
                                    "conditioning_map": "content_schema_v1" if uses_schema else "legacy_raw",
                                    "style_input_kind": arm.encoder_kind,
                                }
                                changed_base = styled_draw != base_draw
                                inside = hard_mask.unsqueeze(0).expand_as(changed_base) if hard_mask is not None else torch.ones_like(
                                    changed_base, dtype=torch.bool
                                )
                                entry.update(
                                    {
                                        "changed_token_ratio_all": float(changed_base.float().mean()),
                                        "changed_token_ratio_inside": float(changed_base[inside].float().mean())
                                        if bool(inside.any())
                                        else None,
                                        "changed_token_ratio_outside": float(changed_base[~inside].float().mean())
                                        if bool((~inside).any())
                                        else None,
                                        "nll_source_base": supervised_nll(base_probs, target_tokens, edit_positions),
                                        "nll_source_styled": supervised_nll(styled_probs, target_tokens, edit_positions),
                                        "tv_styled_vs_base_distribution": float(
                                            0.5 * (styled_probs - base_probs).abs().sum(-1).mean()
                                        ),
                                        "nll_improvement_vs_base": float(
                                            supervised_nll(base_probs, target_tokens, edit_positions)
                                            - supervised_nll(styled_probs, target_tokens, edit_positions)
                                        ),
                                    }
                                )
                                if counterfactual.get("available") and counterfactual.get("variants"):
                                    tvs = []
                                    improvements = []
                                    changes = []
                                    for variant in counterfactual["variants"]:
                                        tvs.append(
                                            float(
                                                0.5
                                                * (styled_probs - variant["result"].probabilities).abs().sum(-1).mean()
                                            )
                                        )
                                        improvements.append(
                                            supervised_nll(variant["result"].probabilities, target_tokens, edit_positions)
                                            - supervised_nll(styled_probs, target_tokens, edit_positions)
                                        )
                                        changes.append(float((styled_draw != variant["tokens"]).float().mean()))
                                    entry.update(
                                        {
                                            "style_input_available": True,
                                            "tv_correct_vs_wrong_style_input": float(np.mean(tvs)),
                                            "nll_improvement_vs_wrong_style_input": float(np.mean(improvements)),
                                            "changed_token_ratio_vs_wrong_style_input": float(np.mean(changes)),
                                        }
                                    )
                                else:
                                    entry.update(
                                        {
                                            "style_input_available": False,
                                            "style_input_reason": counterfactual.get("reason"),
                                            "tv_correct_vs_wrong_style_input": None,
                                            "nll_improvement_vs_wrong_style_input": None,
                                            "changed_token_ratio_vs_wrong_style_input": None,
                                        }
                                    )
                                with torch.inference_mode():
                                    decoded_source = tokenizer.decode_indices(target_tokens)
                                    decoded_base = tokenizer.decode_indices(base_draw)
                                    decoded_styled = tokenizer.decode_indices(styled_draw)
                                comparison = comparison_physics(
                                    source_motion=decoded_source,
                                    base_motion=decoded_base,
                                    styled_motion=decoded_styled,
                                    kinematic=kinematic,
                                    edit_interval=(0, FRAMES),
                                )
                                for name, values in comparison.items():
                                    for key, value in values.items():
                                        if key == "comparison":
                                            continue
                                        entry[f"{name}_{key}"] = value
                                entry["contact_gate_frames"] = (
                                    comparison["base_to_styled"].get("contact_gate_frames")
                                )
                                # The *feature* contacts (the label the tokenizer
                                # carries) next to the geometric gate: a motion whose
                                # gate is empty is not evidence of clean contacts.
                                entry["contact_rate_from_features_source"] = _feature_contact_rate(
                                    decoded_source, context, scale=tokenizer_scale
                                )
                                entry["contact_rate_from_features_base"] = _feature_contact_rate(
                                    decoded_base, context, scale=tokenizer_scale
                                )
                                entry["contact_rate_from_features_styled"] = _feature_contact_rate(
                                    decoded_styled, context, scale=tokenizer_scale
                                )
                                report["rows"].append(entry)
                                if (
                                    float(strength) == 1.0
                                    and int(draw) == 0
                                    and int(steps) == 1
                                    and not args.no_figures
                                ):
                                    record = {
                                        "case_index": int(case_index),
                                        "arm": arm.spec.name,
                                        "region": region,
                                        "mirror": mirror,
                                        "source": decoded_source[0].detach().cpu().numpy(),
                                        "base": decoded_base[0].detach().cpu().numpy(),
                                        "styled": decoded_styled[0].detach().cpu().numpy(),
                                    }
                                    figures.append(record)
                                    if args.save_motions:
                                        motions_dir = args.output / "motions"
                                        motions_dir.mkdir(parents=True, exist_ok=True)
                                        np.savez(
                                            motions_dir
                                            / f"case{int(case_index):03d}_{arm.spec.name}_{region}.npz",
                                            source=record["source"],
                                            base=record["base"],
                                            styled=record["styled"],
                                            source_tokens=target_tokens[0].detach().cpu().numpy(),
                                            styled_tokens=styled_draw[0].detach().cpu().numpy(),
                                            base_tokens=base_draw[0].detach().cpu().numpy(),
                                            meta=json.dumps(
                                                {
                                                    key: record[key]
                                                    for key in ("case_index", "arm", "region", "mirror")
                                                }
                                                | {
                                                    key: case.get(key)
                                                    for key in ("subtask", "source_style", "target_style")
                                                }
                                                | {
                                                    key: query[key]
                                                    for key in ("clip", "content_raw", "content_canonical")
                                                },
                                                ensure_ascii=False,
                                            ),
                                        )
                    print(f"case {case_index} arm {arm.spec.name} region {region} done", flush=True)
        report["figures"] = draw_flipbooks(
            figures, args.output / "flipbooks", context, frames=int(args.flipbook_frames)
        )
        report["style_input_effect"] = summarize_style_input_effect(report["rows"])
        write_summary(args.output / "benchmark.json", report)
        with (args.output / "benchmark_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            keys = sorted({key for row in report["rows"] for key in row})
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(report["rows"])
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rows": len(report["rows"]),
                    "figures": len(report["figures"]),
                    "style_input_effect": report["style_input_effect"],
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
    finally:
        store.close()
    return 0


def summarize_style_input_effect(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per arm, the style-input effect over the rows that could measure it."""
    report: dict[str, Any] = {}
    for arm in sorted({str(row.get("arm")) for row in rows}):
        subset = [row for row in rows if str(row.get("arm")) == arm and row.get("style_input_available")]
        if not subset:
            report[arm] = {
                "rows": 0,
                "reason": next(
                    (row.get("style_input_reason") for row in rows if str(row.get("arm")) == arm and row.get("style_input_reason")),
                    "no rows",
                ),
            }
            continue
        tv = np.asarray([float(row["tv_correct_vs_wrong_style_input"]) for row in subset])
        improvement = np.asarray([float(row["nll_improvement_vs_wrong_style_input"]) for row in subset])
        changed = np.asarray([float(row["changed_token_ratio_vs_wrong_style_input"]) for row in subset])
        report[arm] = {
            "rows": len(subset),
            "tv_mean": float(tv.mean()),
            "nll_improvement_mean": float(improvement.mean()),
            "nll_improvement_median": float(np.median(improvement)),
            "nll_improvement_rows_positive": int((improvement > 0).sum()),
            "changed_token_ratio_mean": float(changed.mean()),
            "changed_token_ratio_rows_nonzero": int((changed > 0).sum()),
        }
    return report


def draw_flipbooks(
    records: list[dict[str, Any]], directory: Path, context: PhysicsContext, *, frames: int
) -> list[dict[str, Any]]:
    """One PNG per (case, arm, region): source/base/styled in a fixed projection."""
    if not records:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for record in records:
        states = {
            name: context.world_state(
                record[name], mirror=bool(record.get("mirror", False)), normalized=True, contact_threshold=None
            ).global_positions
            for name in ("source", "base", "styled")
        }
        count = int(states["source"].shape[0])
        step = max(1, count // max(int(frames), 1))
        indices = list(range(0, count, step))[: int(frames)]
        stacked = np.stack([states[name][indices] for name in states]).reshape(-1, 3)
        margin = 0.15
        lower = stacked.min(axis=0) - margin
        upper = stacked.max(axis=0) + margin
        figure, axes = plt.subplots(3, len(indices), figsize=(1.9 * len(indices), 6.0), squeeze=False)
        for row_index, name in enumerate(("source", "base", "styled")):
            colour = {"source": "#444444", "base": "#1f77b4", "styled": "#d62728"}[name]
            for column, frame in enumerate(indices):
                axis = axes[row_index][column]
                positions = states[name][frame]
                for joint, parent in enumerate(context.parents):
                    if parent < 0:
                        continue
                    axis.plot(
                        [positions[parent, 0], positions[joint, 0]],
                        [positions[parent, 2], positions[joint, 2]],
                        color=colour,
                        linewidth=1.0,
                    )
                axis.set_xlim(lower[0], upper[0])
                axis.set_ylim(lower[2], upper[2])
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_aspect("equal", adjustable="box")
                if row_index == 0:
                    axis.set_title(f"f{frame}", fontsize=8)
            axes[row_index][0].set_ylabel(name, fontsize=9)
        figure_path = directory / f"case{record['case_index']:03d}_{record['arm']}_{record['region']}.png"
        figure.tight_layout()
        figure.savefig(figure_path, dpi=110)
        plt.close(figure)
        written.append(
            {
                "path": str(figure_path),
                **{key: value for key, value in record.items() if key not in ("source", "base", "styled")},
            }
        )
    return written


if __name__ == "__main__":
    raise SystemExit(main())
