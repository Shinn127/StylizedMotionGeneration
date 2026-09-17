#!/usr/bin/env python
"""Measure decoded locality of region-supported token edits (MTS-FSQ plan, R1).

One NEF-FSQ checkpoint answers: when a region's tokens are replaced by a donor's,
which decoded features and joints move, how far the decoder carries the edit and
what it costs at the contacts.  The same report is produced for flat/part
checkpoints so the three representations can be compared on identical windows: a
flat tokenizer can only be edited as a whole, and that contrast is the point.

    python scripts/evaluate_nef_locality.py \
      --checkpoint outputs/nef_fsq_40x9/best.pt \
      --feature-database data/processed/100style_pruned_90/fsq_window_index \
      --split test --parts left_arm right_arm left_leg right_leg \
      --output outputs/nef_locality/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylized_motion.data import open_any_feature_store  # noqa: E402
from stylized_motion.data.sampling import FixedWindowSampler  # noqa: E402
from stylized_motion.learning.nef_eval import validate_checkpoint_store  # noqa: E402
from stylized_motion.learning.nef_layout import NEF_EDIT_PARTS, NEFLayout, nef_edit_streams  # noqa: E402
from stylized_motion.learning.nef_probe import (  # noqa: E402
    KinematicContext,
    json_dumps,
    locality_report,
    model_space_window,
    read_probe_window,
)
from stylized_motion.learning.part_layout import PART_NAMES  # noqa: E402
from stylized_motion.learning.representation import (  # noqa: E402
    FLAT_FSQ_FAMILY,
    NEF_FSQ_FAMILY,
    PART_FSQ_FAMILY,
    load_representation_checkpoint,
)
from stylized_motion.learning.runner import choose_device  # noqa: E402

WHOLE_BODY = "whole_body"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decoded locality report for flat / part / NEF token edits."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat once per representation to build the R1 comparison table "
        "(flat / part / NEF on the same windows).",
    )
    parser.add_argument("--feature-database", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--parts", nargs="+", default=["left_arm", "right_arm", "left_leg", "right_leg"])
    parser.add_argument("--edit", choices=["strict", "full"], default="strict")
    parser.add_argument("--max-clips", type=int, default=64, help="Target windows (0 = all).")
    parser.add_argument("--edit-start", type=int, default=16)
    parser.add_argument("--edit-stop", type=int, default=48)
    parser.add_argument("--include-whole-body", action="store_true", default=True)
    parser.add_argument("--root-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _semantic_part_joints(model, store, part: str) -> list[int]:
    """The part's joints under the NEF naming convention, for flat models."""
    layout = NEFLayout.from_skeleton(store.names, store.parents)
    return sorted(layout.stream_joints(NEF_EDIT_PARTS[part][0]))


def region_support(
    model, store, region: str, *, full_part: bool
) -> tuple[list[slice], list[int], list[int], str]:
    """Coordinate slices, feature support, target joints and scope label."""
    module = model.module
    if region == WHOLE_BODY:
        support = slice(0, int(module.num_coordinates))
        return [support], list(range(int(module.motion_dim))), [], f"all:{module.num_coordinates} coordinates"
    if model.family == NEF_FSQ_FAMILY:
        layout = module.layout
        streams = nef_edit_streams(region, full_part=full_part)
        feature_indices = layout.feature_indices(module.motion_dim)
        return (
            [layout.stream_slices[stream] for stream in streams],
            torch.cat([feature_indices[stream] for stream in streams]).tolist(),
            sorted({joint for stream in streams for joint in layout.stream_joints(stream)}),
            "streams:" + ",".join(streams),
        )
    if model.family == PART_FSQ_FAMILY:
        if region not in PART_NAMES:
            raise ValueError(f"part_fsq has no {region!r} group; available {sorted(PART_NAMES)}")
        layout = module.layout
        feature_indices = layout.feature_indices(module.motion_dim)
        return (
            [layout.group_slices[region]],
            feature_indices[region].tolist(),
            list(layout.part_joint_indices[PART_NAMES.index(region)]),
            f"group:{region}",
        )
    if model.family == FLAT_FSQ_FAMILY:
        # No region contract: the only honest flat edit is the whole token, and
        # locality is measured against the region's joints.
        return (
            [slice(0, 40)],
            list(range(int(module.motion_dim))),
            _semantic_part_joints(model, store, region),
            "flat:whole-token",
        )
    raise ValueError(f"Unsupported representation family for locality: {model.family!r}")


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _aggregate(parts: list[dict[str, object]], support_meta: dict[str, object]) -> dict[str, object]:
    keys = sorted({key for part in parts for key in part})
    result: dict[str, object] = dict(support_meta)
    result["windows"] = len(parts)
    for key in keys:
        values = [part[key] for part in parts]
        numeric = [value for value in values if isinstance(value, (int, float))]
        if len(numeric) == len(values):
            result[key] = _summarize(numeric)
        else:
            result[key] = values[0] if len(set(map(str, values))) == 1 else values
    return result


def evaluate_region(model, tokens, donor_tokens, *, slices, features, joints, kinematic, start, stop):
    reports = []
    for target, donor in zip(tokens, donor_tokens):
        reports.append(
            locality_report(
                model,
                target[None],
                donor[None],
                slices=slices,
                start=start,
                stop=stop,
                target_joints=joints,
                feature_support=features,
                kinematic=kinematic,
            )
        )
    flat: list[dict[str, object]] = []
    for report in reports:
        row: dict[str, object] = {
            "support_coordinates": report["support_coordinates"],
            "support_fraction": report["support_fraction"],
            "support_token_change_fraction": report["support_token_change_fraction"],
            "edit_feature_mean": report["edit_feature_mean"],
            "off_target_feature_mean": report["off_target_feature_mean"],
            "off_target_feature_max": report["off_target_feature_max"],
            "pre_edit_unchanged": report["pre_edit_unchanged"],
            "post_influence_unchanged": report["post_influence_unchanged"],
        }
        kinematics = report.get("kinematics")
        if kinematics:
            row.update(
                {
                    "target_joint_change": kinematics["target_joint_change"],
                    "descendant_joint_change": kinematics["descendant_joint_change"],
                    "non_target_joint_change_mean": kinematics["non_target_joint_change_mean"],
                    "non_target_joint_change_max": kinematics["non_target_joint_change_max"],
                    "root_position_change": kinematics["root_position_change"],
                    "root_rotation_change": kinematics["root_rotation_change"],
                }
            )
            if "contact_flip_rate" in kinematics:
                row["contact_flip_rate"] = kinematics["contact_flip_rate"]
        flat.append(row)
    support_meta = {
        "support_coordinates": flat[0]["support_coordinates"] if flat else 0,
        "support_fraction": flat[0]["support_fraction"] if flat else 0.0,
        "pre_edit_unchanged_all": all(bool(row["pre_edit_unchanged"]) for row in flat),
        "post_influence_unchanged_all": all(
            bool(row["post_influence_unchanged"]) for row in flat
        ),
    }
    return _aggregate(flat, support_meta)


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_path: Path, *, windows_cache) -> dict[str, Any]:
    """Evaluates one representation on the shared window set."""
    device = choose_device(args.device)
    checkpoint, model = load_representation_checkpoint(checkpoint_path, device)
    store = windows_cache["store"]
    validate_checkpoint_store(checkpoint, model, store)
    model = model.to(device).eval()
    module = model.module
    history = int(model.history_frames)
    windows = windows_cache["windows"]
    donors = windows_cache["donors"]
    feature_stats = checkpoint["feature_stats"]
    shards: dict[int, Any] = windows_cache.setdefault("shards", {})
    target_windows, donor_windows = [], []
    with torch.no_grad():
        for request, donor_request in zip(windows, donors):
            for source, sink in ((request, target_windows), (donor_request, donor_windows)):
                window = read_probe_window(store, source, history=history, shards=shards)
                motion = model_space_window(window, store, feature_stats).to(device)
                sink.append(model.encode_indices(motion[None])[0, history:])
    tokens = torch.stack(target_windows)
    donor_tokens = torch.stack(donor_windows)
    kinematic = KinematicContext.from_feature_stats(feature_stats, dt=args.root_dt)

    regions = list(args.parts)
    if args.include_whole_body:
        regions.append(WHOLE_BODY)
    parts: list[dict[str, Any]] = []
    for region in regions:
        slices, features, joints, scope = region_support(
            model, store, region, full_part=args.edit == "full"
        )
        if not joints and region != WHOLE_BODY:
            joints = _semantic_part_joints(model, store, region)
        parts.append(
            {
                "part": region,
                "scope": scope,
                "edit": args.edit if region != WHOLE_BODY else "whole-token",
                **_aggregate_region(
                    model,
                    tokens,
                    donor_tokens,
                    slices=slices,
                    features=features,
                    joints=joints,
                    kinematic=kinematic,
                    start=int(args.edit_start),
                    stop=int(args.edit_stop),
                ),
            }
        )
        print(f"  [{model.family}] {region}: {parts[-1]['scope']}", flush=True)
    layout = module.get_token_layout() if hasattr(module, "get_token_layout") else None
    return {
        "family": model.family,
        "representation_id": model.representation_id,
        "checkpoint": str(checkpoint_path),
        "layout_hash": layout.layout_hash() if layout is not None else None,
        "parts": parts,
    }


def comparison_table(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per (representation, part): the numbers R1 is decided on.

    Per-part metrics are aggregated over windows, so each numeric metric becomes
    two flat columns (``<metric>_mean`` and ``<metric>_max``) — a CSV cell holding
    a summary dict would not be usable in a table or a plot.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        for part in result["parts"]:
            row: dict[str, Any] = {
                "representation": result["family"],
                "representation_id": result["representation_id"],
                "part": part["part"],
                "scope": part["scope"],
                "edit": part.get("edit"),
            }
            for name, value in part.items():
                if name in {"part", "scope", "edit"}:
                    continue
                if isinstance(value, Mapping):
                    for statistic in ("mean", "max"):
                        if statistic in value:
                            row[f"{name}_{statistic}"] = value[statistic]
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    row[name] = value
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = open_any_feature_store(args.feature_database)
    try:
        windows = list(
            FixedWindowSampler(store, args.split, target_frames=64, stride=64, include_tail=True)
        )
        if args.max_clips > 0:
            windows = windows[: args.max_clips]
        if len(windows) < 2:
            raise ValueError("Locality evaluation needs at least two windows for a donor pair")
        donors = windows[len(windows) // 2 :] + windows[: len(windows) // 2]
        cache = {"store": store, "windows": windows, "donors": donors}
        results: list[dict[str, Any]] = []
        for checkpoint_path in args.checkpoint:
            print(f"evaluating {checkpoint_path}", flush=True)
            results.append(evaluate_checkpoint(args, Path(checkpoint_path), windows_cache=cache))
        payload = {
            "kind": "nef_locality",
            "feature_database": str(args.feature_database),
            "split": args.split,
            "windows": len(windows),
            "edit_interval": [int(args.edit_start), int(args.edit_stop)],
            "representations": results,
            "comparison": comparison_table(results),
            "note": (
                "support_fraction is the fraction of the 40 coordinates the "
                "representation can edit for this part (flat has no region "
                "contract, so its honest edit is the whole token); "
                "off_target_feature_max is the decoded feature change outside that "
                "support and non_target_joint_change_* excludes the edit's own "
                "kinematic descendants."
            ),
        }
        output = args.output or Path("outputs/nef_locality/run")
        output.mkdir(parents=True, exist_ok=True)
        (output / "locality.json").write_text(json_dumps(payload) + "\n", encoding="utf-8")
        header = list(payload["comparison"][0]) if payload["comparison"] else []
        if header:
            with (output / "locality.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(payload["comparison"])
        print(json.dumps({"representations": [r["family"] for r in results], "rows": len(payload["comparison"])}, indent=2))
        print(f"wrote {output / 'locality.json'} and {output / 'locality.csv'}")
    finally:
        store.close()


def _aggregate_region(model, tokens, donor_tokens, *, slices, features, joints, kinematic, start, stop):
    return evaluate_region(
        model,
        tokens,
        donor_tokens,
        slices=slices,
        features=features,
        joints=joints,
        kinematic=kinematic,
        start=start,
        stop=stop,
    )


if __name__ == "__main__":
    main()
