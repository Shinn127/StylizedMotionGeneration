#!/usr/bin/env python
"""Build the paper figures and tables from the evaluation artifacts.

    python scripts/plot_mts_figures.py --output outputs/mts_figures
    # optional extra artifacts, repeatable:
    #   --locality  outputs/stage3_eval/locality/locality.csv
    #   --physics   outputs/stage3_eval/physics/physics.csv
    #   --geometry  outputs/nef_eval_1h/nef_geometry/probe_geometry.json
    #   --operator  outputs/mts_operator/<family>/<run>/operator_metrics.json

Everything is read from files the other scripts already write, so a figure can be
regenerated without rerunning a model.  Plan §8.3 asks for six figures; this
builds the ones that are data-driven (1, 3, 4, 5) plus the aggregate tables, and
records which artifacts were missing instead of plotting an empty axis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_LOCALITY = ("outputs/stage3_eval/locality/locality.csv", "outputs/stage3_eval_ah/locality/locality.csv")
DEFAULT_PHYSICS = ("outputs/stage3_eval/physics/physics.csv", "outputs/stage3_eval_ah/physics/physics.csv")
DEFAULT_GEOMETRY = ("outputs/nef_eval_1h/nef_geometry/probe_geometry.json",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MTS paper figures and tables from artifacts.")
    parser.add_argument("--locality", type=Path, action="append", default=None)
    parser.add_argument("--physics", type=Path, action="append", default=None)
    parser.add_argument("--geometry", type=Path, action="append", default=None)
    parser.add_argument("--operator", type=Path, action="append", default=None, help="operator_metrics.json files.")
    parser.add_argument("--output", type=Path, default=Path("outputs/mts_figures"))
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def figure_leakage(locality_rows: list[tuple[str, list[dict[str, str]]]], output: Path) -> Path | None:
    """Figure 3: off-target leakage per representation (the R1 claim)."""
    series: dict[str, dict[str, float]] = defaultdict(dict)
    for source, rows in locality_rows:
        for row in rows:
            if row.get("part") == "whole_body":
                continue
            key = f"{row['representation']}\n({source})"
            series[key][row["part"]] = _number(row["non_target_joint_change_max_max"])
    if not series:
        return None
    parts = sorted({part for values in series.values() for part in values})
    labels = list(series)
    width = 0.8 / max(len(parts), 1)
    figure, axis = plt.subplots(figsize=(9, 4.2))
    for index, part in enumerate(parts):
        values = [series[label].get(part, float("nan")) for label in labels]
        positions = np.arange(len(labels)) + index * width
        axis.bar(positions, values, width, label=part)
    axis.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("off-target joint change (m, max)")
    axis.set_title("Local edit leakage: non-target joints move (flat) or do not (part / NEF)")
    axis.legend(fontsize=7, ncol=4)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output / "fig3_off_target_leakage.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_geometry(geometry_files: list[tuple[str, dict[str, Any]]], output: Path) -> Path | None:
    """Figure 4: adjacent vs far level effect per stream."""
    series: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for source, payload in geometry_files:
        records = payload.get("geometry", payload).get("per_coordinate", [])
        by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_stream[record["stream"]].append(record)
        streams = list(by_stream)
        ratio = [
            float(np.median([record["adjacent_to_far_ratio"] for record in by_stream[stream]]))
            for stream in streams
        ]
        consistency = [
            float(np.mean([record["direction_consistency"] for record in by_stream[stream]]))
            for stream in streams
        ]
        series[source] = (streams, ratio, consistency)
    if not series:
        return None
    source, (streams, ratio, consistency) = next(iter(series.items()))
    order = np.argsort(ratio)
    figure, axis = plt.subplots(figsize=(10, 4.4))
    positions = np.arange(len(streams))
    bars = axis.bar(positions, [ratio[index] for index in order], color="#4878a8", label="median adjacent/far ratio")
    axis.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="far = adjacent")
    axis.set_xticks(positions)
    axis.set_xticklabels([streams[index] for index in order], rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("adjacent / far distance ratio")
    axis.set_title(f"FSQ level geometry per stream ({source})")
    for bar, index in zip(bars, order):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f"{consistency[index]:.2f}", ha="center", fontsize=7, color="#444")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output / "fig4_level_geometry.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_strength(operator_files: list[tuple[str, dict[str, Any]]], output: Path) -> Path | None:
    """Figure 1/5: strength response per operator (and reference sensitivity)."""
    curves = {
        label: payload.get("strength_curve", [])
        for label, payload in operator_files
        if payload.get("strength_curve")
    }
    if not curves:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, curve in curves.items():
        strengths = [float(point["strength"]) for point in curve]
        axes[0].plot(strengths, [float(point["total_variation"]) for point in curve], marker="o", label=label)
        axes[1].plot(strengths, [float(point["changed_token_ratio"]) for point in curve], marker="s", label=label)
    axes[0].set_xlabel("style strength")
    axes[0].set_ylabel("total variation vs base")
    axes[0].set_title("Strength response (distribution)")
    axes[1].set_xlabel("style strength")
    axes[1].set_ylabel("changed-token ratio (CRN)")
    axes[1].set_title("Strength response (coupled draws)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "fig1_strength_response.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_reference(operator_files: list[tuple[str, dict[str, Any]]], output: Path) -> Path | None:
    """Figure 5: correct / wrong / random reference response."""
    rows: list[tuple[str, dict[str, float]]] = []
    for label, payload in operator_files:
        aggregate = payload.get("aggregate", {})
        if "nll_correct" in aggregate:
            rows.append((label, {key: _number(value.get("mean")) for key, value in aggregate.items()}))
    if not rows:
        return None
    keys = ["nll_correct", "nll_wrong", "nll_random"]
    labels = [label for label, _ in rows]
    width = 0.8 / len(keys)
    figure, axis = plt.subplots(figsize=(8, 4))
    for index, key in enumerate(keys):
        values = [values_.get(key, float("nan")) for _, values_ in rows]
        positions = np.arange(len(labels)) + index * width
        axis.bar(positions, values, width, label=key)
    axis.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    axis.set_xticklabels(labels, fontsize=8, rotation=15)
    axis.set_ylabel("masked NLL of the target")
    axis.set_title("Reference sensitivity: correct reference must score best")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = output / "fig5_reference_response.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_tables(
    locality_rows: list[tuple[str, list[dict[str, str]]]],
    physics_rows: list[tuple[str, list[dict[str, str]]]],
    operator_files: list[tuple[str, dict[str, Any]]],
    output: Path,
) -> list[Path]:
    written: list[Path] = []
    if locality_rows:
        merged: list[dict[str, Any]] = []
        for source, rows in locality_rows:
            for row in rows:
                merged.append({"source": source, **row})
        path = output / "table_locality.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
            writer.writeheader()
            writer.writerows(merged)
        written.append(path)
    if physics_rows:
        merged = []
        for source, rows in physics_rows:
            for row in rows:
                merged.append({"source": source, **row})
        path = output / "table_physics.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(merged[0]))
            writer.writeheader()
            writer.writerows(merged)
        written.append(path)
    if operator_files:
        merged = []
        for label, payload in operator_files:
            aggregate = payload.get("aggregate", {})
            row: dict[str, Any] = {
                "label": label,
                "checkpoint": payload.get("checkpoint"),
                "operator": payload.get("operator"),
                "style_encoder": payload.get("style_encoder"),
                "split": payload.get("split"),
                "regions": ",".join(payload.get("regions", [])),
                "graph_radius": payload.get("graph_radius"),
                "frame_range": payload.get("frame_range"),
                "style_retrieval_top1": payload.get("style_retrieval_top1"),
            }
            for key, value in aggregate.items():
                if isinstance(value, dict) and "mean" in value:
                    row[f"{key}_mean"] = value["mean"]
            merged.append(row)
        path = output / "table_operator.csv"
        fields = sorted({key for row in merged for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(merged)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    locality_paths = args.locality or [Path(p) for p in DEFAULT_LOCALITY]
    physics_paths = args.physics or [Path(p) for p in DEFAULT_PHYSICS]
    geometry_paths = args.geometry or [Path(p) for p in DEFAULT_GEOMETRY]

    locality = [(path.parent.parent.name or str(path), _read_csv(path)) for path in locality_paths if path.exists()]
    physics = [(path.parent.parent.name or str(path), _read_csv(path)) for path in physics_paths if path.exists()]
    geometry = [(path.parent.name, json.loads(path.read_text(encoding="utf-8"))) for path in geometry_paths if path.exists()]
    operators = [
        (payload.get("label") or payload.get("operator") or path.parent.name, payload)
        for path in (args.operator or [])
        if path.exists()
        for payload in [json.loads(path.read_text(encoding="utf-8"))]
    ]

    figures = [
        figure_leakage(locality, output),
        figure_geometry(geometry, output),
        figure_strength(operators, output),
        figure_reference(operators, output),
    ]
    tables = write_tables(locality, physics, operators, output)
    missing = []
    if not locality:
        missing.append("locality CSV (scripts/evaluate_nef_locality.py)")
    if not geometry:
        missing.append("probe geometry JSON (scripts/probe_nef_geometry.py)")
    if not operators:
        missing.append("operator metrics (scripts/evaluate_mts_operator.py)")
    report = {
        "figures": [str(path) for path in figures if path is not None],
        "tables": [str(path) for path in tables],
        "missing_artifacts": missing,
        "note": (
            "Figures 2 and 6 of plan §8.3 are visual comparisons (rendered motion strips and "
            "failure cases): generate the token/motion inputs with scripts/generate_mts_operator.py "
            "and render through stylized_motion.anim.somaview."
        ),
    }
    (output / "figures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
