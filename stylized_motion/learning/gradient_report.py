"""Summarize gradient probe JSONL output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize_gradient_probe(path: str | Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Gradient probe records must be JSON objects")
            records.append(value)
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"value": [], "norm": [], "share": [], "projection": [], "cosine": []}
    )
    for record in records:
        edit_part = str(record.get("edit_part") or "none")
        components = record.get("components", {})
        if not isinstance(components, dict):
            continue
        for name, metrics in components.items():
            if not isinstance(metrics, dict):
                continue
            bucket = grouped[(edit_part, str(name))]
            for metric in bucket:
                value = metrics.get(metric)
                if isinstance(value, (float, int)):
                    bucket[metric].append(float(value))
    summary: dict[str, Any] = {
        "records": len(records),
        "steps": [int(record["step"]) for record in records if "step" in record],
        "components": {},
    }
    for (edit_part, name), metrics in sorted(grouped.items()):
        key = f"{edit_part}/{name}"
        summary["components"][key] = {
            metric: {
                "mean": mean(values) if values else 0.0,
                "median": median(values) if values else 0.0,
                "p10": _percentile(values, 0.10),
                "p90": _percentile(values, 0.90),
                "negative_fraction": (
                    sum(value < 0.0 for value in values) / len(values) if values else 0.0
                ) if metric in {"projection", "cosine"} else None,
            }
            for metric, values in metrics.items()
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="gradient_probe.jsonl path")
    args = parser.parse_args(argv)
    print(json.dumps(summarize_gradient_probe(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
