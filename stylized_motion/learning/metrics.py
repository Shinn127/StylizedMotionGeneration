"""Reusable metric callback collection for representation runners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


MetricCallback = Callable[[Any, Any], torch.Tensor | float]


class MetricSuite(dict[str, MetricCallback]):
    """Named scalar callbacks accepted directly by ``RepresentationRunner``."""

    def evaluate(self, output: Any, batch: Any) -> dict[str, torch.Tensor | float]:
        return {name: callback(output, batch) for name, callback in self.items()}


__all__ = ["MetricCallback", "MetricSuite"]
