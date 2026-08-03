"""Shared checkpoint persistence for representation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .representation import RepresentationAdapter, load_representation_checkpoint


class CheckpointManager:
    """Small filesystem-owned saver used by the common representation runner."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, payload: dict[str, Any], name: str = "last.pt", *, is_best: bool = False) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        torch.save(payload, path)
        if is_best:
            torch.save(payload, self.directory / "best.pt")
        return path

    def load(
        self,
        path: str | Path,
        device: torch.device,
        *,
        feature_schema: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], RepresentationAdapter]:
        return load_representation_checkpoint(path, device, feature_schema=feature_schema)


__all__ = ["CheckpointManager", "load_representation_checkpoint"]
