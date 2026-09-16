"""Thin read-only adapter over :class:`NEFLayout` for the MTS modules.

The transport, graph and masking code needs index tables (stream of every
coordinate, edge lists, feature ownership), not the layout's validation logic.
Building those tables here — once, from the layout's own tables — is what keeps
"the operator edits stream s" and "the decoder owns stream s" the same statement
instead of two hand-written slice lists that can drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from stylized_motion.learning.nef_layout import (
    NEF_EDIT_PARTS,
    NEF_FAMILY_STREAMS,
    NEF_STREAM_FAMILY,
    NEF_STREAM_NAMES,
    NEF_WHOLE_BODY_REGION,
    NEFLayout,
    nef_edit_streams,
)

from .contract import TokenSpec


class LayoutAdapter:
    """Index views over one NEF layout, with no duplicated slices."""

    def __init__(self, layout: NEFLayout, *, num_levels: int = 9, family: str = "nef_fsq",
                 representation_id: str = "") -> None:
        if not isinstance(layout, NEFLayout):
            raise TypeError("LayoutAdapter requires a NEFLayout")
        self.layout = layout
        self.num_levels = int(num_levels)
        self._metadata = layout.coordinate_metadata()
        self._stream_index = {stream: index for index, stream in enumerate(NEF_STREAM_NAMES)}
        self._slices = layout.stream_slices

    # -- identity ----------------------------------------------------------
    @property
    def stream_names(self) -> tuple[str, ...]:
        return NEF_STREAM_NAMES

    @property
    def num_streams(self) -> int:
        return len(NEF_STREAM_NAMES)

    @property
    def num_coordinates(self) -> int:
        return self.layout.num_coordinates

    @property
    def num_levels_value(self) -> int:
        return self.num_levels

    @property
    def layout_hash(self) -> str:
        return self.layout.layout_hash()

    def token_spec(self, *, representation_id: str = "") -> TokenSpec:
        return TokenSpec(
            num_coordinates=self.num_coordinates,
            num_levels=self.num_levels,
            num_streams=self.num_streams,
            family="nef_fsq",
            representation_id=representation_id,
            layout_hash=self.layout_hash,
        )

    # -- streams and coordinates -------------------------------------------
    def stream_index(self, stream: str) -> int:
        try:
            return self._stream_index[stream]
        except KeyError as exc:
            raise ValueError(f"Unknown NEF stream {stream!r}") from exc

    def coordinate_slice(self, stream: str) -> slice:
        return self._slices[stream]

    @property
    def stream_slices(self) -> dict[str, slice]:
        return dict(self._slices)

    def stream_of_coordinate(self, coordinate: int) -> str:
        return str(self._metadata[int(coordinate)]["stream"])

    def coordinate_indices(self, streams: Sequence[str]) -> torch.Tensor:
        """Long indices of every coordinate owned by ``streams``, sorted."""
        indices = [
            coordinate
            for coordinate, record in enumerate(self._metadata)
            if record["stream"] in streams
        ]
        return torch.tensor(sorted(indices), dtype=torch.long)

    def coordinate_stream_ids(self, *, device: torch.device | None = None) -> torch.Tensor:
        """``[40]`` long tensor mapping each coordinate to its stream index."""
        return torch.tensor(
            [self.stream_index(str(record["stream"])) for record in self._metadata],
            dtype=torch.long,
            device=device,
        )

    def coordinate_family_ids(self, *, device: torch.device | None = None) -> torch.Tensor:
        """``[40]`` long tensor mapping each coordinate to its family index."""
        families = tuple(NEF_FAMILY_STREAMS)
        index = {family: position for position, family in enumerate(families)}
        return torch.tensor(
            [index[str(record["family"])] for record in self._metadata],
            dtype=torch.long,
            device=device,
        )

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(NEF_FAMILY_STREAMS)

    def family_of_stream(self, stream: str) -> str:
        return NEF_STREAM_FAMILY[stream]

    def family_of_coordinate(self, coordinate: int) -> str:
        return str(self._metadata[int(coordinate)]["family"])

    # -- features ----------------------------------------------------------
    def feature_indices(self, motion_dim: int) -> dict[str, torch.Tensor]:
        return self.layout.feature_indices(int(motion_dim))

    def feature_of_stream(self, motion_dim: int) -> torch.Tensor:
        """``[motion_dim]`` long tensor mapping each feature to its stream index."""
        result = torch.full((int(motion_dim),), -1, dtype=torch.long)
        for stream, index in self.layout.feature_indices(int(motion_dim)).items():
            result[index] = self.stream_index(stream)
        if bool((result < 0).any()):
            raise RuntimeError("NEF feature partition does not cover every motion feature")
        return result

    def stream_joints(self, stream: str) -> tuple[int, ...]:
        return self.layout.stream_joints(stream)

    def joint_owner(self) -> dict[str, str]:
        return {
            self.layout.names[joint]: stream
            for stream in NEF_STREAM_NAMES
            for joint in self.layout.stream_joints(stream)
        }

    # -- graph -------------------------------------------------------------
    def relation_types(self) -> tuple[str, ...]:
        """Distinct relation names, in first-appearance order."""
        seen: list[str] = []
        for _, _, relation in self.layout.stream_graph():
            if relation not in seen:
                seen.append(relation)
        return tuple(seen)

    def edge_index(self, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(source, target)`` stream indices for every skeleton relation."""
        edges = self.layout.stream_graph()
        source = torch.tensor(
            [self.stream_index(parent) for parent, _, _ in edges], dtype=torch.long, device=device
        )
        target = torch.tensor(
            [self.stream_index(child) for _, child, _ in edges], dtype=torch.long, device=device
        )
        return source, target

    def edge_type_ids(self, *, device: torch.device | None = None) -> tuple[torch.Tensor, tuple[str, ...]]:
        types = self.relation_types()
        index = {name: position for position, name in enumerate(types)}
        values = torch.tensor(
            [index[relation] for _, _, relation in self.layout.stream_graph()],
            dtype=torch.long,
            device=device,
        )
        return values, types

    def adjacency(
        self, *, include_self: bool = True, device: torch.device | None = None
    ) -> torch.Tensor:
        """``[13, 13]`` float adjacency of the undirected stream graph."""
        source, target = self.edge_index(device=device)
        matrix = torch.zeros((self.num_streams, self.num_streams), dtype=torch.float32, device=device)
        matrix[source, target] = 1.0
        matrix[target, source] = 1.0
        if include_self:
            matrix.fill_diagonal_(1.0)
        return matrix

    # -- regions -----------------------------------------------------------
    def region_streams(self, regions: str | Sequence[str], *, graph_radius: int = 0) -> tuple[str, ...]:
        return self.layout.region_streams(regions, graph_radius=graph_radius)

    def hard_mask(
        self,
        regions: str | Sequence[str],
        *,
        graph_radius: int = 0,
        frame_range: tuple[int, int] | None = None,
        length: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """``[T, 40]`` support mask, delegated to the layout."""
        return self.layout.make_region_mask(
            regions, graph_radius=graph_radius, frame_range=frame_range, length=length, device=device
        )

    def part_names(self) -> tuple[str, ...]:
        return tuple(NEF_EDIT_PARTS)

    def part_streams(self, part: str, *, full_part: bool = False) -> tuple[str, ...]:
        return nef_edit_streams(part, full_part=full_part)

    def as_dict(self) -> dict[str, Any]:
        return {
            "skeleton": self.layout.skeleton,
            "layout_hash": self.layout_hash,
            "num_coordinates": self.num_coordinates,
            "num_streams": self.num_streams,
            "num_levels": self.num_levels,
            "families": list(self.families),
            "relation_types": list(self.relation_types()),
            "regions": list(self.part_names()) + [NEF_WHOLE_BODY_REGION],
        }


def layout_adapter(layout: NEFLayout, **kwargs: Any) -> LayoutAdapter:
    """Convenience constructor mirroring the module-level adapters."""
    return LayoutAdapter(layout, **kwargs)


__all__ = ["LayoutAdapter", "layout_adapter"]
