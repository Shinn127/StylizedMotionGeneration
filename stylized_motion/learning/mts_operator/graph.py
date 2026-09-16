"""Relation-indexed message passing over the 13 NEF streams.

The tokenizer keeps streams independent, so the coordination the tokenizer does
not model has to live here: a graph block moves information along the skeleton
relations (``node_to_edge`` / ``edge_to_child`` ...) that
:meth:`LayoutAdapter.edge_index` exposes.  A stream's updated state is a
function of its neighbours only, which keeps the operator's support argument
honest: widening the graph is an explicit, measurable choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layout_adapter import LayoutAdapter


class StreamGraphBlock(nn.Module):
    """One relation-aware residual update of the stream states."""

    def __init__(
        self,
        adapter: LayoutAdapter,
        dim: int,
        *,
        dropout: float = 0.0,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if aggregation not in {"mean", "sum"}:
            raise ValueError(f"Unsupported aggregation {aggregation!r}")
        self.adapter = adapter
        self.dim = int(dim)
        self.aggregation = aggregation
        source, target, type_ids, direction, types = adapter.message_edges()
        self.relation_types = types
        self.register_buffer("edge_source", source, persistent=False)
        self.register_buffer("edge_target", target, persistent=False)
        self.register_buffer("edge_type", type_ids, persistent=False)
        self.register_buffer("edge_direction", direction, persistent=False)
        self.num_relations = max(1, len(types))
        self.relation_embedding = nn.Embedding(self.num_relations, self.dim)
        self.direction_embedding = nn.Embedding(2, self.dim)
        self.relation_message = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.num_relations)
        )
        self.norm = nn.LayerNorm(self.dim)
        self.update = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.relation_embedding.weight, std=0.02)
        nn.init.normal_(self.direction_embedding.weight, std=0.02)
        # Small but non-zero: the block must be a real (if gentle) operator at
        # step 0, otherwise a graph_depth ablation would silently measure the
        # identity function.
        nn.init.normal_(self.update[-1].weight, std=0.02)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, T, 13, D]`` -> ``[B, T, 13, D]``."""
        if hidden.ndim != 4 or hidden.shape[2] != self.adapter.num_streams:
            raise ValueError(
                f"stream hidden states must be [B, T, {self.adapter.num_streams}, D], "
                f"got {tuple(hidden.shape)}"
            )
        batch, frames, streams, dim = hidden.shape
        source = self.edge_source
        target = self.edge_target
        messages = hidden[:, :, source, :]  # [B, T, E, D]
        mixed = torch.zeros_like(messages)
        for relation in range(self.num_relations):
            selected = self.edge_type == relation
            if not bool(selected.any()):
                continue
            relation_bias = self.relation_embedding.weight[relation].view(1, 1, 1, dim)
            projected = self.relation_message[relation](messages[:, :, selected, :]) + relation_bias
            mixed[:, :, selected, :] = projected + self.direction_embedding.weight[
                self.edge_direction[selected]
            ].view(1, 1, -1, dim)
        aggregated = torch.zeros(
            batch, frames, streams, dim, dtype=hidden.dtype, device=hidden.device
        )
        aggregated.index_add_(2, target, mixed)
        if self.aggregation == "mean":
            counts = torch.bincount(target, minlength=streams).clamp_min(1).to(hidden.dtype)
            aggregated = aggregated / counts.view(1, 1, streams, 1)
        updated = aggregated + hidden
        return hidden + self.dropout(self.update(self.norm(updated)))


class StreamGraphNetwork(nn.Module):
    """``depth`` stacked graph blocks (no-op when ``depth`` is zero)."""

    def __init__(
        self,
        adapter: LayoutAdapter,
        dim: int,
        *,
        depth: int,
        dropout: float = 0.0,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if int(depth) < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        self.blocks = nn.ModuleList(
            StreamGraphBlock(adapter, dim, dropout=dropout, aggregation=aggregation)
            for _ in range(int(depth))
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


__all__ = ["StreamGraphBlock", "StreamGraphNetwork"]
