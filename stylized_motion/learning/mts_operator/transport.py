"""Style-free base transport: masked content generation over NEF stream tokens.

This is ``p0(Z | C, H)`` of the plan: a generator that never sees style.  It
exists so the style operator has a fixed base distribution to modify, and so
"the operator lost content" can be separated from "the base generator cannot
generate".

Architecture (plan §4.4)::

    13 stream token embeddings            embeddings.StreamTokenEmbedding
            |
    temporal attention per stream         this module (streams folded to batch)
            |
    local graph message passing           graph.StreamGraphNetwork
            |
    40 x 9 per-coordinate output head     this module (no cross-stream mixing)

The head reads each coordinate's logits from its *own* stream's hidden state, so
a token's prediction can only depend on streams the graph actually connected.
No style input, no cross-stream tokenizer fusion: coordination lives in the
graph, which is an explicit, ablatable choice.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .contract import TokenSpec, TransportOutput
from .embeddings import StreamTokenEmbedding
from .graph import StreamGraphNetwork
from .layout_adapter import LayoutAdapter

GRAPH_MODES = ("local_relational", "none")
TEMPORAL_MODES = ("bidirectional", "causal")


class ContentConditioner(nn.Module):
    """Turns an action/trajectory condition into a per-frame additive bias.

    Accepts ``None`` (unconditional), integer class ids ``[B]`` or ``[B, T]``,
    a clip-level feature ``[B, C]`` or a per-frame feature ``[B, T, C]``.  The
    first version deliberately supports these shapes only: a text encoder is not
    part of the operator's contribution and must not leak into it.
    """

    def __init__(
        self,
        dim: int,
        *,
        content_dim: int | None = None,
        content_classes: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.content_dim = None if content_dim is None else int(content_dim)
        self.content_classes = None if content_classes is None else int(content_classes)
        if self.content_dim is not None and self.content_dim <= 0:
            raise ValueError("content_dim must be positive")
        if self.content_classes is not None and self.content_classes <= 0:
            raise ValueError("content_classes must be positive")
        self.embedding = (
            nn.Embedding(self.content_classes, self.dim) if self.content_classes else None
        )
        self.projection = (
            nn.Sequential(
                nn.Linear(self.content_dim, self.dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.dim, self.dim),
            )
            if self.content_dim
            else None
        )
        if self.embedding is None and self.projection is None:
            raise ValueError(
                "ContentConditioner needs content_dim (features) or content_classes (ids)"
            )
        if self.embedding is not None:
            nn.init.normal_(self.embedding.weight, std=0.02)
        if self.projection is not None:
            nn.init.zeros_(self.projection[-1].weight)
            nn.init.zeros_(self.projection[-1].bias)

    @property
    def enabled(self) -> bool:
        return self.embedding is not None or self.projection is not None

    def forward(self, condition: Any) -> torch.Tensor | None:
        if condition is None:
            return None
        if not isinstance(condition, torch.Tensor):
            raise TypeError("content_condition must be a tensor or None")
        if condition.dtype in (torch.long, torch.int64, torch.int32):
            if self.embedding is None:
                raise ValueError("Integer content ids require content_classes")
            if condition.ndim not in (1, 2):
                raise ValueError("Integer content ids must be [B] or [B, T]")
            values = condition.to(self.embedding.weight.device).long()
            if int(values.min()) < 0 or int(values.max()) >= self.embedding.num_embeddings:
                raise ValueError(
                    f"content ids must be in [0, {self.embedding.num_embeddings - 1}]"
                )
            return self.embedding(values)
        if self.projection is None:
            raise ValueError("Continuous content features require content_dim")
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if condition.ndim != 3 or condition.shape[-1] != self.content_dim:
            raise ValueError(
                f"content features must be [B, C] or [B, T, {self.content_dim}], "
                f"got {tuple(condition.shape)}"
            )
        reference = self.projection[0].weight
        return self.projection(condition.to(device=reference.device, dtype=reference.dtype))


class MotionTransportTransformer(nn.Module):
    """Style-free masked token generator over the 13 NEF streams."""

    def __init__(
        self,
        adapter: LayoutAdapter,
        *,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        graph_mode: str = "local_relational",
        graph_depth: int = 2,
        temporal_mode: str = "bidirectional",
        content_dim: int | None = None,
        content_classes: int | None = None,
        feedforward_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode must be one of {GRAPH_MODES}, got {graph_mode!r}")
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODES}, got {temporal_mode!r}")
        self.adapter = adapter
        self.dim = int(dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.graph_mode = str(graph_mode)
        self.temporal_mode = str(temporal_mode)
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("dim, depth and heads must be positive")
        if self.dim % self.heads:
            raise ValueError(f"dim {self.dim} must be divisible by heads {self.heads}")
        if int(graph_depth) < 0:
            raise ValueError("graph_depth must be non-negative")
        self.spec = adapter.token_spec()
        self.num_levels = self.spec.num_levels

        self.embedding = StreamTokenEmbedding(adapter, self.dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.dim * int(feedforward_multiplier),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=self.depth, enable_nested_tensor=False)
        self.temporal_norm = nn.LayerNorm(self.dim)
        self.graph = StreamGraphNetwork(
            adapter,
            self.dim,
            depth=0 if self.graph_mode == "none" else int(graph_depth),
            dropout=float(dropout),
        )
        self.output_head = nn.Linear(self.dim, self.num_levels)
        self.coordinate_bias = nn.Embedding(self.spec.num_coordinates, self.num_levels)
        self.conditioner = (
            ContentConditioner(
                self.dim,
                content_dim=content_dim,
                content_classes=content_classes,
                dropout=dropout,
            )
            if (content_dim is not None or content_classes is not None)
            else None
        )
        self.register_buffer(
            "coordinate_stream_ids", adapter.coordinate_stream_ids(), persistent=False
        )
        nn.init.zeros_(self.coordinate_bias.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        visible_mask: torch.Tensor,
        *,
        content_condition: Any | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> TransportOutput:
        tokens = self.spec.validate_tokens(tokens)
        visible_mask = self.spec.validate_mask(
            visible_mask.to(tokens.device).bool(),
            name="visible_mask",
            batch=tokens.shape[0],
            frames=tokens.shape[1],
        )
        frames = tokens.shape[1]
        valid = None
        if valid_mask is not None:
            valid = self.spec.validate_frame_mask(
                valid_mask.to(tokens.device).bool(), batch=tokens.shape[0], frames=frames
            )

        hidden = self.embedding(tokens, visible_mask)  # [B, T, 13, D]
        if self.conditioner is not None:
            bias = self.conditioner(content_condition)
            if bias is not None:
                if bias.ndim == 2:
                    bias = bias.unsqueeze(1)
                if bias.shape[1] == 1 and frames > 1:
                    bias = bias.expand(-1, frames, -1)
                if bias.shape[1] != frames:
                    raise ValueError(
                        f"content condition has {bias.shape[1]} frames, expected {frames}"
                    )
                hidden = hidden + bias.unsqueeze(2).to(hidden.dtype)

        hidden = self._temporal(hidden, valid)
        hidden = self.graph(hidden)
        logits = self._head(hidden)
        return TransportOutput(
            logits=logits,
            stream_hidden=hidden,
            valid_mask=valid,
            spec=self.spec,
        )

    def _temporal(self, hidden: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        batch, frames, streams, dim = hidden.shape
        flat = hidden.permute(0, 2, 1, 3).reshape(batch * streams, frames, dim)
        padding = None
        if valid_mask is not None:
            padding = ~valid_mask.to(hidden.device)
            padding = padding.unsqueeze(1).expand(batch, streams, frames).reshape(batch * streams, frames)
            if bool(padding.all(dim=1).any()):
                raise ValueError("Every sample needs at least one valid frame for temporal attention")
        causal = self._causal_mask(frames, flat) if self.temporal_mode == "causal" else None
        encoded = self.temporal(flat, mask=causal, src_key_padding_mask=padding)
        encoded = self.temporal_norm(encoded)
        return encoded.reshape(batch, streams, frames, dim).permute(0, 2, 1, 3).contiguous()

    def _causal_mask(self, frames: int, reference: torch.Tensor) -> torch.Tensor:
        """Boolean upper-triangular mask (True = may not attend)."""
        key = (frames, reference.device)
        cached = getattr(self, "_causal_cache", None)
        if cached is None:
            cached = {}
            self._causal_cache = cached
        mask = cached.get(key)
        if mask is None:
            mask = torch.triu(
                torch.ones((frames, frames), dtype=torch.bool, device=reference.device),
                diagonal=1,
            )
            cached[key] = mask
        return mask

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Per-coordinate logits from the owning stream's hidden state."""
        per_stream = self.output_head(hidden)  # [B, T, 13, levels]
        batch, frames, streams, levels = per_stream.shape
        index = self.coordinate_stream_ids.view(1, 1, self.spec.num_coordinates, 1).expand(
            batch, frames, self.spec.num_coordinates, levels
        )
        logits = per_stream.gather(2, index)
        return logits + self.coordinate_bias.weight.view(1, 1, self.spec.num_coordinates, levels)

    def config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "graph_mode": self.graph_mode,
            "graph_depth": len(self.graph.blocks),
            "temporal_mode": self.temporal_mode,
            "content_dim": None if self.conditioner is None else self.conditioner.content_dim,
            "content_classes": None
            if self.conditioner is None
            else self.conditioner.content_classes,
        }


__all__ = [
    "GRAPH_MODES",
    "TEMPORAL_MODES",
    "ContentConditioner",
    "MotionTransportTransformer",
]
