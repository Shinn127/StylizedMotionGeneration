"""Token embeddings for the 13 NEF streams.

One shared level embedding plus a per-coordinate identity embedding, then a
**per-stream** projection: each stream flattens its coordinates in canonical
layout order into ``K_stream * token_embed_dim`` and learns its own linear map to
the hidden width.  Masked positions use a dedicated learned vector in place of the
level term (the coordinate identity is kept), so "not yet known" is never
confused with a legal FSQ level (contract §5.1).

The per-stream projection is what makes two coordinates of one stream
distinguishable.  The old mean-pooled version summed them, and since a swap of two
coordinates' levels cancels in a sum, every coordinate of a stream reached the
network as the *same* vector — the audit's 1.9e-09 "embedding difference" for a
within-stream swap.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layout_adapter import LayoutAdapter


DEFAULT_TOKEN_EMBED_DIM = 16


class StreamTokenEmbedding(nn.Module):
    """``[B, T, 40]`` token levels -> ``[B, T, 13, D]`` stream states."""

    def __init__(
        self,
        adapter: LayoutAdapter,
        dim: int,
        *,
        token_embed_dim: int = DEFAULT_TOKEN_EMBED_DIM,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.dim = int(dim)
        self.token_embed_dim = int(token_embed_dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.token_embed_dim <= 0:
            raise ValueError(f"token_embed_dim must be positive, got {token_embed_dim}")
        width = self.token_embed_dim
        self.level_embedding = nn.Embedding(adapter.num_levels, width)
        self.coordinate_embedding = nn.Embedding(adapter.num_coordinates, width)
        self.stream_embedding = nn.Embedding(adapter.num_streams, self.dim)
        self.mask_embedding = nn.Parameter(torch.zeros(width))
        self.dropout = nn.Dropout(float(dropout))

        stream_ids = adapter.coordinate_stream_ids()
        self.register_buffer("coordinate_stream_ids", stream_ids, persistent=False)
        coordinate_indices = adapter.stream_coordinate_indices()
        for position, indices in enumerate(coordinate_indices):
            self.register_buffer(f"stream_coordinates_{position}", indices, persistent=False)
        self.stream_sizes = tuple(int(indices.numel()) for indices in coordinate_indices)
        # One projection per stream: no family tying, so a left/right mix-up cannot
        # hide behind shared weights, and the local order inside a stream is fixed
        # by the layout rather than by an arithmetic mean.
        self.stream_projection = nn.ModuleList(
            nn.Linear(size * width, self.dim) for size in self.stream_sizes
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embedding.weight, std=0.02)
        nn.init.normal_(self.coordinate_embedding.weight, std=0.02)
        nn.init.normal_(self.stream_embedding.weight, std=0.02)
        nn.init.zeros_(self.mask_embedding)
        for projection in self.stream_projection:
            nn.init.normal_(projection.weight, std=0.02)
            nn.init.zeros_(projection.bias)

    def coordinate_indices(self, stream: int) -> torch.Tensor:
        return getattr(self, f"stream_coordinates_{int(stream)}")

    def forward(self, tokens: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
        """Embeds visible tokens; invisible positions get the mask vector."""
        spec = self.adapter.token_spec()
        tokens = spec.validate_tokens(tokens).to(self.coordinate_stream_ids.device)
        visible = spec.validate_mask(
            visible_mask,
            name="visible_mask",
            batch=tokens.shape[0],
            frames=tokens.shape[1],
        ).to(self.coordinate_stream_ids.device)
        if tokens.shape[1] != visible.shape[1]:
            raise ValueError("tokens and visible_mask must share the frame axis")
        width = self.token_embed_dim
        levels = self.level_embedding(tokens)  # [B, T, 40, E]
        mask_vector = self.mask_embedding.view(1, 1, 1, width)
        # Only the level term is replaced: "which coordinate is this, and is it
        # known" must survive masking, or a hidden token becomes unidentifiable.
        levels = torch.where(visible.unsqueeze(-1), levels, mask_vector.expand_as(levels))
        identity = self.coordinate_embedding.weight.view(
            1, 1, self.adapter.num_coordinates, width
        )
        embedded = levels + identity

        batch, frames, coordinates, _ = embedded.shape
        if coordinates != self.adapter.num_coordinates:
            raise RuntimeError("Embedding lost the coordinate axis")
        streams = []
        for position, projection in enumerate(self.stream_projection):
            indices = self.coordinate_indices(position)
            chunk = embedded.index_select(2, indices)  # [B, T, K_s, E]
            streams.append(projection(chunk.reshape(batch, frames, -1)))
        stacked = torch.stack(streams, dim=2)  # [B, T, 13, D]
        stacked = stacked + self.stream_embedding.weight.view(1, 1, self.adapter.num_streams, self.dim)
        return self.dropout(stacked)


class StreamLevelHead(nn.Module):
    """Per-stream ``D -> K_stream * levels`` heads, scattered back to coordinates.

    The inverse of the per-stream embedding: each stream predicts its own
    coordinates' levels, so two coordinates of one stream are no longer forced to
    differ only by a static bias.
    """

    def __init__(self, adapter: LayoutAdapter, dim: int, num_levels: int) -> None:
        super().__init__()
        self.adapter = adapter
        self.dim = int(dim)
        self.num_levels = int(num_levels)
        coordinate_indices = adapter.stream_coordinate_indices()
        for position, indices in enumerate(coordinate_indices):
            self.register_buffer(f"stream_coordinates_{position}", indices, persistent=False)
        self.stream_sizes = tuple(int(indices.numel()) for indices in coordinate_indices)
        self.stream_heads = nn.ModuleList(
            nn.Linear(self.dim, size * self.num_levels) for size in self.stream_sizes
        )
        self.coordinate_bias = nn.Embedding(adapter.num_coordinates, self.num_levels)
        nn.init.zeros_(self.coordinate_bias.weight)

    def coordinate_indices(self, stream: int) -> torch.Tensor:
        return getattr(self, f"stream_coordinates_{int(stream)}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, T, 13, D]`` -> ``[B, T, 40, levels]``."""
        batch, frames, streams, _ = hidden.shape
        if streams != self.adapter.num_streams:
            raise ValueError(f"hidden has {streams} streams, expected {self.adapter.num_streams}")
        logits = hidden.new_empty(batch, frames, self.adapter.num_coordinates, self.num_levels)
        for position, head in enumerate(self.stream_heads):
            indices = self.coordinate_indices(position)
            per_stream = head(hidden[:, :, position]).reshape(
                batch, frames, self.stream_sizes[position], self.num_levels
            )
            logits = logits.index_copy(2, indices, per_stream)
        return logits + self.coordinate_bias.weight.view(
            1, 1, self.adapter.num_coordinates, self.num_levels
        )


__all__ = ["DEFAULT_TOKEN_EMBED_DIM", "StreamLevelHead", "StreamTokenEmbedding"]
