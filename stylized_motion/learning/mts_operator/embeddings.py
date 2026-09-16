"""Token embeddings for the 13 NEF streams.

One shared level embedding plus a per-coordinate identity embedding, pooled into
the stream slots the layout already defines.  Masked positions use a dedicated
learned vector instead of a level embedding, so "not yet known" is never
confused with a legal FSQ level (contract §5.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layout_adapter import LayoutAdapter


class StreamTokenEmbedding(nn.Module):
    """``[B, T, 40]`` token levels -> ``[B, T, 13, D]`` stream states."""

    def __init__(self, adapter: LayoutAdapter, dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.adapter = adapter
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.level_embedding = nn.Embedding(adapter.num_levels, self.dim)
        self.coordinate_embedding = nn.Embedding(adapter.num_coordinates, self.dim)
        self.stream_embedding = nn.Embedding(adapter.num_streams, self.dim)
        self.mask_embedding = nn.Parameter(torch.zeros(self.dim))
        self.dropout = nn.Dropout(float(dropout))

        stream_ids = adapter.coordinate_stream_ids()
        self.register_buffer("coordinate_stream_ids", stream_ids, persistent=False)
        self.register_buffer(
            "stream_coordinate_counts",
            self._counts(stream_ids).to(torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    @staticmethod
    def _counts(stream_ids: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(stream_ids, minlength=int(stream_ids.max()) + 1)
        return counts.clamp_min(1)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embedding.weight, std=0.02)
        nn.init.normal_(self.coordinate_embedding.weight, std=0.02)
        nn.init.normal_(self.stream_embedding.weight, std=0.02)
        nn.init.zeros_(self.mask_embedding)

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
        levels = self.level_embedding(tokens)  # [B, T, 40, D]
        identity = self.coordinate_embedding.weight.view(1, 1, self.adapter.num_coordinates, self.dim)
        embedded = levels + identity
        mask_vector = self.mask_embedding.view(1, 1, 1, self.dim)
        embedded = torch.where(visible.unsqueeze(-1), embedded, mask_vector.expand_as(embedded))

        batch, frames, coordinates, dim = embedded.shape
        if coordinates != self.adapter.num_coordinates:
            raise RuntimeError("Embedding lost the coordinate axis")
        pooled = torch.zeros(
            batch, frames, self.adapter.num_streams, dim, dtype=embedded.dtype, device=embedded.device
        )
        pooled.index_add_(2, self.coordinate_stream_ids, embedded)
        pooled = pooled / self.stream_coordinate_counts.view(1, 1, self.adapter.num_streams, 1)
        pooled = pooled + self.stream_embedding.weight.view(1, 1, self.adapter.num_streams, self.dim)
        return self.dropout(pooled)


__all__ = ["StreamTokenEmbedding"]
