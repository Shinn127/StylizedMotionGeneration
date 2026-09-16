"""Reference style descriptor: tokens in, one global embedding out.

Plan §5.3: a reference clip's tokens go through the same 13-stream embedding and
a small temporal/graph encoder, then a masked mean/std pooling produces a single
global descriptor.  Two properties matter for the paper's claims:

* the descriptor is **global** — there is no region-specific style vector, so a
  local edit is the *operator's* support mask acting on a global descriptor,
  never a separate style encoder per body part;
* pooling is mask-aware — padded frames never contribute, and a reference with
  only a few valid frames still yields a finite descriptor.

``StyleIDEncoder`` is the Phase 3 stand-in: the same output width driven by a
learned style id, which lets the sandbox separate "the operator cannot do it"
from "the reference encoder did not extract style".
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .contract import TokenSpec
from .embeddings import StreamTokenEmbedding
from .graph import StreamGraphNetwork
from .layout_adapter import LayoutAdapter
from .transport import TEMPORAL_MODES


class GlobalStyleEncoder(nn.Module):
    """``reference_tokens [B, T, 40] -> style_embedding [B, output_dim]``."""

    def __init__(
        self,
        adapter: LayoutAdapter,
        *,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        graph_depth: int = 1,
        temporal_mode: str = "bidirectional",
        output_dim: int | None = None,
        pooling: str = "mean_std",
    ) -> None:
        super().__init__()
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODES}")
        if pooling not in {"mean", "mean_std"}:
            raise ValueError("pooling must be 'mean' or 'mean_std'")
        if int(dim) <= 0 or int(depth) <= 0 or int(heads) <= 0:
            raise ValueError("dim, depth and heads must be positive")
        if int(dim) % int(heads):
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.adapter = adapter
        self.spec: TokenSpec = adapter.token_spec()
        self.dim = int(dim)
        self.output_dim = int(output_dim or dim)
        self.pooling = str(pooling)
        self.temporal_mode = str(temporal_mode)

        self.embedding = StreamTokenEmbedding(adapter, self.dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(heads),
            dim_feedforward=self.dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(depth), enable_nested_tensor=False)
        self.temporal_norm = nn.LayerNorm(self.dim)
        self.graph = StreamGraphNetwork(adapter, self.dim, depth=int(graph_depth), dropout=float(dropout))
        # Pooling keeps the 13 stream slots (a global descriptor is still built
        # from per-stream statistics), so the pooled width carries the stream count.
        streams = int(self.spec.num_streams)
        self.pool_norm = nn.LayerNorm(streams * self.dim)
        pooled_width = streams * self.dim * (2 if self.pooling == "mean_std" else 1)
        self.projection = nn.Sequential(
            nn.Linear(pooled_width, self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim, self.output_dim),
        )

    def forward(
        self,
        reference_tokens: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.spec.validate_tokens(reference_tokens)
        frames = tokens.shape[1]
        if valid_mask is None:
            valid = torch.ones(tokens.shape[0], frames, dtype=torch.bool, device=tokens.device)
        else:
            valid = self.spec.validate_frame_mask(
                valid_mask.to(tokens.device).bool(), batch=tokens.shape[0], frames=frames
            )
        if bool((~valid).all(dim=1).any()):
            raise ValueError("Every reference needs at least one valid frame")
        visible = valid.unsqueeze(-1).expand(-1, -1, self.spec.num_coordinates)
        hidden = self.embedding(tokens, visible)  # [B, T, 13, D]
        hidden = self._temporal(hidden, valid)
        hidden = self.graph(hidden)
        pooled = self._pool(hidden, valid)
        descriptor = self.projection(pooled)
        if not return_diagnostics:
            return descriptor
        return descriptor, {
            "pooled_norm": pooled.norm(dim=-1),
            "valid_frames": valid.sum(dim=1).to(pooled.dtype),
        }

    def _temporal(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, streams, dim = hidden.shape
        flat = hidden.permute(0, 2, 1, 3).reshape(batch * streams, frames, dim)
        padding = (~valid_mask).unsqueeze(1).expand(batch, streams, frames).reshape(
            batch * streams, frames
        )
        causal = self._causal_mask(frames, hidden) if self.temporal_mode == "causal" else None
        encoded = self.temporal(flat, mask=causal, src_key_padding_mask=padding)
        encoded = self.temporal_norm(encoded)
        return encoded.reshape(batch, streams, frames, dim).permute(0, 2, 1, 3).contiguous()

    def _causal_mask(self, frames: int, reference: torch.Tensor) -> torch.Tensor:
        key = (frames, reference.device)
        cache = getattr(self, "_causal_cache", None)
        if cache is None:
            cache = {}
            self._causal_cache = cache
        mask = cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.ones((frames, frames), dtype=torch.bool, device=reference.device), diagonal=1
            )
            cache[key] = mask
        return mask

    def _pool(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.to(hidden.dtype).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        total = weights.sum(dim=1).clamp_min(1.0)
        mean = (hidden * weights).sum(dim=1) / total  # [B, S, D]
        flattened_mean = self.pool_norm(mean.reshape(hidden.shape[0], -1))
        if self.pooling == "mean":
            return flattened_mean
        squared = (hidden.square() * weights).sum(dim=1) / total
        variance = (squared - mean.square()).clamp_min(0.0)
        deviation = variance.sqrt().reshape(hidden.shape[0], -1)
        return torch.cat((flattened_mean, deviation), dim=-1)

    def config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "depth": len(self.temporal.layers),
            "heads": self.temporal.layers[0].self_attn.num_heads,
            "graph_depth": len(self.graph.blocks),
            "temporal_mode": self.temporal_mode,
            "output_dim": self.output_dim,
            "pooling": self.pooling,
        }


class StyleIDEncoder(nn.Module):
    """Phase 3 stand-in: a learned embedding per style id."""

    def __init__(self, *, num_styles: int, output_dim: int) -> None:
        super().__init__()
        if int(num_styles) <= 0 or int(output_dim) <= 0:
            raise ValueError("num_styles and output_dim must be positive")
        self.num_styles = int(num_styles)
        self.output_dim = int(output_dim)
        self.embedding = nn.Embedding(self.num_styles, self.output_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, style_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if style_ids.dtype not in (torch.long, torch.int64, torch.int32):
            raise ValueError("StyleIDEncoder expects integer style ids")
        ids = style_ids.long().reshape(-1)
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_styles:
            raise ValueError(f"style ids must be in [0, {self.num_styles - 1}]")
        return self.embedding(ids)

    def config(self) -> dict[str, Any]:
        return {"num_styles": self.num_styles, "output_dim": self.output_dim, "kind": "style_id"}


__all__ = ["GlobalStyleEncoder", "StyleIDEncoder"]
