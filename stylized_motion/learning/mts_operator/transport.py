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

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .contract import TransportOutput
from .embeddings import DEFAULT_TOKEN_EMBED_DIM, StreamLevelHead, StreamTokenEmbedding
from .graph import StreamGraphNetwork
from .layout_adapter import LayoutAdapter
from .temporal import POSITION_ENCODINGS, SinusoidalPositionEncoding

GRAPH_MODES = ("local_relational", "none")
TEMPORAL_MODES = ("bidirectional", "causal")


#: Bumped whenever the parameter shapes change.  A checkpoint whose revision does
#: not match must be retrained, not loaded with ``strict=False``.
ARCHITECTURE_REVISION = 2


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
        content_vocabulary: Mapping[str, Any] | Any | None = None,
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
        content_vocabulary: Any | None = None,
        token_embed_dim: int = DEFAULT_TOKEN_EMBED_DIM,
        position_encoding: str = "sinusoidal",
        architecture_revision: int | None = None,
        feedforward_multiplier: int = 4,
    ) -> None:
        super().__init__()
        # ``config()`` always emits ``architecture_revision`` and the checkpoint
        # loader replays it, so a *stored* config is checked here; a model built
        # from scratch just states the current revision.
        if architecture_revision is not None and int(architecture_revision) != ARCHITECTURE_REVISION:
            # A stored config from another revision describes different parameter
            # shapes; failing here names the reason instead of leaving a
            # load_state_dict shape error (and never loads with strict=False).
            raise ValueError(
                f"Stored transport architecture revision {int(architecture_revision)} != "
                f"{ARCHITECTURE_REVISION} (per-stream embedding and head); retrain instead of "
                "loading the old weights"
            )
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode must be one of {GRAPH_MODES}, got {graph_mode!r}")
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODES}, got {temporal_mode!r}")
        self.adapter = adapter
        self.dim = int(dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.feedforward_multiplier = int(feedforward_multiplier)
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

        self.token_embed_dim = int(token_embed_dim)
        self.embedding = StreamTokenEmbedding(
            adapter, self.dim, token_embed_dim=self.token_embed_dim, dropout=dropout
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.dim * self.feedforward_multiplier,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=self.depth, enable_nested_tensor=False)
        # Where a frame sits in the window; see temporal.py for why both models
        # need it before their attention.
        self.position_encoding = SinusoidalPositionEncoding(self.dim, kind=str(position_encoding))
        self.temporal_norm = nn.LayerNorm(self.dim)
        self.graph = StreamGraphNetwork(
            adapter,
            self.dim,
            depth=0 if self.graph_mode == "none" else int(graph_depth),
            dropout=float(dropout),
        )
        # Per-stream heads: a stream predicts its own coordinates' levels instead of
        # all 40 sharing one linear map over one pooled stream state.
        self.head = StreamLevelHead(adapter, self.dim, self.num_levels)
        # The action vocabulary travels with the model: the operator and every
        # evaluation script inherit it from the frozen transport instead of
        # rebuilding a map from whatever split happens to be loaded.
        from .windows import ContentVocabulary

        vocabulary = ContentVocabulary.from_dict(content_vocabulary) if isinstance(
            content_vocabulary, Mapping
        ) else content_vocabulary
        self.content_vocabulary = vocabulary
        if vocabulary is not None and content_classes is None:
            content_classes = len(vocabulary.classes)
        if vocabulary is not None and content_classes is not None:
            if len(vocabulary.classes) != int(content_classes):
                raise ValueError(
                    f"content_vocabulary has {len(vocabulary.classes)} actions but "
                    f"content_classes={int(content_classes)}"
                )
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
        hidden = self.position_encoding(hidden)
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
        """Per-coordinate logits from the owning stream's per-stream head."""
        return self.head(hidden)

    def config(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "dropout": self.dropout,
            "feedforward_multiplier": self.feedforward_multiplier,
            "graph_mode": self.graph_mode,
            "graph_depth": len(self.graph.blocks),
            "temporal_mode": self.temporal_mode,
            "architecture_revision": ARCHITECTURE_REVISION,
            "token_embed_dim": self.token_embed_dim,
            "position_encoding": self.position_encoding.kind,
            "content_dim": None if self.conditioner is None else self.conditioner.content_dim,
            "content_classes": None
            if self.conditioner is None
            else self.conditioner.content_classes,
            "content_vocabulary": None
            if self.content_vocabulary is None
            else self.content_vocabulary.as_dict(),
        }


    @torch.no_grad()
    def generate(
        self,
        *,
        frames: int,
        batch: int = 1,
        visible_tokens: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
        support: torch.Tensor | None = None,
        steps: int = 8,
        temperature: float = 1.0,
        sampler: Any | None = None,
        valid_mask: torch.Tensor | None = None,
        content_condition: Any | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Iteratively fills a masked support, leaving everything else alone.

        ``support`` is a ``[T, 40]`` boolean mask; only those coordinates are ever
        written, which is the plan's ``locked_edit`` behaviour: tokens outside the
        support keep the values they were given and are never re-sampled.
        ``sampler`` receives the probabilities and returns tokens; the default is
        a temperature-scaled multinomial draw.
        """
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        device = device or next(self.parameters()).device
        frames = int(frames)
        if visible_tokens is None:
            tokens = torch.zeros(
                (batch, frames, self.spec.num_coordinates), dtype=torch.long, device=device
            )
            visible = torch.zeros(
                (batch, frames, self.spec.num_coordinates), dtype=torch.bool, device=device
            )
        else:
            tokens = self.spec.validate_tokens(visible_tokens).to(device)
            if tokens.shape[:2] != (batch, frames):
                raise ValueError(
                    f"visible_tokens must be {(batch, frames, self.spec.num_coordinates)}, "
                    f"got {tuple(tokens.shape)}"
                )
            visible = (
                self.spec.validate_mask(
                    visible_mask, name="visible_mask", batch=batch, frames=frames
                ).to(device)
                if visible_mask is not None
                else torch.ones(
                    (batch, frames, self.spec.num_coordinates), dtype=torch.bool, device=device
                )
            )
        if support is not None:
            support = support.to(device).bool()
            if support.shape != (frames, self.spec.num_coordinates):
                raise ValueError(
                    f"support must be {(frames, self.spec.num_coordinates)}, got {tuple(support.shape)}"
                )
        else:
            support = torch.ones(
                (frames, self.spec.num_coordinates), dtype=torch.bool, device=device
            )
        self.eval()
        for _ in range(int(steps)):
            output = self(
                tokens, visible, content_condition=content_condition, valid_mask=valid_mask
            )
            probs = (output.logits / float(temperature)).softmax(dim=-1)
            if sampler is not None:
                drawn = sampler(probs)
            else:
                # multinomial requires the generator and its input on one device:
                # draw where the generator lives, then bring the tokens back.
                flat = probs.reshape(-1, probs.shape[-1])
                draw = flat.device if generator is None else generator.device
                drawn = (
                    torch.multinomial(flat.to(draw), 1, generator=generator)
                    .reshape(probs.shape[:-1])
                    .to(probs.device)
                )
            tokens = torch.where(support.unsqueeze(0), drawn, tokens)
            visible = visible | support.unsqueeze(0)
        return tokens

__all__ = [
    "GRAPH_MODES",
    "TEMPORAL_MODES",
    "ContentConditioner",
    "MotionTransportTransformer",
]
