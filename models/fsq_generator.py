from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


LayerKVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class FSQGeneratorCache:
    layers: list[LayerKVCache]
    next_position: int

    @property
    def length(self) -> int:
        if not self.layers:
            return 0
        return int(self.layers[0][0].shape[-2])


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(values.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (values.float() * scale).to(values.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        if theta <= 0.0:
            raise ValueError(f"RoPE theta must be positive, got {theta}")
        frequencies = 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    @staticmethod
    def _rotate_half(values: torch.Tensor) -> torch.Tensor:
        first, second = values.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(positions.float(), self.inverse_frequencies.float())
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos().to(dtype=query.dtype)[None, None]
        sine = angles.sin().to(dtype=query.dtype)[None, None]
        return (
            query * cosine + self._rotate_half(query) * sine,
            key * cosine + self._rotate_half(key) * sine,
        )


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output_projection(F.silu(self.gate_projection(values)) * self.value_projection(values))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float,
        rope_theta: float,
        qk_norm: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if dim % num_query_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_query_heads={num_query_heads}")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads={num_query_heads} must be divisible by num_kv_heads={num_kv_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.dim = int(dim)
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.dim // self.num_query_heads
        self.dropout = float(dropout)
        self.query_projection = nn.Linear(dim, self.num_query_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.query_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)

    def _project(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = values.shape
        query = self.query_projection(values).view(
            batch_size, seq_len, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key_projection(values).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value_projection(values).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        return query, key, value

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        dropout = self.dropout if self.training else 0.0
        if self.num_query_heads == self.num_kv_heads:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=causal,
            )

        # PyTorch's native GQA is currently backend-dependent. MPS uses the same
        # grouped heads with an explicit expansion while CPU/CUDA use native GQA.
        if query.device.type == "mps":
            groups = self.num_query_heads // self.num_kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=causal,
            )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            is_causal=causal,
            enable_gqa=True,
        )

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        cache: LayerKVCache | None,
        use_cache: bool,
        max_cache_length: int,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        query, key, value = self._project(values)
        query = self.query_norm(query)
        key = self.key_norm(key)
        query, key = self.rope(query, key, positions)

        if cache is None:
            attention_key = key
            attention_value = value
            causal = True
        else:
            if values.shape[1] != 1:
                raise ValueError("Cached decoding accepts exactly one frame per call")
            previous_key, previous_value = cache
            attention_key = torch.cat((previous_key, key), dim=-2)
            attention_value = torch.cat((previous_value, value), dim=-2)
            if attention_key.shape[-2] > max_cache_length:
                attention_key = attention_key[..., -max_cache_length:, :]
                attention_value = attention_value[..., -max_cache_length:, :]
            # With a single new query, every cached key is in its past.
            causal = False

        attended = self._attention(query, attention_key, attention_value, causal=causal)
        attended = attended.transpose(1, 2).contiguous().view(values.shape[0], values.shape[1], self.dim)
        output = self.output_projection(attended)
        next_cache = (attention_key, attention_value) if use_cache else None
        return output, next_cache


class FSQTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        dropout: float,
        rope_theta: float,
        qk_norm: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.attention = GroupedQueryAttention(
            dim=dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
        )
        self.feed_forward_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = SwiGLU(dim, ff_dim)
        self.dropout = float(dropout)

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
        cache: LayerKVCache | None,
        use_cache: bool,
        max_cache_length: int,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        attention, next_cache = self.attention(
            self.attention_norm(values),
            positions=positions,
            cache=cache,
            use_cache=use_cache,
            max_cache_length=max_cache_length,
        )
        values = values + F.dropout(attention, p=self.dropout, training=self.training)
        feed_forward = self.feed_forward(self.feed_forward_norm(values))
        values = values + F.dropout(feed_forward, p=self.dropout, training=self.training)
        return values, next_cache


class FSQCausalTransformerGenerator(nn.Module):
    def __init__(
        self,
        num_coordinates: int,
        num_levels: int,
        coordinate_embedding_dim: int = 16,
        dim: int = 256,
        num_layers: int = 6,
        num_query_heads: int = 8,
        num_kv_heads: int = 4,
        ff_dim: int = 768,
        dropout: float = 0.1,
        context_frames: int = 64,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_coordinates <= 0 or num_levels <= 1:
            raise ValueError("num_coordinates must be positive and num_levels must be greater than one")
        if coordinate_embedding_dim <= 0 or dim <= 0 or num_layers <= 0 or ff_dim <= 0:
            raise ValueError("Embedding, model, layer, and feed-forward dimensions must be positive")
        if context_frames <= 0:
            raise ValueError(f"context_frames must be positive, got {context_frames}")

        self.config = {key: value for key, value in locals().items() if key not in {"self", "__class__"}}
        self.num_coordinates = int(num_coordinates)
        self.num_levels = int(num_levels)
        self.coordinate_embedding_dim = int(coordinate_embedding_dim)
        self.dim = int(dim)
        self.context_frames = int(context_frames)

        self.level_embedding = nn.Embedding(
            self.num_coordinates * self.num_levels,
            self.coordinate_embedding_dim,
        )
        coordinate_offsets = torch.arange(self.num_coordinates) * self.num_levels
        self.register_buffer("coordinate_offsets", coordinate_offsets, persistent=False)
        self.frame_projection = nn.Linear(
            self.num_coordinates * self.coordinate_embedding_dim,
            self.dim,
            bias=False,
        )
        self.frame_norm = RMSNorm(self.dim, norm_eps)
        self.blocks = nn.ModuleList(
            [
                FSQTransformerBlock(
                    dim=self.dim,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    qk_norm=qk_norm,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = RMSNorm(self.dim, norm_eps)
        self.level_head = nn.Linear(self.dim, self.num_coordinates * self.num_levels, bias=False)
        self._reset_parameters(num_layers)

    def _reset_parameters(self, num_layers: int) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2.0 * num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output_projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.output_projection.weight, mean=0.0, std=residual_std)

    def _embed_frames(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3 or indices.shape[-1] != self.num_coordinates:
            raise ValueError(
                f"Expected indices [B,T,{self.num_coordinates}], got {tuple(indices.shape)}"
            )
        if indices.shape[1] <= 0:
            raise ValueError("At least one token frame is required")
        embedding_indices = indices.long() + self.coordinate_offsets.view(1, 1, -1)
        embedded = self.level_embedding(embedding_indices)
        embedded = embedded.reshape(indices.shape[0], indices.shape[1], -1)
        return self.frame_norm(self.frame_projection(embedded))

    def forward(
        self,
        indices: torch.Tensor,
        cache: FSQGeneratorCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> dict[str, torch.Tensor | FSQGeneratorCache | None]:
        if cache is not None and not use_cache:
            raise ValueError("cache requires use_cache=True")
        if cache is None and indices.shape[1] > self.context_frames:
            raise ValueError(
                f"Sequence length {indices.shape[1]} exceeds context_frames={self.context_frames}"
            )
        if cache is not None and len(cache.layers) != len(self.blocks):
            raise ValueError(
                f"Cache has {len(cache.layers)} layers, expected {len(self.blocks)}"
            )

        hidden = self._embed_frames(indices)
        start_position = cache.next_position if cache is not None else int(position_offset)
        positions = torch.arange(
            start_position,
            start_position + indices.shape[1],
            device=indices.device,
            dtype=torch.long,
        )
        next_layers: list[LayerKVCache] = []
        for layer_index, block in enumerate(self.blocks):
            layer_cache = cache.layers[layer_index] if cache is not None else None
            hidden, next_cache = block(
                hidden,
                positions=positions,
                cache=layer_cache,
                use_cache=use_cache,
                max_cache_length=self.context_frames,
            )
            if next_cache is not None:
                next_layers.append(next_cache)

        hidden = self.output_norm(hidden)
        logits = self.level_head(hidden).reshape(
            indices.shape[0],
            indices.shape[1],
            self.num_coordinates,
            self.num_levels,
        )
        next_cache = None
        if use_cache:
            next_cache = FSQGeneratorCache(
                layers=next_layers,
                next_position=start_position + indices.shape[1],
            )
        return {"hidden": hidden, "logits": logits, "cache": next_cache}

    def prefill(self, seed_indices: torch.Tensor) -> tuple[torch.Tensor, FSQGeneratorCache]:
        output = self(seed_indices, use_cache=True)
        cache = output["cache"]
        if not isinstance(cache, FSQGeneratorCache):
            raise RuntimeError("Prefill did not produce a KV cache")
        return output["logits"][:, -1], cache

    def decode_step(
        self,
        current_indices: torch.Tensor,
        cache: FSQGeneratorCache,
    ) -> tuple[torch.Tensor, FSQGeneratorCache]:
        if current_indices.ndim == 2:
            current_indices = current_indices[:, None]
        if current_indices.ndim != 3 or current_indices.shape[1] != 1:
            raise ValueError(
                f"decode_step expects [B,K] or [B,1,K], got {tuple(current_indices.shape)}"
            )
        output = self(current_indices, cache=cache, use_cache=True)
        next_cache = output["cache"]
        if not isinstance(next_cache, FSQGeneratorCache):
            raise RuntimeError("decode_step did not produce a KV cache")
        return output["logits"][:, -1], next_cache

    def sample_next(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if logits.shape[-2:] != (self.num_coordinates, self.num_levels):
            raise ValueError(
                f"Expected logits [...,{self.num_coordinates},{self.num_levels}], got {tuple(logits.shape)}"
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if greedy:
            return logits.argmax(dim=-1)
        probabilities = F.softmax(logits / float(temperature), dim=-1)
        flat = probabilities.reshape(-1, self.num_levels)
        sampled = torch.multinomial(flat, num_samples=1, generator=generator)
        return sampled.reshape(*probabilities.shape[:-1])

    def generate(
        self,
        seed_indices: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        next_logits, cache = self.prefill(seed_indices)
        generated = []
        for step in range(num_steps):
            current = self.sample_next(
                next_logits,
                temperature=temperature,
                greedy=greedy,
                generator=generator,
            )
            generated.append(current)
            if step + 1 < num_steps:
                next_logits, cache = self.decode_step(current, cache)
        return torch.stack(generated, dim=1)
