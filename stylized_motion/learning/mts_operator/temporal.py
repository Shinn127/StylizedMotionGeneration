"""Where a frame is in the window: one sinusoidal position table, two models.

The transport and the reference style encoder both attend over a window, and
neither had any notion of *when* a frame was: with every token masked, the
transport's per-frame outputs were structurally identical (the audit's
``all_masked_logits_frame_max_diff = 0.0``), and the reference descriptor was
almost invariant to reordering its frames.  Both models add the same fixed table
before their temporal attention.

The table depends only on the window position, ``[1, T, 1, D]``, so it can be
built for any length (no 64-frame assumption) and it never reads token values or
the batch order.  Padding keeps being handled by the key-padding mask and by
pooling over the valid frames, so absolute positions never let padding into a
descriptor statistic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

POSITION_ENCODINGS = ("sinusoidal",)


def sinusoidal_positions(
    length: int,
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[1, length, 1, dim]`` fixed sinusoidal table.

    Computed in float64 and cast, so the values stay stable for long windows, and
    built for exactly the requested length: there is no maximum to overflow.
    """
    length, dim = int(length), int(dim)
    if length <= 0 or dim <= 0:
        raise ValueError(f"length and dim must be positive, got {length} and {dim}")
    position = torch.arange(length, device=device, dtype=torch.float64).unsqueeze(1)
    half = (dim + 1) // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float64)
        / max(half, 1)
    )
    phase = position * frequency.unsqueeze(0)
    table = torch.empty(length, dim, device=device, dtype=torch.float64)
    table[:, 0::2] = torch.sin(phase[:, 0 : table[:, 0::2].shape[1]])
    table[:, 1::2] = torch.cos(phase[:, 0 : table[:, 1::2].shape[1]])
    return table.to(dtype).view(1, length, 1, dim)


class SinusoidalPositionEncoding(nn.Module):
    """Adds the fixed position table to ``[B, T, S, D]`` hidden states."""

    name = "sinusoidal"

    def __init__(self, dim: int, *, kind: str = "sinusoidal") -> None:
        super().__init__()
        if kind not in POSITION_ENCODINGS:
            raise ValueError(
                f"Unknown position encoding {kind!r}; expected {list(POSITION_ENCODINGS)}"
            )
        if int(dim) <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.kind = str(kind)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[-1] != self.dim:
            raise ValueError(
                f"hidden must be [B, T, S, {self.dim}], got {tuple(hidden.shape)}"
            )
        table = sinusoidal_positions(
            hidden.shape[1], self.dim, device=hidden.device, dtype=hidden.dtype
        )
        return hidden + table

    def config(self) -> dict[str, object]:
        return {"kind": self.kind, "dim": self.dim}


__all__ = ["POSITION_ENCODINGS", "SinusoidalPositionEncoding", "sinusoidal_positions"]
