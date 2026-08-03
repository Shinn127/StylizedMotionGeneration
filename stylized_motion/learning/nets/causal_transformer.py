import torch
import torch.nn as nn


def build_banded_causal_mask(seq_len: int, context_len: int, device) -> torch.Tensor:
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    indices = torch.arange(seq_len, device=device)
    query = indices[:, None]
    key = indices[None, :]
    blocked = (key > query) | ((query - key) >= context_len)
    mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    return mask.masked_fill(blocked, float("-inf"))


class LearnedPositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = int(max_seq_len)
        self.embedding = nn.Parameter(torch.zeros(self.max_seq_len, dim))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        return x + self.embedding[:seq_len].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class BandedCausalTransformerStack(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        context_len: int,
        max_seq_len: int,
        pos_encoding: str = "learned",
        causal: bool = True,
    ) -> None:
        super().__init__()
        if pos_encoding != "learned":
            raise ValueError(f"Unsupported pos_encoding: {pos_encoding}")
        self.causal = bool(causal)
        self.context_len = int(context_len)
        self.pos_encoding = LearnedPositionEncoding(max_seq_len=max_seq_len, dim=dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(x)
        mask = None
        if self.causal:
            mask = build_banded_causal_mask(
                seq_len=x.shape[1],
                context_len=self.context_len,
                device=x.device,
            )
        x = self.encoder(x, mask=mask)
        return self.out_norm(x)


class CausalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        context_len: int,
        max_seq_len: int,
        pos_encoding: str = "learned",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, code_dim)
        self.transformer = BandedCausalTransformerStack(
            dim=code_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            context_len=context_len,
            max_seq_len=max_seq_len,
            pos_encoding=pos_encoding,
            causal=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        x = self.transformer(x)
        return x.transpose(1, 2).contiguous()


class CausalTransformerDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        code_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        context_len: int,
        max_seq_len: int,
        pos_encoding: str = "learned",
    ) -> None:
        super().__init__()
        self.transformer = BandedCausalTransformerStack(
            dim=code_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            context_len=context_len,
            max_seq_len=max_seq_len,
            pos_encoding=pos_encoding,
            causal=True,
        )
        self.output_proj = nn.Linear(code_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.transpose(1, 2).contiguous()
        z = self.transformer(z)
        z = self.output_proj(z)
        return z.transpose(1, 2).contiguous()
