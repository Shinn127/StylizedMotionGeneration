from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_concrete(
    log_alpha: torch.Tensor,
    temperature: float,
    training: bool,
    gamma: float = -0.1,
    zeta: float = 1.1,
    hard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns a sampled mask, deterministic probabilities, and expected L0."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not gamma < 0.0 < 1.0 < zeta:
        raise ValueError("Hard Concrete bounds must satisfy gamma < 0 < 1 < zeta")

    if training:
        uniform = torch.rand_like(log_alpha).clamp_(1e-6, 1.0 - 1e-6)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        concrete = torch.sigmoid((logistic + log_alpha) / temperature)
    else:
        concrete = torch.sigmoid(log_alpha)
    stretched = concrete * (zeta - gamma) + gamma
    soft_mask = stretched.clamp(0.0, 1.0)

    probability = (torch.sigmoid(log_alpha) * (zeta - gamma) + gamma).clamp(0.0, 1.0)
    expected_l0 = torch.sigmoid(log_alpha - temperature * math.log(-gamma / zeta))
    if hard:
        hard_mask = (soft_mask >= 0.5).to(soft_mask.dtype)
        soft_mask = hard_mask + soft_mask - soft_mask.detach() if training else hard_mask
    return soft_mask, probability, expected_l0


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {values.shape[1]} exceeds max_seq_len={self.position_embedding.shape[1]}"
            )
        hidden = self.input_projection(values)
        hidden = hidden + self.position_embedding[:, : hidden.shape[1]]
        hidden = self.encoder(hidden)
        return self.output_norm(hidden.mean(dim=1))


class FSQDynamicStyleGate(nn.Module):
    def __init__(
        self,
        num_coordinates: int,
        num_levels: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.num_coordinates = int(num_coordinates)
        self.num_levels = int(num_levels)
        self.input_dim = self.num_coordinates * self.num_levels
        self.temporal_encoder = TemporalEncoder(
            self.input_dim,
            hidden_dim,
            num_heads,
            num_layers,
            ff_dim,
            dropout,
            max_seq_len,
        )
        self.gate_head = nn.Linear(hidden_dim, self.input_dim)

    def logits(self, token_onehot: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_coordinates, num_levels = token_onehot.shape
        if (num_coordinates, num_levels) != (self.num_coordinates, self.num_levels):
            raise ValueError(
                f"Expected token shape [..., {self.num_coordinates}, {self.num_levels}], "
                f"got {tuple(token_onehot.shape)}"
            )
        features = token_onehot.reshape(batch_size, seq_len, self.input_dim)
        return self.gate_head(self.temporal_encoder(features)).reshape(
            batch_size, self.num_coordinates, self.num_levels
        )

    def forward(
        self,
        token_onehot: torch.Tensor,
        temperature: float,
        stochastic: bool,
        hard: bool,
    ) -> dict[str, torch.Tensor]:
        log_alpha = self.logits(token_onehot)
        mask, probability, expected_l0 = hard_concrete(
            log_alpha,
            temperature=temperature,
            training=stochastic,
            hard=hard,
        )
        return {
            "log_alpha": log_alpha,
            "mask": mask,
            "mask_probability": probability,
            "expected_l0": expected_l0,
        }


class FSQTokenStyleClassifier(nn.Module):
    def __init__(
        self,
        num_coordinates: int,
        num_levels: int,
        num_styles: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = int(num_coordinates) * int(num_levels)
        self.temporal_encoder = TemporalEncoder(
            self.input_dim,
            hidden_dim,
            num_heads,
            num_layers,
            ff_dim,
            dropout,
            max_seq_len,
        )
        self.classifier = nn.Linear(hidden_dim, num_styles)

    def forward(self, token_values: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_values.shape[:2]
        features = token_values.reshape(batch_size, seq_len, self.input_dim)
        return self.classifier(self.temporal_encoder(features))


class FSQStyleGateExperiment(nn.Module):
    def __init__(
        self,
        num_coordinates: int,
        num_levels: int,
        num_styles: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.config = {key: value for key, value in locals().items() if key not in {"self", "__class__"}}
        self.num_coordinates = int(num_coordinates)
        self.num_levels = int(num_levels)
        common = dict(
            num_coordinates=num_coordinates,
            num_levels=num_levels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.gate = FSQDynamicStyleGate(**common)
        self.dynamic_classifier = FSQTokenStyleClassifier(num_styles=num_styles, **common)
        self.full_classifier = FSQTokenStyleClassifier(num_styles=num_styles, **common)
        self.random_classifier = FSQTokenStyleClassifier(num_styles=num_styles, **common)

    def token_onehot(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3 or indices.shape[-1] != self.num_coordinates:
            raise ValueError(
                f"Expected indices [B,T,{self.num_coordinates}], got {tuple(indices.shape)}"
            )
        return F.one_hot(indices.long(), num_classes=self.num_levels).float()

    def gate_tokens(
        self,
        indices: torch.Tensor,
        temperature: float,
        stochastic: bool,
        hard: bool,
    ) -> dict[str, torch.Tensor]:
        token_onehot = self.token_onehot(indices)
        gate_output = self.gate(token_onehot, temperature, stochastic, hard)
        gate_output["token_onehot"] = token_onehot
        return gate_output

    def forward(
        self,
        indices: torch.Tensor,
        temperature: float,
        stochastic: bool | None = None,
        hard: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        stochastic = self.training if stochastic is None else stochastic
        hard = (not self.training) if hard is None else hard
        gate_output = self.gate_tokens(indices, temperature, stochastic, hard)
        token_onehot = gate_output["token_onehot"]
        mask = gate_output["mask"]
        active_probability = gate_output["mask_probability"].mean().detach()
        random_mask = (torch.rand_like(mask) < active_probability).to(mask.dtype)
        return {
            **gate_output,
            "dynamic_logits": self.dynamic_classifier(token_onehot * mask[:, None]),
            "full_logits": self.full_classifier(token_onehot),
            "random_logits": self.random_classifier(token_onehot * random_mask[:, None]),
            "random_mask": random_mask,
        }
