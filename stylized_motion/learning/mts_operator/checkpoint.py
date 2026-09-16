"""Checkpoint format for MTS transport and operator models.

The payload always records the token alphabet (``token_spec``), the tokenizer
fingerprint and the model config, so restoring a model against the wrong
tokenizer fails loudly instead of silently scoring nonsense.  Building the model
itself is the caller's job: this module never guesses a class.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import torch
import torch.nn as nn

from .contract import (
    MTS_CONTRACT_VERSION,
    TokenSpec,
    operator_metadata,
    validate_operator_metadata,
)

MTS_CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KINDS = ("transport", "operator", "style_encoder")


def mts_checkpoint_payload(
    *,
    kind: str,
    model: nn.Module,
    model_config: Mapping[str, object],
    token_spec: TokenSpec,
    tokenizer_metadata: Mapping[str, object] | None,
    metrics: Mapping[str, object] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
    payload: dict[str, object] = {
        "schema_version": MTS_CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "model": model.state_dict(),
        "metadata": operator_metadata(
            token_spec=token_spec,
            tokenizer_metadata=tokenizer_metadata,
            model_config=model_config,
            extra=extra,
        ),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_mts_checkpoint(path: str | Path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return path


def load_mts_checkpoint(
    path: str | Path,
    *,
    kind: str,
    build_model: Callable[[Mapping[str, object]], nn.Module],
    device: torch.device | str = "cpu",
    token_spec: TokenSpec | None = None,
    tokenizer_metadata: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], nn.Module]:
    """Build and restore one MTS model, validating its token identity."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("MTS checkpoint must be a mapping")
    if int(checkpoint.get("schema_version", 0)) != MTS_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MTS checkpoint schema_version {checkpoint.get('schema_version')}"
        )
    if checkpoint.get("kind") != kind:
        raise ValueError(
            f"Expected a {kind!r} checkpoint, got {checkpoint.get('kind')!r}"
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("MTS checkpoint is missing its metadata block")
    if int(metadata.get("mts_contract_version", 0)) != MTS_CONTRACT_VERSION:
        raise ValueError("MTS checkpoint contract version does not match this code")
    validate_operator_metadata(
        metadata, token_spec=token_spec, tokenizer_metadata=tokenizer_metadata
    )
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("MTS checkpoint metadata is missing model_config")
    model = build_model(dict(model_config))
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("MTS checkpoint is missing its model state dict")
    model.load_state_dict(state)
    model.to(torch.device(device)).eval()
    return dict(checkpoint), model


def checkpoint_token_spec(checkpoint: Mapping[str, object]) -> TokenSpec:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("MTS checkpoint is missing its metadata block")
    stored = metadata.get("token_spec")
    if not isinstance(stored, Mapping):
        raise ValueError("MTS checkpoint is missing token_spec")
    return TokenSpec(
        num_coordinates=int(stored["num_coordinates"]),
        num_levels=int(stored["num_levels"]),
        num_streams=int(stored["num_streams"]),
        family=str(stored.get("family", "")),
        representation_id=str(stored.get("representation_id", "")),
        layout_hash=str(stored.get("layout_hash", "")),
    )


__all__ = [
    "CHECKPOINT_KINDS",
    "MTS_CHECKPOINT_SCHEMA_VERSION",
    "checkpoint_token_spec",
    "load_mts_checkpoint",
    "mts_checkpoint_payload",
    "save_mts_checkpoint",
]
