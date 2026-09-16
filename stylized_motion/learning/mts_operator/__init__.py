"""MTS style operator: a generation operator over the NEF-FSQ discrete alphabet.

Modules
-------
``contract``
    Tensor shapes, token identity and the masking rules shared by everything.
``layout_adapter``
    Read-only index views over :class:`~stylized_motion.learning.nef_layout.NEFLayout`.
``embeddings`` / ``graph`` / ``masking`` / ``transport``
    The style-free base generator (Phase 2).
``style_encoder`` / ``operators`` / ``sampling``
    Reference style descriptor and the injection operators (Phase 3/4).

Nothing in this package may change the persisted 40x9 / 13-stream token
contract; it consumes the tokenizer read-only.
"""

from __future__ import annotations

from .contract import (
    MASK_KINDS,
    MTS_CONTRACT_VERSION,
    TokenSpec,
    TransportOutput,
    masked_cross_entropy,
    masked_mean,
    operator_metadata,
    tokenizer_fingerprint,
    validate_operator_metadata,
)
from .layout_adapter import LayoutAdapter, layout_adapter

__all__ = [
    "MASK_KINDS",
    "MTS_CONTRACT_VERSION",
    "LayoutAdapter",
    "TokenSpec",
    "TransportOutput",
    "layout_adapter",
    "masked_cross_entropy",
    "masked_mean",
    "operator_metadata",
    "tokenizer_fingerprint",
    "validate_operator_metadata",
]
