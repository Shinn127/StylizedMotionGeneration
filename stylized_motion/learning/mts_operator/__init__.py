"""MTS style operator: a generation operator over the NEF-FSQ discrete alphabet.

Modules
-------
``contract``
    Tensor shapes, token identity and the masking rules shared by everything.
``layout_adapter``
    Read-only index views over :class:`~stylized_motion.learning.nef_layout.NEFLayout`.
``embeddings`` / ``graph`` / ``masking`` / ``transport``
    The style-free base generator (Phase 2).
``training`` / ``checkpoint``
    Masked-token training loop and the checkpoint format that binds a model to
    its token alphabet.
``operators`` / ``sampling``
    Injection operators (logit field, arbitrary kernel, birth-death CTMC) and
    coupled sampling.

Nothing in this package may change the persisted 40x9 / 13-stream token
contract; it consumes the tokenizer read-only.
"""

from __future__ import annotations

from .checkpoint import (
    CHECKPOINT_KINDS,
    MTS_CHECKPOINT_SCHEMA_VERSION,
    checkpoint_token_spec,
    load_mts_checkpoint,
    mts_checkpoint_payload,
    save_mts_checkpoint,
)
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
from .masking import MaskBatch, MaskConfig, MaskGenerator, apply_hard_support
from .operators import (
    OPERATOR_NAMES,
    AdditiveLogitField,
    ArbitraryKernelOperator,
    BirthDeathCTMCOperator,
    OperatorInputs,
    OperatorOutput,
    StyleOperator,
    build_operator,
)
from .sampling import (
    CommonRandomNumbers,
    inverse_cdf_sample,
    paired_comparison,
    region_support_mask,
    sample_tokens,
)
from .training import TrainerConfig, TransportMetrics, TransportTrainer
from .transport import GRAPH_MODES, TEMPORAL_MODES, ContentConditioner, MotionTransportTransformer

__all__ = [
    "CHECKPOINT_KINDS",
    "GRAPH_MODES",
    "MASK_KINDS",
    "MTS_CHECKPOINT_SCHEMA_VERSION",
    "MTS_CONTRACT_VERSION",
    "OPERATOR_NAMES",
    "TEMPORAL_MODES",
    "AdditiveLogitField",
    "ArbitraryKernelOperator",
    "BirthDeathCTMCOperator",
    "CommonRandomNumbers",
    "ContentConditioner",
    "LayoutAdapter",
    "MaskBatch",
    "MaskConfig",
    "MaskGenerator",
    "MotionTransportTransformer",
    "OperatorInputs",
    "OperatorOutput",
    "StyleOperator",
    "TokenSpec",
    "TrainerConfig",
    "TransportMetrics",
    "TransportOutput",
    "TransportTrainer",
    "apply_hard_support",
    "build_operator",
    "checkpoint_token_spec",
    "inverse_cdf_sample",
    "layout_adapter",
    "load_mts_checkpoint",
    "masked_cross_entropy",
    "masked_mean",
    "mts_checkpoint_payload",
    "operator_metadata",
    "paired_comparison",
    "region_support_mask",
    "sample_tokens",
    "save_mts_checkpoint",
    "tokenizer_fingerprint",
    "validate_operator_metadata",
]
