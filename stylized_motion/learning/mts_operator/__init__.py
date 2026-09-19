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
    METRICS_SCHEMA_VERSION,
    MTS_CHECKPOINT_SCHEMA_VERSION,
    build_provenance,
    checkpoint_action_vocabulary,
    checkpoint_style_index,
    checkpoint_token_spec,
    load_mts_checkpoint,
    load_operator_bundle,
    mts_checkpoint_payload,
    save_mts_checkpoint,
    validate_store_binding,
)
from .contract import (
    MASK_KINDS,
    MTS_CONTRACT_VERSION,
    TokenSpec,
    TransportOutput,
    masked_cross_entropy,
    masked_mean,
    masked_nll_from_probs,
    operator_metadata,
    tokenizer_fingerprint,
    validate_operator_metadata,
)
from .layout_adapter import LayoutAdapter, layout_adapter
from .masking import MaskBatch, MaskConfig, MaskGenerator, apply_hard_support
from .model import MtsStyleOperator, OperatorBatch, OperatorResult
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
from .training import OperatorTrainer, TrainerConfig, TransportMetrics, TransportTrainer
from .style_encoder import GlobalStyleEncoder, StyleIDEncoder
from .transport import GRAPH_MODES, TEMPORAL_MODES, ContentConditioner, MotionTransportTransformer
from .windows import (
    ContentVocabulary,
    PairedBatchSource,
    TokenSource,
    WindowSample,
    read_window_tokens,
    windows_by_clip,
)

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
    "GlobalStyleEncoder",
    "LayoutAdapter",
    "MaskBatch",
    "MaskConfig",
    "MaskGenerator",
    "MotionTransportTransformer",
    "MtsStyleOperator",
    "OperatorBatch",
    "OperatorInputs",
    "OperatorOutput",
    "OperatorResult",
    "OperatorTrainer",
    "StyleIDEncoder",
    "StyleOperator",
    "TokenSpec",
    "TrainerConfig",
    "TransportMetrics",
    "TransportOutput",
    "TransportTrainer",
    "apply_hard_support",
    "build_operator",
    "checkpoint_action_vocabulary",
    "checkpoint_style_index",
    "checkpoint_token_spec",
    "inverse_cdf_sample",
    "layout_adapter",
    "load_mts_checkpoint",
    "load_operator_bundle",
    "masked_cross_entropy",
    "masked_mean",
    "masked_nll_from_probs",
    "mts_checkpoint_payload",
    "build_provenance",
    "operator_metadata",
    "paired_comparison",
    "read_window_tokens",
    "region_support_mask",
    "sample_tokens",
    "save_mts_checkpoint",
    "tokenizer_fingerprint",
    "validate_operator_metadata",
    "ContentVocabulary",
    "PairedBatchSource",
    "TokenSource",
    "WindowSample",
    "windows_by_clip",
]
