"""Checkpoint format for MTS transport and operator models.

The payload always records the token alphabet (``token_spec``), the tokenizer
fingerprint and the model config, so restoring a model against the wrong
tokenizer fails loudly instead of silently scoring nonsense.

Revision 2 also stores the whole *bundle*: the operator payload carries its own
transport weights, so inference rebuilds every MTS model from the operator file
alone and never has to find the transport checkpoint again.  The external
transport file stays as training provenance (its SHA-256 is recorded), and the
artifact identities that make a result reproducible (store hashes, action and
style maps, resolved config, code commit) travel with it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .contract import (
    MTS_CONTRACT_VERSION,
    TokenSpec,
    operator_metadata,
    validate_operator_metadata,
)

MTS_CHECKPOINT_SCHEMA_VERSION = 2
#: Bumped when the *metric* names/definitions change; independent of the
#: checkpoint schema and of the architecture revision.
METRICS_SCHEMA_VERSION = 2
CHECKPOINT_KINDS = ("transport", "operator", "style_encoder")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity(root: str | Path | None = None) -> dict[str, object]:
    """Commit and dirty flag of the working tree, best effort.

    A dirty tree is recorded rather than refused: the point is that a reader can
    tell whether a checkpoint came from committed code.
    """
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git, no matter
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "code_commit": commit,
        "working_tree_dirty": None if status is None else bool(status),
        "source_digest": source_digest(root),
    }


#: Source and configuration files whose contents identify a run.  Explicit globs,
#: so no unrelated or sensitive file is ever read, and untracked modules count.
CODE_DIGEST_PATTERNS: tuple[str, ...] = (
    "stylized_motion/learning/mts_operator/*.py",
    "stylized_motion/data/*.py",
    "scripts/*mts*.py",
    "scripts/evaluate_mts_operator.py",
    "scripts/generate_mts_operator.py",
    "data/configs/mts_revision2*.yaml",
)


def source_digest(
    root: str | Path | None = None, patterns: Sequence[str] = CODE_DIGEST_PATTERNS
) -> dict[str, object]:
    """A per-file digest list plus one combined digest of the run's code.

    ``HEAD + dirty`` cannot tell two dirty revisions apart; this can.  The files
    are read, hashed, and reduced to a stable combined digest, so a checkpoint
    names the exact code that produced it without requiring a commit.
    """
    base = Path(root) if root is not None else Path.cwd()
    files: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path.is_file():
                files[str(path.relative_to(base))] = file_sha256(path)
    combined = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items())).encode("utf-8")
    ).hexdigest()
    return {"files": files, "combined": combined, "count": len(files)}


def build_provenance(
    *,
    store_identity: Mapping[str, object] | None = None,
    action_vocabulary: Mapping[str, object] | None = None,
    style_index: Mapping[str, int] | None = None,
    resolved_config: Mapping[str, object] | None = None,
    seed: int | None = None,
    upstream_transport_sha256: str | None = None,
    training_protocol_id: str | None = None,
    content_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The artifact-binding block every revision-2 checkpoint carries.

    ``content_schema`` is the raw->canonical label map the run was trained with
    (version, source file and its SHA, and how many rows it renamed).  The action
    ids in ``action_to_id`` are positions in the *canonical* class list, so a
    checkpoint without the map alongside it cannot be re-labelled after the fact.
    """
    provenance: dict[str, object] = {
        "metrics_version": METRICS_SCHEMA_VERSION,
        "store_identity": dict(store_identity or {}),
        "action_to_id": None if action_vocabulary is None else dict(action_vocabulary),
        "style_to_id": None if style_index is None else {str(k): int(v) for k, v in style_index.items()},
        "resolved_config": dict(resolved_config or {}),
        "seed": None if seed is None else int(seed),
        "upstream_transport_sha256": upstream_transport_sha256,
        "training_protocol_id": training_protocol_id,
        "content_schema": None if content_schema is None else dict(content_schema),
    }
    provenance.update(code_identity())
    return provenance


def training_exposure(
    *,
    trainable_styles: Sequence[str] | None = None,
    held_out_styles: Sequence[str] | None = None,
    actions: Sequence[str] | None = None,
    pairs: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """What the operator could actually learn from, recorded with the checkpoint.

    An evaluation may *claim* a style was held out; this is the fact it is checked
    against.  ``exposure_unknown`` is set when the run cannot report it, so a
    reader never mistakes a missing record for a measured one.
    """
    styles = None if trainable_styles is None else sorted(str(style) for style in trainable_styles)
    held = None if held_out_styles is None else sorted(str(style) for style in held_out_styles)
    return {
        "trainable_styles": styles,
        "held_out_styles": held,
        "actions": None if actions is None else sorted(str(action) for action in actions),
        "pairs_per_style": None if pairs is None else {str(k): int(v) for k, v in pairs.items()},
        "exposure_unknown": styles is None and held is None,
    }


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
    provenance: Mapping[str, object] | None = None,
    tokenizer_checkpoint: str | Path | None = None,
) -> dict[str, object]:
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
    payload: dict[str, object] = {
        "schema_version": MTS_CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "metrics_version": METRICS_SCHEMA_VERSION,
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
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if tokenizer_checkpoint is not None:
        payload["tokenizer_checkpoint_sha256"] = file_sha256(tokenizer_checkpoint)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_mts_checkpoint(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Writes the checkpoint atomically: a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_mts_checkpoint(
    path: str | Path,
    *,
    kind: str,
    build_model: Callable[[Mapping[str, object]], nn.Module],
    device: torch.device | str = "cpu",
    token_spec: TokenSpec | None = None,
    tokenizer_metadata: Mapping[str, object] | None = None,
    tokenizer_checkpoint: str | Path | None = None,
    tokenizer_checkpoint_sha256: str | None = None,
    require_tokenizer: bool = True,
) -> tuple[dict[str, object], nn.Module]:
    """Build and restore one MTS model, validating its token identity.

    ``tokenizer_checkpoint`` (or its ``tokenizer_checkpoint_sha256``) is compared
    with the SHA the checkpoint recorded, so a same-structure, different-weights
    tokenizer is refused at load time -- this is the entry point the operator's
    frozen upstream and the transport's warm start both go through.  The metadata
    block only describes the *alphabet*, which two NEF tokenizers share; the file
    is what separates them.

    ``require_tokenizer=False`` exists for callers that have no tokenizer artifact
    to bind (unit fixtures); a checkpoint that recorded no SHA is refused either
    way, and the formal entry points keep the check on.
    """
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
    if require_tokenizer:
        require_tokenizer_checkpoint(
            checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            tokenizer_checkpoint_sha256=tokenizer_checkpoint_sha256,
            where=f"MTS {kind} checkpoint",
        )
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


def require_tokenizer_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    tokenizer_checkpoint: str | Path | None = None,
    tokenizer_checkpoint_sha256: str | None = None,
    where: str = "MTS checkpoint",
) -> str:
    """Compares the tokenizer file's SHA-256 with the one the checkpoint recorded.

    Recording a hash is not binding anything: this is the function that makes a
    same-structure, different-weights tokenizer fail.  Exactly one of the two
    arguments is required, and a checkpoint without a recorded SHA is refused
    instead of being waved through.
    """
    recorded = checkpoint.get("tokenizer_checkpoint_sha256")
    if not recorded:
        raise ValueError(
            f"{where} records no tokenizer_checkpoint_sha256, so the tokenizer it was trained "
            "with cannot be identified; retrain or re-export with the SHA recorded"
        )
    if tokenizer_checkpoint is None and tokenizer_checkpoint_sha256 is None:
        raise ValueError(
            f"{where} requires the tokenizer file (tokenizer_checkpoint=...) or its sha256 to "
            "verify the binding; passing neither silently skips the check"
        )
    actual = (
        str(tokenizer_checkpoint_sha256)
        if tokenizer_checkpoint is None
        else file_sha256(tokenizer_checkpoint)
    )
    if str(actual) != str(recorded):
        source = "the given sha256" if tokenizer_checkpoint is None else str(tokenizer_checkpoint)
        raise ValueError(
            f"{where}: tokenizer sha256 mismatch — the checkpoint was trained with "
            f"{recorded}, {source} has {actual}. Same structure is not the same weights."
        )
    return str(actual)


def load_operator_bundle(
    path: str | Path,
    *,
    adapter: Any,
    tokenizer_identity: Mapping[str, object] | None = None,
    tokenizer_checkpoint: str | Path | None = None,
    tokenizer_checkpoint_sha256: str | None = None,
    require_tokenizer: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, object], nn.Module]:
    """Rebuilds the whole style operator from an operator checkpoint.

    Revision 2 stores the transport weights inside the operator payload, so this
    needs no external transport file: the operator's own config rebuilds the
    transport, the encoder and the operator, and the state dict is loaded
    strictly.  ``tokenizer_identity`` is the live tokenizer's representation
    metadata (the same block a transport load validates against).
    """
    from .model import MtsStyleOperator
    from .operators import build_operator
    from .style_encoder import ConstantStyleEncoder, GlobalStyleEncoder, StyleIDEncoder
    from .transport import MotionTransportTransformer

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("MTS checkpoint must be a mapping")
    if int(checkpoint.get("schema_version", 0)) != MTS_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MTS checkpoint schema_version {checkpoint.get('schema_version')}; "
            f"revision 2 requires {MTS_CHECKPOINT_SCHEMA_VERSION}. Old checkpoints were trained "
            "under a different loss, mask and architecture, so they are audit-only."
        )
    if checkpoint.get("kind") != "operator":
        raise ValueError(f"Expected an 'operator' checkpoint, got {checkpoint.get('kind')!r}")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("MTS operator checkpoint is missing its metadata block")
    if int(metadata.get("mts_contract_version", 0)) != MTS_CONTRACT_VERSION:
        raise ValueError("MTS operator checkpoint contract version does not match this code")
    if require_tokenizer:
        require_tokenizer_checkpoint(
            checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            tokenizer_checkpoint_sha256=tokenizer_checkpoint_sha256,
            where="MTS operator checkpoint",
        )
    # The alphabet is checked structurally *and* through the tokenizer identity:
    # the representation id comes from the stored spec (the live tokenizer is
    # checked against it by ``validate_operator_metadata``), while the counts,
    # family and layout hash must match the adapter exactly.
    stored_spec = checkpoint_token_spec(checkpoint)
    expected_spec = adapter.token_spec(representation_id=stored_spec.representation_id)
    if expected_spec.as_dict() != stored_spec.as_dict():
        raise ValueError(
            "The checkpoint's token alphabet does not match this layout adapter: "
            f"stored {stored_spec.as_dict()} vs adapter {expected_spec.as_dict()}"
        )
    validate_operator_metadata(
        metadata,
        token_spec=expected_spec,
        tokenizer_metadata=tokenizer_identity,
    )
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("MTS operator checkpoint is missing model_config")
    transport_config = dict(model_config.get("transport") or {})
    if not transport_config:
        raise ValueError("MTS operator checkpoint records no transport config")
    transport = MotionTransportTransformer(adapter, **transport_config)

    encoder_config = dict(model_config.get("style_encoder") or {})
    encoder_kind = str(encoder_config.pop("kind", "reference"))
    if encoder_kind == "style_id":
        style_encoder: nn.Module = StyleIDEncoder(
            num_styles=int(encoder_config["num_styles"]),
            output_dim=int(encoder_config["output_dim"]),
        )
    elif encoder_kind == "reference":
        style_encoder = GlobalStyleEncoder(adapter, **encoder_config)
    elif encoder_kind == "constant":
        style_encoder = ConstantStyleEncoder(output_dim=int(encoder_config["output_dim"]))
    else:
        raise ValueError(f"Unknown stored style encoder kind {encoder_kind!r}")

    operator_block = dict(model_config.get("operator") or {})
    operator_name = str(operator_block.get("name", ""))
    if not operator_name:
        raise ValueError("MTS operator checkpoint records no operator name")
    # ``describe()`` splits the constructor arguments (``config``) from the derived
    # facts; only the constructor arguments are replayed, and the level count and
    # stream width come from the live layout and the stored transport.
    operator_config = dict(operator_block.get("config") or {})
    declared_style_dim = operator_config.get("style_dim")
    encoder_output_dim = int(
        getattr(style_encoder, "output_dim", getattr(style_encoder, "dim", 0))
    )
    if declared_style_dim is not None and int(declared_style_dim) != encoder_output_dim:
        raise ValueError(
            f"The stored operator expects style_dim={int(declared_style_dim)} but the stored "
            f"encoder produces {encoder_output_dim} dimensions"
        )
    if declared_style_dim is None:
        hidden = int(operator_config.get("hidden_dim", 0))
        if hidden and encoder_output_dim != hidden:
            raise ValueError(
                f"The stored operator has no style_dim, so its style embedding must be "
                f"hidden_dim={hidden}, but the stored encoder produces {encoder_output_dim}; "
                "the checkpoint is internally inconsistent"
            )
    operator_config.pop("stream_dim", None)  # the stored transport owns the width
    operator = build_operator(
        operator_name,
        num_levels=adapter.num_levels,
        stream_dim=int(getattr(transport, "dim")),
        **operator_config,
    )
    model = MtsStyleOperator(
        adapter,
        transport=transport,
        style_encoder=style_encoder,
        operator=operator,
        freeze_transport=bool(model_config.get("freeze_transport", True)),
        freeze_style_encoder=bool(model_config.get("freeze_style_encoder", False)),
    )
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("MTS operator checkpoint is missing its model state dict")
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    if model.freeze_transport:
        model.transport.eval()
    if model.freeze_style_encoder:
        model.style_encoder.eval()
    return dict(checkpoint), model


def checkpoint_style_index(checkpoint: Mapping[str, object]) -> dict[str, int] | None:
    """The style-id map a style-ID checkpoint was trained with, if any."""
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    stored = provenance.get("style_to_id")
    if not isinstance(stored, Mapping):
        return None
    return {str(key): int(value) for key, value in stored.items()}


def checkpoint_action_vocabulary(checkpoint: Mapping[str, object]) -> dict[str, object] | None:
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    stored = provenance.get("action_to_id")
    return None if not isinstance(stored, Mapping) else dict(stored)


#: Which identities each store kind must be able to report.  A feature store has
#: no tokenizer identity of its own, and demanding one there would turn an
#: unrelated field into a new blocker (C02 item 3).
STORE_IDENTITY_FIELDS = {
    "token": ("feature_schema_hash", "normalization_hash", "split_manifest_hash", "representation_id"),
    "feature": ("feature_schema_hash", "normalization_hash", "split_manifest_hash", "skeleton_hash"),
}


def require_token_store_binding(
    store: Any,
    *,
    tokenizer_checkpoint: str | Path | None = None,
    tokenizer_checkpoint_sha256: str | None = None,
    checkpoint: Mapping[str, object] | None = None,
    where: str = "token store",
) -> dict[str, object]:
    """Three-way tokenizer check for a token store: store == file == checkpoint.

    A token store is created *by* one tokenizer, so the store's recorded
    ``checkpoint_sha256``, the tokenizer file the run will use, and (when a model
    is being loaded) the MTS checkpoint's recorded SHA must all agree.  This is
    the check that catches "same structure, different weights".
    """
    path = Path(store) if isinstance(store, (str, Path)) else None
    manifest: Mapping[str, object] = {}
    if isinstance(getattr(store, "manifest", None), Mapping):
        manifest = store.manifest
    elif path is not None and (path / "manifest.json").exists():
        import json as _json

        manifest = _json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    recorded = getattr(store, "checkpoint_sha256", None) or manifest.get("checkpoint_sha256")
    if not recorded:
        raise ValueError(
            f"{where}: the store records no checkpoint_sha256, so the tokenizer that created "
            "its tokens cannot be identified"
        )
    if tokenizer_checkpoint is None and tokenizer_checkpoint_sha256 is None:
        raise ValueError(
            f"{where}: pass the tokenizer file or its sha256; passing neither silently skips "
            "the binding"
        )
    actual = (
        str(tokenizer_checkpoint_sha256)
        if tokenizer_checkpoint is None
        else file_sha256(tokenizer_checkpoint)
    )
    if str(actual) != str(recorded):
        raise ValueError(
            f"{where}: the tokens were created by tokenizer {recorded}, but the supplied "
            f"tokenizer is {actual}; same structure is not the same weights"
        )
    if checkpoint is not None:
        expected = checkpoint.get("tokenizer_checkpoint_sha256")
        if not expected:
            raise ValueError(
                f"{where}: the MTS checkpoint records no tokenizer_checkpoint_sha256, so the "
                "store/checkpoint binding cannot be verified"
            )
        if str(expected) != str(recorded):
            raise ValueError(
                f"{where}: the MTS checkpoint was trained with tokenizer {expected}, the store "
                f"was built by {recorded}"
            )
    return {
        "checkpoint_sha256": str(recorded),
        "tokenizer_sha256": str(actual),
        "representation_id": getattr(store, "representation_id", None),
    }


def store_identity_block(
    store: Any, *, store_kind: str, store_path: str | Path | None = None
) -> dict[str, object]:
    """Everything a checkpoint should record about the store it trained on.

    A store *path* is not an identity: the same path can hold a rebuilt store, and
    two stores built from different tokenizers sit at different paths while
    agreeing on every structural field.  This records the declared identities, the
    schema version, the clip count, the motion width and a digest of the split
    table, so the artifact behind a checkpoint can be named.  Optional identities
    the store cannot report are recorded as ``None`` instead of being invented.
    """
    observed = validate_store_binding(store, store_kind=store_kind)
    block: dict[str, object] = {
        "store_path": None if store_path is None else str(store_path),
        "store_kind": str(store_kind),
    }
    for name in (
        "feature_schema_hash",
        "normalization_hash",
        "split_manifest_hash",
        "skeleton_hash",
        "representation_id",
    ):
        block[name] = observed.get(name)
    schema = getattr(store, "data_schema_version", None)
    if schema is None and isinstance(getattr(store, "manifest", None), Mapping):
        schema = store.manifest.get("data_schema_version")
    block["data_schema_version"] = None if schema is None else int(schema)
    if hasattr(store, "num_clips"):
        block["num_clips"] = int(getattr(store, "num_clips"))
    elif hasattr(store, "range_names"):
        block["num_clips"] = len(store.range_names)
    else:  # pragma: no cover - stores without a clip table
        block["num_clips"] = None
    block["motion_dim"] = int(getattr(store, "motion_dim", 0) or 0)
    block.update(split_table_identity(store))
    return block


def split_table_identity(store: Any) -> dict[str, object]:
    """A digest plus per-split counts of the store's own split table.

    Two stores can report the same schema hashes and still divide their clips
    differently; the split table is what a "held out" claim rests on, so it is
    recorded with the checkpoint.
    """
    if hasattr(store, "clip_split"):
        values = np.asarray(store.clip_split)
    elif hasattr(store, "split_ids"):
        values = np.asarray(store.split_ids)
    else:
        return {"split_table_sha256": None, "split_counts": None}
    values = np.ascontiguousarray(values.astype(np.uint8).reshape(-1))
    counts = {
        name: int(np.count_nonzero(values == index))
        for index, name in enumerate(("train", "val", "test"))
    }
    return {
        "split_table_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "split_counts": counts,
        "split_table_length": int(values.size),
    }


def validate_store_binding(
    store: Any,
    *,
    tokenizer_identity: Mapping[str, object] | None = None,
    expected_data_identity: Mapping[str, object] | None = None,
    store_kind: str | None = None,
) -> dict[str, object]:
    """Checks a store against the identities a checkpoint/evaluation declares.

    ``expected_data_identity`` may carry ``feature_schema_hash``,
    ``normalization_hash``, ``split_manifest_hash``, ``skeleton_hash`` and
    ``representation_id``; every field that is present must match the store.
    A store that cannot report a requested identity fails instead of being
    waved through ("no actor labels" is not the same as "actor-disjoint").
    """
    expected = dict(expected_data_identity or {})
    if store_kind is not None and store_kind not in STORE_IDENTITY_FIELDS:
        raise ValueError(f"Unknown store kind {store_kind!r}; expected {sorted(STORE_IDENTITY_FIELDS)}")
    observed: dict[str, object] = {}
    for name, attribute in (
        ("feature_schema_hash", "feature_schema_hash"),
        ("normalization_hash", "normalization_hash"),
        ("split_manifest_hash", "split_manifest_hash"),
        ("skeleton_hash", "skeleton_hash"),
        ("representation_id", "representation_id"),
    ):
        value = getattr(store, attribute, None)
        if value is None and isinstance(getattr(store, "manifest", None), Mapping):
            value = store.manifest.get(attribute)
        observed[name] = value
    if tokenizer_identity is not None:
        from .contract import tokenizer_fingerprint

        stored = expected.get("representation_id") or tokenizer_fingerprint(
            tokenizer_identity
        ).get("representation_id")
        expected.setdefault("representation_id", stored)
    if isinstance(getattr(store, "manifest", None), Mapping):
        normalization = store.manifest.get("normalization_hash")
        if normalization is not None:
            observed.setdefault("normalization_hash", normalization)
        observed.setdefault("split_manifest_hash", store.manifest.get("split_manifest_hash"))
        observed.setdefault("skeleton_hash", store.manifest.get("skeleton_hash"))
    required = STORE_IDENTITY_FIELDS.get(str(store_kind)) if store_kind else None
    for name in required or ():
        if name not in expected:
            expected[name] = observed.get(name)
    for name, wanted in expected.items():
        if wanted in (None, ""):
            continue
        actual = observed.get(name)
        if actual in (None, ""):
            raise ValueError(
                f"The store does not report {name}, so the binding to it cannot be verified "
                f"(expected {wanted!r})"
            )
        if str(actual) != str(wanted):
            raise ValueError(
                f"Store {name} {actual!r} does not match the expected {wanted!r}"
            )
    return {name: value for name, value in observed.items() if value not in (None, "")}


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
    "METRICS_SCHEMA_VERSION",
    "MTS_CHECKPOINT_SCHEMA_VERSION",
    "build_provenance",
    "checkpoint_action_vocabulary",
    "checkpoint_style_index",
    "checkpoint_token_spec",
    "code_identity",
    "file_sha256",
    "load_mts_checkpoint",
    "load_operator_bundle",
    "CODE_DIGEST_PATTERNS",
    "require_token_store_binding",
    "source_digest",
    "split_table_identity",
    "store_identity_block",
    "training_exposure",
    "require_tokenizer_checkpoint",
    "STORE_IDENTITY_FIELDS",
    "mts_checkpoint_payload",
    "save_mts_checkpoint",
    "validate_store_binding",
]
