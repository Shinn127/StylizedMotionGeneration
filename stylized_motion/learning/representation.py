"""Canonical representation registry and metadata contract.

The four FSQ families are intentionally the only representations exposed by
the new learning path.  Concrete model classes are imported here and nowhere
else in the workflow layer; callers receive :class:`RepresentationAdapter`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import yaml

from .fsq import FSQMotionAutoencoder
from .latent_residual_fsq import LatentResidualPartFSQMotionAutoencoder
from .part_fsq import HierarchicalPartFSQMotionAutoencoder
from .part_layout import GROUP_COORDINATES as PART_GROUP_COORDINATES
from .part_layout import GROUP_NAMES as PART_GROUP_NAMES
from .residual_part_fsq import (
    GROUP_COORDINATES as RESIDUAL_GROUP_COORDINATES,
    GROUP_NAMES as RESIDUAL_GROUP_NAMES,
    ResidualPartFSQMotionAutoencoder,
)


FLAT_FSQ_FAMILY = "flat_fsq"
PART_FSQ_FAMILY = "part_fsq"
RESIDUAL_PART_FSQ_FAMILY = "residual_part_fsq"
LATENT_RESIDUAL_FSQ_FAMILY = "latent_residual_fsq"
REPRESENTATION_FAMILIES = frozenset(
    {
        FLAT_FSQ_FAMILY,
        PART_FSQ_FAMILY,
        RESIDUAL_PART_FSQ_FAMILY,
        LATENT_RESIDUAL_FSQ_FAMILY,
    }
)

LEGACY_MODEL_FAMILY = {
    FLAT_FSQ_FAMILY: "fsq",
    PART_FSQ_FAMILY: "part_fsq",
    RESIDUAL_PART_FSQ_FAMILY: "residual_part_fsq",
    LATENT_RESIDUAL_FSQ_FAMILY: "latent_residual_part_fsq",
}


class RepresentationProtocol(Protocol):
    family: str
    variant: str
    representation_id: str
    motion_dim: int
    num_coordinates: int
    num_levels: int
    receptive_field: int
    context_left: int
    lookahead_frames: int

    def forward(self, motion: torch.Tensor) -> dict[str, Any]: ...

    def encode_to_indices(self, motion: torch.Tensor) -> torch.Tensor: ...

    def encode_to_codes(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor: ...

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor: ...

    def representation_metadata(self) -> dict[str, object]: ...

    def compute_representation_losses(
        self, output: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class RepresentationSpec:
    family: str
    variant: str
    representation_id: str
    coordinate_order: tuple[str, ...]
    coordinate_counts: tuple[tuple[str, int], ...]
    num_coordinates: int
    num_levels: int
    temporal_downsample: int = 1
    frame_rate: int = 60
    receptive_field: int = 64
    context_left: int = 63
    lookahead_frames: int = 0
    decoder_passes_inference: int = 1
    architecture_version: int | None = None

    @property
    def coordinate_layout(self) -> dict[str, int]:
        return dict(self.coordinate_counts)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "family": self.family,
            "variant": self.variant,
            "representation_id": self.representation_id,
            "coordinate_order": list(self.coordinate_order),
            "coordinate_counts": self.coordinate_layout,
            "num_coordinates": self.num_coordinates,
            "num_levels": self.num_levels,
            "temporal_downsample": self.temporal_downsample,
            "frame_rate": self.frame_rate,
            "receptive_field": self.receptive_field,
            "context_left": self.context_left,
            "lookahead_frames": self.lookahead_frames,
            "decoder_passes_inference": self.decoder_passes_inference,
        }
        if self.architecture_version is not None:
            result["architecture_version"] = self.architecture_version
        return result


def _default_layout(family: str, coordinates: int) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    if family == FLAT_FSQ_FAMILY:
        return ("flat",), (("flat", coordinates),)
    if family == PART_FSQ_FAMILY:
        return tuple(PART_GROUP_NAMES), tuple(
            (name, int(PART_GROUP_COORDINATES[name])) for name in PART_GROUP_NAMES
        )
    if family in {RESIDUAL_PART_FSQ_FAMILY, LATENT_RESIDUAL_FSQ_FAMILY}:
        return tuple(RESIDUAL_GROUP_NAMES), tuple(
            (name, int(RESIDUAL_GROUP_COORDINATES[name])) for name in RESIDUAL_GROUP_NAMES
        )
    raise ValueError(f"Unsupported representation family {family!r}")


def _read_model_config(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Representation model config does not exist: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Representation model config must contain a mapping: {path}")
        return dict(loaded)
    raise TypeError("representation.config must be a mapping or YAML path")


def _canonical_model_config(value: Mapping[str, object]) -> dict[str, object]:
    # Configs are part of the canonical contract.  An unknown or historical
    # key should fail at the concrete constructor instead of being translated
    # silently at the registry boundary.
    return dict(value)


def _config_family(config: Mapping[str, object]) -> tuple[str, str, dict[str, object]]:
    representation = config.get("representation")
    if not isinstance(representation, Mapping):
        raise ValueError("Config must contain a representation mapping")
    family = representation.get("family")
    variant = representation.get("variant")
    if family not in REPRESENTATION_FAMILIES:
        raise ValueError(f"representation.family must be one of {sorted(REPRESENTATION_FAMILIES)}")
    if not isinstance(variant, str) or not variant:
        raise ValueError("representation.variant must be a non-empty string")
    model_config = _canonical_model_config(_read_model_config(representation.get("config")))
    return str(family), variant, model_config


def _attach_dataset_config(model_config: dict[str, object], feature_store: Any | None, family: str) -> None:
    if feature_store is None:
        return
    model_config["motion_dim"] = int(feature_store.motion_dim)
    if family != FLAT_FSQ_FAMILY:
        model_config["names"] = list(feature_store.names)
        model_config["parents"] = [int(value) for value in feature_store.parents.tolist()]


def _spec_from_values(
    family: str,
    variant: str,
    model_config: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> RepresentationSpec:
    metadata = metadata or {}
    coordinates = int(metadata.get("num_coordinates", model_config.get("num_coordinates", 40)))
    levels = int(metadata.get("num_levels", model_config.get("num_levels", 9)))
    if coordinates != 40 or levels != 9:
        raise ValueError("Draft 0.3 representations require exactly 40 coordinates and 9 levels")
    order_value = metadata.get("coordinate_order")
    counts_value = metadata.get("coordinate_counts")
    default_order, default_counts = _default_layout(family, coordinates)
    order = tuple(str(item) for item in order_value) if isinstance(order_value, Sequence) and not isinstance(order_value, (str, bytes)) else default_order
    if isinstance(counts_value, Mapping):
        counts = tuple((name, int(counts_value[name])) for name in order if name in counts_value)
        if len(counts) != len(order):
            raise ValueError("coordinate_counts must define every coordinate_order entry")
    else:
        counts = default_counts
    if order != default_order or counts != default_counts:
        raise ValueError(
            f"{family} coordinate layout must be {list(default_order)} / {dict(default_counts)}"
        )
    if sum(value for _, value in counts) != coordinates:
        raise ValueError("coordinate_counts must sum to num_coordinates")
    architecture_version = metadata.get("architecture_version")
    if family == LATENT_RESIDUAL_FSQ_FAMILY:
        architecture_version = 2 if architecture_version is None else int(architecture_version)
        if architecture_version != 2:
            raise ValueError("Latent Residual-FSQ requires architecture_version=2")
    expected_variant = {
        FLAT_FSQ_FAMILY: "flat",
        PART_FSQ_FAMILY: "hierarchical",
        RESIDUAL_PART_FSQ_FAMILY: "default",
        LATENT_RESIDUAL_FSQ_FAMILY: "v2",
    }[family]
    if variant != expected_variant:
        raise ValueError(f"{family} requires variant={expected_variant!r}")
    representation_id = str(
        metadata.get("representation_id")
        or f"{family}_{coordinates}x{levels}"
    )
    expected_id = f"{family}_{coordinates}x{levels}"
    if representation_id != expected_id:
        raise ValueError(f"representation_id={representation_id!r} does not match {expected_id!r}")
    temporal_downsample = int(metadata.get("temporal_downsample", 1))
    frame_rate = int(metadata.get("frame_rate", 60))
    receptive_field = int(metadata.get("receptive_field", 64))
    context_left = int(metadata.get("context_left", 63))
    lookahead_frames = int(metadata.get("lookahead_frames", 0))
    decoder_passes = int(
        metadata.get(
            "decoder_passes_inference",
            2 if family == RESIDUAL_PART_FSQ_FAMILY else 1,
        )
    )
    if temporal_downsample != 1:
        raise ValueError("Canonical FSQ representations cannot temporally downsample")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    if (receptive_field, context_left, lookahead_frames) != (64, 63, 0):
        raise ValueError("Canonical FSQ representations require RF=64, context_left=63, lookahead=0")
    expected_decoder_passes = 2 if family == RESIDUAL_PART_FSQ_FAMILY else 1
    if decoder_passes != expected_decoder_passes:
        raise ValueError(
            f"{family} requires decoder_passes_inference={expected_decoder_passes}"
        )
    return RepresentationSpec(
        family=family,
        variant=variant,
        representation_id=representation_id,
        coordinate_order=order,
        coordinate_counts=counts,
        num_coordinates=coordinates,
        num_levels=levels,
        temporal_downsample=temporal_downsample,
        frame_rate=frame_rate,
        receptive_field=receptive_field,
        context_left=context_left,
        lookahead_frames=lookahead_frames,
        decoder_passes_inference=decoder_passes,
        architecture_version=architecture_version,
    )


def representation_spec(config: Mapping[str, object]) -> RepresentationSpec:
    family, variant, model_config = _config_family(config)
    metadata = config.get("representation")
    assert isinstance(metadata, Mapping)
    return _spec_from_values(family, variant, model_config, metadata)


class RepresentationAdapter(nn.Module):
    """Uniform protocol surface around one registered concrete model."""

    def __init__(
        self,
        module: nn.Module,
        spec: RepresentationSpec,
        feature_schema: Mapping[str, object] | None,
    ) -> None:
        super().__init__()
        self.module = module
        self._spec = spec
        self.feature_schema = dict(feature_schema or {})
        self.family = spec.family
        self.variant = spec.variant
        self.representation_id = spec.representation_id
        self.motion_dim = int(getattr(module, "motion_dim"))
        self.num_coordinates = spec.num_coordinates
        self.num_levels = spec.num_levels
        self.receptive_field = spec.receptive_field
        self.context_left = spec.context_left
        self.lookahead_frames = spec.lookahead_frames
        self.config = dict(getattr(module, "config", {}))

    def forward(self, motion: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        output = self.module(motion, **kwargs)
        if not isinstance(output, Mapping):
            raise TypeError("Representation forward() must return a mapping")
        result = dict(output)
        if "codes" not in result:
            result["codes"] = result.get("fsq_codes")
        if result.get("codes") is None:
            raise ValueError("Representation output is missing codes")
        metrics = result.get("representation_metrics")
        if metrics is None:
            standard = {
                "recon_state", "indices", "codes", "fsq_codes", "commit_loss",
                "group_codes", "group_indices", "base_codes", "base_indices",
                "part_codes", "part_indices", "base_recon_state", "edit_recon_state",
                "part_residuals", "part_latent_residuals", "latent_residual_energy",
                "group_coordinate_change_rates",
            }
            metrics = {key: value for key, value in result.items() if key not in standard}
        result["representation_metrics"] = metrics
        return result

    def encode_to_indices(self, motion: torch.Tensor) -> torch.Tensor:
        return self.module.encode_to_indices(motion)

    def encode_to_codes(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.module.encode_to_codes(motion)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.module.decode_from_indices(indices)

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.module.decode_from_codes(codes)

    def representation_metadata(self) -> dict[str, object]:
        result = self._spec.as_dict()
        result["feature_schema"] = dict(self.feature_schema)
        result["capabilities"] = {
            "encode": True,
            "decode": True,
            "part_edit": self.family != FLAT_FSQ_FAMILY,
            "generator": True,
        }
        if self.family == LATENT_RESIDUAL_FSQ_FAMILY:
            result["part_latent_dims"] = list(getattr(self.module, "part_latent_dims"))
        return result

    def compute_representation_losses(
        self, output: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        hook = getattr(self.module, "compute_representation_losses", None)
        if hook is None:
            return {}
        return dict(hook(output, batch))


def build_representation(
    config: Mapping[str, object],
    *,
    feature_store: Any | None = None,
    feature_schema: Mapping[str, object] | None = None,
) -> RepresentationAdapter:
    """Build the configured canonical representation.

    ``feature_store`` supplies skeleton names, parents and motion dimension for
    the part families.  No workflow caller is allowed to instantiate a
    concrete representation class directly.
    """
    family, variant, model_config = _config_family(config)
    _attach_dataset_config(model_config, feature_store, family)
    if family == FLAT_FSQ_FAMILY:
        model_config.setdefault("num_coordinates", 40)
    model_config.setdefault("num_levels", 9)
    if family == LATENT_RESIDUAL_FSQ_FAMILY:
        model_config.setdefault("architecture_version", 2)
    spec = _spec_from_values(family, variant, model_config, config["representation"])
    model_config.pop("architecture_version", None)
    for metadata_key in (
        "family", "variant", "representation_id", "coordinate_order", "coordinate_counts",
        "temporal_downsample", "frame_rate", "receptive_field", "context_left",
        "lookahead_frames", "decoder_passes_inference",
    ):
        model_config.pop(metadata_key, None)
    classes: dict[str, type[nn.Module]] = {
        FLAT_FSQ_FAMILY: FSQMotionAutoencoder,
        PART_FSQ_FAMILY: HierarchicalPartFSQMotionAutoencoder,
        RESIDUAL_PART_FSQ_FAMILY: ResidualPartFSQMotionAutoencoder,
        LATENT_RESIDUAL_FSQ_FAMILY: LatentResidualPartFSQMotionAutoencoder,
    }
    model = classes[family](**model_config)
    if int(getattr(model, "motion_dim")) != int(model_config["motion_dim"]):
        raise ValueError("Representation model motion_dim differs from config")
    if int(getattr(model, "num_coordinates")) != spec.num_coordinates:
        raise ValueError("Representation model coordinate count differs from metadata")
    model_levels = getattr(model, "num_levels", None)
    if model_levels is None:
        model_levels = getattr(getattr(model, "quantizer", None), "num_levels", None)
    if model_levels is None or int(model_levels) != spec.num_levels:
        raise ValueError("Representation model level count differs from metadata")
    if int(getattr(model, "lookahead_frames")) != 0:
        raise ValueError("All canonical FSQ representations must have lookahead_frames=0")
    return RepresentationAdapter(model, spec, feature_schema)


def representation_metadata(
    config: Mapping[str, object],
    *,
    representation: RepresentationProtocol | None = None,
    feature_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if representation is not None:
        result = dict(representation.representation_metadata())
        if feature_schema is not None:
            result["feature_schema"] = dict(feature_schema)
        return result
    spec = representation_spec(config)
    result = spec.as_dict()
    result["feature_schema"] = dict(feature_schema or {})
    return result


def checkpoint_metadata(
    config: Mapping[str, object],
    representation: RepresentationProtocol,
    feature_schema: Mapping[str, object],
) -> dict[str, object]:
    metadata = representation_metadata(config, representation=representation, feature_schema=feature_schema)
    if not metadata.get("feature_schema"):
        raise ValueError("New checkpoints require non-empty feature_schema metadata")
    return metadata


def load_representation_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    feature_schema: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], RepresentationAdapter]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping) or int(checkpoint.get("schema_version", 0)) != 2:
        raise ValueError("Only schema_version=2 canonical representation checkpoints are supported")
    config = checkpoint.get("config")
    representation_metadata_value = checkpoint.get("representation")
    model_config = checkpoint.get("model_config")
    if (
        not isinstance(config, Mapping)
        or not isinstance(config.get("representation"), Mapping)
        or not isinstance(representation_metadata_value, Mapping)
        or not isinstance(model_config, Mapping)
    ):
        raise ValueError("Checkpoint must contain representation and model_config mappings")
    configured_representation = dict(config["representation"])
    if configured_representation.get("family") != representation_metadata_value.get("family"):
        raise ValueError("Checkpoint config and representation metadata have different families")
    if configured_representation.get("variant") != representation_metadata_value.get("variant"):
        raise ValueError("Checkpoint config and representation metadata have different variants")
    configured_representation["config"] = dict(model_config)
    resolved_config = dict(config)
    resolved_config["representation"] = configured_representation
    adapter = build_representation(
        resolved_config,
        feature_schema=feature_schema or representation_metadata_value.get("feature_schema"),
    )
    expected = adapter.representation_metadata()
    for key in (
        "family", "variant", "representation_id", "num_coordinates", "num_levels",
        "coordinate_order", "coordinate_counts", "temporal_downsample", "lookahead_frames",
        "receptive_field", "context_left", "decoder_passes_inference", "architecture_version",
    ):
        if representation_metadata_value.get(key) != expected.get(key):
            raise ValueError(f"Checkpoint representation metadata mismatch at {key!r}")
    checkpoint_schema = representation_metadata_value.get("feature_schema")
    top_level_schema = checkpoint.get("feature_schema")
    if not isinstance(checkpoint_schema, Mapping) or not isinstance(top_level_schema, Mapping):
        raise ValueError("Canonical checkpoints require representation and top-level feature_schema mappings")
    if dict(top_level_schema) != dict(checkpoint_schema):
        raise ValueError("Checkpoint feature_schema differs between top-level and representation metadata")
    if feature_schema is not None and dict(checkpoint_schema) != dict(feature_schema):
        raise ValueError("Checkpoint feature_schema does not match the feature database")
    family = str(representation_metadata_value["family"])
    if checkpoint.get("model_family") != LEGACY_MODEL_FAMILY[family]:
        raise ValueError("Checkpoint model_family does not match its canonical representation family")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Canonical checkpoints require a model state_dict")
    adapter.load_state_dict(state_dict)
    adapter.to(device).eval()
    return dict(checkpoint), adapter


__all__ = [
    "FLAT_FSQ_FAMILY",
    "PART_FSQ_FAMILY",
    "RESIDUAL_PART_FSQ_FAMILY",
    "LATENT_RESIDUAL_FSQ_FAMILY",
    "REPRESENTATION_FAMILIES",
    "RepresentationAdapter",
    "RepresentationProtocol",
    "RepresentationSpec",
    "build_representation",
    "checkpoint_metadata",
    "load_representation_checkpoint",
    "representation_metadata",
    "representation_spec",
]
