from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.residual_part_fsq import (
    GROUP_COORDINATES,
    GROUP_NAMES,
    ResidualPartFSQMotionAutoencoder,
)


class LatentResidualPartFSQMotionAutoencoder(ResidualPartFSQMotionAutoencoder):
    """Causal 40x9 holistic-base + local-latent-residual FSQ tokenizer.

    Each part quantizes the local state that cannot be predicted from the
    quantized holistic base. Part embeddings are projected into fixed,
    non-overlapping slices of the base latent and added before one decoder.
    """

    def __init__(
        self,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        motion_dim: int | None = None,
        base_code_dim: int = 128,
        base_width: int = 512,
        part_state_dim: int = 64,
        part_predictor_hidden_dim: int = 128,
        latent_projector_hidden_dim: int = 128,
        part_latent_dims: Mapping[str, int] | Sequence[int] | None = None,
        num_levels: int = 9,
        activation: str = "relu",
        norm: str | None = None,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
        part_membership: Mapping[str, Sequence[int | str]] | None = None,
    ) -> None:
        super().__init__(
            names=names,
            parents=parents,
            motion_dim=motion_dim,
            base_code_dim=base_code_dim,
            base_width=base_width,
            part_state_dim=part_state_dim,
            residual_decoder_dim=base_code_dim,
            residual_decoder_width=base_width,
            residual_hidden_dim=part_state_dim,
            num_levels=num_levels,
            activation=activation,
            norm=norm,
            fsq_scale=fsq_scale,
            fsq_preserve_symmetry=fsq_preserve_symmetry,
            fsq_noise_dropout=fsq_noise_dropout,
            residual_group_dropout=0.0,
            part_membership=part_membership,
        )
        for obsolete_name in (
            "latent_residual_norm",
            "local_projections",
            "part_fuse",
            "residual_fuse",
            "residual_group_embeddings",
            "residual_decoder",
            "residual_output_heads",
        ):
            delattr(self, obsolete_name)

        self.part_predictor_hidden_dim = int(part_predictor_hidden_dim)
        self.latent_projector_hidden_dim = int(latent_projector_hidden_dim)
        if self.part_predictor_hidden_dim <= 0 or self.latent_projector_hidden_dim <= 0:
            raise ValueError("Part predictor and latent projector hidden dimensions must be positive")

        self.part_latent_dims = self._resolve_part_latent_dims(part_latent_dims)
        self.part_latent_slices: dict[str, slice] = {}
        start = 0
        for part in PART_NAMES:
            end = start + self.part_latent_dims[part]
            self.part_latent_slices[part] = slice(start, end)
            start = end

        feature_indices = self.layout.feature_indices(self.motion_dim)
        self.local_state_encoders = nn.ModuleDict(
            {
                "torso": self._state_mlp(feature_indices["torso"].numel(), self.part_predictor_hidden_dim),
                "leg": self._state_mlp(feature_indices["left_leg"].numel(), self.part_predictor_hidden_dim),
                "arm": self._state_mlp(feature_indices["left_arm"].numel(), self.part_predictor_hidden_dim),
            }
        )
        self.base_part_predictors = nn.ModuleDict(
            {
                part: self._state_mlp(self.base_code_dim, self.part_predictor_hidden_dim)
                for part in PART_NAMES
            }
        )
        self.part_residual_norms = nn.ModuleDict(
            {part: nn.LayerNorm(self.part_state_dim) for part in PART_NAMES}
        )
        self.latent_residual_projectors = nn.ModuleDict(
            {
                part: nn.Sequential(
                    nn.Linear(self.part_state_dim, self.latent_projector_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.latent_projector_hidden_dim, self.part_latent_dims[part]),
                )
                for part in PART_NAMES
            }
        )
        for projector in self.latent_residual_projectors.values():
            output = projector[-1]
            assert isinstance(output, nn.Linear)
            nn.init.normal_(output.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(output.bias)

        self.config = {
            "names": list(self.layout.names),
            "parents": list(self.layout.parents),
            "motion_dim": self.motion_dim,
            "base_code_dim": self.base_code_dim,
            "base_width": self.base_width,
            "part_state_dim": self.part_state_dim,
            "part_predictor_hidden_dim": self.part_predictor_hidden_dim,
            "latent_projector_hidden_dim": self.latent_projector_hidden_dim,
            "part_latent_dims": dict(self.part_latent_dims),
            "num_levels": self.num_levels,
            "activation": activation,
            "norm": norm,
            "fsq_scale": fsq_scale,
            "fsq_preserve_symmetry": bool(fsq_preserve_symmetry),
            "fsq_noise_dropout": float(fsq_noise_dropout),
            "part_membership": {
                part: list(indices) for part, indices in zip(PART_NAMES, self.layout.part_joint_indices)
            },
        }

    def _state_mlp(self, input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.part_state_dim),
        )

    def _resolve_part_latent_dims(
        self, values: Mapping[str, int] | Sequence[int] | None
    ) -> dict[str, int]:
        if values is None:
            if self.base_code_dim < len(PART_NAMES):
                raise ValueError("base_code_dim must provide at least one latent channel per part")
            weights = [GROUP_COORDINATES[part] for part in PART_NAMES]
            total_weight = sum(weights)
            raw = [self.base_code_dim * weight / total_weight for weight in weights]
            dims = [max(1, math.floor(value)) for value in raw]
            order = sorted(range(len(raw)), key=lambda index: raw[index] - dims[index], reverse=True)
            for index in order[: self.base_code_dim - sum(dims)]:
                dims[index] += 1
            resolved = dict(zip(PART_NAMES, dims))
        elif isinstance(values, Mapping):
            if set(values) != set(PART_NAMES):
                raise ValueError(f"part_latent_dims must contain exactly {PART_NAMES}")
            resolved = {part: int(values[part]) for part in PART_NAMES}
        else:
            dims = [int(value) for value in values]
            if len(dims) != len(PART_NAMES):
                raise ValueError(f"part_latent_dims must contain {len(PART_NAMES)} values")
            resolved = dict(zip(PART_NAMES, dims))
        if any(value <= 0 for value in resolved.values()):
            raise ValueError(f"All part latent dimensions must be positive, got {resolved}")
        if sum(resolved.values()) != self.base_code_dim:
            raise ValueError(
                f"Part latent dimensions must sum to base_code_dim={self.base_code_dim}, got {resolved}"
            )
        return resolved

    @property
    def decoder(self) -> nn.Module:
        return self.base_decoder

    def _validate_embeddings(self, embeddings: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = set(GROUP_NAMES) - set(embeddings)
        if missing:
            raise ValueError(f"Missing latent Residual Part-FSQ embeddings: {sorted(missing)}")
        base = embeddings["base"]
        if base.ndim != 3 or base.shape[-1] != self.base_code_dim:
            raise ValueError(f"Base embedding must have shape [B, T, {self.base_code_dim}]")
        for part in PART_NAMES:
            expected = (*base.shape[:2], self.part_state_dim)
            if embeddings[part].shape != expected:
                raise ValueError(
                    f"Part embedding {part!r} must have shape {expected}, got {tuple(embeddings[part].shape)}"
                )
        return base

    def _encode(self, x: torch.Tensor, collect_metrics: bool):
        self._validate_motion(x)
        h = self.base_encoder(x.permute(0, 2, 1).float()).permute(0, 2, 1).contiguous()
        embeddings: dict[str, torch.Tensor] = {}
        codes: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        stats: dict[str, tuple[torch.Tensor, ...]] = {}
        embeddings["base"], codes["base"], indices["base"], stats["base"] = self._quantize(
            self.base_quantizer, h, collect_metrics
        )
        detached_base = embeddings["base"].detach()
        for part in PART_NAMES:
            family = self._part_family(part)
            local_features = x.index_select(-1, self._feature_index(part))
            local_state = self.local_state_encoders[family](local_features)
            predicted_state = self.base_part_predictors[part](detached_base)
            residual_state = self.part_residual_norms[part](local_state - predicted_state)
            embeddings[part], codes[part], indices[part], stats[part] = self._quantize(
                self._quantizer_for_group(part), residual_state, collect_metrics
            )
        return embeddings, codes, indices, stats

    def _project_part_residuals(
        self, embeddings: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        base = self._validate_embeddings(embeddings)
        residuals = {}
        for part in PART_NAMES:
            slot = self.latent_residual_projectors[part](embeddings[part])
            part_slice = self.part_latent_slices[part]
            residuals[part] = F.pad(
                slot,
                (part_slice.start, self.base_code_dim - part_slice.stop),
            )
            if residuals[part].shape != base.shape:
                raise RuntimeError(f"Projected latent residual {part!r} has an invalid shape")
        return residuals

    def _fuse_embeddings(
        self, embeddings: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        base = self._validate_embeddings(embeddings)
        residuals = self._project_part_residuals(embeddings)
        total_residual = torch.stack([residuals[part] for part in PART_NAMES], dim=0).sum(dim=0)
        return base + total_residual, residuals, total_residual

    def _decode_latent(self, embedding: torch.Tensor) -> torch.Tensor:
        return self._decode_base_embedding(embedding)

    def _decode_embeddings(
        self,
        embeddings: Mapping[str, torch.Tensor],
        decode_base: bool = False,
        edit_part: str | None = None,
        donor_permutation: torch.Tensor | None = None,
        collect_diagnostics: bool = False,
    ):
        base = self._validate_embeddings(embeddings)
        fused, residuals, total_residual = self._fuse_embeddings(embeddings)
        if (edit_part is None) != (donor_permutation is None):
            raise ValueError("edit_part and donor_permutation must be supplied together")

        latent_batches: list[tuple[str, torch.Tensor]] = []
        if decode_base:
            latent_batches.append(("base", base))
        latent_batches.append(("recon", fused))
        if edit_part is not None:
            if edit_part not in PART_NAMES:
                raise ValueError(f"Unknown edit part {edit_part!r}")
            assert donor_permutation is not None
            if donor_permutation.ndim != 1 or donor_permutation.shape[0] != base.shape[0]:
                raise ValueError("donor_permutation must have shape [B]")
            donor_permutation = donor_permutation.to(device=base.device, dtype=torch.long)
            if donor_permutation.numel() and (
                int(donor_permutation.min()) < 0 or int(donor_permutation.max()) >= base.shape[0]
            ):
                raise ValueError("donor_permutation contains an out-of-range batch index")
            edited = fused - residuals[edit_part] + residuals[edit_part].index_select(0, donor_permutation)
            latent_batches.append(("edit", edited))

        decoded = self._decode_latent(torch.cat([value for _, value in latent_batches], dim=0))
        decoded_by_name = {
            name: value for (name, _), value in zip(latent_batches, decoded.split(base.shape[0], dim=0))
        }
        residual_energy = torch.stack(
            [value.square().mean(dim=-1) for value in residuals.values()], dim=0
        ).mean(dim=0)
        if collect_diagnostics:
            base_rms = base.square().mean().sqrt().clamp_min(1e-7)
            latent_to_base_ratio = total_residual.square().mean().sqrt() / base_rms
        else:
            latent_to_base_ratio = None
        return (
            decoded_by_name["recon"],
            decoded_by_name.get("base"),
            decoded_by_name.get("edit"),
            residuals,
            residual_energy,
            latent_to_base_ratio,
        )

    def forward(
        self,
        x: torch.Tensor,
        collect_metrics: bool = False,
        decode_base: bool = False,
        edit_part: str | None = None,
        donor_permutation: torch.Tensor | None = None,
    ) -> dict[str, object]:
        embeddings, codes, indices, stats = self._encode(x, collect_metrics)
        recon, base_recon, edit_recon, residuals, residual_energy, latent_to_base_ratio = (
            self._decode_embeddings(
                embeddings,
                decode_base=decode_base,
                edit_part=edit_part,
                donor_permutation=donor_permutation,
                collect_diagnostics=collect_metrics,
            )
        )
        output: dict[str, object] = {
            "recon_state": recon,
            "fsq_codes": torch.cat([codes[group] for group in GROUP_NAMES], dim=-1),
            "indices": torch.cat([indices[group] for group in GROUP_NAMES], dim=-1),
            "base_codes": codes["base"],
            "base_indices": indices["base"],
            "part_codes": {part: codes[part] for part in PART_NAMES},
            "part_indices": {part: indices[part] for part in PART_NAMES},
            "part_latent_residuals": residuals,
            "latent_residual_energy": residual_energy,
            "latent_to_base_ratio": latent_to_base_ratio,
            "commit_loss": x.new_zeros(()),
        }
        output["codes"] = output["fsq_codes"]
        if base_recon is not None:
            output["base_recon_state"] = base_recon
        if edit_recon is not None:
            output["edit_recon_state"] = edit_recon
        if collect_metrics:
            output.update(self._metrics(indices, stats))
        return output

    def compute_representation_losses(self, output, batch):
        from .losses import compute_latent_residual_representation_losses

        return compute_latent_residual_representation_losses(output, batch, self.layout)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        recon, _, _, _, _, _ = self._decode_embeddings(
            self._decode_indices_to_embeddings(indices), decode_base=False
        )
        return recon

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        recon, _, _, _, _, _ = self._decode_embeddings(
            self._decode_codes_to_embeddings(codes), decode_base=False
        )
        return recon

    def decode_base_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self._validate_code_tensor(indices, "indices")
        base = self.base_quantizer.dequantize(indices[..., self.group_slices["base"]])
        return self._decode_latent(base.permute(0, 2, 1).contiguous())

    def decode_base_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        self._validate_code_tensor(codes, "codes")
        base = self.base_quantizer.project_codes_to_latent(codes[..., self.group_slices["base"]])
        return self._decode_latent(base.permute(0, 2, 1).contiguous())
