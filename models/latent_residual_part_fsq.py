from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from models.part_layout import PART_NAMES
from models.residual_part_fsq import (
    GROUP_NAMES,
    ResidualPartFSQMotionAutoencoder,
)


class LatentResidualPartFSQMotionAutoencoder(ResidualPartFSQMotionAutoencoder):
    """Causal 40x9 holistic-base + latent part-residual FSQ tokenizer.

    The encoder and coordinate layout match ResidualPartFSQMotionAutoencoder,
    but the five part embeddings are projected into the holistic base latent
    and fused before a single shared temporal decoder.  The inherited feature
    residual decoder is removed from this model.
    """

    def __init__(
        self,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        motion_dim: int | None = None,
        base_code_dim: int = 128,
        base_width: int = 512,
        part_state_dim: int = 64,
        latent_fusion_hidden_dim: int = 128,
        num_levels: int = 9,
        activation: str = "relu",
        norm: str | None = None,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
        residual_group_dropout: float = 0.1,
        part_membership: Mapping[str, Sequence[int | str]] | None = None,
    ) -> None:
        # Build the common encoder, quantizers, and holistic decoder with the
        # legacy implementation, then discard its feature-space residual path.
        # Keeping the common module names also permits selective warm-starting
        # from a feature-side Residual Part-FSQ checkpoint.
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
            residual_group_dropout=residual_group_dropout,
            part_membership=part_membership,
        )
        del self.residual_fuse
        del self.residual_group_embeddings
        del self.residual_decoder
        del self.residual_output_heads

        self.latent_fusion_hidden_dim = int(latent_fusion_hidden_dim)
        if self.latent_fusion_hidden_dim <= 0:
            raise ValueError(
                f"latent_fusion_hidden_dim must be positive, got {latent_fusion_hidden_dim}"
            )

        self.latent_group_embeddings = nn.Parameter(
            torch.randn(len(PART_NAMES), self.part_state_dim) * 0.02
        )
        self.latent_residual_fuse = nn.ModuleDict(
            {
                family: nn.Sequential(
                    nn.Linear(self.part_state_dim + self.base_code_dim, self.latent_fusion_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.latent_fusion_hidden_dim, self.base_code_dim),
                )
                for family in ("torso", "leg", "arm")
            }
        )
        for projector in self.latent_residual_fuse.values():
            output = projector[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

        self.config = {
            "names": list(self.layout.names),
            "parents": list(self.layout.parents),
            "motion_dim": self.motion_dim,
            "base_code_dim": self.base_code_dim,
            "base_width": self.base_width,
            "part_state_dim": self.part_state_dim,
            "latent_fusion_hidden_dim": self.latent_fusion_hidden_dim,
            "num_levels": self.num_levels,
            "activation": activation,
            "norm": norm,
            "fsq_scale": fsq_scale,
            "fsq_preserve_symmetry": bool(fsq_preserve_symmetry),
            "fsq_noise_dropout": float(fsq_noise_dropout),
            "residual_group_dropout": self.residual_group_dropout,
            "part_membership": {
                part: list(indices) for part, indices in zip(PART_NAMES, self.layout.part_joint_indices)
            },
        }

    @property
    def decoder(self) -> nn.Module:
        """The sole temporal decoder used by the latent-fusion model."""
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

    def _fuse_embeddings(
        self,
        embeddings: Mapping[str, torch.Tensor],
        apply_dropout: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        base = self._validate_embeddings(embeddings)
        residuals: dict[str, torch.Tensor] = {}
        ordered = []
        for index, part in enumerate(PART_NAMES):
            family = self._part_family(part)
            part_embedding = embeddings[part] + self.latent_group_embeddings[index].view(1, 1, -1)
            residual = self.latent_residual_fuse[family](torch.cat((part_embedding, base), dim=-1))
            residuals[part] = residual
            ordered.append(residual)

        stacked = torch.stack(ordered, dim=2)
        dropped = self._apply_residual_dropout(stacked) if apply_dropout else stacked
        total_residual = dropped.sum(dim=2) / math.sqrt(float(len(PART_NAMES)))
        return base + total_residual, residuals, total_residual

    def _decode_latent(self, embedding: torch.Tensor) -> torch.Tensor:
        return self._decode_base_embedding(embedding)

    def _decode_embeddings(
        self,
        embeddings: Mapping[str, torch.Tensor],
        apply_dropout: bool = False,
        decode_base: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        base = self._validate_embeddings(embeddings)
        fused, residuals, total_residual = self._fuse_embeddings(embeddings, apply_dropout=apply_dropout)
        if decode_base:
            decoded = self._decode_latent(torch.cat((base, fused), dim=0))
            base_recon, recon = decoded.chunk(2, dim=0)
        else:
            recon = self._decode_latent(fused)
            base_recon = None

        residual_energy = torch.stack([value.square().mean() for value in residuals.values()]).mean()
        base_rms = base.square().mean().sqrt().clamp_min(1e-7)
        latent_to_base_ratio = total_residual.square().mean().sqrt() / base_rms
        return recon, base_recon, residuals, residual_energy, latent_to_base_ratio

    def forward(
        self,
        x: torch.Tensor,
        collect_metrics: bool = False,
        decode_base: bool = False,
    ) -> dict[str, object]:
        embeddings, codes, indices, stats = self._encode(x, collect_metrics)
        recon, base_recon, residuals, residual_energy, latent_to_base_ratio = self._decode_embeddings(
            embeddings,
            apply_dropout=True,
            decode_base=decode_base,
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
        if base_recon is not None:
            output["base_recon_state"] = base_recon
        if collect_metrics:
            output.update(self._metrics(indices, stats))
        return output

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        recon, _, _, _, _ = self._decode_embeddings(
            self._decode_indices_to_embeddings(indices),
            apply_dropout=False,
            decode_base=False,
        )
        return recon

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        recon, _, _, _, _ = self._decode_embeddings(
            self._decode_codes_to_embeddings(codes),
            apply_dropout=False,
            decode_base=False,
        )
        return recon

    def decode_base_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode only the first 20 holistic base indices."""
        self._validate_code_tensor(indices, "indices")
        base = self.base_quantizer.dequantize(indices[..., self.group_slices["base"]])
        return self._decode_latent(base.permute(0, 2, 1).contiguous())

    def decode_base_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode only the first 20 holistic base codes."""
        self._validate_code_tensor(codes, "codes")
        base = self.base_quantizer.project_codes_to_latent(codes[..., self.group_slices["base"]])
        return self._decode_latent(base.permute(0, 2, 1).contiguous())
