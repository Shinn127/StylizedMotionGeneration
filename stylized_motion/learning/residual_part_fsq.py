from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from stylized_motion.learning.nets.causal_cnn import FrameCausalDecoder1D, FrameCausalEncoder1D
from stylized_motion.learning.fsq import MotionFSQ
from stylized_motion.learning.part_layout import PART_NAMES, PartFSQLayout


GROUP_NAMES = ("base", *PART_NAMES)
GROUP_COORDINATES = {
    "base": 20,
    "torso": 4,
    "left_leg": 4,
    "right_leg": 4,
    "left_arm": 4,
    "right_arm": 4,
}


class ResidualPartFSQMotionAutoencoder(nn.Module):
    """Causal holistic-base + local-residual 40x9 FSQ tokenizer."""

    def __init__(
        self,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        motion_dim: int | None = None,
        base_code_dim: int = 128,
        base_width: int = 512,
        part_state_dim: int = 64,
        residual_decoder_dim: int = 128,
        residual_decoder_width: int = 128,
        residual_hidden_dim: int = 64,
        num_levels: int = 9,
        activation: str = "relu",
        norm: str | None = None,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
        residual_group_dropout: float = 0.1,
        part_membership: Mapping[str, Sequence[int | str]] | None = None,
    ) -> None:
        super().__init__()
        self.layout = PartFSQLayout.from_skeleton(names, parents, part_membership=part_membership)
        expected_motion_dim = 9 * self.layout.num_joints + 5
        self.motion_dim = expected_motion_dim if motion_dim is None else int(motion_dim)
        self.layout.validate_motion_dim(self.motion_dim)
        self.base_code_dim = int(base_code_dim)
        self.base_width = int(base_width)
        self.part_state_dim = int(part_state_dim)
        self.residual_decoder_dim = int(residual_decoder_dim)
        self.residual_decoder_width = int(residual_decoder_width)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.num_levels = int(num_levels)
        self.residual_group_dropout = float(residual_group_dropout)
        dimensions = (
            self.base_code_dim,
            self.base_width,
            self.part_state_dim,
            self.residual_decoder_dim,
            self.residual_decoder_width,
            self.residual_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError(f"All model dimensions must be positive, got {dimensions}")
        if self.num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")
        if not 0.0 <= self.residual_group_dropout < 1.0:
            raise ValueError("residual_group_dropout must be in [0, 1)")

        self.config = {
            "names": list(self.layout.names),
            "parents": list(self.layout.parents),
            "motion_dim": self.motion_dim,
            "base_code_dim": self.base_code_dim,
            "base_width": self.base_width,
            "part_state_dim": self.part_state_dim,
            "residual_decoder_dim": self.residual_decoder_dim,
            "residual_decoder_width": self.residual_decoder_width,
            "residual_hidden_dim": self.residual_hidden_dim,
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

        feature_indices = self.layout.feature_indices(self.motion_dim)
        for group, index in feature_indices.items():
            self.register_buffer(f"_{group}_feature_indices", index, persistent=False)

        self.base_encoder = FrameCausalEncoder1D(
            input_dim=self.motion_dim,
            code_dim=self.base_code_dim,
            width=self.base_width,
            activation=activation,
            norm=norm,
        )
        self.base_quantizer = MotionFSQ(
            code_dim=self.base_code_dim,
            num_coordinates=GROUP_COORDINATES["base"],
            num_levels=self.num_levels,
            scale=fsq_scale,
            preserve_symmetry=fsq_preserve_symmetry,
            noise_dropout=fsq_noise_dropout,
        )
        self.base_decoder = FrameCausalDecoder1D(
            output_dim=self.motion_dim,
            code_dim=self.base_code_dim,
            width=self.base_width,
            activation=activation,
            norm=norm,
        )
        self.latent_residual_norm = nn.LayerNorm(self.base_code_dim)

        self.local_projections = nn.ModuleDict(
            {
                "torso": nn.Linear(feature_indices["torso"].numel(), self.part_state_dim),
                "leg": nn.Linear(feature_indices["left_leg"].numel(), self.part_state_dim),
                "arm": nn.Linear(feature_indices["left_arm"].numel(), self.part_state_dim),
            }
        )
        part_fuse_input = self.base_code_dim * 2 + self.part_state_dim
        self.part_fuse = nn.ModuleDict(
            {
                family: nn.Sequential(
                    nn.Linear(part_fuse_input, self.base_code_dim),
                    nn.SiLU(),
                    nn.Linear(self.base_code_dim, self.part_state_dim),
                )
                for family in ("torso", "leg", "arm")
            }
        )
        self.part_quantizers = nn.ModuleDict(
            {
                "torso": self._make_part_quantizer("torso", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
                "leg": self._make_part_quantizer("left_leg", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
                "arm": self._make_part_quantizer("left_arm", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
            }
        )
        self.residual_fuse = nn.ModuleDict(
            {
                family: nn.Linear(self.part_state_dim + self.base_code_dim, self.residual_decoder_dim)
                for family in ("torso", "leg", "arm")
            }
        )
        self.residual_group_embeddings = nn.Parameter(torch.randn(len(PART_NAMES), self.residual_decoder_dim) * 0.02)
        self.residual_decoder = FrameCausalDecoder1D(
            output_dim=self.residual_hidden_dim,
            code_dim=self.residual_decoder_dim,
            width=self.residual_decoder_width,
            activation=activation,
            norm=norm,
        )
        self.residual_output_heads = nn.ModuleDict(
            {
                part: nn.Linear(self.residual_hidden_dim, feature_indices[part].numel())
                for part in PART_NAMES
            }
        )
        for head in self.residual_output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Both base and residual paths are encoder RF31 + decoder RF34.
        self.receptive_field, self.lookahead_frames = 64, 0

    @property
    def num_coordinates(self) -> int:
        return sum(GROUP_COORDINATES.values())

    @property
    def group_slices(self) -> dict[str, slice]:
        start = 0
        result = {}
        for group in GROUP_NAMES:
            end = start + GROUP_COORDINATES[group]
            result[group] = slice(start, end)
            start = end
        return result

    @staticmethod
    def _part_family(part: str) -> str:
        if part == "torso":
            return "torso"
        if part.endswith("leg"):
            return "leg"
        if part.endswith("arm"):
            return "arm"
        raise KeyError(f"Unknown residual part {part!r}")

    def _feature_index(self, group: str) -> torch.Tensor:
        return getattr(self, f"_{group}_feature_indices")

    def _make_part_quantizer(self, group: str, scale, preserve_symmetry, noise_dropout) -> MotionFSQ:
        return MotionFSQ(
            code_dim=self.part_state_dim,
            num_coordinates=GROUP_COORDINATES[group],
            num_levels=self.num_levels,
            scale=scale,
            preserve_symmetry=preserve_symmetry,
            noise_dropout=noise_dropout,
        )

    def _quantizer_for_group(self, group: str) -> MotionFSQ:
        return self.part_quantizers[self._part_family(group)]

    def _validate_motion(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.motion_dim:
            raise ValueError(f"Expected motion [B, T, {self.motion_dim}], got {tuple(x.shape)}")

    def _validate_code_tensor(self, values: torch.Tensor, name: str) -> None:
        if values.ndim != 3 or values.shape[-1] != self.num_coordinates:
            raise ValueError(f"Expected {name} [B, T, {self.num_coordinates}], got {tuple(values.shape)}")

    @staticmethod
    def _quantize(
        quantizer: MotionFSQ, state_bt: torch.Tensor, collect_metrics: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        embedding, codes, indices, _, *stats = quantizer(
            state_bt.permute(0, 2, 1).contiguous(),
            collect_stats=collect_metrics,
            collect_sequence_stats=False,
        )
        return embedding.permute(0, 2, 1).contiguous(), codes, indices, tuple(stats)

    def _encode(
        self, x: torch.Tensor, collect_metrics: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, ...]]]:
        self._validate_motion(x)
        h = self.base_encoder(x.permute(0, 2, 1).float()).permute(0, 2, 1).contiguous()
        embeddings: dict[str, torch.Tensor] = {}
        codes: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        stats: dict[str, tuple[torch.Tensor, ...]] = {}
        embeddings["base"], codes["base"], indices["base"], stats["base"] = self._quantize(
            self.base_quantizer, h, collect_metrics
        )
        latent_residual = self.latent_residual_norm(h - embeddings["base"])
        for part in PART_NAMES:
            family = self._part_family(part)
            local = self.local_projections[family](x.index_select(-1, self._feature_index(part)))
            state = self.part_fuse[family](torch.cat((latent_residual, embeddings["base"], local), dim=-1))
            embeddings[part], codes[part], indices[part], stats[part] = self._quantize(
                self._quantizer_for_group(part), state, collect_metrics
            )
        return embeddings, codes, indices, stats

    def _apply_residual_dropout(self, streams: torch.Tensor) -> torch.Tensor:
        if not self.training or self.residual_group_dropout <= 0.0:
            return streams
        keep_probability = 1.0 - self.residual_group_dropout
        mask = torch.rand(
            streams.shape[0], 1, streams.shape[2], 1, device=streams.device, dtype=streams.dtype
        ) < keep_probability
        return streams * mask.to(streams.dtype) / keep_probability

    def _decode_embeddings(
        self, embeddings: Mapping[str, torch.Tensor], apply_dropout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        missing = set(GROUP_NAMES) - set(embeddings)
        if missing:
            raise ValueError(f"Missing residual Part-FSQ embeddings: {sorted(missing)}")
        base = embeddings["base"]
        if base.ndim != 3 or base.shape[-1] != self.base_code_dim:
            raise ValueError(f"Base embedding must have shape [B, T, {self.base_code_dim}]")
        base_recon = self._decode_base_embedding(base)

        streams = []
        for part in PART_NAMES:
            embedding = embeddings[part]
            if embedding.shape != (*base.shape[:2], self.part_state_dim):
                raise ValueError(f"Part embedding {part!r} has invalid shape {tuple(embedding.shape)}")
            family = self._part_family(part)
            streams.append(F.silu(self.residual_fuse[family](torch.cat((embedding, base), dim=-1))))
        stacked = torch.stack(streams, dim=2)
        stacked = stacked + self.residual_group_embeddings.view(1, 1, len(PART_NAMES), self.residual_decoder_dim)
        if apply_dropout:
            stacked = self._apply_residual_dropout(stacked)
        batch_size, seq_len = stacked.shape[:2]
        flat = stacked.permute(0, 2, 3, 1).reshape(
            batch_size * len(PART_NAMES), self.residual_decoder_dim, seq_len
        )
        decoded = self.residual_decoder(flat)
        decoded = decoded.reshape(
            batch_size, len(PART_NAMES), self.residual_hidden_dim, seq_len
        ).permute(0, 3, 1, 2).contiguous()

        recon = base_recon.clone()
        residuals = {}
        for index, part in enumerate(PART_NAMES):
            residual = self.residual_output_heads[part](decoded[:, :, index])
            residuals[part] = residual
            recon[..., self._feature_index(part)] = recon[..., self._feature_index(part)] + residual
        return recon, base_recon, residuals

    def _decode_base_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 3 or embedding.shape[-1] != self.base_code_dim:
            raise ValueError(f"Base embedding must have shape [B, T, {self.base_code_dim}]")
        return self.base_decoder(embedding.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

    def _decode_indices_to_embeddings(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_code_tensor(indices, "indices")
        embeddings = {
            "base": self.base_quantizer.dequantize(indices[..., self.group_slices["base"]])
            .permute(0, 2, 1)
            .contiguous()
        }
        for part in PART_NAMES:
            embeddings[part] = self._quantizer_for_group(part).dequantize(
                indices[..., self.group_slices[part]]
            ).permute(0, 2, 1).contiguous()
        return embeddings

    def _decode_codes_to_embeddings(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_code_tensor(codes, "codes")
        embeddings = {
            "base": self.base_quantizer.project_codes_to_latent(codes[..., self.group_slices["base"]])
            .permute(0, 2, 1)
            .contiguous()
        }
        for part in PART_NAMES:
            embeddings[part] = self._quantizer_for_group(part).project_codes_to_latent(
                codes[..., self.group_slices[part]]
            ).permute(0, 2, 1).contiguous()
        return embeddings

    def _metrics(self, indices_by_group, stats_by_group) -> dict[str, torch.Tensor]:
        reference = indices_by_group["base"]
        weights = reference.new_tensor(
            [GROUP_COORDINATES[group] for group in GROUP_NAMES], dtype=torch.float32
        )
        metric_names = (
            "level_perplexity",
            "level_usage",
            "level_perplexity_min",
            "level_perplexity_max",
            "level_usage_min",
            "level_usage_max",
        )
        metrics = {}
        for metric_index, name in enumerate(metric_names):
            values = torch.stack([stats_by_group[group][metric_index] for group in GROUP_NAMES])
            metrics[name] = (values * weights).sum() / weights.sum()
        flat = torch.cat([indices_by_group[group] for group in GROUP_NAMES], dim=-1)
        with torch.no_grad():
            tuples = flat.reshape(-1, self.num_coordinates)
            metrics["tuple_unique_ratio"] = flat.new_tensor(
                torch.unique(tuples, dim=0).shape[0] / max(tuples.shape[0], 1), dtype=torch.float32
            )
            if flat.shape[1] < 2:
                metrics["tuple_change_rate"] = metrics["tuple_unique_ratio"].new_zeros(())
                metrics["coordinate_change_rate"] = metrics["tuple_unique_ratio"].new_zeros(())
                metrics["group_coordinate_change_rates"] = flat.new_zeros((len(GROUP_NAMES),), dtype=torch.float32)
            else:
                changes = flat[:, 1:] != flat[:, :-1]
                metrics["tuple_change_rate"] = changes.any(dim=-1).float().mean()
                metrics["coordinate_change_rate"] = changes.float().mean()
                metrics["group_coordinate_change_rates"] = torch.stack(
                    [
                        (indices_by_group[group][:, 1:] != indices_by_group[group][:, :-1]).float().mean()
                        for group in GROUP_NAMES
                    ]
                )
        return metrics

    def forward(self, x: torch.Tensor, collect_metrics: bool = False) -> dict[str, object]:
        embeddings, codes, indices, stats = self._encode(x, collect_metrics)
        recon, base_recon, residuals = self._decode_embeddings(embeddings, apply_dropout=True)
        output: dict[str, object] = {
            "recon_state": recon,
            "base_recon_state": base_recon,
            "fsq_codes": torch.cat([codes[group] for group in GROUP_NAMES], dim=-1),
            "indices": torch.cat([indices[group] for group in GROUP_NAMES], dim=-1),
            "base_codes": codes["base"],
            "base_indices": indices["base"],
            "part_codes": {part: codes[part] for part in PART_NAMES},
            "part_indices": {part: indices[part] for part in PART_NAMES},
            "part_residuals": residuals,
            "commit_loss": x.new_zeros(()),
        }
        output["codes"] = output["fsq_codes"]
        if collect_metrics:
            output.update(self._metrics(indices, stats))
        return output

    def compute_representation_losses(self, output, batch):
        from .losses import compute_residual_representation_losses

        return compute_residual_representation_losses(output, batch, self.layout)

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        _, _, indices, _ = self._encode(x, collect_metrics=False)
        return torch.cat([indices[group] for group in GROUP_NAMES], dim=-1)

    def encode_to_codes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, codes, indices, _ = self._encode(x, collect_metrics=False)
        return torch.cat([codes[group] for group in GROUP_NAMES], dim=-1), torch.cat(
            [indices[group] for group in GROUP_NAMES], dim=-1
        )

    def encode_to_embeddings(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        embeddings, _, indices, _ = self._encode(x, collect_metrics=False)
        return embeddings, torch.cat([indices[group] for group in GROUP_NAMES], dim=-1)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self._decode_embeddings(self._decode_indices_to_embeddings(indices), apply_dropout=False)
        return recon

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self._decode_embeddings(self._decode_codes_to_embeddings(codes), apply_dropout=False)
        return recon

    def decode_base_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode only the holistic base path from a full 40-coordinate index tensor."""
        embeddings = self._decode_indices_to_embeddings(indices)
        return self._decode_base_embedding(embeddings["base"])

    def decode_base_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode only the holistic base path from a full 40-coordinate code tensor."""
        embeddings = self._decode_codes_to_embeddings(codes)
        return self._decode_base_embedding(embeddings["base"])
