from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from stylized_motion.learning.fsq import MotionFSQ
from stylized_motion.learning.nets.causal_cnn import FrameCausalDecoder1D, FrameCausalEncoder1D
from stylized_motion.learning.part_layout import PART_NAMES
from stylized_motion.learning.residual_part_fsq import GROUP_COORDINATES, GROUP_NAMES


class LatentResidualPartFSQV2MotionAutoencoder(nn.Module):
    """Holistic base FSQ plus dense full-latent local residual FSQs.

    V2 keeps the decoder latent unstructured: every local residual is projected
    into the full base latent and added to it before one shared decoder pass.
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
        part_encoder_width: int = 128,
        num_levels: int = 9,
        activation: str = "relu",
        norm: str | None = None,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
        part_membership: Mapping[str, Sequence[int | str]] | None = None,
    ) -> None:
        super().__init__()
        from stylized_motion.learning.part_layout import PartFSQLayout

        self.layout = PartFSQLayout.from_skeleton(names, parents, part_membership=part_membership)
        expected_motion_dim = 9 * self.layout.num_joints + 5
        self.motion_dim = expected_motion_dim if motion_dim is None else int(motion_dim)
        self.layout.validate_motion_dim(self.motion_dim)
        self.base_code_dim = int(base_code_dim)
        self.base_width = int(base_width)
        self.part_state_dim = int(part_state_dim)
        self.part_predictor_hidden_dim = int(part_predictor_hidden_dim)
        self.part_encoder_width = int(part_encoder_width)
        self.num_levels = int(num_levels)
        if min(self.base_code_dim, self.base_width, self.part_state_dim, self.part_predictor_hidden_dim, self.part_encoder_width) <= 0:
            raise ValueError("V2 model dimensions must be positive")

        feature_indices = self.layout.feature_indices(self.motion_dim)
        for part in PART_NAMES:
            self.register_buffer(f"_{part}_feature_indices", feature_indices[part], persistent=False)

        self.base_encoder = FrameCausalEncoder1D(self.motion_dim, self.base_code_dim, self.base_width, activation, norm)
        self.base_quantizer = MotionFSQ(self.base_code_dim, GROUP_COORDINATES["base"], self.num_levels, fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout)
        self.base_decoder = FrameCausalDecoder1D(self.motion_dim, self.base_code_dim, self.base_width, activation, norm)

        self.part_quantizers = nn.ModuleDict({
            family: MotionFSQ(self.part_state_dim, GROUP_COORDINATES[part], self.num_levels, fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout)
            for part, family in (("torso", "torso"), ("left_leg", "leg"), ("left_arm", "arm"))
        })
        input_dims = {part: int(feature_indices[part].numel()) for part in PART_NAMES}
        self.part_encoders = nn.ModuleDict({
            family: FrameCausalEncoder1D(input_dims[part], self.part_state_dim, self.part_encoder_width, activation, norm)
            for part, family in (("torso", "torso"), ("left_leg", "leg"), ("left_arm", "arm"))
        })
        self.base_part_predictors = nn.ModuleDict({
            part: nn.Sequential(
                nn.Linear(self.base_code_dim, self.part_predictor_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.part_predictor_hidden_dim, self.part_state_dim),
            ) for part in PART_NAMES
        })
        self.latent_residual_projectors = nn.ModuleDict({
            part: nn.Linear(self.part_state_dim, self.base_code_dim, bias=False)
            for part in PART_NAMES
        })
        for projector in self.latent_residual_projectors.values():
            nn.init.xavier_uniform_(projector.weight, gain=len(PART_NAMES) ** -0.5)

        self.receptive_field, self.lookahead_frames = 64, 0
        self.config = {
            "names": list(self.layout.names), "parents": list(self.layout.parents), "motion_dim": self.motion_dim,
            "base_code_dim": self.base_code_dim, "base_width": self.base_width, "part_state_dim": self.part_state_dim,
            "part_predictor_hidden_dim": self.part_predictor_hidden_dim, "part_encoder_width": self.part_encoder_width,
            "num_levels": self.num_levels, "activation": activation, "norm": norm, "fsq_scale": fsq_scale,
            "fsq_preserve_symmetry": bool(fsq_preserve_symmetry), "fsq_noise_dropout": float(fsq_noise_dropout),
            "part_membership": {part: list(indices) for part, indices in zip(PART_NAMES, self.layout.part_joint_indices)},
        }

    @property
    def decoder(self) -> nn.Module:
        return self.base_decoder

    @property
    def num_coordinates(self) -> int:
        return sum(GROUP_COORDINATES.values())

    @property
    def group_slices(self) -> dict[str, slice]:
        start, result = 0, {}
        for group in GROUP_NAMES:
            result[group] = slice(start, start + GROUP_COORDINATES[group])
            start += GROUP_COORDINATES[group]
        return result

    def _part_family(self, part: str) -> str:
        if part == "torso": return "torso"
        if part.endswith("leg"): return "leg"
        if part.endswith("arm"): return "arm"
        raise KeyError(part)

    def _feature_index(self, part: str) -> torch.Tensor:
        return getattr(self, f"_{part}_feature_indices")

    def _part_quantizer(self, part: str) -> MotionFSQ:
        return self.part_quantizers[self._part_family(part)]

    def _quantize(self, quantizer: MotionFSQ, state: torch.Tensor, collect_metrics: bool):
        embedding, codes, indices, _, *stats = quantizer(state.permute(0, 2, 1).contiguous(), collect_stats=collect_metrics)
        return embedding.permute(0, 2, 1).contiguous(), codes, indices, tuple(stats)

    def _encode(self, x: torch.Tensor, collect_metrics: bool):
        if x.ndim != 3 or x.shape[-1] != self.motion_dim:
            raise ValueError(f"Expected motion [B,T,{self.motion_dim}], got {tuple(x.shape)}")
        h = self.base_encoder(x.permute(0, 2, 1).float()).permute(0, 2, 1).contiguous()
        embeddings, codes, indices, stats = {}, {}, {}, {}
        embeddings["base"], codes["base"], indices["base"], stats["base"] = self._quantize(self.base_quantizer, h, collect_metrics)
        base_for_condition = embeddings["base"].detach()
        for part in PART_NAMES:
            local = self.part_encoders[self._part_family(part)](x.index_select(-1, self._feature_index(part)).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
            predicted = self.base_part_predictors[part](base_for_condition)
            embeddings[part], codes[part], indices[part], stats[part] = self._quantize(self._part_quantizer(part), local - predicted, collect_metrics)
        return embeddings, codes, indices, stats

    def _decode_embeddings(self, embeddings: Mapping[str, torch.Tensor], *, edit_part: str | None = None, donor_embeddings: Mapping[str, torch.Tensor] | None = None):
        base = embeddings["base"]
        delta_by_part = {part: self.latent_residual_projectors[part](embeddings[part]) for part in PART_NAMES}
        if edit_part is not None:
            if donor_embeddings is None or edit_part not in PART_NAMES:
                raise ValueError("V2 edit requires a valid part and donor embeddings")
            donor_state = self.base_part_predictors[edit_part](donor_embeddings["base"].detach()) + donor_embeddings[edit_part]
            target_prediction = self.base_part_predictors[edit_part](base.detach())
            delta_by_part[edit_part] = self.latent_residual_projectors[edit_part](donor_state - target_prediction)
        delta = torch.zeros_like(base)
        for value in delta_by_part.values():
            delta = delta + value
        return self._decode_latent(base + delta), delta_by_part

    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.base_decoder(latent.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

    def _fuse_embeddings(self, embeddings: Mapping[str, torch.Tensor]):
        base = embeddings["base"]
        delta_by_part = {
            part: self.latent_residual_projectors[part](embeddings[part]) for part in PART_NAMES
        }
        total_delta = torch.zeros_like(base)
        for delta in delta_by_part.values():
            total_delta = total_delta + delta
        return base + total_delta, delta_by_part

    def _build_compensated_edit_latent(
        self,
        embeddings: Mapping[str, torch.Tensor],
        fused: torch.Tensor,
        delta_by_part: Mapping[str, torch.Tensor],
        part: str,
        donor_permutation: torch.Tensor,
    ) -> torch.Tensor:
        if part not in PART_NAMES:
            raise ValueError(f"Unknown edit part {part!r}")
        base = embeddings["base"]
        if donor_permutation.ndim != 1 or donor_permutation.shape[0] != base.shape[0]:
            raise ValueError("donor_permutation must have shape [B]")
        donor_permutation = donor_permutation.to(device=base.device, dtype=torch.long)
        donor_base = base.index_select(0, donor_permutation)
        donor_residual = embeddings[part].index_select(0, donor_permutation)
        donor_state = self.base_part_predictors[part](donor_base.detach()) + donor_residual
        target_prediction = self.base_part_predictors[part](base.detach())
        edited_delta = self.latent_residual_projectors[part](donor_state - target_prediction)
        return fused - delta_by_part[part] + edited_delta

    def forward(
        self,
        x: torch.Tensor,
        collect_metrics: bool = False,
        decode_base: bool = False,
        edit_part: str | None = None,
        donor_permutation: torch.Tensor | None = None,
    ) -> dict[str, object]:
        embeddings, codes, indices, stats = self._encode(x, collect_metrics)
        if (edit_part is None) != (donor_permutation is None):
            raise ValueError("edit_part and donor_permutation must be supplied together")
        fused, delta_by_part = self._fuse_embeddings(embeddings)
        latent_batches: list[tuple[str, torch.Tensor]] = []
        if decode_base:
            latent_batches.append(("base", embeddings["base"]))
        latent_batches.append(("recon", fused))
        if edit_part is not None:
            assert donor_permutation is not None
            latent_batches.append(("edit", self._build_compensated_edit_latent(
                embeddings, fused, delta_by_part, edit_part, donor_permutation
            )))
        decoded = self._decode_latent(torch.cat([latent for _, latent in latent_batches], dim=0))
        decoded_by_name = {
            name: value for (name, _), value in zip(latent_batches, decoded.split(x.shape[0], dim=0))
        }
        output = {
            "recon_state": decoded_by_name["recon"], "fsq_codes": torch.cat([codes[group] for group in GROUP_NAMES], dim=-1),
            "indices": torch.cat([indices[group] for group in GROUP_NAMES], dim=-1), "base_codes": codes["base"],
            "base_indices": indices["base"], "part_codes": {part: codes[part] for part in PART_NAMES},
            "part_indices": {part: indices[part] for part in PART_NAMES}, "part_latent_residuals": delta_by_part,
            "commit_loss": x.new_zeros(()),
        }
        output["codes"] = output["fsq_codes"]
        if "base" in decoded_by_name:
            output["base_recon_state"] = decoded_by_name["base"]
        if "edit" in decoded_by_name:
            output["edit_recon_state"] = decoded_by_name["edit"]
            output["edit_part"] = edit_part
            output["donor_permutation"] = donor_permutation
        if collect_metrics:
            output.update(self._metrics(indices, stats))
        return output

    def compute_representation_losses(self, output, batch):
        from stylized_motion.learning.losses import compute_latent_residual_v2_representation_losses

        return compute_latent_residual_v2_representation_losses(output, batch, self.layout)

    def _metrics(self, indices_by_group, stats_by_group):
        weights = indices_by_group["base"].new_tensor(
            [GROUP_COORDINATES[group] for group in GROUP_NAMES], dtype=torch.float32
        )
        names = ("level_perplexity", "level_usage", "level_perplexity_min", "level_perplexity_max", "level_usage_min", "level_usage_max")
        metrics = {}
        for metric_index, name in enumerate(names):
            values = torch.stack([stats_by_group[group][metric_index] for group in GROUP_NAMES])
            metrics[name] = (values * weights).sum() / weights.sum()
        if indices_by_group["base"].shape[1] < 2:
            metrics["group_coordinate_change_rates"] = weights.new_zeros(len(GROUP_NAMES))
        else:
            metrics["group_coordinate_change_rates"] = torch.stack([
                (indices_by_group[group][:, 1:] != indices_by_group[group][:, :-1]).float().mean()
                for group in GROUP_NAMES
            ])
        return metrics

    def encode_to_indices(self, x):
        _, _, indices, _ = self._encode(x, False)
        return torch.cat([indices[group] for group in GROUP_NAMES], dim=-1)

    def encode_to_codes(self, x):
        _, codes, indices, _ = self._encode(x, False)
        return torch.cat([codes[group] for group in GROUP_NAMES], dim=-1), torch.cat([indices[group] for group in GROUP_NAMES], dim=-1)

    def _decode_code_groups(self, values: torch.Tensor, indices: bool):
        if values.ndim != 3 or values.shape[-1] != self.num_coordinates:
            raise ValueError(f"Expected [B,T,{self.num_coordinates}] values")
        embeddings = {"base": self.base_quantizer.dequantize(values[..., self.group_slices["base"]]) if indices else self.base_quantizer.project_codes_to_latent(values[..., self.group_slices["base"]])}
        for part in PART_NAMES:
            quantizer = self._part_quantizer(part)
            embeddings[part] = quantizer.dequantize(values[..., self.group_slices[part]]) if indices else quantizer.project_codes_to_latent(values[..., self.group_slices[part]])
            embeddings[part] = embeddings[part].permute(0, 2, 1).contiguous()
        embeddings["base"] = embeddings["base"].permute(0, 2, 1).contiguous()
        return embeddings

    def decode_from_indices(self, indices):
        return self._decode_embeddings(self._decode_code_groups(indices, True))[0]

    def decode_from_codes(self, codes):
        return self._decode_embeddings(self._decode_code_groups(codes, False))[0]

    def decode_base_from_indices(self, indices):
        return self._decode_latent(self._decode_code_groups(indices, True)["base"])

    def decode_base_from_codes(self, codes):
        return self._decode_latent(self._decode_code_groups(codes, False)["base"])

    def decode_from_indices_with_part_edit(self, target_indices, donor_indices, part):
        target = self._decode_code_groups(target_indices, True)
        donor = self._decode_code_groups(donor_indices, True)
        if target["base"].shape != donor["base"].shape:
            raise ValueError("Target and donor code batches must have the same shape")
        fused, delta_by_part = self._fuse_embeddings(target)
        donor_state = self.base_part_predictors[part](donor["base"].detach()) + donor[part]
        target_prediction = self.base_part_predictors[part](target["base"].detach())
        edited_delta = self.latent_residual_projectors[part](donor_state - target_prediction)
        edited = fused - delta_by_part[part] + edited_delta
        return self._decode_latent(edited)
