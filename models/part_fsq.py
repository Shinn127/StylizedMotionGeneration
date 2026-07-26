from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.causal_cnn import FrameCausalDecoder1D, FrameCausalEncoder1D
from models.fsq import MotionFSQ
from models.part_layout import FEATURE_GROUP_NAMES, GROUP_COORDINATES, GROUP_NAMES, PART_NAMES, PartFSQLayout


class HierarchicalPartFSQMotionAutoencoder(nn.Module):
    """Dense causal 40x9 Part-FSQ without graph message passing.

    Six static streams are processed by one shared causal encoder and decoder
    after being folded into the batch dimension.  This keeps the intended body
    routing while avoiding per-joint GNN kernels and sparse reductions.
    """

    def __init__(
        self,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        motion_dim: int | None = None,
        stream_dim: int = 64,
        num_levels: int = 9,
        activation: str = "relu",
        norm: str | None = None,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
        part_membership: Mapping[str, Sequence[int | str]] | None = None,
    ) -> None:
        super().__init__()
        self.layout = PartFSQLayout.from_skeleton(names, parents, part_membership=part_membership)
        expected_motion_dim = 9 * self.layout.num_joints + 5
        self.motion_dim = expected_motion_dim if motion_dim is None else int(motion_dim)
        self.layout.validate_motion_dim(self.motion_dim)
        self.stream_dim = int(stream_dim)
        self.num_levels = int(num_levels)
        if self.stream_dim <= 0:
            raise ValueError(f"stream_dim must be positive, got {stream_dim}")
        if self.num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")

        self.config = {
            "names": list(self.layout.names),
            "parents": list(self.layout.parents),
            "motion_dim": self.motion_dim,
            "stream_dim": self.stream_dim,
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

        feature_indices = self.layout.feature_indices(self.motion_dim)
        for group, index in feature_indices.items():
            self.register_buffer(f"_{group}_feature_indices", index, persistent=False)

        self.input_projections = nn.ModuleDict(
            {
                "global": nn.Linear(feature_indices["global"].numel(), self.stream_dim),
                "torso": nn.Linear(feature_indices["torso"].numel(), self.stream_dim),
                "leg": nn.Linear(feature_indices["left_leg"].numel(), self.stream_dim),
                "arm": nn.Linear(feature_indices["left_arm"].numel(), self.stream_dim),
            }
        )
        self.stream_embeddings = nn.Parameter(torch.randn(len(FEATURE_GROUP_NAMES), self.stream_dim) * 0.02)
        self.stream_encoder = FrameCausalEncoder1D(
            input_dim=self.stream_dim,
            code_dim=self.stream_dim,
            width=self.stream_dim,
            activation=activation,
            norm=norm,
        )
        self.stream_decoder = FrameCausalDecoder1D(
            output_dim=self.stream_dim,
            code_dim=self.stream_dim,
            width=self.stream_dim,
            activation=activation,
            norm=norm,
        )
        self.group_quantizers = nn.ModuleDict(
            {
                "global": self._make_quantizer("global", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
                "sync": self._make_quantizer("sync", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
                "torso": self._make_quantizer("torso", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
                "leg": self._make_quantizer("left_leg", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
                "arm": self._make_quantizer("left_arm", fsq_scale, fsq_preserve_symmetry, fsq_noise_dropout),
            }
        )
        self.sync_fuse = nn.Linear(self.stream_dim * len(FEATURE_GROUP_NAMES), self.stream_dim)
        self.part_fuse = nn.ModuleDict(
            {
                "torso": nn.Linear(self.stream_dim * 3, self.stream_dim),
                "leg": nn.Linear(self.stream_dim * 3, self.stream_dim),
                "arm": nn.Linear(self.stream_dim * 3, self.stream_dim),
            }
        )
        self.global_decode_fuse = nn.Linear(self.stream_dim * 2, self.stream_dim)
        self.part_decode_fuse = nn.ModuleDict(
            {
                "torso": nn.Linear(self.stream_dim * 3, self.stream_dim),
                "leg": nn.Linear(self.stream_dim * 3, self.stream_dim),
                "arm": nn.Linear(self.stream_dim * 3, self.stream_dim),
            }
        )
        self.output_heads = nn.ModuleDict(
            {
                "global": nn.Linear(self.stream_dim, feature_indices["global"].numel()),
                "torso": nn.Linear(self.stream_dim, feature_indices["torso"].numel()),
                "leg": nn.Linear(self.stream_dim, feature_indices["left_leg"].numel()),
                "arm": nn.Linear(self.stream_dim, feature_indices["left_arm"].numel()),
            }
        )
        # Frame encoder RF=31 and frame decoder RF=34; their composition has
        # RF=64, left context=63, and no future access.
        self.receptive_field, self.context_left, self.lookahead_frames = 64, 63, 0

    @property
    def num_coordinates(self) -> int:
        return self.layout.num_coordinates

    def _make_quantizer(
        self,
        group: str,
        fsq_scale: float | None,
        fsq_preserve_symmetry: bool,
        fsq_noise_dropout: float,
    ) -> MotionFSQ:
        return MotionFSQ(
            code_dim=self.stream_dim,
            num_coordinates=GROUP_COORDINATES[group],
            num_levels=self.num_levels,
            scale=fsq_scale,
            preserve_symmetry=fsq_preserve_symmetry,
            noise_dropout=fsq_noise_dropout,
        )

    @staticmethod
    def _part_family(part: str) -> str:
        if part == "torso":
            return "torso"
        if part.endswith("leg"):
            return "leg"
        if part.endswith("arm"):
            return "arm"
        raise KeyError(f"Unknown Part-FSQ body part {part!r}")

    @staticmethod
    def _quantizer_key(group: str) -> str:
        if group in {"global", "sync", "torso"}:
            return group
        if group.endswith("leg"):
            return "leg"
        if group.endswith("arm"):
            return "arm"
        raise KeyError(f"Unknown Part-FSQ group {group!r}")

    def _feature_index(self, group: str) -> torch.Tensor:
        return getattr(self, f"_{group}_feature_indices")

    def _validate_motion(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.motion_dim:
            raise ValueError(f"Expected motion [B, T, {self.motion_dim}], got {tuple(x.shape)}")

    def _apply_shared_temporal(self, module: nn.Module, streams: torch.Tensor) -> torch.Tensor:
        expected = (len(FEATURE_GROUP_NAMES), self.stream_dim)
        if streams.ndim != 4 or streams.shape[2:] != expected:
            raise ValueError(f"Expected part streams [B, T, {expected[0]}, {expected[1]}], got {tuple(streams.shape)}")
        batch_size, seq_len = streams.shape[:2]
        flat = streams.permute(0, 2, 3, 1).reshape(batch_size * len(FEATURE_GROUP_NAMES), self.stream_dim, seq_len)
        result = module(flat)
        return result.reshape(batch_size, len(FEATURE_GROUP_NAMES), self.stream_dim, seq_len).permute(0, 3, 1, 2).contiguous()

    def _encode_streams(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_motion(x)
        streams = []
        for group in FEATURE_GROUP_NAMES:
            family = group if group in {"global", "torso"} else self._part_family(group)
            streams.append(self.input_projections[family](x.index_select(-1, self._feature_index(group))))
        stacked = torch.stack(streams, dim=2)
        stacked = stacked + self.stream_embeddings.view(1, 1, len(FEATURE_GROUP_NAMES), self.stream_dim)
        return self._apply_shared_temporal(self.stream_encoder, stacked)

    def _quantize_group(
        self, group: str, state: torch.Tensor, collect_metrics: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        quantizer = self.group_quantizers[self._quantizer_key(group)]
        quantized, codes, indices, _, *stats = quantizer(
            state.permute(0, 2, 1).contiguous(), collect_stats=collect_metrics
        )
        return quantized.permute(0, 2, 1).contiguous(), codes, indices, tuple(stats)

    def _encode(
        self, x: torch.Tensor, collect_metrics: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, ...]]]:
        encoded = self._encode_streams(x.float())
        states = {group: encoded[:, :, index] for index, group in enumerate(FEATURE_GROUP_NAMES)}
        embeddings: dict[str, torch.Tensor] = {}
        codes: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        stats: dict[str, tuple[torch.Tensor, ...]] = {}

        embeddings["global"], codes["global"], indices["global"], stats["global"] = self._quantize_group(
            "global", states["global"], collect_metrics
        )
        sync_state = F.silu(self.sync_fuse(torch.cat([states[group] for group in FEATURE_GROUP_NAMES], dim=-1)))
        embeddings["sync"], codes["sync"], indices["sync"], stats["sync"] = self._quantize_group(
            "sync", sync_state, collect_metrics
        )
        for part in PART_NAMES:
            family = self._part_family(part)
            part_state = F.silu(
                self.part_fuse[family](torch.cat((states[part], embeddings["global"], embeddings["sync"]), dim=-1))
            )
            embeddings[part], codes[part], indices[part], stats[part] = self._quantize_group(
                part, part_state, collect_metrics
            )
        return embeddings, codes, indices, stats

    def _decode_embeddings(self, embeddings: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = set(GROUP_NAMES) - set(embeddings)
        if missing:
            raise ValueError(f"Missing Part-FSQ group embeddings: {sorted(missing)}")
        reference = embeddings["global"]
        if reference.ndim != 3 or reference.shape[-1] != self.stream_dim:
            raise ValueError(f"Part-FSQ embeddings must have shape [B, T, {self.stream_dim}]")
        if any(embeddings[group].shape != reference.shape for group in GROUP_NAMES):
            raise ValueError("All Part-FSQ group embeddings must have the same [B, T, C] shape")

        streams = [F.silu(self.global_decode_fuse(torch.cat((embeddings["global"], embeddings["sync"]), dim=-1)))]
        for part in PART_NAMES:
            family = self._part_family(part)
            streams.append(
                F.silu(self.part_decode_fuse[family](torch.cat((embeddings[part], embeddings["global"], embeddings["sync"]), dim=-1)))
            )
        decoded = torch.stack(streams, dim=2)
        decoded = decoded + self.stream_embeddings.view(1, 1, len(FEATURE_GROUP_NAMES), self.stream_dim)
        decoded = self._apply_shared_temporal(self.stream_decoder, decoded)

        recon = reference.new_zeros((*reference.shape[:2], self.motion_dim))
        for index, group in enumerate(FEATURE_GROUP_NAMES):
            family = group if group in {"global", "torso"} else self._part_family(group)
            recon[..., self._feature_index(group)] = self.output_heads[family](decoded[:, :, index])
        return recon

    def _validate_code_tensor(self, values: torch.Tensor, name: str) -> None:
        if values.ndim != 3 or values.shape[-1] != self.num_coordinates:
            raise ValueError(f"Expected Part-FSQ {name} [B, T, {self.num_coordinates}], got {tuple(values.shape)}")

    def _decode_indices_to_embeddings(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_code_tensor(indices, "indices")
        return {
            group: self.group_quantizers[self._quantizer_key(group)]
            .dequantize(indices[..., group_slice])
            .permute(0, 2, 1)
            .contiguous()
            for group, group_slice in self.layout.group_slices.items()
        }

    def _decode_codes_to_embeddings(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_code_tensor(codes, "codes")
        return {
            group: self.group_quantizers[self._quantizer_key(group)]
            .project_codes_to_latent(codes[..., group_slice])
            .permute(0, 2, 1)
            .contiguous()
            for group, group_slice in self.layout.group_slices.items()
        }

    def _metrics(
        self,
        indices_by_group: Mapping[str, torch.Tensor],
        stats_by_group: Mapping[str, tuple[torch.Tensor, ...]],
    ) -> dict[str, torch.Tensor]:
        weights = torch.tensor(
            [GROUP_COORDINATES[group] for group in GROUP_NAMES],
            dtype=torch.float32,
            device=next(iter(indices_by_group.values())).device,
        )
        metrics = {}
        metric_names = (
            "level_perplexity",
            "level_usage",
            "level_perplexity_min",
            "level_perplexity_max",
            "level_usage_min",
            "level_usage_max",
        )
        for metric_index, metric_name in enumerate(metric_names):
            values = torch.stack([stats_by_group[group][metric_index] for group in GROUP_NAMES])
            metrics[metric_name] = (values * weights).sum() / weights.sum()
        flat_indices = torch.cat([indices_by_group[group] for group in GROUP_NAMES], dim=-1)
        with torch.no_grad():
            tuples = flat_indices.reshape(-1, self.num_coordinates)
            tuple_unique_ratio = flat_indices.new_tensor(
                torch.unique(tuples, dim=0).shape[0] / max(tuples.shape[0], 1), dtype=torch.float32
            )
            if flat_indices.shape[1] < 2:
                tuple_change_rate = tuple_unique_ratio.new_zeros(())
                coordinate_change_rate = tuple_unique_ratio.new_zeros(())
                group_changes = tuple_unique_ratio.new_zeros((len(GROUP_NAMES),))
            else:
                changes = flat_indices[:, 1:] != flat_indices[:, :-1]
                tuple_change_rate = changes.any(dim=-1).float().mean()
                coordinate_change_rate = changes.float().mean()
                group_changes = torch.stack(
                    [(indices_by_group[group][:, 1:] != indices_by_group[group][:, :-1]).float().mean() for group in GROUP_NAMES]
                )
        metrics.update(
            {
                "tuple_unique_ratio": tuple_unique_ratio,
                "tuple_change_rate": tuple_change_rate,
                "coordinate_change_rate": coordinate_change_rate,
                "group_coordinate_change_rates": group_changes,
            }
        )
        return metrics

    def forward(self, x: torch.Tensor, collect_metrics: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        embeddings, codes, indices, stats = self._encode(x, collect_metrics=collect_metrics)
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "recon_state": self._decode_embeddings(embeddings),
            "fsq_codes": torch.cat([codes[group] for group in GROUP_NAMES], dim=-1),
            "indices": torch.cat([indices[group] for group in GROUP_NAMES], dim=-1),
            "commit_loss": x.new_zeros(()),
            "group_codes": codes,
            "group_indices": indices,
        }
        if collect_metrics:
            output.update(self._metrics(indices, stats))
        return output

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
        return self._decode_embeddings(self._decode_indices_to_embeddings(indices))

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self._decode_embeddings(self._decode_codes_to_embeddings(codes))
