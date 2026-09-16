"""NEF-FSQ: Node-Edge Factorized FSQ tokenizer with 13 independent streams.

There is no full-body latent, base addition, sync token or cross-stream
attention: every stream is projected, encoded, quantized and decoded on its
own.  The 13 streams share one temporal encoder and one temporal decoder by
folding the stream axis into the batch axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from stylized_motion.learning.fsq import MotionFSQ
from stylized_motion.learning.nef_layout import (
    NEF_FAMILIES,
    NEF_FAMILY_STREAMS,
    NEF_STREAM_COORDINATES,
    NEF_STREAM_FAMILY,
    NEF_STREAM_NAMES,
    NEF_VARIANT,
    NEFLayout,
)
from stylized_motion.learning.nets.causal_cnn import FrameCausalDecoder1D, FrameCausalEncoder1D


# FrameCausalEncoder1D has RF=31, FrameCausalDecoder1D has RF=34; their
# composition is exactly the 64-frame lookahead-0 contract of the family.
NEF_ENCODER_RECEPTIVE_FIELD = 31
NEF_DECODER_RECEPTIVE_FIELD = 34


class NEFMotionAutoencoder(nn.Module):
    def __init__(
        self,
        names: Sequence[str],
        parents: Sequence[int] | torch.Tensor,
        motion_dim: int | None = None,
        stream_dim: int = 128,
        num_levels: int = 9,
        activation: str = "relu",
        norm: str | None = None,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
        encoder_receptive_field: int = NEF_ENCODER_RECEPTIVE_FIELD,
        decoder_receptive_field: int = NEF_DECODER_RECEPTIVE_FIELD,
    ) -> None:
        super().__init__()
        self.layout = NEFLayout.from_skeleton(names, parents)
        self.motion_dim = 9 * self.layout.num_joints + 5 if motion_dim is None else int(motion_dim)
        self.layout.validate_motion_dim(self.motion_dim)
        self.stream_dim = int(stream_dim)
        self.num_levels = int(num_levels)
        if self.stream_dim <= 0:
            raise ValueError(f"stream_dim must be positive, got {stream_dim}")
        if self.num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")
        if (int(encoder_receptive_field), int(decoder_receptive_field)) != (
            NEF_ENCODER_RECEPTIVE_FIELD,
            NEF_DECODER_RECEPTIVE_FIELD,
        ):
            raise ValueError(
                "NEF-FSQ requires the shared FrameCausalEncoder1D/FrameCausalDecoder1D "
                f"receptive fields {NEF_ENCODER_RECEPTIVE_FIELD}/{NEF_DECODER_RECEPTIVE_FIELD}"
            )

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
            "encoder_receptive_field": NEF_ENCODER_RECEPTIVE_FIELD,
            "decoder_receptive_field": NEF_DECODER_RECEPTIVE_FIELD,
        }

        feature_indices = self.layout.feature_indices(self.motion_dim)
        for stream, index in feature_indices.items():
            self.register_buffer(f"_{stream}_feature_indices", index, persistent=False)

        for family, streams in NEF_FAMILY_STREAMS.items():
            widths = {int(feature_indices[stream].numel()) for stream in streams}
            coordinates = {NEF_STREAM_COORDINATES[stream] for stream in streams}
            if len(widths) != 1 or len(coordinates) != 1:
                raise ValueError(
                    f"NEF family {family!r} requires equal feature width and coordinate count "
                    f"across {list(streams)}"
                )
        self.family_feature_dims = {
            family: int(feature_indices[streams[0]].numel()) for family, streams in NEF_FAMILY_STREAMS.items()
        }
        self.family_coordinates = {
            family: NEF_STREAM_COORDINATES[streams[0]] for family, streams in NEF_FAMILY_STREAMS.items()
        }

        self.input_projections = nn.ModuleDict(
            {
                family: nn.Linear(self.family_feature_dims[family], self.stream_dim)
                for family in NEF_FAMILIES
            }
        )
        self.output_heads = nn.ModuleDict(
            {
                family: nn.Linear(self.stream_dim, self.family_feature_dims[family])
                for family in NEF_FAMILIES
            }
        )
        # One learned embedding per stream: left/right family sharing keeps the
        # projections symmetric while the embeddings keep the sides distinct.
        self.stream_embeddings = nn.Parameter(torch.randn(len(NEF_STREAM_NAMES), self.stream_dim) * 0.02)
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
        self.stream_quantizers = nn.ModuleDict(
            {
                family: MotionFSQ(
                    code_dim=self.stream_dim,
                    num_coordinates=self.family_coordinates[family],
                    num_levels=self.num_levels,
                    scale=fsq_scale,
                    preserve_symmetry=fsq_preserve_symmetry,
                    noise_dropout=fsq_noise_dropout,
                )
                for family in NEF_FAMILIES
            }
        )

        self.variant = NEF_VARIANT
        self.encoder_receptive_field = NEF_ENCODER_RECEPTIVE_FIELD
        self.decoder_receptive_field = NEF_DECODER_RECEPTIVE_FIELD
        self.receptive_field = self.encoder_receptive_field + self.decoder_receptive_field - 1
        self.lookahead_frames = 0

    @property
    def num_coordinates(self) -> int:
        return self.layout.num_coordinates

    def _feature_index(self, stream: str) -> torch.Tensor:
        return getattr(self, f"_{stream}_feature_indices")

    def _validate_motion(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.motion_dim:
            raise ValueError(f"Expected motion [B, T, {self.motion_dim}], got {tuple(x.shape)}")

    def _apply_shared_temporal(self, module: nn.Module, streams: torch.Tensor) -> torch.Tensor:
        expected = (len(NEF_STREAM_NAMES), self.stream_dim)
        if streams.ndim != 4 or streams.shape[2:] != expected:
            raise ValueError(
                f"Expected NEF streams [B, T, {expected[0]}, {expected[1]}], got {tuple(streams.shape)}"
            )
        batch_size, seq_len = streams.shape[:2]
        flat = streams.permute(0, 2, 3, 1).reshape(batch_size * len(NEF_STREAM_NAMES), self.stream_dim, seq_len)
        result = module(flat)
        return result.reshape(batch_size, len(NEF_STREAM_NAMES), self.stream_dim, seq_len).permute(0, 3, 1, 2).contiguous()

    def _encode_streams(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_motion(x)
        projected = []
        for index, stream in enumerate(NEF_STREAM_NAMES):
            family = NEF_STREAM_FAMILY[stream]
            state = self.input_projections[family](x.index_select(-1, self._feature_index(stream)))
            projected.append(state + self.stream_embeddings[index])
        return self._apply_shared_temporal(self.stream_encoder, torch.stack(projected, dim=2))

    def _quantize_stream(
        self, stream: str, state: torch.Tensor, collect_metrics: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        quantizer = self.stream_quantizers[NEF_STREAM_FAMILY[stream]]
        quantized, codes, indices, _, *stats = quantizer(
            state.permute(0, 2, 1).contiguous(),
            collect_stats=collect_metrics,
            collect_sequence_stats=False,
        )
        return quantized.permute(0, 2, 1).contiguous(), codes, indices, tuple(stats)

    def _encode(
        self, x: torch.Tensor, collect_metrics: bool
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, tuple[torch.Tensor, ...]],
    ]:
        encoded = self._encode_streams(x.float())
        embeddings: dict[str, torch.Tensor] = {}
        codes: dict[str, torch.Tensor] = {}
        indices: dict[str, torch.Tensor] = {}
        stats: dict[str, tuple[torch.Tensor, ...]] = {}
        for index, stream in enumerate(NEF_STREAM_NAMES):
            embeddings[stream], codes[stream], indices[stream], stats[stream] = self._quantize_stream(
                stream, encoded[:, :, index], collect_metrics
            )
        return embeddings, codes, indices, stats

    def _decode_embeddings(self, embeddings: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = set(NEF_STREAM_NAMES) - set(embeddings)
        if missing:
            raise ValueError(f"Missing NEF stream embeddings: {sorted(missing)}")
        reference = embeddings[NEF_STREAM_NAMES[0]]
        if reference.ndim != 3 or reference.shape[-1] != self.stream_dim:
            raise ValueError(f"NEF embeddings must have shape [B, T, {self.stream_dim}]")
        if any(embeddings[stream].shape != reference.shape for stream in NEF_STREAM_NAMES):
            raise ValueError("All NEF stream embeddings must have the same [B, T, C] shape")

        decoded = torch.stack([embeddings[stream] for stream in NEF_STREAM_NAMES], dim=2)
        decoded = decoded + self.stream_embeddings.view(1, 1, len(NEF_STREAM_NAMES), self.stream_dim)
        decoded = self._apply_shared_temporal(self.stream_decoder, decoded)

        recon = reference.new_zeros((*reference.shape[:2], self.motion_dim))
        for index, stream in enumerate(NEF_STREAM_NAMES):
            recon[..., self._feature_index(stream)] = self.output_heads[NEF_STREAM_FAMILY[stream]](
                decoded[:, :, index]
            )
        return recon

    def _validate_code_tensor(self, values: torch.Tensor, name: str) -> None:
        if values.ndim != 3 or values.shape[-1] != self.num_coordinates:
            raise ValueError(f"Expected NEF {name} [B, T, {self.num_coordinates}], got {tuple(values.shape)}")

    def _decode_indices_to_embeddings(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_code_tensor(indices, "indices")
        slices = self.layout.stream_slices
        return {
            stream: self.stream_quantizers[NEF_STREAM_FAMILY[stream]]
            .dequantize(indices[..., slices[stream]])
            .permute(0, 2, 1)
            .contiguous()
            for stream in NEF_STREAM_NAMES
        }

    def _decode_codes_to_embeddings(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_code_tensor(codes, "codes")
        slices = self.layout.stream_slices
        return {
            stream: self.stream_quantizers[NEF_STREAM_FAMILY[stream]]
            .project_codes_to_latent(codes[..., slices[stream]])
            .permute(0, 2, 1)
            .contiguous()
            for stream in NEF_STREAM_NAMES
        }

    def _metrics(
        self,
        indices_by_stream: Mapping[str, torch.Tensor],
        stats_by_stream: Mapping[str, tuple[torch.Tensor, ...]],
    ) -> dict[str, torch.Tensor]:
        weights = torch.tensor(
            [NEF_STREAM_COORDINATES[stream] for stream in NEF_STREAM_NAMES],
            dtype=torch.float32,
            device=next(iter(indices_by_stream.values())).device,
        )
        metrics: dict[str, torch.Tensor] = {}
        metric_names = (
            "level_perplexity",
            "level_usage",
            "level_perplexity_min",
            "level_perplexity_max",
            "level_usage_min",
            "level_usage_max",
        )
        for metric_index, metric_name in enumerate(metric_names):
            values = torch.stack([stats_by_stream[stream][metric_index] for stream in NEF_STREAM_NAMES])
            metrics[metric_name] = (values * weights).sum() / weights.sum()
        flat_indices = torch.cat([indices_by_stream[stream] for stream in NEF_STREAM_NAMES], dim=-1)
        with torch.no_grad():
            tuples = flat_indices.reshape(-1, self.num_coordinates)
            tuple_unique_ratio = flat_indices.new_tensor(
                torch.unique(tuples, dim=0).shape[0] / max(tuples.shape[0], 1), dtype=torch.float32
            )
            if flat_indices.shape[1] < 2:
                tuple_change_rate = tuple_unique_ratio.new_zeros(())
                coordinate_change_rate = tuple_unique_ratio.new_zeros(())
                stream_changes = tuple_unique_ratio.new_zeros((len(NEF_STREAM_NAMES),))
            else:
                changes = flat_indices[:, 1:] != flat_indices[:, :-1]
                tuple_change_rate = changes.any(dim=-1).float().mean()
                coordinate_change_rate = changes.float().mean()
                stream_changes = torch.stack(
                    [
                        (indices_by_stream[stream][:, 1:] != indices_by_stream[stream][:, :-1]).float().mean()
                        for stream in NEF_STREAM_NAMES
                    ]
                )
        metrics.update(
            {
                "tuple_unique_ratio": tuple_unique_ratio,
                "tuple_change_rate": tuple_change_rate,
                "coordinate_change_rate": coordinate_change_rate,
                "stream_coordinate_change_rates": stream_changes,
            }
        )
        return metrics

    def forward(
        self, x: torch.Tensor, collect_metrics: bool = False
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        embeddings, codes, indices, stats = self._encode(x, collect_metrics=collect_metrics)
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "recon_state": self._decode_embeddings(embeddings),
            "fsq_codes": torch.cat([codes[stream] for stream in NEF_STREAM_NAMES], dim=-1),
            "indices": torch.cat([indices[stream] for stream in NEF_STREAM_NAMES], dim=-1),
            "commit_loss": x.new_zeros(()),
            "stream_codes": codes,
            "stream_indices": indices,
        }
        output["codes"] = output["fsq_codes"]
        if collect_metrics:
            output.update(self._metrics(indices, stats))
        return output

    def compute_representation_losses(self, output, batch):
        # NEF-FSQ v1 trains on the weighted reconstruction and delta terms only.
        return {}

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        _, _, indices, _ = self._encode(x, collect_metrics=False)
        return torch.cat([indices[stream] for stream in NEF_STREAM_NAMES], dim=-1)

    def encode_to_codes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, codes, indices, _ = self._encode(x, collect_metrics=False)
        return (
            torch.cat([codes[stream] for stream in NEF_STREAM_NAMES], dim=-1),
            torch.cat([indices[stream] for stream in NEF_STREAM_NAMES], dim=-1),
        )

    def encode_to_embeddings(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        embeddings, _, indices, _ = self._encode(x, collect_metrics=False)
        return embeddings, torch.cat([indices[stream] for stream in NEF_STREAM_NAMES], dim=-1)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._decode_embeddings(self._decode_indices_to_embeddings(indices))

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self._decode_embeddings(self._decode_codes_to_embeddings(codes))

    # -- research-facing aliases -------------------------------------------
    # Read-only shortcuts for probes, generators and style operators.  They do
    # not change the persisted contract: the same index tensor in means the same
    # decode out, and ``lengths`` is validated but never used to truncate.
    def get_token_layout(self) -> NEFLayout:
        """The live layout object, without copying or re-validating it."""
        return self.layout

    @staticmethod
    def _validate_lengths(lengths: torch.Tensor | Sequence[int] | None, batch: int, frames: int) -> None:
        if lengths is None:
            return
        values = torch.as_tensor(lengths).detach().reshape(-1)
        if values.numel() != batch:
            raise ValueError(f"lengths must have {batch} entries, got {values.numel()}")
        if bool((values <= 0).any()) or bool((values > frames).any()):
            raise ValueError(
                f"lengths must be in [1, {frames}]; padded frames stay in the tensor until a "
                "padding-aware API exists"
            )

    @torch.no_grad()
    def encode_indices(
        self, motion: torch.Tensor, *, lengths: torch.Tensor | Sequence[int] | None = None
    ) -> torch.Tensor:
        self._validate_motion(motion)
        self._validate_lengths(lengths, motion.shape[0], motion.shape[1])
        return self.encode_to_indices(motion)

    @torch.no_grad()
    def decode_indices(
        self, indices: torch.Tensor, *, lengths: torch.Tensor | Sequence[int] | None = None
    ) -> torch.Tensor:
        self._validate_code_tensor(indices, "indices")
        self._validate_lengths(lengths, indices.shape[0], indices.shape[1])
        return self.decode_from_indices(indices)


__all__ = ["NEFMotionAutoencoder", "NEF_DECODER_RECEPTIVE_FIELD", "NEF_ENCODER_RECEPTIVE_FIELD"]
