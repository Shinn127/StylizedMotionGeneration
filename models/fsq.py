import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from models.causal_cnn import FrameCausalDecoder1D, FrameCausalEncoder1D


class MotionFSQ(nn.Module):
    def __init__(
        self,
        code_dim: int,
        num_coordinates: int = 20,
        num_levels: int = 9,
        scale: float | None = None,
        preserve_symmetry: bool = False,
        noise_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_coordinates <= 0:
            raise ValueError(f"num_coordinates must be positive, got {num_coordinates}")
        if num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")

        self.code_dim = int(code_dim)
        self.num_coordinates = int(num_coordinates)
        self.num_levels = int(num_levels)
        self.fsq = FSQ(
            levels=[self.num_levels],
            dim=self.code_dim,
            num_codebooks=self.num_coordinates,
            keep_num_codebooks_dim=True,
            scale=scale,
            preserve_symmetry=preserve_symmetry,
            noise_dropout=noise_dropout,
        )

    def _usage_stats(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        counts = torch.stack(
            [
                torch.bincount(indices[:, :, token].reshape(-1), minlength=self.num_levels).float()
                for token in range(self.num_coordinates)
            ],
            dim=0,
        )
        probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        per_coordinate_perplexity = torch.exp(-(probs * torch.log(probs + 1e-7)).sum(dim=-1))
        per_coordinate_usage = (counts > 0).float().mean(dim=-1)
        return (
            per_coordinate_perplexity.mean(),
            per_coordinate_usage.mean(),
            per_coordinate_perplexity.min(),
            per_coordinate_perplexity.max(),
            per_coordinate_usage.min(),
            per_coordinate_usage.max(),
        )

    def _sequence_stats(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summarizes reuse and temporal variation of complete FSQ coordinate tuples."""
        with torch.no_grad():
            flat_indices = indices.reshape(-1, self.num_coordinates)
            tuple_unique_ratio = indices.new_tensor(
                torch.unique(flat_indices, dim=0).shape[0] / max(flat_indices.shape[0], 1), dtype=torch.float32
            )
            if indices.shape[1] < 2:
                tuple_change_rate = tuple_unique_ratio.new_zeros(())
                coordinate_change_rate = tuple_unique_ratio.new_zeros(())
            else:
                changes = indices[:, 1:] != indices[:, :-1]
                tuple_change_rate = changes.any(dim=-1).float().mean()
                coordinate_change_rate = changes.float().mean()
        return tuple_unique_ratio, tuple_change_rate, coordinate_change_rate

    def project_to_coordinates(self, z: torch.Tensor) -> torch.Tensor:
        """Projects [B, C, T] encoder latents to continuous FSQ coordinates."""
        z_bt = z.permute(0, 2, 1).contiguous()
        return self.fsq.project_in(z_bt)

    def quantize_coordinates(self, projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantizes [B, T, K] coordinates with FSQ straight-through gradients."""
        if projected.ndim != 3 or projected.shape[-1] != self.num_coordinates:
            raise ValueError(
                "Expected projected coordinates with shape [B, T, "
                f"{self.num_coordinates}], got {tuple(projected.shape)}"
            )
        codes_4d = self.fsq.quantize(projected.unsqueeze(-1))
        indices = self.fsq.codes_to_indices(codes_4d).long()
        return codes_4d.squeeze(-1), indices

    def project_codes_to_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Maps quantized FSQ coordinates [B, T, K] back to [B, C, T]."""
        if codes.ndim != 3 or codes.shape[-1] != self.num_coordinates:
            raise ValueError(
                "Expected quantized codes with shape [B, T, "
                f"{self.num_coordinates}], got {tuple(codes.shape)}"
            )
        return self.fsq.project_out(codes).permute(0, 2, 1).contiguous()

    def forward(self, z: torch.Tensor):
        projected = self.project_to_coordinates(z)
        codes, indices = self.quantize_coordinates(projected)
        quantized = self.project_codes_to_latent(codes)
        (
            level_perplexity,
            level_usage,
            level_perplexity_min,
            level_perplexity_max,
            level_usage_min,
            level_usage_max,
        ) = self._usage_stats(indices)
        tuple_unique_ratio, tuple_change_rate, coordinate_change_rate = self._sequence_stats(indices)
        commit_loss = quantized.new_zeros(())
        return (
            quantized,
            codes,
            indices,
            commit_loss,
            level_perplexity,
            level_usage,
            level_perplexity_min,
            level_perplexity_max,
            level_usage_min,
            level_usage_max,
            tuple_unique_ratio,
            tuple_change_rate,
            coordinate_change_rate,
        )

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.fsq.indices_to_codes(indices.long()).permute(0, 2, 1).contiguous()

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.fsq.indices_to_level_indices(indices.long())


class FSQMotionAutoencoder(nn.Module):
    def __init__(
        self,
        motion_dim: int = 230,
        code_dim: int = 256,
        width: int = 512,
        activation: str = "relu",
        norm: str | None = None,
        num_coordinates: int = 20,
        num_levels: int = 9,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = {key: value for key, value in locals().items() if key not in {"self", "__class__"}}
        self.motion_dim = int(motion_dim)
        self.code_dim = int(code_dim)
        self.encoder = FrameCausalEncoder1D(
            input_dim=motion_dim,
            code_dim=code_dim,
            width=width,
            activation=activation,
            norm=norm,
        )
        self.quantizer = MotionFSQ(
            code_dim=code_dim,
            num_coordinates=num_coordinates,
            num_levels=num_levels,
            scale=fsq_scale,
            preserve_symmetry=fsq_preserve_symmetry,
            noise_dropout=fsq_noise_dropout,
        )
        self.decoder = FrameCausalDecoder1D(
            output_dim=motion_dim,
            code_dim=code_dim,
            width=width,
            activation=activation,
            norm=norm,
        )
        self.receptive_field, self.context_left, self.lookahead_frames = 64, 63, 0

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float()

    def _decode_output(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encoder(self._encode_input(x))
        (
            z_q,
            codes,
            indices,
            commit_loss,
            level_perplexity,
            level_usage,
            level_perplexity_min,
            level_perplexity_max,
            level_usage_min,
            level_usage_max,
            tuple_unique_ratio,
            tuple_change_rate,
            coordinate_change_rate,
        ) = self.quantizer(z_e)
        recon = self._decode_output(self.decoder(z_q))
        return {
            "recon_state": recon,
            "fsq_codes": codes,
            "indices": indices,
            "commit_loss": commit_loss,
            "level_perplexity": level_perplexity,
            "level_usage": level_usage,
            "level_perplexity_min": level_perplexity_min,
            "level_perplexity_max": level_perplexity_max,
            "level_usage_min": level_usage_min,
            "level_usage_max": level_usage_max,
            "tuple_unique_ratio": tuple_unique_ratio,
            "tuple_change_rate": tuple_change_rate,
            "coordinate_change_rate": coordinate_change_rate,
        }

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(self._encode_input(x))
        _, _, indices, *_ = self.quantizer(z_e)
        return indices

    def encode_to_codes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(self._encode_input(x))
        _, codes, indices, *_ = self.quantizer(z_e)
        return codes, indices

    def encode_to_embeddings(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(self._encode_input(x))
        z_q, _, indices, *_ = self.quantizer(z_e)
        return z_q, indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.dequantize(indices)
        return self._decode_output(self.decoder(z_q))

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.project_codes_to_latent(codes)
        return self._decode_output(self.decoder(z_q))

    def decode_from_embeddings(self, z_q: torch.Tensor) -> torch.Tensor:
        return self._decode_output(self.decoder(z_q))
