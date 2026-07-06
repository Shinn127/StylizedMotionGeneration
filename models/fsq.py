import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ

from models.causal_cnn import FrameCausalDecoder1D, FrameCausalEncoder1D


class MotionFSQ(nn.Module):
    def __init__(
        self,
        code_dim: int,
        num_latent_tokens: int = 40,
        num_levels: int = 9,
        scale: float | None = None,
        preserve_symmetry: bool = False,
        noise_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_latent_tokens <= 0:
            raise ValueError(f"num_latent_tokens must be positive, got {num_latent_tokens}")
        if num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")

        self.code_dim = int(code_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        self.num_levels = int(num_levels)
        self.fsq = FSQ(
            levels=[self.num_levels],
            dim=self.code_dim,
            num_codebooks=self.num_latent_tokens,
            keep_num_codebooks_dim=True,
            scale=scale,
            preserve_symmetry=preserve_symmetry,
            noise_dropout=noise_dropout,
        )

    def _usage_stats(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.stack(
            [
                torch.bincount(indices[:, :, token].reshape(-1), minlength=self.num_levels).float()
                for token in range(self.num_latent_tokens)
            ],
            dim=0,
        )
        probs = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-7)).sum(dim=-1)).mean()
        usage = (counts > 0).float().mean()
        return perplexity, usage

    def forward(self, z: torch.Tensor):
        z_bt = z.permute(0, 2, 1).contiguous()
        quantized, indices = self.fsq(z_bt)
        indices = indices.long()
        level_perplexity, level_usage = self._usage_stats(indices)
        commit_loss = quantized.new_zeros(())
        return (
            quantized.permute(0, 2, 1).contiguous(),
            indices,
            commit_loss,
            level_perplexity,
            level_usage,
        )

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.fsq.indices_to_codes(indices.long())

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.fsq.indices_to_level_indices(indices.long())


class FSQMotionAutoencoder(nn.Module):
    def __init__(
        self,
        motion_dim: int = 230,
        code_dim: int = 256,
        width: int = 512,
        depth: int = 6,
        dilation_growth_rate: int = 2,
        activation: str = "relu",
        norm: str | None = None,
        num_latent_tokens: int = 40,
        num_levels: int = 9,
        fsq_scale: float | None = None,
        fsq_preserve_symmetry: bool = False,
        fsq_noise_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.code_dim = int(code_dim)
        self.model_family = "fsq"
        self.encoder = FrameCausalEncoder1D(
            input_dim=motion_dim,
            code_dim=code_dim,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.quantizer = MotionFSQ(
            code_dim=code_dim,
            num_latent_tokens=num_latent_tokens,
            num_levels=num_levels,
            scale=fsq_scale,
            preserve_symmetry=fsq_preserve_symmetry,
            noise_dropout=fsq_noise_dropout,
        )
        self.decoder = FrameCausalDecoder1D(
            output_dim=motion_dim,
            code_dim=code_dim,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).float()

    def _decode_output(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encoder(self._encode_input(x))
        z_q, indices, commit_loss, level_perplexity, level_usage = self.quantizer(z_e)
        recon = self._decode_output(self.decoder(z_q))
        return {
            "recon_state": recon,
            "indices": indices,
            "commit_loss": commit_loss,
            "level_perplexity": level_perplexity,
            "level_usage": level_usage,
        }

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(self._encode_input(x))
        _, indices, _, _, _ = self.quantizer(z_e)
        return indices

    def encode_to_embeddings(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(self._encode_input(x))
        z_q, indices, _, _, _ = self.quantizer(z_e)
        return z_q, indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.dequantize(indices).permute(0, 2, 1).contiguous()
        return self._decode_output(self.decoder(z_q))

    def decode_from_embeddings(self, z_q: torch.Tensor) -> torch.Tensor:
        return self._decode_output(self.decoder(z_q))
