import torch
import torch.nn as nn

from models.causal_cnn import CausalDecoder1D, CausalEncoder1D, CondCausalDecoder1D
from models.quantizer import MultiHeadEMAVQ


class CausalMotionVQVAE(nn.Module):
    def __init__(
        self,
        motion_dim=230,
        root_cond_dim=6,
        use_root_cond=True,
        code_dim=256,
        codebook_size=512,
        num_heads=8,
        decay=0.99,
        down_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.root_cond_dim = root_cond_dim
        self.use_root_cond = use_root_cond
        self.upsample_factor = 2 ** down_t

        encoder_input_dim = motion_dim
        decoder_cond_dim = root_cond_dim if use_root_cond else 0

        self.encoder = CausalEncoder1D(
            input_dim=encoder_input_dim,
            code_dim=code_dim,
            down_t=down_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.quantizer = MultiHeadEMAVQ(
            code_dim=code_dim,
            num_heads=num_heads,
            codebook_size=codebook_size,
            decay=decay,
        )
        if use_root_cond:
            self.decoder = CondCausalDecoder1D(
                output_dim=motion_dim,
                code_dim=code_dim,
                cond_dim=decoder_cond_dim,
                down_t=down_t,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
        else:
            self.decoder = CausalDecoder1D(
                output_dim=motion_dim,
                code_dim=code_dim,
                down_t=down_t,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

    def _encode_input(self, x):
        return x.permute(0, 2, 1).float()

    def _decode_output(self, x):
        return x.permute(0, 2, 1).contiguous()

    def _validate_root_cond(self, root_cond, expected_batch, expected_frames):
        if not self.use_root_cond:
            return
        if root_cond is None:
            raise ValueError("root_cond is required when use_root_cond=True")
        if root_cond.ndim != 3:
            raise ValueError(f"root_cond must be rank-3, got shape {tuple(root_cond.shape)}")
        if root_cond.shape[0] != expected_batch or root_cond.shape[1] != expected_frames or root_cond.shape[2] != self.root_cond_dim:
            raise ValueError(
                f"root_cond shape must be ({expected_batch}, {expected_frames}, {self.root_cond_dim}), "
                f"got {tuple(root_cond.shape)}"
            )

    def forward(self, x, root_cond=None):
        if self.use_root_cond:
            self._validate_root_cond(root_cond, x.shape[0], x.shape[1])
        z_e = self.encoder(self._encode_input(x))
        z_q, indices, commit_loss, mean_head_perplexity = self.quantizer(z_e)
        if self.use_root_cond:
            recon = self.decoder(z_q, root_cond)
        else:
            recon = self.decoder(z_q)
        recon = self._decode_output(recon)
        return {
            "recon_state": recon,
            "indices": indices,
            "commit_loss": commit_loss,
            "mean_head_perplexity": mean_head_perplexity,
        }

    def encode_to_indices(self, x):
        z_e = self.encoder(self._encode_input(x))
        _, indices, _, _ = self.quantizer(z_e)
        return indices

    def encode_to_embeddings(self, x):
        z_e = self.encoder(self._encode_input(x))
        z_q, indices, _, _ = self.quantizer(z_e)
        return z_q, indices

    def decode_from_indices(self, indices, root_cond=None):
        z_q = self.quantizer.dequantize(indices).permute(0, 2, 1).contiguous()
        if self.use_root_cond:
            self._validate_root_cond(root_cond, indices.shape[0], indices.shape[1] * self.upsample_factor)
            recon = self.decoder(z_q, root_cond)
        else:
            recon = self.decoder(z_q)
        return self._decode_output(recon)

    def decode_from_embeddings(self, z_q, root_cond=None):
        if self.use_root_cond:
            self._validate_root_cond(root_cond, z_q.shape[0], z_q.shape[-1] * self.upsample_factor)
            recon = self.decoder(z_q, root_cond)
        else:
            recon = self.decoder(z_q)
        return self._decode_output(recon)
