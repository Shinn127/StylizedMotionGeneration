import torch.nn as nn

from models.causal_cnn import CausalDecoder1D, CausalEncoder1D, FrameCausalDecoder1D, FrameCausalEncoder1D
from models.quantizer import MultiHeadEMAVQ


class CausalMotionVQVAE(nn.Module):
    def __init__(
        self,
        motion_dim=230,
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
        model_type="causal_cnn",
    ):
        super().__init__()
        if model_type not in {"causal_cnn", "frame_causal_cnn"}:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model_type = model_type
        self.motion_dim = motion_dim
        self.upsample_factor = 1 if model_type == "frame_causal_cnn" else 2 ** down_t

        if model_type == "frame_causal_cnn":
            self.encoder = FrameCausalEncoder1D(
                input_dim=motion_dim,
                code_dim=code_dim,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation,
                norm=norm,
            )
        else:
            self.encoder = CausalEncoder1D(
                input_dim=motion_dim,
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
        if model_type == "frame_causal_cnn":
            self.decoder = FrameCausalDecoder1D(
                output_dim=motion_dim,
                code_dim=code_dim,
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

    def forward(self, x):
        z_e = self.encoder(self._encode_input(x))
        z_q, indices, commit_loss, mean_head_perplexity = self.quantizer(z_e)
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

    def decode_from_indices(self, indices):
        z_q = self.quantizer.dequantize(indices).permute(0, 2, 1).contiguous()
        recon = self.decoder(z_q)
        return self._decode_output(recon)

    def decode_from_embeddings(self, z_q):
        recon = self.decoder(z_q)
        return self._decode_output(recon)
