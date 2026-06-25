import torch
import torch.nn as nn

from vector_quantize_pytorch import VectorQuantize


class MultiHeadEMAVQ(nn.Module):
    def __init__(
        self,
        code_dim,
        num_heads,
        codebook_size,
        decay=0.99,
        kmeans_init=True,
        threshold_ema_dead_code=1,
    ):
        super().__init__()
        if code_dim % num_heads != 0:
            raise ValueError(f"code_dim={code_dim} must be divisible by num_heads={num_heads}")
        self.code_dim = code_dim
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.head_dim = code_dim // num_heads
        self.vq = VectorQuantize(
            dim=code_dim,
            codebook_dim=self.head_dim,
            heads=num_heads,
            separate_codebook_per_head=True,
            codebook_size=codebook_size,
            accept_image_fmap=False,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            kmeans_init=kmeans_init,
        )

    def forward(self, z):
        z_bt = z.permute(0, 2, 1).contiguous()
        quantized, indices, commit_loss = self.vq(z_bt)
        avg_probs = torch.stack(
            [
                torch.bincount(indices[:, :, i].reshape(-1), minlength=self.codebook_size).float()
                for i in range(self.num_heads)
            ],
            dim=0,
        )
        avg_probs = avg_probs / avg_probs.sum(dim=-1, keepdim=True).clamp_min(1e-7)
        mean_head_perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-7)).sum(dim=-1)).mean()
        return quantized.permute(0, 2, 1).contiguous(), indices, commit_loss.mean(), mean_head_perplexity

    def dequantize(self, indices):
        codes = self.vq.get_codes_from_indices(indices)
        return codes.reshape(indices.shape[0], indices.shape[1], self.code_dim)
