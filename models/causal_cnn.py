import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import CausalResnet1D


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalEncoder1D(nn.Module):
    def __init__(
        self,
        input_dim,
        code_dim,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = [CausalConv1d(input_dim, width, 3, 1, 1), nn.ReLU()]
        filter_t = stride_t * 2
        for _ in range(down_t):
            blocks.append(
                nn.Sequential(
                    CausalConv1d(width, width, filter_t, stride_t, 1),
                    CausalResnet1D(
                        channels=width,
                        depth=depth,
                        dilation_growth_rate=dilation_growth_rate,
                        activation=activation,
                        norm=norm,
                    ),
                )
            )
        blocks.append(CausalConv1d(width, code_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class CausalDecoder1D(nn.Module):
    def __init__(
        self,
        output_dim,
        code_dim,
        down_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = [CausalConv1d(code_dim, width, 3, 1, 1), nn.ReLU()]
        for _ in range(down_t):
            blocks.append(
                nn.Sequential(
                    CausalResnet1D(
                        channels=width,
                        depth=depth,
                        dilation_growth_rate=dilation_growth_rate,
                        reverse_dilation=True,
                        activation=activation,
                        norm=norm,
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    CausalConv1d(width, width, 3, 1, 1),
                )
            )
        blocks.extend([CausalConv1d(width, width, 3, 1, 1), nn.ReLU(), CausalConv1d(width, output_dim, 3, 1, 1)])
        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        return self.model(z)


class CondCausalDecoder1D(nn.Module):
    def __init__(
        self,
        output_dim,
        code_dim,
        cond_dim,
        down_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self.down_t = down_t
        self.input = nn.Sequential(CausalConv1d(code_dim, width, 3, 1, 1), nn.ReLU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    CausalResnet1D(
                        channels=width,
                        depth=depth,
                        dilation_growth_rate=dilation_growth_rate,
                        reverse_dilation=True,
                        activation=activation,
                        norm=norm,
                    ),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    CausalConv1d(width, width, 3, 1, 1),
                )
                for _ in range(down_t)
            ]
        )
        self.cond_fusions = nn.ModuleList([nn.Sequential(nn.Linear(width + cond_dim, width), nn.ReLU()) for _ in range(down_t + 1)])
        self.out = nn.Sequential(CausalConv1d(width, width, 3, 1, 1), nn.ReLU(), CausalConv1d(width, output_dim, 3, 1, 1))

    def _fuse_cond(self, h, cond, fusion):
        cond_resized = F.interpolate(cond.transpose(1, 2), size=h.shape[-1], mode="nearest").transpose(1, 2)
        h = h.transpose(1, 2)
        h = fusion(torch.cat([h, cond_resized], dim=-1))
        return h.transpose(1, 2)

    def forward(self, z, cond):
        h = self.input(z)
        h = self._fuse_cond(h, cond, self.cond_fusions[0])
        for i, block in enumerate(self.blocks):
            h = block(h)
            h = self._fuse_cond(h, cond, self.cond_fusions[i + 1])
        return self.out(h)
