import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalResConv1DBlock(nn.Module):
    def __init__(self, channels, dilation=1, activation="relu", norm=None):
        super().__init__()
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(32, channels)
            self.norm2 = nn.GroupNorm(32, channels)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(channels)
            self.norm2 = nn.BatchNorm1d(channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            self.activation1 = nn.SiLU()
            self.activation2 = nn.SiLU()

        self.left_padding = (3 - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)

    def _apply_norm_act(self, x, norm, activation):
        if self.norm == "LN":
            x = norm(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = norm(x)
        return activation(x)

    def forward(self, x):
        residual = x
        x = self._apply_norm_act(x, self.norm1, self.activation1)
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv1(x)
        x = self._apply_norm_act(x, self.norm2, self.activation2)
        x = self.conv2(x)
        return x + residual


class CausalResnet1D(nn.Module):
    def __init__(self, channels, depth, dilation_growth_rate=1, reverse_dilation=False, activation="relu", norm=None):
        super().__init__()
        blocks = [
            CausalResConv1DBlock(
                channels=channels,
                dilation=dilation_growth_rate ** i,
                activation=activation,
                norm=norm,
            )
            for i in range(depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
