from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    norm = str(norm).lower()
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "group":
        groups = min(8, num_channels)
        while groups > 1 and (num_channels % groups) != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm='{norm}'. Use one of ['batch', 'group', 'none'].")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: str = "batch",
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        if str(activation).lower() == "silu":
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            act,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            act,
            nn.Dropout2d(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, norm=norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        norm: str = "batch",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, norm=norm, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MOSUNetSmall(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
        norm: str = "batch",
    ):
        super().__init__()
        bc = int(base_channels)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = bc
        self.dropout = float(dropout)
        self.norm = norm

        self.enc1 = ConvBlock(self.in_channels, bc, norm=norm, dropout=dropout)
        self.down1 = DownBlock(bc, bc * 2, norm=norm, dropout=dropout)
        self.down2 = DownBlock(bc * 2, bc * 4, norm=norm, dropout=dropout)

        self.bottleneck = ConvBlock(bc * 4, bc * 8, norm=norm, dropout=dropout)

        self.up2 = UpBlock(bc * 8, bc * 4, bc * 4, norm=norm, dropout=dropout)
        self.up1 = UpBlock(bc * 4, bc * 2, bc * 2, norm=norm, dropout=dropout)
        self.up0 = UpBlock(bc * 2, bc, bc, norm=norm, dropout=dropout)

        self.out_conv = nn.Conv2d(bc, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        xb = self.bottleneck(x3)

        u2 = self.up2(xb, x3)
        u1 = self.up1(u2, x2)
        u0 = self.up0(u1, x1)

        logits = self.out_conv(u0)
        return logits

