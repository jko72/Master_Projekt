from __future__ import annotations

import torch
import torch.nn as nn


class ResContextBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        res_a = self.conv2(shortcut)
        res_a = self.act2(res_a)
        res_a1 = self.bn1(res_a)

        res_a = self.conv3(res_a1)
        res_a = self.act3(res_a)
        res_a2 = self.bn2(res_a)

        output = shortcut + res_a2
        return output


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        dropout_rate: float,
        kernel_size=(3, 3),
        stride: int = 1,
        pooling: bool = True,
        drop_out: bool = True,
    ):
        super().__init__()
        self.pooling = bool(pooling)
        self.drop_out = bool(drop_out)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout = nn.Dropout2d(p=float(dropout_rate))
        if self.pooling:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        res_a = self.conv2(x)
        res_a = self.act2(res_a)
        res_a1 = self.bn1(res_a)

        res_a = self.conv3(res_a1)
        res_a = self.act3(res_a)
        res_a2 = self.bn2(res_a)

        res_a = self.conv4(res_a2)
        res_a = self.act4(res_a)
        res_a3 = self.bn3(res_a)

        concat = torch.cat((res_a1, res_a2, res_a3), dim=1)
        res_a = self.conv5(concat)
        res_a = self.act5(res_a)
        res_a = self.bn4(res_a)
        res_a = shortcut + res_a

        if self.pooling:
            if self.drop_out:
                res_b = self.dropout(res_a)
            else:
                res_b = res_a
            res_b = self.pool(res_b)
            return res_b, res_a

        if self.drop_out:
            res_b = self.dropout(res_a)
        else:
            res_b = res_a
        return res_b


class UpBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, dropout_rate: float, drop_out: bool = True):
        super().__init__()
        self.drop_out = bool(drop_out)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.dropout1 = nn.Dropout2d(p=float(dropout_rate))
        self.dropout2 = nn.Dropout2d(p=float(dropout_rate))

        self.conv1 = nn.Conv2d(in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=float(dropout_rate))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_a = self.pixel_shuffle(x)
        if self.drop_out:
            up_a = self.dropout1(up_a)

        up_b = torch.cat((up_a, skip), dim=1)
        if self.drop_out:
            up_b = self.dropout2(up_b)

        up_e = self.conv1(up_b)
        up_e = self.act1(up_e)
        up_e1 = self.bn1(up_e)

        up_e = self.conv2(up_e1)
        up_e = self.act2(up_e)
        up_e2 = self.bn2(up_e)

        up_e = self.conv3(up_e2)
        up_e = self.act3(up_e)
        up_e3 = self.bn3(up_e)

        concat = torch.cat((up_e1, up_e2, up_e3), dim=1)
        up_e = self.conv4(concat)
        up_e = self.act4(up_e)
        up_e = self.bn4(up_e)
        if self.drop_out:
            up_e = self.dropout3(up_e)

        return up_e


class SalsaNextMOS(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        self.down_cntx = ResContextBlock(self.in_channels, 32)
        self.down_cntx2 = ResContextBlock(32, 32)
        self.down_cntx3 = ResContextBlock(32, 32)

        self.res_block1 = ResBlock(32, 2 * 32, self.dropout, pooling=True, drop_out=False)
        self.res_block2 = ResBlock(2 * 32, 2 * 2 * 32, self.dropout, pooling=True)
        self.res_block3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, self.dropout, pooling=True)
        self.res_block4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, self.dropout, pooling=True)
        self.res_block5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, self.dropout, pooling=False)

        self.up_block1 = UpBlock(2 * 4 * 32, 4 * 32, self.dropout)
        self.up_block2 = UpBlock(4 * 32, 4 * 32, self.dropout)
        self.up_block3 = UpBlock(4 * 32, 2 * 32, self.dropout)
        self.up_block4 = UpBlock(2 * 32, 32, self.dropout, drop_out=False)

        self.logits = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x[:, -1]
        elif x.ndim != 4:
            raise ValueError(
                f"SalsaNextMOS expects input with 4D [B,C,H,W] or 5D [B,T,C,H,W], got shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SalsaNextMOS expected {self.in_channels} input channels, but got {x.shape[1]} "
                f"for tensor shape={tuple(x.shape)}"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)

        down_cntx = self.down_cntx(x)
        down_cntx = self.down_cntx2(down_cntx)
        down_cntx = self.down_cntx3(down_cntx)

        down0c, down0b = self.res_block1(down_cntx)
        down1c, down1b = self.res_block2(down0c)
        down2c, down2b = self.res_block3(down1c)
        down3c, down3b = self.res_block4(down2c)
        down5c = self.res_block5(down3c)

        up4e = self.up_block1(down5c, down3b)
        up3e = self.up_block2(up4e, down2b)
        up2e = self.up_block3(up3e, down1b)
        up1e = self.up_block4(up2e, down0b)

        logits = self.logits(up1e)
        return logits


if __name__ == "__main__":
    model = SalsaNextMOS(in_channels=2, num_classes=2)
    x = torch.randn(2, 2, 64, 512)
    y = model(x)
    print(y.shape)
    assert y.shape == (2, 2, 64, 512)
