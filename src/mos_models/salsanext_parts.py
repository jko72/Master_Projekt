from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, Sequence, Tuple

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


class SalsaNextEncoder(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.dropout = float(dropout)

        self.down_cntx = ResContextBlock(self.in_channels, 32)
        self.down_cntx2 = ResContextBlock(32, 32)
        self.down_cntx3 = ResContextBlock(32, 32)

        self.res_block1 = ResBlock(32, 2 * 32, self.dropout, pooling=True, drop_out=False)
        self.res_block2 = ResBlock(2 * 32, 2 * 2 * 32, self.dropout, pooling=True)
        self.res_block3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, self.dropout, pooling=True)
        self.res_block4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, self.dropout, pooling=True)
        self.res_block5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, self.dropout, pooling=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        down_cntx = self.down_cntx(x)
        down_cntx = self.down_cntx2(down_cntx)
        down_cntx = self.down_cntx3(down_cntx)

        down0c, down0b = self.res_block1(down_cntx)
        down1c, down1b = self.res_block2(down0c)
        down2c, down2b = self.res_block3(down1c)
        down3c, down3b = self.res_block4(down2c)
        down5c = self.res_block5(down3c)

        skips = (down0b, down1b, down2b, down3b)
        return down5c, skips


class SalsaNextDecoder(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        self.up_block1 = UpBlock(2 * 4 * 32, 4 * 32, self.dropout)
        self.up_block2 = UpBlock(4 * 32, 4 * 32, self.dropout)
        self.up_block3 = UpBlock(4 * 32, 2 * 32, self.dropout)
        self.up_block4 = UpBlock(2 * 32, 32, self.dropout, drop_out=False)
        self.logits = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(skips) != 4:
            raise ValueError(f"SalsaNextDecoder expects 4 skip tensors, got {len(skips)}")
        down0b, down1b, down2b, down3b = skips

        up4e = self.up_block1(bottleneck, down3b)
        up3e = self.up_block2(up4e, down2b)
        up2e = self.up_block3(up3e, down1b)
        up1e = self.up_block4(up2e, down0b)
        logits = self.logits(up1e)
        return logits


def adapt_input_conv_weight(pretrained_weight: torch.Tensor, target_in_channels: int, mode: str = "mean") -> torch.Tensor:
    if pretrained_weight.ndim != 4:
        raise ValueError(
            f"Expected conv weight tensor with 4 dims [out,in,kH,kW], got shape={tuple(pretrained_weight.shape)}"
        )
    if target_in_channels <= 0:
        raise ValueError(f"target_in_channels must be > 0, got {target_in_channels}")

    old_out, old_in, k_h, k_w = pretrained_weight.shape
    if old_in == int(target_in_channels):
        return pretrained_weight

    mode = str(mode).lower()
    if mode not in {"mean", "zero", "random"}:
        raise ValueError(f"Unsupported mode='{mode}'. Use one of: 'mean', 'zero', 'random'.")

    new_weight = pretrained_weight.new_zeros((old_out, int(target_in_channels), k_h, k_w))
    copy_in = min(old_in, int(target_in_channels))
    new_weight[:, :copy_in] = pretrained_weight[:, :copy_in]

    if int(target_in_channels) > old_in:
        if mode == "mean":
            fill = pretrained_weight.mean(dim=1, keepdim=True)
            new_weight[:, old_in:] = fill
        elif mode == "zero":
            new_weight[:, old_in:] = 0.0
        else:  # random
            nn.init.kaiming_normal_(new_weight[:, old_in:], nonlinearity="leaky_relu")

    return new_weight


def _find_conv_key(state_dict: Mapping[str, torch.Tensor], conv_key_hint: str) -> Optional[str]:
    if conv_key_hint in state_dict:
        return conv_key_hint

    suffix_matches = [k for k in state_dict.keys() if k.endswith(conv_key_hint)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        return sorted(suffix_matches, key=len)[0]

    contains_matches = [k for k in state_dict.keys() if conv_key_hint in k]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1:
        return sorted(contains_matches, key=len)[0]

    return None


def adapt_first_conv_in_channels(
    state_dict: Mapping[str, torch.Tensor],
    target_in_channels: int,
    conv_key_hint: str = "down_cntx.conv1.weight",
    mode: str = "mean",
) -> "OrderedDict[str, torch.Tensor]":
    adapted_state = OrderedDict(state_dict.items())

    conv_key = _find_conv_key(adapted_state, conv_key_hint)
    if conv_key is None:
        raise KeyError(
            f"Could not find first encoder conv key with hint '{conv_key_hint}'. "
            f"Available keys example: {list(adapted_state.keys())[:8]}"
        )

    weight = adapted_state[conv_key]
    if not torch.is_tensor(weight):
        raise TypeError(f"Expected tensor for '{conv_key}', got {type(weight)}")

    old_in = int(weight.shape[1]) if weight.ndim == 4 else None
    new_weight = adapt_input_conv_weight(weight, target_in_channels=target_in_channels, mode=mode)
    adapted_state[conv_key] = new_weight

    if old_in is not None and old_in != int(target_in_channels):
        print(
            f"[PRETRAIN] Adapted first encoder conv from {old_in} to {int(target_in_channels)} input channels. "
            f"New channels initialized with mode='{mode}'."
        )

    return adapted_state
