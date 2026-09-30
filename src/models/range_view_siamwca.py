"""Shared SalsaNext encoder with local, shifted range-view cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mos_models.salsanext_parts import SalsaNextEncoder


class WindowCrossAttention2D(nn.Module):
    """Update current tokens from previous tokens in local azimuth/elevation windows."""

    def __init__(self, channels: int, heads: int = 8, window_h: int = 4, window_w: int = 8):
        super().__init__()
        if channels % heads:
            raise ValueError("Cross-attention channels must be divisible by heads")
        self.window_h, self.window_w = int(window_h), int(window_w)
        if min(self.window_h, self.window_w) < 1:
            raise ValueError("Cross-attention window dimensions must be positive")
        tokens = self.window_h * self.window_w
        self.position = nn.Parameter(torch.zeros(1, tokens, channels))
        self.absolute_position = nn.Conv2d(3, channels, kernel_size=1)
        nn.init.normal_(self.position, std=0.02)
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(channels, heads, batch_first=True) for _ in range(2)]
        )
        self.norm_attention = nn.ModuleList([nn.LayerNorm(channels) for _ in range(2)])
        self.norm_mlp = nn.ModuleList([nn.LayerNorm(channels) for _ in range(2)])
        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU(), nn.Linear(2 * channels, channels)) for _ in range(2)]
        )

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        batch, channels, height, width = x.shape
        pad_h = (-height) % self.window_h
        pad_w = (-width) % self.window_w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        nh, nw = (height + pad_h) // self.window_h, (width + pad_w) // self.window_w
        x = x.view(batch, channels, nh, self.window_h, nw, self.window_w)
        tokens = x.permute(0, 2, 4, 3, 5, 1).reshape(batch * nh * nw, self.window_h * self.window_w, channels)
        return tokens, (batch, nh, nw, height, width)

    def _restore(self, tokens: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        batch, nh, nw, height, width = shape
        channels = tokens.shape[-1]
        x = tokens.reshape(batch, nh, nw, self.window_h, self.window_w, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, nh * self.window_h, nw * self.window_w)
        return x[:, :, :height, :width]

    def forward(
        self, current: torch.Tensor, previous: torch.Tensor,
        current_valid: torch.Tensor, previous_valid: torch.Tensor,
    ) -> torch.Tensor:
        if current.shape != previous.shape:
            raise ValueError("Current and previous feature maps must have identical shapes")
        if current_valid.shape != previous_valid.shape or current_valid.shape != current[:, :1].shape:
            raise ValueError("Cross-attention validity masks must be [B,1,H,W]")
        height, width = current.shape[-2:]
        row = torch.linspace(-1.0, 1.0, height, device=current.device, dtype=current.dtype)
        azimuth = torch.arange(width, device=current.device, dtype=current.dtype) * (2.0 * torch.pi / width)
        coordinates = torch.stack((
            row[:, None].expand(height, width),
            torch.sin(azimuth)[None, :].expand(height, width),
            torch.cos(azimuth)[None, :].expand(height, width),
        ), dim=0).unsqueeze(0)
        absolute_position = self.absolute_position(coordinates)
        result = current
        for layer, (shift_h, shift_w) in enumerate(((0, 0), (self.window_h // 2, self.window_w // 2))):
            shift = (shift_h, shift_w)
            query_map = torch.roll(result, (-shift_h, -shift_w), dims=(-2, -1)) if any(shift) else result
            key_map = torch.roll(previous, (-shift_h, -shift_w), dims=(-2, -1)) if any(shift) else previous
            query_valid = torch.roll(current_valid, (-shift_h, -shift_w), dims=(-2, -1)) if any(shift) else current_valid
            key_valid = torch.roll(previous_valid, (-shift_h, -shift_w), dims=(-2, -1)) if any(shift) else previous_valid
            pos_map = torch.roll(absolute_position, (-shift_h, -shift_w), dims=(-2, -1)) if any(shift) else absolute_position
            query, shape = self._partition(query_map)
            key, _ = self._partition(key_map)
            pos_tokens, _ = self._partition(pos_map.expand(current.shape[0], -1, -1, -1))
            qmask, _ = self._partition(query_valid.float())
            kmask, _ = self._partition(key_valid.float())
            qmask = qmask[..., 0] > 0.5
            kmask = kmask[..., 0] > 0.5
            has_reference = kmask.any(dim=1)
            # MultiheadAttention needs one unmasked key even for an empty window.
            safe_key_mask = ~kmask.clone()
            safe_key_mask[~has_reference, 0] = False
            q = query + self.position.to(query.dtype) + pos_tokens
            k = key + self.position.to(key.dtype) + pos_tokens
            attention_mask = None
            query_has_reference = has_reference[:, None]
            if shift_h:
                original_rows = torch.arange(height, device=current.device).view(1, 1, height, 1)
                original_rows = original_rows.expand(1, 1, height, width)
                original_rows = torch.roll(original_rows, -shift_h, dims=-2)
                wrapped, _ = self._partition((original_rows < shift_h).float())
                wrapped = wrapped[..., 0] > 0.5
                wrapped = wrapped.repeat(current.shape[0], 1)
                group0 = (kmask & ~wrapped).any(dim=1)
                group1 = (kmask & wrapped).any(dim=1)
                query_has_reference = torch.where(wrapped, group1[:, None], group0[:, None])
                first0 = (~wrapped).int().argmax(dim=1)
                first1 = wrapped.int().argmax(dim=1)
                rows = torch.arange(wrapped.shape[0], device=wrapped.device)
                safe_key_mask[rows[~group0], first0[~group0]] = False
                missing_group1 = (~group1) & wrapped.any(dim=1)
                safe_key_mask[rows[missing_group1], first1[missing_group1]] = False
                blocked = wrapped[:, :, None] != wrapped[:, None, :]
                attention_mask = blocked.repeat_interleave(self.attention[layer].num_heads, dim=0)
            attended, _ = self.attention[layer](
                q, k, key, key_padding_mask=safe_key_mask,
                attn_mask=attention_mask, need_weights=False,
            )
            update = qmask & query_has_reference
            fused = self.norm_attention[layer](query + torch.where(update[..., None], attended, 0.0))
            fused = self.norm_mlp[layer](fused + self.mlp[layer](fused))
            # Empty and masked current tokens are absent from the encoder's sparse analogue.
            fused = torch.where(update[..., None], fused, query)
            result = self._restore(fused, shape)
            if any(shift):
                result = torch.roll(result, shift, dims=(-2, -1))
        return result


class RangeViewSiamWCA(nn.Module):
    """Two shared SalsaNext branches; previous features supply K/V to current Q."""

    def __init__(self, in_channels: int, dropout: float = 0.2, heads: int = 8):
        super().__init__()
        self.encoder = SalsaNextEncoder(in_channels=in_channels, dropout=dropout)
        self.cross_attention = nn.ModuleDict({
            "skip2": WindowCrossAttention2D(256, heads),
            "skip3": WindowCrossAttention2D(256, heads),
            "bottleneck": WindowCrossAttention2D(256, heads),
        })

    @staticmethod
    def _valid_at(valid: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool2d(valid.float(), feature.shape[-2:]) > 0.5

    def forward(self, pair: torch.Tensor, visible: torch.Tensor | None = None):
        if pair.ndim != 5 or pair.shape[1] != 2:
            raise ValueError("RangeViewSiamWCA expects [B,2,C,H,W]")
        if visible is None:
            visible = (pair[:, :, :3].abs().sum(dim=2, keepdim=True) > 0).float()
        if visible.shape != (pair.shape[0], 2, 1, pair.shape[-2], pair.shape[-1]):
            raise ValueError("visible must have shape [B,2,1,H,W]")
        batch, _, channels, height, width = pair.shape
        encoded, skips = self.encoder(pair.reshape(batch * 2, channels, height, width))
        previous, current = encoded.reshape(batch, 2, *encoded.shape[1:]).unbind(dim=1)
        prev_mask = self._valid_at(visible[:, 0], previous)
        curr_mask = self._valid_at(visible[:, 1], current)
        current = self.cross_attention["bottleneck"](
            current * curr_mask, previous * prev_mask, curr_mask, prev_mask
        )
        current_skips = []
        for i, skip in enumerate(skips):
            past, present = skip.reshape(batch, 2, *skip.shape[1:]).unbind(dim=1)
            key = f"skip{i}"
            present_valid = self._valid_at(visible[:, 1], present)
            present = present * present_valid
            if key in self.cross_attention:
                past_valid = self._valid_at(visible[:, 0], past)
                present = self.cross_attention[key](
                    present, past * past_valid, present_valid, past_valid
                )
            current_skips.append(present)
        return current, tuple(current_skips)
