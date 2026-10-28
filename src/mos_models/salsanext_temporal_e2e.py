"""Shared temporal E2E architecture for MOS and masked five-channel reconstruction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.range_view_local_temporal_attention import LocalHistoricalAttention
from .salsanext_parts import SalsaNextDecoder, SalsaNextEncoder


class SalsaNextTemporalE2E(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, dropout=0.1,
                 attention_heads=8, radius=1, compare_channels=64):
        super().__init__()
        if in_channels not in {4, 5} or num_classes not in {2, 5} or (num_classes == 5 and in_channels != 5):
            raise ValueError("Temporal E2E requires XYZR/XYZRI MOS (2 outputs) or XYZRI reconstruction (5 outputs)")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder = SalsaNextEncoder(in_channels, dropout)
        self.temporal = LocalHistoricalAttention(256, attention_heads, radius, compare_channels)
        self.decoder = SalsaNextDecoder(num_classes, dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, dt: torch.Tensor,
                return_diagnostics: bool = False):
        if x.ndim != 5 or x.shape[1] < 2 or x.shape[2] != self.in_channels:
            raise ValueError(f"x must be [B,T>=2,{self.in_channels},H,W]")
        b, t, c, h, w = x.shape
        if h % 16 or w % 16:
            raise ValueError("SalsaNext requires H and W divisible by 16")
        if valid.shape == (b, t, 1, h, w):
            valid = valid[:, :, 0]
        if valid.shape != (b, t, h, w) or dt.shape != (b, t):
            raise ValueError("valid must be [B,T,H,W] and dt [B,T]")
        if not torch.isfinite(dt).all() or not (dt[:, :-1] < 0).all() or not (dt[:, -1] == 0).all():
            raise ValueError("Past dt must be negative seconds and current dt zero")
        encoded, skips = self.encoder(x.reshape(b * t, c, h, w))
        _, f, fh, fw = encoded.shape
        encoded = encoded.reshape(b, t, f, fh, fw)
        scaled_valid = F.adaptive_max_pool2d(valid.reshape(b * t, 1, h, w).float(), (fh, fw)) > 0.5
        scaled_valid = scaled_valid.reshape(b, t, 1, fh, fw)
        result = self.temporal(encoded[:, -1], encoded[:, :-1], scaled_valid[:, -1],
                               scaled_valid[:, :-1], dt[:, :-1], return_diagnostics)
        if return_diagnostics:
            fused, diagnostics = result
        else:
            fused = result
        current_skips = tuple(s.reshape(b, t, *s.shape[1:])[:, -1] for s in skips)
        logits = self.decoder(fused, current_skips)
        if logits.shape != (b, self.num_classes, h, w):
            raise RuntimeError(f"Unexpected decoder output {tuple(logits.shape)}")
        return (logits, diagnostics) if return_diagnostics else logits
