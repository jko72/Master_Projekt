"""Temporal SalsaNext MOS using the same range-view SiamWCA as T-MAE."""

from __future__ import annotations

import torch

from models.range_view_siamwca import RangeViewSiamWCA
from .salsanext_mos import SalsaNextMOS


class SalsaNextTemporalMOS(SalsaNextMOS):
    def __init__(self, in_channels: int = 4, num_classes: int = 2,
                 dropout: float = 0.2, cross_attention_heads: int = 8):
        if in_channels != 4:
            raise ValueError("Temporal MOS uses [x,y,z,range] without normals or intensity")
        super().__init__(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
        self.backbone = RangeViewSiamWCA(in_channels, dropout, cross_attention_heads)
        self.encoder = self.backbone.encoder
        self.cross_attention = self.backbone.cross_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 2 or x.shape[2] != self.in_channels:
            raise ValueError("SalsaNextTemporalMOS expects [B,2,4,H,W]")
        bottleneck, skips = self.backbone(x)
        return self.decoder(bottleneck, skips)

    def load_pretrained_backbone(self, encoder_state_dict=None, decoder_state_dict=None,
                                 cross_attention_state_dict=None, **kwargs):
        result = super().load_pretrained_backbone(
            encoder_state_dict=encoder_state_dict,
            decoder_state_dict=decoder_state_dict,
            **kwargs,
        )
        if cross_attention_state_dict is None:
            raise ValueError("Temporal MOS needs cross_attention_state_dict from the T-MAE checkpoint")
        self.cross_attention.load_state_dict(cross_attention_state_dict, strict=True)
        result["cross_attention_loaded"] = True
        return result
