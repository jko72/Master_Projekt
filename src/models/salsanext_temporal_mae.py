"""T-MAE-style temporal pretraining in the existing 2D LiDAR range view."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.range_view_siamwca import RangeViewSiamWCA
from mos_models.salsanext_parts import SalsaNextDecoder


class SalsaNextTemporalMAE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get("model_params", {})
        self.in_channels = int(model_cfg.get("grid_channels", 5))
        if int(model_cfg.get("input_horizon", 2)) != 2:
            raise ValueError("T-MAE uses exactly one previous and one current scan")
        self.out_channels = int(model_cfg.get("output_channels", 4))
        if self.in_channels != 5 or self.out_channels != 4:
            raise ValueError("T-MAE requires five input channels and four geometric reconstruction channels")
        dropout = float(model_cfg.get("dropout_prob", 0.2))
        heads = int(model_cfg.get("cross_attention_heads", 8))
        self.backbone = RangeViewSiamWCA(self.in_channels, dropout, heads)
        self.encoder = self.backbone.encoder
        self.cross_attention = self.backbone.cross_attention
        self.decoder = SalsaNextDecoder(num_classes=self.out_channels, dropout=dropout)

    def forward(self, masked_hist_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if masked_hist_features.ndim != 5 or masked_hist_features.shape[1] != 2:
            raise ValueError("Expected masked pair [B,2,C,H,W]")
        if masked_hist_features.shape[2] != self.in_channels:
            raise ValueError("Input channel count does not match grid_channels")
        if mask.shape != masked_hist_features[:, 1, :1].shape:
            raise ValueError("mask must have shape [B,1,H,W]")
        pair = masked_hist_features.clone()
        pair[:, 1] = pair[:, 1].masked_fill(mask.bool(), 0.0)
        visible = (pair[:, :, :3].abs().sum(dim=2, keepdim=True) > 0).float()
        bottleneck, skips = self.backbone(pair, visible)
        return self.decoder(bottleneck, skips)

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()

    def get_decoder_state_dict(self):
        return self.decoder.state_dict()

    def get_cross_attention_state_dict(self):
        return self.cross_attention.state_dict()

    def get_backbone_state_dict(self):
        return {
            "encoder": self.get_encoder_state_dict(),
            "cross_attention": self.get_cross_attention_state_dict(),
            "decoder": self.get_decoder_state_dict(),
        }
