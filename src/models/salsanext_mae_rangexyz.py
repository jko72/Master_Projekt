"""SalsaNext masked autoencoder for four-channel RangeXYZ views."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from mos_models.salsanext_parts import SalsaNextDecoder, SalsaNextEncoder


class SalsaNextMAERangeXYZ(nn.Module):
    """Reconstruct ``[x,y,z,range]`` without adding a mask input channel."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model_params", {}) or {}
        in_channels = int(model_cfg.get("grid_channels", 4))
        if in_channels != 4:
            raise ValueError(f"MAE-RangeXYZ requires model_params.grid_channels=4, got {in_channels}.")
        dropout = float(model_cfg.get("dropout_prob", model_cfg.get("dropout", 0.2)))
        self.in_channels = 4
        self.out_channels = 4
        self.encoder = SalsaNextEncoder(in_channels=4, dropout=dropout)
        self.decoder = SalsaNextDecoder(num_classes=4, dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 4, 1, 1))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"Expected x=[B,4,H,W], got {tuple(x.shape)}")
        if (
            mask.ndim != 4
            or mask.shape[1] != 1
            or mask.shape[0] != x.shape[0]
            or mask.shape[-2:] != x.shape[-2:]
        ):
            raise ValueError(f"Expected mask=[B,1,H,W] matching x, got x={tuple(x.shape)}, mask={tuple(mask.shape)}")
        mask = mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        x_masked = x * (1.0 - mask) + self.mask_token.to(dtype=x.dtype) * mask
        bottleneck, skips = self.encoder(x_masked)
        return self.decoder(bottleneck, skips)

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()

    def get_decoder_state_dict(self):
        return self.decoder.state_dict()

    def get_backbone_state_dict(self):
        return {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict()}

    def load_encoder_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def load_decoder_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        return self.decoder.load_state_dict(state_dict, strict=strict)
