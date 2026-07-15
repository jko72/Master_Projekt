"""SalsaNext temporal masked autoencoder for range-view LiDAR histories."""

from __future__ import annotations

import os
import sys
from typing import Mapping

import torch
import torch.nn as nn

try:
    from models.acc_models import MidMetaNet
    from mos_models.salsanext_parts import SalsaNextDecoder, SalsaNextEncoder
except ModuleNotFoundError:  # direct file execution fallback
    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)
    from models.acc_models import MidMetaNet
    from mos_models.salsanext_parts import SalsaNextDecoder, SalsaNextEncoder


class SalsaNextTemporalMAE(nn.Module):
    """Temporal MAE that reconstructs current geometry/features plus residual maps."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model_params", {}) or {}
        pretrain_cfg = cfg.get("pretrain_params", {}) or {}
        residual_cfg = pretrain_cfg.get("residual_targets", {}) or {}
        mask_cfg = pretrain_cfg.get("mask", {}) or {}

        self.input_horizon = int(model_cfg.get("input_horizon", 5))
        self.in_channels = int(model_cfg.get("grid_channels", 7))
        self.grid_height = int(model_cfg.get("grid_height", 64))
        self.grid_width = int(model_cfg.get("grid_width", 512))
        self.residual_offsets = [int(v) for v in residual_cfg.get("offsets", [1])]
        self.residual_channels = len(self.residual_offsets)
        self.out_channels = int(model_cfg.get("output_channels", self.in_channels + self.residual_channels))
        self.mask_apply_to = str(mask_cfg.get("apply_to", "current")).lower()

        if self.input_horizon <= 0:
            raise ValueError(f"input_horizon must be positive, got {self.input_horizon}.")
        if self.in_channels != 7:
            raise ValueError(f"Temporal MAE expects model_params.grid_channels=7 for this step, got {self.in_channels}.")
        if self.residual_channels < 1:
            raise ValueError("Temporal MAE expects at least one residual target channel.")
        if self.out_channels != self.in_channels + self.residual_channels:
            raise ValueError(
                "model_params.output_channels must equal grid_channels + len(residual_offsets): "
                f"got {self.out_channels}, expected {self.in_channels + self.residual_channels}."
            )
        if self.mask_apply_to not in {"current", "all"}:
            raise ValueError("pretrain_params.mask.apply_to must be 'current' or 'all'.")

        dropout = float(model_cfg.get("dropout_prob", model_cfg.get("dropout", 0.2)))
        self.encoder = SalsaNextEncoder(in_channels=self.in_channels, dropout=dropout)
        self.decoder = SalsaNextDecoder(num_classes=self.out_channels, dropout=dropout)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.in_channels, 1, 1))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self.encoder_bottleneck_channels = int(model_cfg.get("salsa_encoder_bottleneck_channels", 256))
        temporal_hidden = int(model_cfg.get("temporal_hidden_channels", 256))
        temporal_depth = max(int(model_cfg.get("temporal_depth", 8)), 2)
        self.temporal = MidMetaNet(
            channel_in=self.input_horizon * self.encoder_bottleneck_channels,
            channel_hid=temporal_hidden,
            N2=temporal_depth,
        )

    def forward(self, masked_hist_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if masked_hist_features.ndim != 5:
            raise ValueError(
                "SalsaNextTemporalMAE expects masked_hist_features=[B,T,C,H,W], "
                f"got {tuple(masked_hist_features.shape)}"
            )
        batch_size, timesteps, channels, height, width = masked_hist_features.shape
        if timesteps != self.input_horizon:
            raise ValueError(f"Expected T={self.input_horizon}, got T={timesteps}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={channels}.")
        if (
            mask.ndim != 4
            or mask.shape[0] != batch_size
            or mask.shape[1] != 1
            or mask.shape[-2:] != (height, width)
        ):
            raise ValueError(
                f"Expected mask=[B,1,H,W] matching input, got input={tuple(masked_hist_features.shape)}, "
                f"mask={tuple(mask.shape)}"
            )

        mask = mask.to(device=masked_hist_features.device, dtype=masked_hist_features.dtype).clamp(0.0, 1.0)
        x_seq = masked_hist_features
        token = self.mask_token.to(device=x_seq.device, dtype=x_seq.dtype)
        if self.mask_apply_to == "all":
            mask_5d = mask.unsqueeze(1)
            x_seq = x_seq * (1.0 - mask_5d) + token * mask_5d
        else:
            x_seq = x_seq.clone()
            x_seq[:, -1] = x_seq[:, -1] * (1.0 - mask) + token[:, 0] * mask

        x_flat = x_seq.reshape(batch_size * timesteps, channels, height, width)
        bottleneck, skips_flat = self.encoder(x_flat)
        bottleneck_channels, bottleneck_h, bottleneck_w = bottleneck.shape[1:]
        if bottleneck_channels != self.encoder_bottleneck_channels:
            raise ValueError(
                "Unexpected SalsaNext bottleneck channels: "
                f"got {bottleneck_channels}, expected {self.encoder_bottleneck_channels}."
            )

        z = bottleneck.view(batch_size, timesteps, bottleneck_channels, bottleneck_h, bottleneck_w)
        z_temporal = self.temporal(z)
        current_bottleneck = z_temporal[:, -1].contiguous()

        current_skips = []
        for skip in skips_flat:
            skip_shape = skip.shape[1:]
            current_skips.append(skip.view(batch_size, timesteps, *skip_shape)[:, -1].contiguous())

        pred = self.decoder(current_bottleneck, tuple(current_skips))
        expected_shape = (batch_size, self.out_channels, height, width)
        if tuple(pred.shape) != expected_shape:
            raise ValueError(f"Temporal MAE decoder returned {tuple(pred.shape)}, expected {expected_shape}.")
        return pred

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()

    def get_decoder_state_dict(self):
        return self.decoder.state_dict()

    def get_temporal_state_dict(self):
        return self.temporal.state_dict()

    def get_backbone_state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "temporal": self.temporal.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_encoder_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def load_decoder_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        return self.decoder.load_state_dict(state_dict, strict=strict)
