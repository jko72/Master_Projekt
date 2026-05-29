from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.acc_models import MidMetaNet
    from mos_models.salsanext_parts import SalsaNextEncoder
except ModuleNotFoundError:  # direct file execution fallback
    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)
    from models.acc_models import MidMetaNet
    from mos_models.salsanext_parts import SalsaNextEncoder


def _time_resample_5d(x: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Resample [B, T, C, H, W] to [B, target_frames, C, H, W] over time axis."""
    bsz, t_in, channels, height, width = x.shape
    target_frames = int(target_frames)
    if t_in == target_frames:
        return x

    x_t = x.permute(0, 3, 4, 2, 1).reshape(bsz * height * width * channels, 1, t_in)
    x_t = F.interpolate(x_t, size=target_frames, mode="linear", align_corners=True)
    x_t = x_t.reshape(bsz, height, width, channels, target_frames).permute(0, 4, 3, 1, 2).contiguous()
    return x_t


def _extract_range_limits(cfg: Dict) -> Tuple[float, float]:
    data_cfg = cfg.get("DATA_CONFIG", {}) or {}
    model_params = cfg.get("model_params", {}) or {}
    data_params = cfg.get("data_params", {}) or {}
    stats = data_params.get("stats", {}) or {}

    min_range = (
        data_cfg.get("MIN_RANGE", None)
        if "MIN_RANGE" in data_cfg
        else data_cfg.get("min_range", None)
    )
    max_range = (
        data_cfg.get("MAX_RANGE", None)
        if "MAX_RANGE" in data_cfg
        else data_cfg.get("max_range", None)
    )
    if min_range is None:
        min_range = model_params.get("MIN_RANGE", model_params.get("min_range", stats.get("min_range", 0.0)))
    if max_range is None:
        max_range = model_params.get("MAX_RANGE", model_params.get("max_range", stats.get("max_range", 80.0)))

    min_range = float(min_range)
    max_range = float(max_range)
    if max_range <= min_range:
        max_range = min_range + 1.0
    return min_range, max_range


class _RangeUpsampleDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.head(x)
        return x


class SalsaNextACCForecast(nn.Module):
    """
    Self-supervised range forecasting model with:
      - spatial encoder per frame: SalsaNextEncoder
      - temporal translation on encoder bottlenecks: MidMetaNet
      - direct range regression decoder: [B, F, H, W]
    """

    supports_mdn = False

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

        model_params = cfg.get("model_params", {}) or {}
        self.input_horizon = int(model_params.get("input_horizon", 10))
        self.forecast_horizon = int(model_params.get("forecast_horizon", 3))
        self.grid_height = int(model_params.get("grid_height", 64))
        self.grid_width = int(model_params.get("grid_width", 512))
        self.in_channels = int(model_params.get("grid_channels", 4))
        self.dropout = float(model_params.get("dropout_prob", model_params.get("dropout", 0.2)))

        self.min_range, self.max_range = _extract_range_limits(cfg)

        self.encoder = SalsaNextEncoder(in_channels=self.in_channels, dropout=self.dropout)

        self.encoder_bottleneck_channels = int(model_params.get("salsa_encoder_bottleneck_channels", 256))
        temporal_hidden = int(model_params.get("salsa_acc_hid_T", 256))
        temporal_depth = int(model_params.get("salsa_acc_temporal_depth", 8))
        if temporal_depth < 2:
            temporal_depth = 2

        self.temporal = MidMetaNet(
            channel_in=self.input_horizon * self.encoder_bottleneck_channels,
            channel_hid=temporal_hidden,
            N2=temporal_depth,
        )
        self.range_decoder = _RangeUpsampleDecoder(in_channels=self.encoder_bottleneck_channels, out_channels=1)

        self._dbg_once = False

    def _prepare_input(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 5:
            raise ValueError(
                f"SalsaNextACCForecast expects 5D input [B,T,C,H,W], got shape={tuple(x_seq.shape)}"
            )
        if x_seq.shape[2] != self.in_channels:
            raise ValueError(
                f"SalsaNextACCForecast expected C={self.in_channels}, got C={x_seq.shape[2]} "
                f"for shape={tuple(x_seq.shape)}"
            )
        return x_seq

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_seq = self._prepare_input(x_seq)
        bsz, t_in, channels, height, width = x_seq.shape

        x_flat = x_seq.reshape(bsz * t_in, channels, height, width)
        bottleneck, _ = self.encoder(x_flat)  # [B*T, Cb, Hb, Wb]
        c_b, h_b, w_b = bottleneck.shape[1], bottleneck.shape[2], bottleneck.shape[3]

        if c_b != self.encoder_bottleneck_channels:
            raise ValueError(
                f"Unexpected SalsaNext bottleneck channels: got {c_b}, expected {self.encoder_bottleneck_channels}."
            )

        z = bottleneck.view(bsz, t_in, c_b, h_b, w_b)  # [B,T,Cb,Hb,Wb]
        z_temporal_in = _time_resample_5d(z, target_frames=self.input_horizon)
        z_temporal = self.temporal(z_temporal_in)  # [B,T_cfg,Cb,Hb,Wb]
        z_future = _time_resample_5d(z_temporal, target_frames=self.forecast_horizon)  # [B,F,Cb,Hb,Wb]

        z_future_flat = z_future.reshape(bsz * self.forecast_horizon, c_b, h_b, w_b)
        raw_ranges = self.range_decoder(z_future_flat)  # [B*F,1,H,W]
        if raw_ranges.shape[-2:] != (height, width):
            raw_ranges = F.interpolate(raw_ranges, size=(height, width), mode="bilinear", align_corners=False)

        raw_ranges = raw_ranges.view(bsz, self.forecast_horizon, height, width)
        pred_ranges = self.min_range + torch.sigmoid(raw_ranges) * (self.max_range - self.min_range)

        if not self._dbg_once:
            self._dbg_once = True
            print(f"[DBG SALSA_ACC] input shape: {list(x_seq.shape)}")
            print(f"[DBG SALSA_ACC] encoder bottleneck shape per frame: {list(bottleneck.shape)}")
            print(f"[DBG SALSA_ACC] temporal feature shape: {list(z_temporal.shape)}")
            print(f"[DBG SALSA_ACC] predicted range shape: {list(pred_ranges.shape)}")
            print(f"[DBG SALSA_ACC] forecast_horizon F={self.forecast_horizon}")
            print(f"[DBG SALSA_ACC] min_range/max_range: {self.min_range}/{self.max_range}")

        return pred_ranges

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, state_dict, strict: bool = True):
        return self.encoder.load_state_dict(state_dict, strict=strict)

    def build_mixture(self, cfg, output):
        raise ValueError("SalsaNextACCForecast does not support MDN. Set model_params.use_mdn=false.")


if __name__ == "__main__":
    cfg = {
        "model_params": {
            "name": "salsanext_acc_forecast",
            "use_mdn": False,
            "input_horizon": 10,
            "forecast_horizon": 3,
            "grid_channels": 4,
            "grid_height": 64,
            "grid_width": 512,
            "dropout_prob": 0.2,
        },
        "DATA_CONFIG": {"MIN_RANGE": 0.0, "MAX_RANGE": 80.0},
    }
    model = SalsaNextACCForecast(cfg)
    x = torch.randn(2, 10, 4, 64, 512)
    y = model(x)
    print(y.shape)
    assert y.shape == (2, 3, 64, 512)
