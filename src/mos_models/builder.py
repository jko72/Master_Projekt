from __future__ import annotations

from .mos_unet import MOSUNetSmall
from .salsanext_mos import SalsaNextMOS


def build_mos_model(cfg):
    params = (cfg or {}).get("mos_model_params", {}) or {}

    name = str(params.get("name", "unet_small")).lower()
    in_channels = int(params.get("in_channels", 2))
    num_classes = int(params.get("num_classes", 2))
    base_channels = int(params.get("base_channels", 32))
    dropout = float(params.get("dropout", 0.1))
    norm = str(params.get("norm", "batch"))

    if name == "unet_small":
        return MOSUNetSmall(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
            norm=norm,
        )
    if name in {"salsanext", "salsanext_mos"}:
        # SalsaNextMOS returns raw logits [B, num_classes, H, W], compatible with CrossEntropyLoss.
        return SalsaNextMOS(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(f"Unknown MOS model name: {name}")
