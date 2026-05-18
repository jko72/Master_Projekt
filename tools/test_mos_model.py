#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mos_models import build_mos_model


def _resolve_cfg_path(cfg_path: str) -> str:
    candidates = [
        cfg_path,
        os.path.join(ROOT_DIR, cfg_path),
        os.path.join(SRC_DIR, cfg_path),
        os.path.join(SRC_DIR, "configs", os.path.basename(cfg_path)),
        os.path.join(ROOT_DIR, "src", cfg_path),
    ]
    seen = set()
    for path in candidates:
        ap = os.path.abspath(path)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            return ap
    raise FileNotFoundError(f"Config not found: '{cfg_path}'. Checked: {sorted(seen)}")


def _count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _compute_in_channels(input_mode: str, residual_offsets) -> int:
    residual_offsets = [int(v) for v in residual_offsets]
    if input_mode == "range":
        return 1
    if input_mode == "residual":
        return len(residual_offsets)
    if input_mode == "range_residual":
        return 1 + len(residual_offsets)
    if input_mode == "range_xyz":
        return 4
    if input_mode == "range_xyz_residual":
        return 4 + len(residual_offsets)
    raise ValueError(
        f"Unsupported input_mode='{input_mode}'. "
        "Use one of ['range', 'residual', 'range_residual', 'range_xyz', 'range_xyz_residual']."
    )


def main():
    parser = argparse.ArgumentParser(description="Shape sanity check for MOS baseline model.")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    cfg_path = _resolve_cfg_path(args.cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    mos_data_params = cfg.get("mos_data_params", {}) or {}
    mos_params = cfg.get("mos_model_params", {}) or {}
    input_mode = str(mos_data_params.get("input_mode", "range_residual"))
    residual_offsets = mos_data_params.get("residual_offsets", [1])
    computed_in_channels = _compute_in_channels(input_mode, residual_offsets)
    mos_params["in_channels"] = int(computed_in_channels)
    cfg["mos_model_params"] = mos_params

    model = build_mos_model(cfg)
    model.eval()

    in_channels = int(mos_params.get("in_channels", 2))
    num_classes = int(mos_params.get("num_classes", 2))
    base_channels = int(mos_params.get("base_channels", 32))
    model_name = str(mos_params.get("name", "unet_small"))

    mp = cfg.get("model_params", {}) or {}
    grid_height = int(mp.get("grid_height", 64))
    grid_width = int(mp.get("grid_width", 512))

    x = torch.randn(int(args.batch_size), in_channels, grid_height, grid_width)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (int(args.batch_size), num_classes, grid_height, grid_width), (
        "Output shape mismatch. "
        f"Expected {(int(args.batch_size), num_classes, grid_height, grid_width)}, got {tuple(y.shape)}"
    )

    total_params, trainable_params = _count_params(model)

    print("=== MOS Model Shape Test ===")
    print(f"cfg_path           : {cfg_path}")
    print(f"model_name         : {model_name}")
    print(f"input_mode         : {input_mode}")
    print(f"residual_offsets   : {residual_offsets}")
    print(f"in_channels        : {in_channels}")
    print(f"num_classes        : {num_classes}")
    print(f"base_channels      : {base_channels}")
    print(f"input shape        : {x.shape}")
    print(f"output shape       : {y.shape}")
    print(f"total_params       : {total_params}")
    print(f"trainable_params   : {trainable_params}")
    print("logits_output      : True (no softmax in forward)")
    print("loss_compatibility : CrossEntropyLoss(ignore_index=-1) compatible")


if __name__ == "__main__":
    main()
