"""Losses dedicated to MAE-RangeXYZ pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _loss_config(cfg: dict) -> dict:
    if "pretrain_params" in cfg:
        return (cfg.get("pretrain_params", {}) or {}).get("loss", {}) or {}
    return cfg.get("loss", cfg) or {}


def mae_rangexyz_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: dict,
):
    """Compute channel-group losses only at finite, valid, masked pixels."""
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 4:
        raise ValueError(f"pred and target must both be [B,4,H,W], got {pred.shape} and {target.shape}")
    expected_mask_shape = (pred.shape[0], 1, pred.shape[2], pred.shape[3])
    if tuple(mask.shape) != expected_mask_shape or tuple(valid_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"mask and valid_mask must be {expected_mask_shape}, got {tuple(mask.shape)} and {tuple(valid_mask.shape)}"
        )

    loss_cfg = _loss_config(cfg)
    name = str(loss_cfg.get("name", "smooth_l1")).lower()
    xyz_weight = float(loss_cfg.get("xyz_weight", 0.5))
    range_weight = float(loss_cfg.get("range_weight", 1.0))
    loss_on_mask_only = bool(loss_cfg.get("loss_on_mask_only", True))

    finite = torch.isfinite(pred).all(dim=1, keepdim=True) & torch.isfinite(target).all(dim=1, keepdim=True)
    selected = (valid_mask > 0.5) & finite
    if loss_on_mask_only:
        selected = selected & (mask > 0.5)

    finite_channels = finite.expand_as(pred)
    safe_pred = torch.where(finite_channels, pred, torch.zeros_like(pred))
    safe_target = torch.where(finite_channels, target, torch.zeros_like(target))
    if name in {"smooth_l1", "huber"}:
        per_channel = F.smooth_l1_loss(safe_pred, safe_target, reduction="none")
    elif name in {"l1", "mae"}:
        per_channel = F.l1_loss(safe_pred, safe_target, reduction="none")
    else:
        raise ValueError(f"Unsupported MAE loss '{name}'. Use 'smooth_l1' or 'l1'.")

    selected_expanded = selected.expand_as(per_channel)
    per_channel = torch.where(selected_expanded, per_channel, torch.zeros_like(per_channel))
    selected_f = selected.to(per_channel.dtype)
    selected_pixels = selected_f.sum()
    loss_xyz = (per_channel[:, :3] * selected_f).sum() / (selected_pixels * 3.0).clamp_min(1.0)
    loss_range = (per_channel[:, 3:4] * selected_f).sum() / selected_pixels.clamp_min(1.0)
    loss_total = xyz_weight * loss_xyz + range_weight * loss_range

    valid_ratio = (valid_mask > 0.5).float().mean()
    masked_valid_ratio = (
        ((valid_mask > 0.5) & (mask > 0.5)).float().sum()
        / (valid_mask > 0.5).float().sum().clamp_min(1.0)
    )
    loss_dict = {
        "loss_xyz": loss_xyz.detach(),
        "loss_range": loss_range.detach(),
        "loss_total": loss_total.detach(),
        "masked_valid_ratio": masked_valid_ratio.detach(),
        "valid_ratio": valid_ratio.detach(),
    }
    return loss_total, loss_dict
