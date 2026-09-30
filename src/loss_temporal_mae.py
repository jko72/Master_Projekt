"""Losses for temporal MAE pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _loss_config(cfg: dict) -> dict:
    if "pretrain_params" in cfg:
        return (cfg.get("pretrain_params", {}) or {}).get("loss", {}) or {}
    return cfg.get("loss", cfg) or {}


def _aux_config(cfg: dict, name: str) -> dict:
    pretrain_cfg = cfg.get("pretrain_params", {}) if "pretrain_params" in cfg else {}
    auxiliary_cfg = (pretrain_cfg or {}).get("auxiliary_tasks", {}) or {}
    return auxiliary_cfg.get(name, {}) or {}


def temporal_mae_loss(
    pred: torch.Tensor,
    target_current: torch.Tensor,
    target_residuals: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    residual_valid_mask: torch.Tensor,
    cfg: dict,
):
    """Compute current-frame and weighted residual reconstruction losses."""
    if pred.ndim != 4 or target_current.ndim != 4 or target_residuals.ndim != 4:
        raise ValueError(
            "pred, target_current, and target_residuals must be 4D tensors; "
            f"got {pred.shape}, {target_current.shape}, {target_residuals.shape}"
        )
    batch_size, current_channels, height, width = target_current.shape
    residual_channels = int(target_residuals.shape[1])
    if current_channels != 4:
        raise ValueError(f"target_current must have C=4, got C={current_channels}.")

    expected_pred_shape = (batch_size, current_channels + residual_channels, height, width)
    if tuple(pred.shape) != expected_pred_shape:
        raise ValueError(f"pred must be {expected_pred_shape}, got {tuple(pred.shape)}")
    expected_mask_shape = (batch_size, 1, height, width)
    if tuple(mask.shape) != expected_mask_shape or tuple(valid_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"mask and valid_mask must be {expected_mask_shape}, got {tuple(mask.shape)} and {tuple(valid_mask.shape)}"
        )
    if tuple(residual_valid_mask.shape) != expected_mask_shape:
        raise ValueError(f"residual_valid_mask must be {expected_mask_shape}, got {tuple(residual_valid_mask.shape)}")

    pred_current = pred[:, :current_channels]
    pred_residuals = pred[:, current_channels : current_channels + residual_channels]

    loss_cfg = _loss_config(cfg)
    current_loss_name = str(loss_cfg.get("name", "smooth_l1")).lower()
    xyz_weight = float(loss_cfg.get("xyz_weight", 0.5))
    range_weight = float(loss_cfg.get("range_weight", 1.0))
    current_loss_on_mask_only = bool(loss_cfg.get("loss_on_mask_only", True))

    residual_cfg = _aux_config(cfg, "residual_reconstruction")
    residual_enabled = bool(residual_cfg.get("enabled", False)) and residual_channels > 0
    residual_weight = float(residual_cfg.get("weight", 0.2))
    residual_loss_name = str(residual_cfg.get("loss", "smooth_l1")).lower()
    residual_loss_on_mask_only = bool(residual_cfg.get("loss_on_mask_only", False))
    positive_threshold = float(residual_cfg.get("positive_threshold", 0.02))
    positive_weight = float(residual_cfg.get("positive_weight", 5.0))

    finite_current = torch.isfinite(pred_current).all(dim=1, keepdim=True) & torch.isfinite(target_current).all(
        dim=1,
        keepdim=True,
    )
    selected_current = (valid_mask > 0.5) & finite_current
    if current_loss_on_mask_only:
        selected_current = selected_current & (mask > 0.5)

    safe_pred_current = torch.where(
        finite_current.expand_as(pred_current),
        pred_current,
        torch.zeros_like(pred_current),
    )
    safe_target_current = torch.where(
        finite_current.expand_as(target_current),
        target_current,
        torch.zeros_like(target_current),
    )
    if current_loss_name in {"smooth_l1", "huber"}:
        current_per_channel = F.smooth_l1_loss(safe_pred_current, safe_target_current, reduction="none")
    elif current_loss_name in {"l1", "mae"}:
        current_per_channel = F.l1_loss(safe_pred_current, safe_target_current, reduction="none")
    else:
        raise ValueError(f"Unsupported temporal MAE current loss '{current_loss_name}'. Use 'smooth_l1' or 'l1'.")

    selected_current_f = selected_current.to(current_per_channel.dtype)
    selected_current_pixels = selected_current_f.sum()
    current_per_channel = torch.where(
        selected_current.expand_as(current_per_channel),
        current_per_channel,
        torch.zeros_like(current_per_channel),
    )
    loss_xyz = (current_per_channel[:, :3] * selected_current_f).sum() / (
        selected_current_pixels * 3.0
    ).clamp_min(1.0)
    loss_range = (current_per_channel[:, 3:4] * selected_current_f).sum() / selected_current_pixels.clamp_min(1.0)

    finite_residual = torch.isfinite(pred_residuals) & torch.isfinite(target_residuals)
    selected_residual = (residual_valid_mask > 0.5).expand_as(target_residuals) & finite_residual
    if residual_loss_on_mask_only:
        selected_residual = selected_residual & (mask > 0.5).expand_as(target_residuals)

    safe_pred_residuals = torch.where(finite_residual, pred_residuals, torch.zeros_like(pred_residuals))
    safe_target_residuals = torch.where(finite_residual, target_residuals, torch.zeros_like(target_residuals))
    if residual_channels == 0:
        residual_per_channel = pred_residuals
    elif residual_loss_name in {"smooth_l1", "huber"}:
        residual_per_channel = F.smooth_l1_loss(safe_pred_residuals, safe_target_residuals, reduction="none")
    elif residual_loss_name in {"l1", "mae"}:
        residual_per_channel = F.l1_loss(safe_pred_residuals, safe_target_residuals, reduction="none")
    else:
        raise ValueError(f"Unsupported residual reconstruction loss '{residual_loss_name}'. Use 'smooth_l1' or 'l1'.")

    positive = safe_target_residuals > positive_threshold
    residual_weights = torch.where(
        positive,
        torch.full_like(residual_per_channel, positive_weight),
        torch.ones_like(residual_per_channel),
    )
    residual_weights = torch.where(selected_residual, residual_weights, torch.zeros_like(residual_weights))
    residual_weight_sum = residual_weights.sum().clamp_min(1.0)
    loss_residual = (residual_per_channel * residual_weights).sum() / residual_weight_sum
    if not residual_enabled:
        loss_residual = pred.new_tensor(0.0)

    selected_residual_count = selected_residual.float().sum().clamp_min(1.0)
    residual_pos_ratio = (positive & selected_residual).float().sum() / selected_residual_count
    loss_total = (
        xyz_weight * loss_xyz
        + range_weight * loss_range
        + residual_weight * loss_residual
    )

    valid_ratio = (valid_mask > 0.5).float().mean()
    masked_valid_ratio = (
        ((valid_mask > 0.5) & (mask > 0.5)).float().sum()
        / (valid_mask > 0.5).float().sum().clamp_min(1.0)
    )
    loss_dict = {
        "loss_total": loss_total.detach(),
        "loss_xyz": loss_xyz.detach(),
        "loss_range": loss_range.detach(),
        "loss_residual": loss_residual.detach(),
        "residual_pos_ratio": residual_pos_ratio.detach(),
        "masked_valid_ratio": masked_valid_ratio.detach(),
        "valid_ratio": valid_ratio.detach(),
    }
    return loss_total, loss_dict
