"""Losses dedicated to MAE-RangeXYZ pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _loss_config(cfg: dict) -> dict:
    if "pretrain_params" in cfg:
        return (cfg.get("pretrain_params", {}) or {}).get("loss", {}) or {}
    return cfg.get("loss", cfg) or {}


def _surface_normals_config(cfg: dict) -> dict:
    pretrain_cfg = cfg.get("pretrain_params", {}) if "pretrain_params" in cfg else {}
    auxiliary_cfg = (pretrain_cfg or {}).get("auxiliary_tasks", {}) or {}
    return auxiliary_cfg.get("surface_normals", {}) or {}


def _residual_inputs_config(cfg: dict) -> dict:
    pretrain_cfg = cfg.get("pretrain_params", {}) if "pretrain_params" in cfg else {}
    return (pretrain_cfg or {}).get("residual_inputs", {}) or {}


def _residual_loss_config(cfg: dict) -> dict:
    pretrain_cfg = cfg.get("pretrain_params", {}) if "pretrain_params" in cfg else {}
    auxiliary_cfg = (pretrain_cfg or {}).get("auxiliary_tasks", {}) or {}
    return auxiliary_cfg.get("residual_reconstruction", {}) or {}


def mae_rangexyz_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
    cfg: dict,
):
    """Compute channel-group losses only at finite, valid, masked pixels."""
    normals_cfg = _surface_normals_config(cfg)
    normals_enabled = bool(normals_cfg.get("enabled", False))
    residual_inputs_cfg = _residual_inputs_config(cfg)
    residual_inputs_enabled = bool(residual_inputs_cfg.get("enabled", False))
    residual_offsets = [int(v) for v in residual_inputs_cfg.get("offsets", [1])] if residual_inputs_enabled else []
    residual_channels = len(residual_offsets)
    base_channels = 7 if normals_enabled else 4
    expected_channels = base_channels + residual_channels

    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != expected_channels:
        raise ValueError(
            "pred and target must both be [B,C,H,W] with C matching configured MAE channels, "
            f"got {pred.shape} and {target.shape}; expected C={expected_channels}"
        )
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
    normal_weight = float(normals_cfg.get("weight", 0.1))
    normal_loss_name = str(normals_cfg.get("loss", "cosine")).lower()
    residual_cfg = _residual_loss_config(cfg)
    residual_loss_enabled = bool(residual_cfg.get("enabled", residual_channels > 0))
    residual_weight = float(residual_cfg.get("weight", 0.2))
    residual_loss_name = str(residual_cfg.get("loss", "smooth_l1")).lower()
    residual_loss_on_mask_only = bool(residual_cfg.get("loss_on_mask_only", True))
    positive_threshold = float(residual_cfg.get("positive_threshold", 0.02))
    positive_weight = float(residual_cfg.get("positive_weight", 1.0))

    pred_base = pred[:, :base_channels]
    target_base = target[:, :base_channels]
    finite = torch.isfinite(pred_base).all(dim=1, keepdim=True) & torch.isfinite(target_base).all(dim=1, keepdim=True)
    selected = (valid_mask > 0.5) & finite
    if loss_on_mask_only:
        selected = selected & (mask > 0.5)

    finite_channels = finite.expand_as(pred_base)
    safe_pred = torch.where(finite_channels, pred_base, torch.zeros_like(pred_base))
    safe_target = torch.where(finite_channels, target_base, torch.zeros_like(target_base))
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
    if base_channels == 7:
        if normal_loss_name == "cosine":
            pred_normals = safe_pred[:, 4:7]
            target_normals = safe_target[:, 4:7]
            normal_per_pixel = 1.0 - F.cosine_similarity(pred_normals, target_normals, dim=1, eps=1e-8)
            selected_normals = selected[:, 0]
            normal_per_pixel = torch.where(
                selected_normals,
                normal_per_pixel,
                torch.zeros_like(normal_per_pixel),
            )
            loss_normals = normal_per_pixel.sum() / selected_pixels.clamp_min(1.0)
        elif normal_loss_name in {"smooth_l1", "huber"}:
            loss_normals = (per_channel[:, 4:7] * selected_f).sum() / (selected_pixels * 3.0).clamp_min(1.0)
        else:
            raise ValueError("Unsupported surface normal loss. Use 'cosine' or 'smooth_l1'.")
    else:
        loss_normals = pred.new_tensor(0.0)

    if residual_channels > 0:
        pred_residuals = pred[:, base_channels:base_channels + residual_channels]
        target_residuals = target[:, base_channels:base_channels + residual_channels]
        finite_residual = torch.isfinite(pred_residuals) & torch.isfinite(target_residuals)
        selected_residual = (valid_mask > 0.5).expand_as(target_residuals) & finite_residual
        if residual_loss_on_mask_only:
            selected_residual = selected_residual & (mask > 0.5).expand_as(target_residuals)

        safe_pred_residuals = torch.where(finite_residual, pred_residuals, torch.zeros_like(pred_residuals))
        safe_target_residuals = torch.where(finite_residual, target_residuals, torch.zeros_like(target_residuals))
        if residual_loss_name in {"smooth_l1", "huber"}:
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
        if not residual_loss_enabled:
            loss_residual = pred.new_tensor(0.0)
        selected_residual_count = selected_residual.float().sum().clamp_min(1.0)
        residual_pos_ratio = (positive & selected_residual).float().sum() / selected_residual_count
    else:
        loss_residual = pred.new_tensor(0.0)
        residual_pos_ratio = pred.new_tensor(0.0)

    loss_total = (
        xyz_weight * loss_xyz
        + range_weight * loss_range
        + normal_weight * loss_normals
        + residual_weight * loss_residual
    )

    valid_ratio = (valid_mask > 0.5).float().mean()
    masked_valid_ratio = (
        ((valid_mask > 0.5) & (mask > 0.5)).float().sum()
        / (valid_mask > 0.5).float().sum().clamp_min(1.0)
    )
    loss_dict = {
        "loss_xyz": loss_xyz.detach(),
        "loss_range": loss_range.detach(),
        "loss_normals": loss_normals.detach(),
        "loss_residual": loss_residual.detach(),
        "residual_pos_ratio": residual_pos_ratio.detach(),
        "loss_total": loss_total.detach(),
        "masked_valid_ratio": masked_valid_ratio.detach(),
        "valid_ratio": valid_ratio.detach(),
    }
    return loss_total, loss_dict
