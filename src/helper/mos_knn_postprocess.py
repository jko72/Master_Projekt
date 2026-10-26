"""Optional, LMNet-style range-view kNN MOS point-cloud postprocessing.

Use AFTER range-view inference; requires NO retraining. The class convention
is 0=static, 1=moving (not LMNet's internal 1=static, 2=moving).

This is an independent implementation of the range-neighborhood procedure
used in PRBonn/LiDAR-MOS: search window, weighted range discrepancy, k-nearest
voting and a distance cutoff. LMNet's center-range override is retained.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from helper.mos_point_projection import ProjectionConfig, original_points_to_pixels


@torch.no_grad()
def knn_unproject_prediction(
    pred_hw: np.ndarray,
    projected_range_hw: np.ndarray,
    points_xyzi: np.ndarray,
    cfg: ProjectionConfig,
    *,
    device: str | torch.device = "cpu",
    knn: int = 5,
    search: int = 5,
    sigma: float = 1.0,
    cutoff: float = 1.0,
    chunk_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    """Map binary [H,W] MOS labels to [N] original-point labels with LMNet-style kNN.

    projected_range_hw MUST be the current scan's raw ranges in meters, not
    standardized, with invalid pixels <= 0. The model's range image already
    meets this requirement when normalization='intensity_only'.

    Returns (point_pred: uint8[N], valid: bool[N]); invalid points default to 0.
    """
    pred = np.asarray(pred_hw)
    proj_range = np.asarray(projected_range_hw)
    points = np.asarray(points_xyzi)
    if pred.shape != (cfg.height, cfg.width) or proj_range.shape != pred.shape:
        raise ValueError("Prediction and raw range must both have shape [H,W]")
    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Expected 0=static, 1=moving predictions")
    if search < 1 or search % 2 != 1 or not (1 <= knn <= search * search):
        raise ValueError("Search must be odd and 1 <= knn <= search*search")
    if not np.isfinite(sigma) or sigma <= 0 or chunk_size < 1:
        raise ValueError("Sigma must be finite and positive; chunk_size must be positive")
    if not np.isfinite(cutoff):
        raise ValueError("Cutoff must be finite")

    rows, cols, valid = original_points_to_pixels(points, cfg)
    point_pred = np.zeros(len(points), dtype=np.uint8)
    valid_index = np.flatnonzero(valid)
    if len(valid_index) == 0:
        return point_pred, valid

    dev = torch.device(device)
    # Local projections use zero for an empty pixel, while LMNet's original
    # projector uses a negative sentinel. Treat both as invalid before KNN.
    ranges_image = torch.as_tensor(proj_range, dtype=torch.float32, device=dev)
    label_image = torch.as_tensor(pred, dtype=torch.float32, device=dev)
    padding = search // 2
    # [search**2, H*W]. Invalid pixels and zero-padding are removed after gather.
    patches_range = F.unfold(ranges_image[None, None], search, padding=padding)[0]
    patches_label = F.unfold(label_image[None, None], search, padding=padding)[0]

    coord = torch.arange(search, device=dev, dtype=torch.float32) - padding
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma**2))
    gaussian /= gaussian.sum()
    spatial_weight = (1.0 - gaussian.flatten())[:, None]
    center = search * search // 2

    # Fixed-size chunks keep GPU memory bounded for scans with >100k points.
    for start in range(0, len(valid_index), chunk_size):
        orig = valid_index[start:start + chunk_size]
        flat_pixel = torch.as_tensor(
            rows[orig].astype(np.int64) * cfg.width + cols[orig].astype(np.int64),
            dtype=torch.long, device=dev,
        )
        depth = np.linalg.norm(points[orig, :3].astype(np.float32), axis=1)
        depth = torch.as_tensor(depth, dtype=torch.float32, device=dev)

        nbr_range = patches_range[:, flat_pixel].clone()
        nbr_range[~torch.isfinite(nbr_range) | (nbr_range <= 0)] = float("inf")
        # LMNet uses the ORIGINAL point range at the center, not the projected winner.
        nbr_range[center] = depth
        distance = (nbr_range - depth[None, :]).abs() * spatial_weight
        knn_idx = distance.topk(knn, dim=0, largest=False, sorted=False).indices
        selected_dist = distance.gather(0, knn_idx)
        selected_labels = patches_label[:, flat_pixel].gather(0, knn_idx).long()

        accepted = (selected_dist <= cutoff) if cutoff > 0 else torch.isfinite(selected_dist)
        n_static = ((selected_labels == 0) & accepted).sum(dim=0)
        n_moving = ((selected_labels == 1) & accepted).sum(dim=0)
        # Static wins ties or when no neighbors pass cutoff (LMNet's class-1 fallback).
        point_pred[orig] = (n_moving > n_static).to(torch.uint8).cpu().numpy()
    return point_pred, valid
