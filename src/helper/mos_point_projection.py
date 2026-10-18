"""Original SemanticKITTI point to active range-view pixel correspondence."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionConfig:
    height: int
    width: int
    fov_up: float
    fov_down: float
    min_range: float = 0.0
    max_range: float | None = None
    # Frame range_xyz projects without max_range; temporal features filter it first.
    filter_max_before_projection: bool = False


def original_points_to_pixels(points_xyzi: np.ndarray, cfg: ProjectionConfig):
    """Return row, column, and usable mask for *every* original point, in order.

    Binning mirrors ``utils_torch.spherical_projection`` exactly. Elevation outside
    the configured FoV is clamped by that function, not discarded here.
    """
    points = np.asarray(points_xyzi, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected [N,>=3] points, got {points.shape}")
    if cfg.height < 2 or cfg.width < 2 or cfg.fov_up <= cfg.fov_down:
        raise ValueError("Invalid projection grid or FoV")
    xyz = points[:, :3]
    with np.errstate(invalid="ignore", over="ignore"):
        ranges = np.sqrt(xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1]
                         + xyz[:, 2] * xyz[:, 2])
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(ranges) & (ranges > cfg.min_range)
    if cfg.max_range is not None:
        valid &= ranges < cfg.max_range
    rows = np.zeros(len(points), dtype=np.int32)
    cols = np.zeros(len(points), dtype=np.int32)
    if not np.any(valid):
        return rows, cols, valid
    selected = xyz[valid]
    distance = ranges[valid]
    phi = np.arctan2(selected[:, 1], selected[:, 0])
    theta = np.arcsin(selected[:, 2] / np.maximum(distance, 1e-8))
    bins_h = np.linspace(np.deg2rad(cfg.fov_down), np.deg2rad(cfg.fov_up), cfg.height, dtype=np.float32)
    bins_w = np.linspace(-np.pi, np.pi, cfg.width, dtype=np.float32)
    rows[valid] = cfg.height - 1 - np.clip(np.searchsorted(bins_h, theta, side="right") - 1, 0, cfg.height - 1)
    cols[valid] = cfg.width - 1 - np.clip(np.searchsorted(bins_w, phi, side="right") - 1, 0, cfg.width - 1)
    return rows, cols, valid


def unproject_prediction(pred_hw: np.ndarray, points_xyzi: np.ndarray, cfg: ProjectionConfig):
    pred = np.asarray(pred_hw)
    if pred.shape != (cfg.height, cfg.width) or not np.isin(pred, [0, 1]).all():
        raise ValueError(f"Expected binary prediction of shape {(cfg.height, cfg.width)}")
    rows, cols, valid = original_points_to_pixels(points_xyzi, cfg)
    point_pred = np.zeros(len(rows), dtype=np.uint8)
    point_pred[valid] = pred[rows[valid], cols[valid]]
    return point_pred, valid
