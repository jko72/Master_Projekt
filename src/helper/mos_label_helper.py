import os
from typing import Dict, Tuple

import numpy as np


# SemanticKITTI moving ids for MOS.
MOVING_LABELS = np.array([251, 252, 253, 254, 255, 256, 257, 258, 259], dtype=np.uint16)

# SemanticKITTI known static ids used by MOS mapping.
STATIC_LABELS = np.array(
    [
        9, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
        40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72,
        80, 81, 99,
    ],
    dtype=np.uint16,
)


def load_semantickitti_labels(label_path) -> np.ndarray:
    """Load raw SemanticKITTI labels from .label file as uint32."""
    return np.fromfile(label_path, dtype=np.uint32)


def split_semantic_instance(raw_labels) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split raw uint32 labels into semantic id (lower 16 bit) and instance id (upper 16 bit).
    """
    raw = np.asarray(raw_labels, dtype=np.uint32)
    semantic_labels = (raw & 0xFFFF).astype(np.uint16)
    instance_ids = (raw >> 16).astype(np.uint16)
    return semantic_labels, instance_ids


def map_semantickitti_to_mos(
    semantic_labels,
    ignore_index: int = -1,
    static_index: int = 0,
    moving_index: int = 1,
) -> np.ndarray:
    """
    Map SemanticKITTI semantic ids to MOS point labels.

    Mapping:
    - 0 / 1 => ignore
    - 251..259 => moving
    - known valid non-moving ids => static
    - unknown ids => ignore
    """
    sem = np.asarray(semantic_labels, dtype=np.uint16)
    mos = np.full(sem.shape, fill_value=ignore_index, dtype=np.int16)

    moving_mask = np.isin(sem, MOVING_LABELS)
    static_mask = np.isin(sem, STATIC_LABELS)

    mos[static_mask] = np.int16(static_index)
    mos[moving_mask] = np.int16(moving_index)
    return mos


def _theta_range_from_cfg(cfg: Dict) -> Tuple[float, float]:
    mp = cfg["model_params"]
    if "theta_range" in mp and mp["theta_range"] is not None:
        tmin, tmax = mp["theta_range"]
        return float(tmin), float(tmax)
    fov_up = float(mp.get("FOV_UP", 3.0))
    fov_down = float(mp.get("FOV_DOWN", -25.0))
    return fov_down * np.pi / 180.0, fov_up * np.pi / 180.0


def _project_point_indices(points_xyzi, cfg: Dict):
    """
    Project points onto range-view grid with the same convention as utils_torch.spherical_projection.
    Returns a proj_idx image with indices into the ORIGINAL input array.
    """
    mp = cfg["model_params"]
    mosp = cfg.get("mos_label_params", {})
    H = int(mp["grid_height"])
    W = int(mp["grid_width"])
    min_range = float(mosp.get("min_range", 0.0))
    max_range = float(mosp.get("max_range", 80.0))
    theta_min, theta_max = _theta_range_from_cfg(cfg)

    points = np.asarray(points_xyzi, dtype=np.float32)
    proj_idx = np.full((H, W), -1, dtype=np.int32)
    proj_range = np.full((H, W), -1.0, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        return proj_idx, proj_range

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(r)
    valid &= (r > float(min_range)) & (r < float(max_range))
    if not np.any(valid):
        return proj_idx, proj_range

    orig_idx = np.nonzero(valid)[0].astype(np.int32)
    points_v = points[valid]
    r_v = r[valid]

    # Keep collision behavior equivalent to spherical_projection:
    # sort descending, so nearer points are written later and win collisions.
    order = np.argsort(r_v)[::-1]
    points_s = points_v[order]
    r_s = r_v[order]
    idx_s = orig_idx[order]

    x_s = points_s[:, 0]
    y_s = points_s[:, 1]
    z_s = points_s[:, 2]

    phi = np.arctan2(y_s, x_s)
    theta = np.arcsin(z_s / np.maximum(r_s, 1e-8))

    if theta_max <= theta_min:
        raise ValueError(f"Invalid theta range: [{theta_min}, {theta_max}]")
    if (theta_max - theta_min) < 1e-8:
        theta_max = theta_min + 1e-8

    phi_min, phi_max = -np.pi, np.pi
    bins_h_asc = np.linspace(theta_min, theta_max, num=H, dtype=np.float32)
    bins_w_asc = np.linspace(phi_min, phi_max, num=W, dtype=np.float32)

    idx_h = np.searchsorted(bins_h_asc, theta, side="right") - 1
    idx_h = np.clip(idx_h, 0, H - 1)
    row = (H - 1) - idx_h

    idx_w = np.searchsorted(bins_w_asc, phi, side="right") - 1
    idx_w = np.clip(idx_w, 0, W - 1)
    col = (W - 1) - idx_w

    proj_idx[row, col] = idx_s
    proj_range[row, col] = r_s
    return proj_idx, proj_range


def project_mos_labels_to_range(points_xyzi, mos_point_labels, cfg: Dict) -> np.ndarray:
    """
    Project MOS point labels to range-view image [H, W] using pipeline projection convention.
    """
    mp = cfg["model_params"]
    mosp = cfg.get("mos_label_params", {})
    H = int(mp["grid_height"])
    W = int(mp["grid_width"])
    ignore_index = int(mosp.get("ignore_index", -1))

    labels = np.asarray(mos_point_labels)
    mos_img = np.full((H, W), fill_value=ignore_index, dtype=np.int16)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    proj_idx, _ = _project_point_indices(points_xyzi, cfg)
    valid = proj_idx >= 0
    if np.any(valid):
        idx = proj_idx[valid]
        in_bounds = (idx >= 0) & (idx < labels.shape[0])
        if np.any(in_bounds):
            v_coords = np.where(valid)
            rows = v_coords[0][in_bounds]
            cols = v_coords[1][in_bounds]
            mos_img[rows, cols] = labels[idx[in_bounds]].astype(np.int16)
    return mos_img


def mos_label_path_for_frame(seq_dir, frame_stem, cfg: Dict) -> str:
    folder_name = cfg.get("mos_label_params", {}).get("folder_name", "mos_labels")
    return os.path.join(seq_dir, folder_name, f"{frame_stem}.npy")


def compute_mos_label_stats(
    mos_label_image,
    ignore_index: int,
    static_index: int,
    moving_index: int,
) -> Dict:
    img = np.asarray(mos_label_image)
    total = int(img.size) if img.size > 0 else 1
    ignore_count = int(np.sum(img == ignore_index))
    static_count = int(np.sum(img == static_index))
    moving_count = int(np.sum(img == moving_index))
    unique_values = sorted(int(v) for v in np.unique(img))
    return {
        "ignore_count": ignore_count,
        "static_count": static_count,
        "moving_count": moving_count,
        "ignore_ratio": ignore_count / total,
        "static_ratio": static_count / total,
        "moving_ratio": moving_count / total,
        "unique_values": unique_values,
        "shape": tuple(img.shape),
    }
