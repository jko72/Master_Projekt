import os
from typing import Dict

import numpy as np

from utils_torch import spherical_projection


def load_scan_xyzi(path) -> np.ndarray:
    scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return scan


def transform_points(points_xyzi, T) -> np.ndarray:
    if points_xyzi.shape[0] == 0:
        return points_xyzi.astype(np.float32)
    xyz = points_xyzi[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    xyz_t = (T @ xyz_h.T).T[:, :3]
    out = points_xyzi.copy().astype(np.float32)
    out[:, :3] = xyz_t.astype(np.float32)
    return out


def _theta_range_from_cfg(cfg: Dict):
    fov_up = float(cfg["model_params"].get("FOV_UP", 3.0))
    fov_down = float(cfg["model_params"].get("FOV_DOWN", -25.0))
    return [fov_down * np.pi / 180.0, fov_up * np.pi / 180.0]


def project_range_for_residual(points_xyzi, cfg) -> np.ndarray:
    mp = cfg["model_params"]
    rp = cfg["residual_params"]
    H = int(mp["grid_height"])
    W = int(mp["grid_width"])
    theta_range = _theta_range_from_cfg(cfg)
    pj_img, _, _, _ = spherical_projection(
        points_xyzi.astype(np.float32),
        height=H,
        width=W,
        theta_range=theta_range,
        max_range=float(rp["max_range"]),
    )
    xyz = pj_img[:, :, :3]
    rng = np.linalg.norm(xyz, axis=2).astype(np.float32)
    return rng


def compute_residual_image(current_scan, past_scan, current_pose, past_pose, cfg, offset) -> np.ndarray:
    _ = offset  # Documented by caller; computation uses explicit past frame already.
    rp = cfg["residual_params"]
    invalid_value = np.float32(rp.get("invalid_value", 0.0))
    min_range = float(rp["min_range"])
    max_range = float(rp["max_range"])
    normalize = bool(rp.get("normalize", True))

    residual = np.full(
        (int(cfg["model_params"]["grid_height"]), int(cfg["model_params"]["grid_width"])),
        invalid_value,
        dtype=np.float32,
    )

    # MOS-style ego-motion compensation:
    # transform S_(t-offset) into current frame t with inv(T_t) * T_(t-offset).
    T = np.linalg.inv(current_pose) @ past_pose
    past_scan_in_current = transform_points(past_scan, T)

    # IMPORTANT: keep projection identical to current pipeline (same function + FoV + grid).
    current_range = project_range_for_residual(current_scan, cfg)
    past_range_warped = project_range_for_residual(past_scan_in_current, cfg)

    valid_mask = (
        (current_range > min_range)
        & (current_range < max_range)
        & (past_range_warped > min_range)
        & (past_range_warped < max_range)
    )
    if np.any(valid_mask):
        diff = np.abs(current_range[valid_mask] - past_range_warped[valid_mask])
        if normalize:
            denom = np.maximum(current_range[valid_mask], 1e-6)
            diff = diff / denom
        residual[valid_mask] = diff.astype(np.float32)

    return residual


def residual_path_for_frame(seq_dir, frame_stem, offset, cfg) -> str:
    folder_template = cfg["residual_params"].get("folder_template", "residual_images_{offset}")
    folder = folder_template.format(offset=offset)
    return os.path.join(seq_dir, folder, f"{frame_stem}.npy")
