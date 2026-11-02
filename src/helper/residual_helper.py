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
    from helper.mos_projection import projection_mode, lmnet_projection
    if projection_mode(cfg) == 'lmnet':
        result = lmnet_projection(points_xyzi, H, W, mp.get('FOV_UP', 3), mp.get('FOV_DOWN', -25),
                                  min_range=2., max_range=50., check_intensity=False)
        image = np.zeros((H, W), np.float32)
        image[result.occupied] = result.ranges[result.indices[result.occupied]]
        return image
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


def projection_ranges(projection):
    """Physical metres from an existing full-range LMNet projection."""
    image = np.zeros(projection.indices.shape, np.float32)
    image[projection.occupied] = projection.ranges[projection.indices[projection.occupied]]
    return image


def compute_residual_target(current_clean, history_clean, *, current_input=None,
                            input_original_ids=None, history_available=True,
                            settings=None, return_diagnostics=False):
    """Online relative target and explicit mask; never changes legacy residuals.

    All projections use the same full-range LMNet grid after shared geometry.
    Range limits apply AFTER winner selection, only to target validity. An input
    winner must retain the clean winner's ORIGINAL point ID after deletion.
    """
    spec = dict(min_range=2., max_range=50., epsilon=1e-6)
    spec.update(settings or {})
    current = projection_ranges(current_clean)
    history = projection_ranges(history_clean)
    if current.shape != history.shape:
        raise ValueError('Residual projection grids must match')
    clean_valid = (current_clean.occupied & history_clean.occupied &
                   np.isfinite(current) & np.isfinite(history) &
                   (current > spec['min_range']) & (current < spec['max_range']) &
                   (history > spec['min_range']) & (history < spec['max_range']))
    if not history_available:
        clean_valid[:] = False
    same_winner = current_clean.occupied.copy()
    if current_input is not None:
        if current_input.indices.shape != current.shape:
            raise ValueError('Input and clean grids must match')
        ids = np.asarray(input_original_ids)
        if ids.ndim != 1 or len(ids) != len(current_input.usable) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Expected original IDs for every surviving current point')
        if (ids < 0).any() or (ids >= len(current_clean.usable)).any() or len(np.unique(ids)) != len(ids):
            raise ValueError('Invalid original point IDs')
        winner_ids = np.full(current.shape, -1, np.int64)
        winner_ids[current_input.occupied] = ids[current_input.indices[current_input.occupied]]
        same_winner = current_input.occupied & (winner_ids == current_clean.indices)
    valid = clean_valid & same_winner
    target = np.zeros(current.shape, np.float32)
    target[valid] = np.abs(current[valid] - history[valid]) / np.maximum(current[valid], spec['epsilon'])
    if return_diagnostics:
        stats = dict(current_pixels=int(current_clean.occupied.sum()),
                     history_pixels=int(history_clean.occupied.sum()),
                     clean_range_valid_pixels=int(clean_valid.sum()),
                     dropped_or_changed_winner_pixels=int((clean_valid & ~same_winner).sum()))
        return target, valid, stats
    return target, valid
