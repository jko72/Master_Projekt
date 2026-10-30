"""Explicit MOS projection dispatch. Legacy callers never change implicitly.

LMNet equations independently implemented from PRBonn/LiDAR-MOS commit
4f2e96f46f5ac267cf00e404ec94514f91ebfbf1, common/laserscan.py.
"""
from dataclasses import dataclass
import numpy as np
import torch

REFERENCE_COMMIT = "4f2e96f46f5ac267cf00e404ec94514f91ebfbf1"


def projection_mode(cfg):
    mode = cfg.get("mos_data_params", {}).get("projection_mode", "legacy")
    if mode not in ("legacy", "lmnet"):
        raise ValueError("mos_data_params.projection_mode must be legacy or lmnet")
    if (cfg.get('mos_augmentation') or {}).get('mode') == 'lmnet' and mode != 'lmnet':
        raise ValueError('LMNet raw augmentation requires projection_mode=lmnet')
    version = cfg.get('mos_data_params', {}).get('alignment_version')
    if version is not None and (version != 'lmnet_v1' or mode != 'lmnet'):
        raise ValueError('alignment_version=lmnet_v1 requires projection_mode=lmnet')
    if mode == 'lmnet':
        data = cfg.get('mos_data_params', {})
        if any(data.get(k, False) for k in ('require_moving_train', 'require_moving_val', 'use_moving_frame_sampler')):
            raise ValueError('LMNet requires all frames without moving-frame filtering/sampling')
        mp = cfg.get('model_params', {})
        if mp.get('min_range', mp.get('MIN_RANGE', 0)) != 0 or mp.get('max_range', mp.get('MAX_RANGE')) is not None:
            raise ValueError('LMNet feature filtering requires min_range=0 and max_range=null')
        lp = cfg.get('mos_label_params', {})
        if lp.get('min_range', 0) != 0 or lp.get('max_range') is not None:
            raise ValueError('LMNet label filtering requires min_range=0 and max_range=null')
        if (lp.get('ignore_index', -1), lp.get('static_index', 0), lp.get('moving_index', 1),
            lp.get('label_mapping', 'semantickitti_mos')) != (-1, 0, 1, 'semantickitti_mos'):
            raise ValueError('LMNet requires SemanticKITTI MOS mapping static=0, moving=1, ignore=-1')
        rp = cfg.get('residual_params', {})
        if rp and (rp.get('min_range', 2) != 2 or rp.get('max_range', 50) != 50 or
                   not rp.get('normalize', True) or rp.get('invalid_value', 0) != 0):
            raise ValueError('LMNet residuals require 2<r<50, normalization and invalid_value=0')
    return mode


@dataclass
class Projection:
    indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    usable: np.ndarray
    ranges: np.ndarray

    @property
    def occupied(self):
        return self.indices >= 0


def lmnet_projection(points, height, width, fov_up=3., fov_down=-25., *,
                     min_range=0., max_range=None, check_intensity=True):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected points [N,>=3]")
    if height < 1 or width < 1 or not np.isfinite([fov_up, fov_down]).all() or fov_up <= fov_down:
        raise ValueError("Invalid projection grid/FoV")
    xyz = points[:, :3]
    with np.errstate(invalid="ignore", over="ignore"):
        ranges = np.linalg.norm(xyz, axis=1)
    usable = np.isfinite(xyz).all(1) & np.isfinite(ranges) & (ranges > min_range) & (ranges > 0)
    if check_intensity and points.shape[1] >= 4:
        usable &= np.isfinite(points[:, 3])
    if max_range is not None:
        usable &= ranges < max_range
    rows, cols = np.zeros(len(points), np.int32), np.zeros(len(points), np.int32)
    selected = np.flatnonzero(usable)
    if len(selected):
        yaw = -np.arctan2(xyz[selected, 1], xyz[selected, 0])
        pitch = np.arcsin(np.clip(xyz[selected, 2] / ranges[selected], -1, 1))
        up, down = np.deg2rad([fov_up, fov_down])
        cols[selected] = np.clip(np.floor(.5 * (yaw / np.pi + 1) * width), 0, width-1).astype(np.int32)
        rows[selected] = np.clip(np.floor((1 - (pitch + abs(down)) / (abs(up)+abs(down))) * height), 0, height-1).astype(np.int32)
    indices = np.full((height, width), -1, np.int32)
    order = selected[np.argsort(ranges[selected])[::-1]]
    indices[rows[order], cols[order]] = order
    return Projection(indices, rows, cols, usable, ranges)


def project_features(points, height, width, theta_range, include_intensity=True):
    p = lmnet_projection(points, height, width, np.rad2deg(theta_range[1]), np.rad2deg(theta_range[0]))
    out = np.zeros((5 if include_intensity else 4, height, width), np.float32)
    winner = p.indices[p.occupied]
    out[:3, p.occupied] = np.asarray(points)[winner, :3].T
    out[3, p.occupied] = p.ranges[winner]
    if include_intensity:
        out[4, p.occupied] = np.asarray(points)[winner, 3]
    return torch.from_numpy(out), torch.from_numpy(p.occupied[None].copy()).float(), p


def project_targets(labels, projection):
    labels = np.asarray(labels)
    if len(labels) != len(projection.usable):
        raise ValueError("Point/label length mismatch")
    out = np.full(projection.indices.shape, -1, np.int64)
    out[projection.occupied] = labels[projection.indices[projection.occupied]]
    return out
