#!/usr/bin/env python3
"""Separate SemanticKITTI MOS point evaluation and official prediction export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SEMANTICKITTI_VALIDATOR = ROOT / "third_party" / "semantic_kitti_api" / "validate_submission.py"
sys.path.insert(0, str(ROOT / "src"))

from helper.egoMotion_compensation_calib import parse_calibration, parse_poses
from helper.mos_benchmark_eval import (TEST_SEQUENCES, add_counts, point_counts,
    point_metrics, read_point_gt, read_scan, validate_export, write_prediction,
    write_submission_zip)
from helper.mos_eval_utils import compute_counts_from_pred_target, empty_counts, metrics_from_counts
from helper.mos_knn_postprocess import knn_unproject_prediction, knn_unproject_probability_sweep
from helper.mos_point_projection import (ProjectionConfig, original_points_to_pixels,
    unproject_prediction)
from helper.mos_label_helper import map_semantickitti_to_movable
from helper.normal_helper import build_normal_xyz
from helper.residual_helper import transform_points
from helper.mos_input import channel_count, normalization, normalize_channels, normalize_feature_tensor
from mos_models import build_mos_model
from temporal_mae_dataset import project_range_features
from utils_torch import spherical_projection


def sequence_root(path):
    path = Path(path).expanduser().resolve()
    if (path / "dataset" / "sequences").is_dir():
        path = path / "dataset" / "sequences"
    elif (path / "sequences").is_dir():
        path = path / "sequences"
    if not path.is_dir():
        raise FileNotFoundError(f"Sequence root missing: {path}")
    return path


def index_sequences(root, seq_ids):
    indexed = {}
    for seq_id in seq_ids:
        folder = root / seq_id / "velodyne"
        scans = sorted(folder.glob("*.bin"))
        if not scans or any(not p.stem.isdigit() for p in scans):
            raise ValueError(f"Missing or invalid .bin scans: {folder}")
        if len({p.stem for p in scans}) != len(scans):
            raise ValueError(f"Duplicate frame stems: {folder}")
        indexed[seq_id] = scans
    return indexed


def load_poses(seq_dir, n_scans):
    pose_path, calib_path = seq_dir / "poses.txt", seq_dir / "calib.txt"
    if not pose_path.is_file() or not calib_path.is_file():
        raise FileNotFoundError(f"Temporal egomotion needs poses.txt AND calib.txt in {seq_dir}; provide estimated odometry for test 11–21")
    calib = parse_calibration(str(calib_path))
    if "Tr" not in calib:
        raise ValueError(f"Calibration lacks Tr: {calib_path}")
    poses = parse_poses(str(pose_path), calib)
    if len(poses) != n_scans or any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
        raise ValueError(f"Pose/scan count or pose values invalid: {pose_path}")
    return poses


def frame_features(points, cfg, data, return_raw_range: bool = False):
    from helper.mos_projection import projection_mode, project_features
    if projection_mode(cfg) == 'lmnet':
        mp = cfg['model_params']
        theta = np.deg2rad([mp.get('FOV_DOWN', -25), mp.get('FOV_UP', 3)])
        feature, mask, _ = project_features(points, int(mp['grid_height']), int(mp['grid_width']), theta, True)
        raw_range = feature[3].numpy().copy()
        mode = data.get('input_mode', 'range_residual')
        feature = normalize_feature_tensor(feature, mask, normalization(cfg, mode))
        if mode.startswith('range_xyz_intensity'):
            channels = list(feature.numpy())
        elif mode.startswith('range_xyz') and 'normal' not in mode:
            channels = list(feature[:4].numpy())
        elif mode in {'range', 'range_residual'}:
            channels = [raw_range]
        elif mode == 'residual':
            channels = []
        else:
            raise ValueError(f'Unsupported LMNet input mode: {mode}')
        return (channels, raw_range) if return_raw_range else channels
    mp = cfg["model_params"]
    h, w = int(mp["grid_height"]), int(mp["grid_width"])
    theta = (np.deg2rad(float(mp.get("FOV_DOWN", -25))), np.deg2rad(float(mp.get("FOV_UP", 3))))
    projected, _, _, _ = spherical_projection(points, height=h, width=w, theta_range=theta)
    xyz = projected[..., :3].copy()
    ranges = np.sqrt(xyz[..., 0] * xyz[..., 0] + xyz[..., 1] * xyz[..., 1]
                     + xyz[..., 2] * xyz[..., 2]).astype(np.float32)
    invalid = ~np.isfinite(xyz).all(axis=-1) | ~np.isfinite(ranges) | (ranges <= 0)
    mode = data.get("input_mode", "range_residual")
    if mode.startswith("range_xyz_normal"):
        min_range = float(mp.get("min_range", mp.get("MIN_RANGE", 0)))
        max_range = mp.get("max_range", mp.get("MAX_RANGE"))
        invalid |= ranges <= min_range
        if max_range is not None:
            invalid |= ranges >= float(max_range)
        normals = build_normal_xyz(xyz)
        normals[invalid] = 0
        channels = [xyz[..., i] for i in range(3)] + [ranges] + [normals[..., i] for i in range(3)]
    elif mode.startswith("range_xyz_intensity"):
        invalid |= ~np.isfinite(projected[..., 3])
        channels = [xyz[..., i] for i in range(3)] + [ranges, projected[..., 3].copy()]
    elif mode.startswith("range_xyz"):
        channels = [xyz[..., i] for i in range(3)] + [ranges]
    elif mode.startswith("range"):
        channels = [ranges]
    elif mode == "residual":
        channels = []
    else:
        raise ValueError(f"Unsupported input mode: {mode}")
    for channel in channels:
        channel[invalid] = 0
    if "intensity" in mode:
        normalize_channels(channels, ["x", "y", "z", "range", "intensity"], ~invalid,
                           normalization(cfg, mode))
    if return_raw_range:
        # This is deliberately captured before input normalization. KNN must
        # compare physical ranges in metres, never an input feature channel.
        return channels, ranges.copy()
    return channels


def single_frame_input(scan, cfg, seq_dir, stem, return_projected_range: bool = False):
    from helper.mos_projection import projection_mode
    lmnet = projection_mode(cfg) == 'lmnet'
    data = cfg["mos_data_params"]
    mode = data.get("input_mode", "range_residual")
    features = frame_features(read_scan(scan), cfg, data, return_raw_range=return_projected_range or lmnet)
    if return_projected_range or lmnet:
        channels, raw_range = features
    else:
        channels = features
    if "residual" in mode:
        offsets = data.get("residual_offsets", [1])
        if isinstance(offsets, int):
            offsets = [offsets]
        for off in sorted({int(v) for v in offsets}):
            template = data.get("residual_folder_template", "residual_images_{offset}")
            from helper.mos_cache import check_cache
            check_cache(seq_dir / template.format(offset=off), cfg, 'residual', seq_dir, off)
            path = seq_dir / template.format(offset=off) / f"{stem}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Required residual image missing: {path}")
            residual = np.load(path)
            if lmnet and not np.isfinite(residual).all():
                raise ValueError(f'Non-finite LMNet residual cache: {path}')
            expected_shape = (int(cfg["model_params"]["grid_height"]), int(cfg["model_params"]["grid_width"]))
            if residual.shape != expected_shape:
                raise ValueError(f"Residual shape mismatch: {path}")
            channels.append(residual.astype(np.float32))
    if not channels:
        raise ValueError("No input channels")
    if lmnet:
        channels = [np.where(raw_range > 0, channel, 0).astype(np.float32) for channel in channels]
        if not all(np.isfinite(channel).all() for channel in channels):
            raise ValueError(f'Non-finite LMNet features: {scan}')
    tensor = torch.from_numpy(np.stack(channels).astype(np.float32))[None]
    return (tensor, raw_range) if return_projected_range else tensor


def temporal_input(scans, index, poses, cfg, history_init, e2e, return_projected_range: bool = False):
    from helper.mos_projection import projection_mode
    lmnet = projection_mode(cfg) == 'lmnet'
    data, mp = cfg["mos_data_params"], cfg["model_params"]
    offsets = sorted({int(v) for v in data.get("previous_offsets", [3])}, reverse=True) if e2e else [int(data.get("previous_offset", 3))]
    h, w = int(mp["grid_height"]), int(mp["grid_width"])
    theta = (np.deg2rad(float(mp.get("FOV_DOWN", -25))), np.deg2rad(float(mp.get("FOV_UP", 3))))
    min_range = float(mp.get("min_range", mp.get("MIN_RANGE", 0)))
    max_range = mp.get("max_range", mp.get("MAX_RANGE"))
    max_range = None if max_range is None else float(max_range)
    features, masks = [], []
    current_raw_range = None
    bootstrapped = False
    for off in offsets + [0]:
        previous = index - off
        missing_history = previous < 0
        if missing_history and lmnet:
            features.append(torch.zeros((5 if 'intensity' in data.get('input_mode', '') else 4, h, w)))
            masks.append(torch.zeros((h, w), dtype=torch.bool))
            bootstrapped = True
            continue
        if missing_history:
            if history_init == "strict":
                raise ValueError(f"Missing history at frame {index}, offset {off}")
            previous, bootstrapped = 0, True
        pts = read_scan(scans[previous])
        if previous != index:
            pts = transform_points(pts, np.linalg.solve(poses[index], poses[previous]))
        feature, mask = project_range_features(pts, h, w, theta, min_range, max_range,
                                                include_intensity="intensity" in data.get("input_mode", ""),
                                                explicit_occupancy=e2e, projection_mode=projection_mode(cfg))
        if off == 0:
            # project_range_features builds [x, y, z, range, (intensity)]
            # before normalization, so channel three is metres.
            current_raw_range = feature[3].numpy().copy()
        feature = normalize_feature_tensor(feature, mask[0], normalization(cfg, data.get("input_mode", "range_xyz")))
        features.append(feature)
        # E2E models require strictly negative past dt. Mark repeated scans as
        # absent history so their nominal dt cannot claim a real observation.
        masks.append(mask[0].bool() & (not (e2e and missing_history)))
    x = torch.stack(features)[None]
    if e2e:
        dt = torch.tensor([[-off / float(data.get("scan_rate_hz", 10)) for off in offsets] + [0.0]], dtype=torch.float32)
        inputs = (x, torch.stack(masks)[None], dt)
    else:
        inputs = (x,)
    if return_projected_range:
        if current_raw_range is None:
            raise RuntimeError("Current projected range was not constructed")
        return inputs, bootstrapped, current_raw_range
    return inputs, bootstrapped


def load_adapter(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint lacks model_state_dict")
    cfg = payload.get("cfg", payload.get("config"))
    if not isinstance(cfg, dict) or "mos_model_params" not in cfg:
        raise ValueError("Checkpoint lacks a MOS config")
    name = cfg["mos_model_params"].get("name")
    if name not in {"salsanext_mos", "salsanext", "salsanext_temporal_mos", "salsanext_temporal_e2e", "unet_small"}:
        raise ValueError(f"Unsupported benchmark model: {name}")
    mode = cfg["mos_data_params"].get("input_mode", "range_residual")
    expected = channel_count(mode, cfg["mos_data_params"].get("residual_offsets", [1]))
    actual = int(cfg["mos_model_params"].get("in_channels", -1))
    if actual != expected:
        raise ValueError(f"Checkpoint config: {mode} requires {expected} channels, model has {actual}")
    normalization(cfg, mode)
    model = build_mos_model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg, payload


def semantic_gate_ablation_metadata(model, disable=False):
    """Validate the runtime option AFTER load_adapter has strictly loaded weights."""
    from mos_models.salsanext_temporal_e2e import SalsaNextTemporalE2E
    fusion = getattr(model, "semantic_fusion", "none")
    supported = (isinstance(model, SalsaNextTemporalE2E)
                 and fusion == "bottleneck_gate" and hasattr(model, "semantic_gate"))
    if supported and any(not torch.isfinite(parameter).all() for parameter in model.semantic_gate.parameters()):
        raise ValueError("Checkpoint semantic_gate contains non-finite parameters")
    if disable and not supported:
        raise ValueError("--disable-semantic-gate requires a strictly loaded SalsaNextTemporalE2E "
                         "checkpoint with semantic_fusion=bottleneck_gate and semantic_gate")
    return dict(supported=supported, checkpoint_fusion=fusion,
                gate_enabled=bool(supported and not disable), disable_semantic_gate=bool(disable),
                mode="runtime_forward_bypass", checkpoint_unchanged=True)


TEMPORAL_ABLATIONS = ('none', 'mask_history', 'bypass_temporal')


def temporal_ablation_metadata(model, mode='none', disable_semantic_gate=False):
    """Describe a runtime intervention after strict loading, without touching weights."""
    from mos_models.salsanext_temporal_e2e import SalsaNextTemporalE2E
    if mode not in TEMPORAL_ABLATIONS:
        raise ValueError(f'Unknown temporal ablation: {mode}')
    supported = isinstance(model, SalsaNextTemporalE2E)
    if mode != 'none' and not supported:
        raise ValueError('--temporal-ablation requires a strictly loaded SalsaNextTemporalE2E checkpoint')
    if mode != 'none' and disable_semantic_gate:
        raise ValueError('--disable-semantic-gate cannot be combined with a temporal ablation')
    return dict(mode=mode, checkpoint_unchanged=True, historical_inputs_masked=mode == 'mask_history',
                temporal_module_executed=supported and mode != 'bypass_temporal',
                semantic_gate_unchanged=not disable_semantic_gate)


def apply_temporal_ablation(inputs, mode='none'):
    """Mask only copies of historical features/occupancy immediately before forward."""
    if mode not in TEMPORAL_ABLATIONS:
        raise ValueError(f'Unknown temporal ablation: {mode}')
    if mode != 'mask_history':
        return inputs
    if len(inputs) != 3 or inputs[0].ndim != 5 or inputs[0].shape[1] < 2:
        raise ValueError('mask_history requires temporal E2E inputs (x[B,T,C,H,W], valid, dt)')
    x, valid, dt = inputs
    x, valid = x.clone(), valid.clone()
    x[:, :-1] = 0
    valid[:, :-1] = False
    return x, valid, dt


def temporal_gamma_metadata(model):
    """Read the loaded attention scale; never modify parameters or RNG state."""
    gamma = getattr(getattr(model, 'temporal', None), 'gamma', None)
    if gamma is None:
        return None
    gamma = gamma.detach().float().cpu()
    if not torch.isfinite(gamma).all():
        raise ValueError('Checkpoint temporal.gamma contains non-finite values')
    return dict(gamma_mean=float(gamma.mean()), gamma_mean_abs=float(gamma.abs().mean()),
                gamma_min=float(gamma.min()), gamma_max=float(gamma.max()))


def save_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


SWEEP_THRESHOLDS = tuple([round(value * 0.05, 2) for value in range(1, 20)] + [0.975, 0.99])
MOVABLE_REFERENCE_BASELINE = {"tp_moving": 1204696, "fp_moving": 345786,
                              "fn_moving": 459425, "point_moving_iou": 0.599379}


def moving_prediction_from_probability(moving_probability_hw, threshold: float):
    """Convert moving probabilities to binary labels; ties stay static like argmax."""
    probability = np.asarray(moving_probability_hw)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("Moving probability must be a finite [H,W] array")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Moving threshold must be finite and within [0, 1]")
    return (probability > threshold).astype(np.uint8)


def moving_probability_from_logits(logits: torch.Tensor):
    """Return [H,W] moving probabilities from a single two-class model inference."""
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError("Expected logits shaped [1,2,H,W]")
    return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()


def online_lmnet_pixel_target(points, point_target, cfg):
    """Project original SemanticKITTI MOS point labels into the LMNet range view.

    LMNet training generates these targets online instead of reading a cached
    ``mos_labels/*.npy`` image.  Benchmark pixel metrics must use the same
    nearest-point-wins projection to remain comparable with training.
    """
    from helper.mos_projection import lmnet_projection, project_targets

    mp = cfg["model_params"]
    projection = lmnet_projection(
        points, int(mp["grid_height"]), int(mp["grid_width"]),
        float(mp.get("FOV_UP", 3)), float(mp.get("FOV_DOWN", -25)),
        min_range=float(mp.get("min_range", 0)), max_range=mp.get("max_range"),
    )
    return project_targets(point_target, projection)


def movable_scores_for_points(movable_probability_hw, points, projection, projected_range):
    """Map the auxiliary score to original points without changing their order.

    A usable point still needs an occupied current-frame pixel.  In particular,
    this keeps out-of-range / filtered points from being accidentally gated by a
    numerically valid value in an otherwise unused output pixel.
    """
    score = np.asarray(movable_probability_hw, dtype=np.float32)
    ranges = np.asarray(projected_range)
    if score.shape != (projection.height, projection.width) or ranges.shape != score.shape:
        raise ValueError("Movable score and current projected range must match the projection grid")
    rows, cols, usable = original_points_to_pixels(points, projection)
    valid = usable.copy()
    valid[usable] &= np.isfinite(score[rows[usable], cols[usable]])
    valid[usable] &= np.isfinite(ranges[rows[usable], cols[usable]]) & (ranges[rows[usable], cols[usable]] > 0)
    point_score = np.full(len(points), np.nan, dtype=np.float32)
    point_score[valid] = score[rows[valid], cols[valid]]
    return point_score, valid


def movable_gate_counts(mos_pred, mos_target, movable_score, movable_score_valid,
                        threshold, movable_target=None):
    """Apply the diagnostic-only gate and return invariant-friendly counters."""
    mos_pred = np.asarray(mos_pred, dtype=np.uint8)
    mos_target = np.asarray(mos_target, dtype=np.int16)
    score = np.asarray(movable_score)
    score_valid = np.asarray(movable_score_valid, dtype=bool)
    if (mos_pred.ndim != 1 or mos_pred.shape != mos_target.shape or score.shape != mos_pred.shape
            or score_valid.shape != mos_pred.shape):
        raise ValueError("MOS prediction, targets, scores, and score mask must be aligned one-dimensional arrays")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Movable threshold must be finite and within [0, 1]")
    remove = (mos_pred == 1) & score_valid & (score <= threshold)
    gated = (mos_pred == 1) & ~remove
    base, gated_counts = point_counts(mos_pred, mos_target), point_counts(gated.astype(np.uint8), mos_target)
    gt_valid = mos_target != -1
    result = {
        **{f"baseline_{key}": value for key, value in base.items()},
        **{f"gated_{key}": value for key, value in gated_counts.items()},
        "removed_moving_predictions": int(remove.sum()),
        "removed_mos_false_positives": int((remove & gt_valid & (mos_target == 0)).sum()),
        "lost_mos_true_positives": int((remove & gt_valid & (mos_target == 1)).sum()),
        "removed_ignored_mos_points": int((remove & ~gt_valid).sum()),
        "movable_score_valid_points": int(score_valid.sum()),
        "movable_score_missing_points": int((~score_valid).sum()),
        "movable_score_valid_mos_eval_points": int((score_valid & gt_valid).sum()),
    }
    if movable_target is not None:
        movable_target = np.asarray(movable_target, dtype=np.int16)
        if movable_target.shape != mos_pred.shape or not np.isin(movable_target, [-1, 0, 1]).all():
            raise ValueError("Invalid movable ground truth")
        aux_valid = movable_target != -1
        result.update({
            "removed_aux_movable_points": int((remove & aux_valid & (movable_target == 1)).sum()),
            "removed_aux_nonmovable_points": int((remove & aux_valid & (movable_target == 0)).sum()),
            "removed_aux_ignored_points": int((remove & ~aux_valid).sum()),
        })
    return gated.astype(np.uint8), remove, result


def movable_diagnostic_rows(by_threshold):
    rows = []
    for threshold, counts in by_threshold.items():
        row = {"movable_threshold": threshold, **counts}
        row.update({f"baseline_{key}": value for key, value in point_metrics({
            key: counts[f"baseline_{key}"] for key in ("tp_moving", "fp_moving", "fn_moving", "tn_static")
        }).items()})
        row.update({f"gated_{key}": value for key, value in point_metrics({
            key: counts[f"gated_{key}"] for key in ("tp_moving", "fp_moving", "fn_moving", "tn_static")
        }).items()})
        row["fp_removed_fraction"] = (row["removed_mos_false_positives"] / row["baseline_fp_moving"]
                                      if row["baseline_fp_moving"] else 0.0)
        row["tp_lost_fraction"] = (row["lost_mos_true_positives"] / row["baseline_tp_moving"]
                                   if row["baseline_tp_moving"] else 0.0)
        rows.append(row)
    return rows


def threshold_sweep_rows(point_by_threshold, pixel_by_threshold):
    """Produce aggregate rows using the established point and pixel metric helpers."""
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        point = point_by_threshold[threshold]
        pixel = pixel_by_threshold[threshold]
        point_row = point_metrics(point)
        pixel_row = metrics_from_counts(pixel)
        rows.append({
            "threshold": threshold,
            "point_tp_moving": point["tp_moving"], "point_fp_moving": point["fp_moving"],
            "point_fn_moving": point["fn_moving"], "point_tn_static": point["tn_static"],
            "point_n_gt_valid": point["n_gt_valid"], **point_row,
            "pixel_tp_moving": pixel["tp_moving"], "pixel_fp_moving": pixel["fp_moving"],
            "pixel_fn_moving": pixel["fn_moving"], "pixel_tn_static": pixel["tn_static"],
            "pixel_valid_pixels": pixel["valid_pixels"],
            **{f"pixel_{key}": value for key, value in pixel_row.items()},
        })
    return rows


def select_best_threshold(rows):
    """Maximize point moving IoU; resolve exact ties by proximity to 0.5."""
    if not rows:
        raise ValueError("No threshold rows to select from")
    return max(rows, key=lambda row: (row["point_moving_iou"], -abs(row["threshold"] - 0.5)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-semantic-gate", action="store_true",
                        help="Bypass only the trained semantic bottleneck gate at runtime")
    parser.add_argument('--temporal-ablation', choices=TEMPORAL_ABLATIONS, default='none',
                        help='Inference-only history masking or complete temporal attention bypass')
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+", default=["08"])
    parser.add_argument("--mode", choices=["pixel", "point", "both"], default="point")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--submission-zip", action="store_true")
    parser.add_argument("--moving-threshold", type=float, default=0.5,
                        help="Moving-class probability threshold (default: 0.5)")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Evaluate fixed moving thresholds on validation sequence 08")
    parser.add_argument("--movable-diagnostics", action="store_true",
                        help="Gate point-level MOS predictions with the movable auxiliary head")
    parser.add_argument("--movable-thresholds", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="Movable-score gate thresholds (default: 0.1 0.3 0.5 0.7 0.9)")
    parser.add_argument("--postprocess", choices=("direct", "knn"), default="direct",
                        help="Point-level range-view unprojection method")
    parser.add_argument("--knn-k", type=int, default=5, help="LMNet KNN neighbors")
    parser.add_argument("--knn-search", type=int, default=5, help="Odd LMNet KNN window width")
    parser.add_argument("--knn-sigma", type=float, default=1.0, help="LMNet spatial Gaussian sigma")
    parser.add_argument("--knn-cutoff", type=float, default=1.0,
                        help="LMNet weighted-distance cutoff; <=0 disables filtering")
    parser.add_argument("--knn-chunk-size", type=int, default=32768,
                        help="Original points processed together during KNN unprojection")
    parser.add_argument("--history-init", choices=["repeat_earliest", "strict"], default="repeat_earliest")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--official-api-dir", type=Path,
                        help="Optional SemanticKITTI API checkout overriding the vendored official validator")
    parser.add_argument("--max-frames", type=int, help="Smoke test only; disables export")
    args = parser.parse_args(argv)
    if args.temporal_ablation != 'none':
        if args.disable_semantic_gate:
            parser.error('--disable-semantic-gate cannot be combined with --temporal-ablation != none')
        if args.threshold_sweep or args.moving_threshold != 0.5 or args.movable_diagnostics:
            parser.error('Temporal ablations require fixed --moving-threshold 0.5 without threshold sweep or movable diagnostics')
    ids = [str(s).zfill(2) for s in args.sequences]
    if len(set(ids)) != len(ids):
        parser.error("Duplicate sequence IDs")
    if args.submission_zip and (not args.export_predictions or set(ids) != set(TEST_SEQUENCES)):
        parser.error("--submission-zip requires export and all test sequences 11..21")
    if args.max_frames is not None and (args.max_frames < 1 or args.export_predictions):
        parser.error("--max-frames must be positive and cannot be used with export")
    if not np.isfinite(args.moving_threshold) or not 0.0 <= args.moving_threshold <= 1.0:
        parser.error("--moving-threshold must be finite and within [0, 1]")
    if args.threshold_sweep:
        if ids != ["08"]:
            parser.error("--threshold-sweep is restricted to validation sequence 08")
        if args.mode != "both":
            parser.error("--threshold-sweep requires --mode both for pixel and point metrics")
        if args.export_predictions or args.submission_zip:
            parser.error("--threshold-sweep does not export predictions or submissions")
    if args.movable_diagnostics:
        if args.mode not in {"point", "both"}:
            parser.error("--movable-diagnostics requires --mode point or both")
        if args.export_predictions or args.submission_zip:
            parser.error("--movable-diagnostics is evaluation-only and cannot export predictions")
        if (not args.movable_thresholds or any(not np.isfinite(v) or not 0 <= v <= 1
                                               for v in args.movable_thresholds)
                or len(set(args.movable_thresholds)) != len(args.movable_thresholds)):
            parser.error("--movable-thresholds must be unique finite values within [0, 1]")
        # 0.5 is the pre-registered operating point and is always reported,
        # even when a caller supplies a narrower exploratory sweep.
        args.movable_thresholds = [0.5] + [v for v in args.movable_thresholds if v != 0.5]
    if args.knn_k < 1 or args.knn_search < 1 or args.knn_search % 2 != 1:
        parser.error("--knn-k must be positive and --knn-search must be positive and odd")
    if args.knn_k > args.knn_search * args.knn_search:
        parser.error("--knn-k must not exceed --knn-search squared")
    if args.knn_sigma <= 0 or not np.isfinite(args.knn_sigma) or not np.isfinite(args.knn_cutoff):
        parser.error("--knn-sigma must be finite and positive; --knn-cutoff must be finite")
    if args.knn_chunk_size < 1:
        parser.error("--knn-chunk-size must be positive")
    root = sequence_root(args.dataset_root)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == root.parent or root.parent in output_dir.parents:
        parser.error("--output-dir must be outside the original dataset")
    args.output_dir = output_dir
    indexed = index_sequences(root, ids)
    device = torch.device(args.device)
    model, cfg, payload = load_adapter(args.checkpoint, device)
    gate_metadata = semantic_gate_ablation_metadata(model, args.disable_semantic_gate)
    temporal_metadata = temporal_ablation_metadata(model, args.temporal_ablation, args.disable_semantic_gate)
    gamma_metadata = temporal_gamma_metadata(model)
    forward_options = {"disable_semantic_gate": True} if args.disable_semantic_gate else {}
    if args.temporal_ablation == 'bypass_temporal':
        forward_options['disable_temporal_fusion'] = True
    if args.movable_diagnostics:
        from helper.mos_auxiliary import validate_auxiliary_config
        auxiliary = validate_auxiliary_config(cfg)
        if not auxiliary["enabled"]:
            parser.error("--movable-diagnostics requires a checkpoint with enabled movable auxiliary supervision")
        if auxiliary["task"] != "movable":
            parser.error("--movable-diagnostics requires a movable auxiliary checkpoint; residual checkpoints are unsupported")
    from helper.mos_projection import projection_mode
    from helper.mos_cache import check_cache
    lmnet = projection_mode(cfg) == 'lmnet'
    name = cfg["mos_model_params"]["name"]
    temporal = name in {"salsanext_temporal_mos", "salsanext_temporal_e2e"}
    e2e = name == "salsanext_temporal_e2e"
    mp = cfg["model_params"]
    input_mode = cfg["mos_data_params"].get("input_mode", "range_residual")
    projection = ProjectionConfig(int(mp["grid_height"]), int(mp["grid_width"]),
        float(mp.get("FOV_UP", 3)), float(mp.get("FOV_DOWN", -25)),
        float(mp.get("min_range", mp.get("MIN_RANGE", 0))) if (temporal or "normal" in input_mode) else 0.0,
        float(mp.get("max_range", mp.get("MAX_RANGE"))) if (temporal or "normal" in input_mode) and mp.get("max_range", mp.get("MAX_RANGE")) is not None else None,
        temporal, projection_mode(cfg))
    poses = {seq: load_poses(root / seq, len(scans)) for seq, scans in indexed.items()} if temporal else {}
    for seq, scans in indexed.items():
        if args.mode in {'pixel', 'both'} and not lmnet:
            check_cache(root / seq / cfg['mos_data_params'].get('mos_label_folder', 'mos_labels'),
                        cfg, 'labels', root / seq)
        needs_original_labels = args.mode in {"point", "both"} or (lmnet and args.mode in {"pixel", "both"})
        if needs_original_labels and any(not (root / seq / "labels" / f"{p.stem}.label").is_file() for p in scans):
            if not args.export_predictions or seq not in TEST_SEQUENCES:
                raise FileNotFoundError(f"Original point GT missing for {seq}; use --export-predictions with --mode point for test 11..21")
            print(f"{seq}: no public GT; export only, no point metric")
        if args.movable_diagnostics and any(not (root / seq / "labels" / f"{p.stem}.label").is_file() for p in scans):
            raise FileNotFoundError(f"--movable-diagnostics requires original .label ground truth: {root / seq / 'labels'}")
        if (args.mode in {"pixel", "both"} and not lmnet
                and any(not (root / seq / cfg["mos_data_params"].get("mos_label_folder", "mos_labels") / f"{p.stem}.npy").is_file() for p in scans)):
            raise FileNotFoundError(f"Pixel GT missing for {seq}; use --mode point")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    totals, pixel_total, per_seq, seq_summary, coverage, pixel_rows = {}, empty_counts(), [], [], [], []
    sweep_point = {threshold: {} for threshold in SWEEP_THRESHOLDS} if args.threshold_sweep else None
    sweep_pixel = {threshold: empty_counts() for threshold in SWEEP_THRESHOLDS} if args.threshold_sweep else None
    movable_totals = {threshold: {} for threshold in args.movable_thresholds} if args.movable_diagnostics else None
    total_scans = 0
    point_postprocess_seconds = 0.0
    with torch.inference_mode():
        for seq, scans in indexed.items():
            seq_counts, seq_pixel = {}, empty_counts()
            bootstrapped = 0
            selected = scans[:args.max_frames] if args.max_frames else scans
            for index, scan in enumerate(selected):
                if temporal:
                    temporal_result = temporal_input(
                        scans, index, poses[seq], cfg, args.history_init, e2e,
                        return_projected_range=args.postprocess == "knn" or args.movable_diagnostics,
                    )
                    if args.postprocess == "knn" or args.movable_diagnostics:
                        inputs, boot, projected_range = temporal_result
                    else:
                        inputs, boot = temporal_result
                    bootstrapped += int(boot)
                else:
                    single_result = single_frame_input(
                        scan, cfg, root / seq, scan.stem,
                        return_projected_range=args.postprocess == "knn" or args.movable_diagnostics,
                    )
                    if args.postprocess == "knn" or args.movable_diagnostics:
                        model_input, projected_range = single_result
                    else:
                        model_input = single_result
                    inputs = (model_input,)
                inputs = apply_temporal_ablation(inputs, args.temporal_ablation)
                if args.movable_diagnostics:
                    output = model(*(part.to(device) for part in inputs), return_aux=True, **forward_options)
                    if not isinstance(output, tuple) or len(output) != 2:
                        raise ValueError("Movable checkpoint did not return (mos_logits, movable_logits)")
                    logits, movable_logits = output
                    if movable_logits.shape != (1, 2, projection.height, projection.width):
                        raise ValueError(f"Unexpected movable-head output shape: {tuple(movable_logits.shape)}")
                    movable_probability = moving_probability_from_logits(movable_logits)
                else:
                    logits = model(*(part.to(device) for part in inputs), **forward_options)
                if logits.shape != (1, 2, projection.height, projection.width):
                    raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
                # Softmax is computed once. A threshold of 0.5 with strict
                # comparison is equivalent to two-class argmax, including ties.
                moving_probability = moving_probability_from_logits(logits)
                pred_hw = moving_prediction_from_probability(moving_probability, args.moving_threshold)
                points = read_scan(scan)
                postprocess_start = time.perf_counter()
                if args.postprocess == "knn":
                    pred, valid = knn_unproject_prediction(
                        pred_hw, projected_range, points, projection, device=device,
                        knn=args.knn_k, search=args.knn_search, sigma=args.knn_sigma,
                        cutoff=args.knn_cutoff, chunk_size=args.knn_chunk_size,
                    )
                else:
                    pred, valid = unproject_prediction(pred_hw, points, projection)
                postprocess_seconds = time.perf_counter() - postprocess_start
                point_postprocess_seconds += postprocess_seconds
                coverage.append({"sequence": seq, "frame": scan.stem, "n_points": len(points),
                    'history_available': ([index >= off for off in sorted(cfg['mos_data_params'].get('previous_offsets', []), reverse=True)] if e2e else []),
                    'history_used': ([bool(mask.any()) and args.temporal_ablation != 'bypass_temporal'
                                      for mask in inputs[1][0, :-1]] if e2e else []),
                    "invalid_points": int((~valid).sum()), "fallback_points": int((~valid).sum()),
                    "coverage_rate": float(valid.mean()) if len(valid) else 0.0,
                    "postprocess_seconds": postprocess_seconds})
                gt_path = root / seq / "labels" / f"{scan.stem}.label"
                point_target = read_point_gt(gt_path, len(points)) if gt_path.is_file() else None
                if args.mode in {"point", "both"} and point_target is not None:
                    add_counts(seq_counts, point_counts(pred, point_target))
                    seq_counts["n_points"] = seq_counts.get("n_points", 0) + len(points)
                    seq_counts["fallback_points"] = seq_counts.get("fallback_points", 0) + int((~valid).sum())
                if args.movable_diagnostics:
                    raw_labels = np.fromfile(gt_path, dtype="<u4")
                    if len(raw_labels) != len(points):
                        raise ValueError(f"GT/scan point count mismatch: {gt_path}: {len(raw_labels)} != {len(points)}")
                    movable_target = map_semantickitti_to_movable(raw_labels)
                    movable_score, movable_valid = movable_scores_for_points(
                        movable_probability, points, projection, projected_range)
                    coverage[-1].update({
                        "movable_score_valid_points": int(movable_valid.sum()),
                        "movable_score_missing_points": int((~movable_valid).sum()),
                        "movable_score_coverage_rate": float(movable_valid.mean()) if len(movable_valid) else 0.0,
                    })
                    for threshold in args.movable_thresholds:
                        _, _, counts = movable_gate_counts(pred, point_target, movable_score,
                                                           movable_valid, threshold, movable_target)
                        add_counts(movable_totals[threshold], counts)
                pixel_target = None
                if args.mode in {"pixel", "both"}:
                    pixel_target = (online_lmnet_pixel_target(points, point_target, cfg) if lmnet else np.load(
                        root / seq / cfg["mos_data_params"].get("mos_label_folder", "mos_labels") / f"{scan.stem}.npy"))
                    if pixel_target.shape != pred_hw.shape or not np.isin(pixel_target, [-1, 0, 1]).all():
                        raise ValueError(f"Invalid pixel GT: {seq}/{scan.stem}")
                    if e2e:
                        pixel_target = pixel_target.copy()
                        pixel_target[~inputs[1][0, -1].numpy()] = -1
                    counts = compute_counts_from_pred_target(torch.from_numpy(pred_hw), torch.from_numpy(pixel_target))
                    for key in seq_pixel:
                        seq_pixel[key] += counts[key]
                if args.threshold_sweep:
                    # No second model pass: each candidate uses these already computed probabilities.
                    knn_sweep_predictions = None
                    if args.postprocess == "knn":
                        knn_sweep_predictions, _ = knn_unproject_probability_sweep(
                            moving_probability, SWEEP_THRESHOLDS, projected_range, points, projection, device=device,
                            knn=args.knn_k, search=args.knn_search, sigma=args.knn_sigma,
                            cutoff=args.knn_cutoff, chunk_size=args.knn_chunk_size,
                        )
                    for threshold_index, threshold in enumerate(SWEEP_THRESHOLDS):
                        threshold_hw = moving_prediction_from_probability(moving_probability, threshold)
                        if args.postprocess == "knn":
                            threshold_point = knn_sweep_predictions[threshold_index]
                        else:
                            threshold_point, _ = unproject_prediction(threshold_hw, points, projection)
                        add_counts(sweep_point[threshold], point_counts(threshold_point, point_target))
                        threshold_counts = compute_counts_from_pred_target(
                            torch.from_numpy(threshold_hw), torch.from_numpy(pixel_target),
                        )
                        for key in sweep_pixel[threshold]:
                            sweep_pixel[threshold][key] += threshold_counts[key]
                if args.export_predictions:
                    dest = args.output_dir / "predictions" / "sequences" / seq / "predictions" / f"{scan.stem}.label"
                    write_prediction(dest, pred)
                total_scans += 1
                if total_scans % 100 == 0:
                    print(f"Processed {total_scans} scans", flush=True)
            if seq_counts:
                add_counts(totals, seq_counts)
                per_seq.append({"sequence": seq, "n_scans": len(selected), "num_bootstrapped_frames": bootstrapped,
                                **seq_counts, **point_metrics(seq_counts)})
            if args.mode in {"pixel", "both"}:
                for key in pixel_total:
                    pixel_total[key] += seq_pixel[key]
                pixel_rows.append({"sequence": seq, "n_scans": len(selected), **seq_pixel,
                    **{k if k.startswith("pixel_") else f"pixel_{k}": v
                       for k, v in metrics_from_counts(seq_pixel).items()}})
            seq_summary.append({"sequence": seq, "n_scans": len(selected),
                                "num_bootstrapped_frames": bootstrapped,
                                "has_point_gt": bool(seq_counts)})
            print(f"{seq}: {len(selected)} scans, {bootstrapped} bootstrapped", flush=True)
    if args.export_predictions:
        validate_export(args.output_dir / "predictions", indexed)
    if args.submission_zip:
        write_submission_zip(args.output_dir / "submission.zip", args.output_dir / "predictions", indexed)
        validator = (args.official_api_dir / "validate_submission.py"
                     if args.official_api_dir else LOCAL_SEMANTICKITTI_VALIDATOR)
        if not validator.is_file():
            raise FileNotFoundError(f"Official validator missing: {validator}")
        subprocess.run([sys.executable, str(validator), str(args.output_dir / "submission.zip"),
            str(root.parent), "--task", "segmentation"], check=True)
    if totals:
        save_csv(args.output_dir / "point_metrics.csv", [{"n_scans": total_scans, **totals, **point_metrics(totals)}])
        save_csv(args.output_dir / "point_per_sequence.csv", per_seq)
        print(f"point_moving_iou={point_metrics(totals)['point_moving_iou']:.9f}")
        if args.movable_diagnostics:
            observed = point_metrics(totals)
            print("baseline deviation vs archived Val08 reference: " + ", ".join(
                f"{key}={totals.get(key, observed.get(key)) - reference:+.9f}"
                for key, reference in MOVABLE_REFERENCE_BASELINE.items()))
    if pixel_rows:
        pixel_rows.insert(0, {"sequence": "ALL", "n_scans": total_scans, **pixel_total,
            **{k if k.startswith("pixel_") else f"pixel_{k}": v
               for k, v in metrics_from_counts(pixel_total).items()}})
        save_csv(args.output_dir / "pixel_metrics.csv", pixel_rows)
    save_csv(args.output_dir / "projection_coverage.csv", coverage)
    save_csv(args.output_dir / "sequence_summary.csv", seq_summary)
    sweep_rows, best_threshold = None, None
    if args.threshold_sweep:
        sweep_rows = threshold_sweep_rows(sweep_point, sweep_pixel)
        best_threshold = select_best_threshold(sweep_rows)
        at_half = next(row for row in sweep_rows if row["threshold"] == 0.5)
        best_yaml = {
            "selection_sequence": "08",
            "selection_metric": "point_moving_iou",
            "tie_break": "closest_to_0.5",
            "best_threshold": best_threshold["threshold"],
            "best": best_threshold,
            "threshold_0.5": at_half,
            "improvement_vs_0.5": {
                "point_moving_iou": best_threshold["point_moving_iou"] - at_half["point_moving_iou"],
                "pixel_moving_iou": best_threshold["pixel_moving_iou"] - at_half["pixel_moving_iou"],
            },
        }
        save_csv(args.output_dir / "threshold_sweep.csv", sweep_rows)
        with (args.output_dir / "threshold_best.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump(best_yaml, handle, sort_keys=False)
        print(f"best_threshold={best_threshold['threshold']:.3f} "
              f"point_moving_iou={best_threshold['point_moving_iou']:.9f}")
    movable_rows, movable_best = None, None
    if args.movable_diagnostics:
        movable_rows = movable_diagnostic_rows(movable_totals)
        save_csv(args.output_dir / "movable_diagnostics.csv", movable_rows)
        at_half = next(row for row in movable_rows if row["movable_threshold"] == 0.5)
        movable_best = max(movable_rows, key=lambda row: (row["gated_point_moving_iou"],
                                                           -abs(row["movable_threshold"] - 0.5)))
        print("movable diagnostic baseline: "
              f"IoU={at_half['baseline_point_moving_iou']:.9f} TP={at_half['baseline_tp_moving']} "
              f"FP={at_half['baseline_fp_moving']} FN={at_half['baseline_fn_moving']}")
        print("movable diagnostic threshold=0.500: "
              f"IoU={at_half['gated_point_moving_iou']:.9f} FP_removed={at_half['removed_mos_false_positives']} "
              f"TP_lost={at_half['lost_mos_true_positives']}")
        print(f"movable diagnostic exploratory best={movable_best['movable_threshold']:.3f} "
              f"IoU={movable_best['gated_point_moving_iou']:.9f}")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    checkpoint_hash = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_hash.update(block)
    metadata = {"checkpoint_sha256": checkpoint_hash.hexdigest(),
        "semantic_gate_ablation": gate_metadata, "checkpoint_model_config": cfg,
        "temporal_ablation": temporal_metadata, "temporal_attention_gamma": gamma_metadata,
        "history_reference_offsets": cfg['mos_data_params'].get('previous_offsets', []) if e2e else [],
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_train_sequences": cfg["mos_data_params"].get("train_sequences"),
        "eval_sequences": ids, "sequence_07_role": next((role for role in ('train', 'val', 'test')
            if '07' in {str(s).zfill(2) for s in cfg['mos_data_params'].get(role+'_sequences', [])}), 'unused'),
        "dataset_root": str(root), "model": name, "input_mode": input_mode,
        "channel_names": (["x", "y", "z", "range", "intensity"] if "intensity" in input_mode else None),
        "normalization": normalization(cfg, input_mode),
        "moving_threshold": args.moving_threshold,
        "threshold_sweep": ({"enabled": True, "selection_sequence": "08",
            "thresholds": list(SWEEP_THRESHOLDS), "selection_metric": "point_moving_iou",
            "tie_break": "closest_to_0.5", "best_threshold": best_threshold["threshold"],
            "improvement_vs_0.5": best_threshold["point_moving_iou"] - next(
                row["point_moving_iou"] for row in sweep_rows if row["threshold"] == 0.5)}
            if args.threshold_sweep else {"enabled": False}),
        "movable_diagnostics": ({"enabled": True, "thresholds": args.movable_thresholds,
            "filter": "remove = mos_pred_moving & movable_score_valid & (p_movable <= tau)",
            "score_mapping": "direct original-point-to-current-pixel lookup; unmapped scores leave MOS unchanged",
            "primary": "MOS KNN point prediction + directly unprojected movable probability",
            "threshold_0.5": at_half, "exploratory_best": movable_best,
            "archived_val08_baseline_reference": MOVABLE_REFERENCE_BASELINE,
            "threshold_selection_note": "Val08 sweep is exploratory; fix a threshold before independent-test evaluation."}
            if args.movable_diagnostics else {"enabled": False}),
        "projection": vars(projection), "history_init": ('zero_missing' if projection_mode(cfg) == 'lmnet' else args.history_init) if temporal else None,
        "pose_source": "poses.txt + calib.txt (LiDAR frame)" if temporal else None,
        "gt_mapping": "PRBonn semantic-kitti-mos learning_map (local mos_label_helper)",
        "unmapped_point_fallback": "static", "n_scans": total_scans,
        "point_postprocess_seconds_total": point_postprocess_seconds,
        "point_postprocess_seconds_per_scan": point_postprocess_seconds / total_scans if total_scans else 0.0,
        "postprocessing": {"method": args.postprocess,
            **({"knn": args.knn_k, "search": args.knn_search, "sigma": args.knn_sigma,
                "cutoff": args.knn_cutoff, "chunk_size": args.knn_chunk_size}
               if args.postprocess == "knn" else {})},
        "partial_smoke_test": bool(args.max_frames), "pipeline_git_commit": git_commit}
    with (args.output_dir / "benchmark_eval_config.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    print(f"Saved benchmark output: {args.output_dir}")


if __name__ == "__main__":
    main()
