#!/usr/bin/env python3
"""Separate SemanticKITTI MOS point evaluation and official prediction export."""

from __future__ import annotations

import argparse
import csv
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
from helper.mos_knn_postprocess import knn_unproject_prediction
from helper.mos_point_projection import ProjectionConfig, unproject_prediction
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
    data = cfg["mos_data_params"]
    mode = data.get("input_mode", "range_residual")
    features = frame_features(read_scan(scan), cfg, data, return_raw_range=return_projected_range)
    if return_projected_range:
        channels, raw_range = features
    else:
        channels = features
    if "residual" in mode:
        offsets = data.get("residual_offsets", [1])
        if isinstance(offsets, int):
            offsets = [offsets]
        for off in sorted({int(v) for v in offsets}):
            template = data.get("residual_folder_template", "residual_images_{offset}")
            path = seq_dir / template.format(offset=off) / f"{stem}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Required residual image missing: {path}")
            residual = np.load(path)
            expected_shape = (int(cfg["model_params"]["grid_height"]), int(cfg["model_params"]["grid_width"]))
            if residual.shape != expected_shape:
                raise ValueError(f"Residual shape mismatch: {path}")
            channels.append(residual.astype(np.float32))
    if not channels:
        raise ValueError("No input channels")
    tensor = torch.from_numpy(np.stack(channels).astype(np.float32))[None]
    return (tensor, raw_range) if return_projected_range else tensor


def temporal_input(scans, index, poses, cfg, history_init, e2e, return_projected_range: bool = False):
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
        if missing_history:
            if history_init == "strict":
                raise ValueError(f"Missing history at frame {index}, offset {off}")
            previous, bootstrapped = 0, True
        pts = read_scan(scans[previous])
        if previous != index:
            pts = transform_points(pts, np.linalg.solve(poses[index], poses[previous]))
        feature, mask = project_range_features(pts, h, w, theta, min_range, max_range,
                                                include_intensity="intensity" in data.get("input_mode", ""),
                                                explicit_occupancy=e2e)
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


def save_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequences", nargs="+", default=["08"])
    parser.add_argument("--mode", choices=["pixel", "point", "both"], default="point")
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument("--submission-zip", action="store_true")
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
    ids = [str(s).zfill(2) for s in args.sequences]
    if len(set(ids)) != len(ids):
        parser.error("Duplicate sequence IDs")
    if args.submission_zip and (not args.export_predictions or set(ids) != set(TEST_SEQUENCES)):
        parser.error("--submission-zip requires export and all test sequences 11..21")
    if args.max_frames is not None and (args.max_frames < 1 or args.export_predictions):
        parser.error("--max-frames must be positive and cannot be used with export")
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
    name = cfg["mos_model_params"]["name"]
    temporal = name in {"salsanext_temporal_mos", "salsanext_temporal_e2e"}
    e2e = name == "salsanext_temporal_e2e"
    mp = cfg["model_params"]
    input_mode = cfg["mos_data_params"].get("input_mode", "range_residual")
    projection = ProjectionConfig(int(mp["grid_height"]), int(mp["grid_width"]),
        float(mp.get("FOV_UP", 3)), float(mp.get("FOV_DOWN", -25)),
        float(mp.get("min_range", mp.get("MIN_RANGE", 0))) if (temporal or "normal" in input_mode) else 0.0,
        float(mp.get("max_range", mp.get("MAX_RANGE"))) if (temporal or "normal" in input_mode) and mp.get("max_range", mp.get("MAX_RANGE")) is not None else None,
        temporal)
    poses = {seq: load_poses(root / seq, len(scans)) for seq, scans in indexed.items()} if temporal else {}
    for seq, scans in indexed.items():
        if args.mode in {"point", "both"} and any(not (root / seq / "labels" / f"{p.stem}.label").is_file() for p in scans):
            if not args.export_predictions or seq not in TEST_SEQUENCES:
                raise FileNotFoundError(f"Original point GT missing for {seq}; use --export-predictions with --mode point for test 11..21")
            print(f"{seq}: no public GT; export only, no point metric")
        if args.mode in {"pixel", "both"} and any(not (root / seq / cfg["mos_data_params"].get("mos_label_folder", "mos_labels") / f"{p.stem}.npy").is_file() for p in scans):
            raise FileNotFoundError(f"Pixel GT missing for {seq}; use --mode point")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    totals, pixel_total, per_seq, seq_summary, coverage, pixel_rows = {}, empty_counts(), [], [], [], []
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
                        return_projected_range=args.postprocess == "knn",
                    )
                    if args.postprocess == "knn":
                        inputs, boot, projected_range = temporal_result
                    else:
                        inputs, boot = temporal_result
                    bootstrapped += int(boot)
                else:
                    single_result = single_frame_input(
                        scan, cfg, root / seq, scan.stem,
                        return_projected_range=args.postprocess == "knn",
                    )
                    if args.postprocess == "knn":
                        model_input, projected_range = single_result
                    else:
                        model_input = single_result
                    inputs = (model_input,)
                logits = model(*(part.to(device) for part in inputs))
                if logits.shape != (1, 2, projection.height, projection.width):
                    raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
                pred_hw = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
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
                    "invalid_points": int((~valid).sum()), "fallback_points": int((~valid).sum()),
                    "coverage_rate": float(valid.mean()) if len(valid) else 0.0,
                    "postprocess_seconds": postprocess_seconds})
                gt_path = root / seq / "labels" / f"{scan.stem}.label"
                if args.mode in {"point", "both"} and gt_path.is_file():
                    add_counts(seq_counts, point_counts(pred, read_point_gt(gt_path, len(points))))
                    seq_counts["n_points"] = seq_counts.get("n_points", 0) + len(points)
                    seq_counts["fallback_points"] = seq_counts.get("fallback_points", 0) + int((~valid).sum())
                if args.mode in {"pixel", "both"}:
                    label = np.load(root / seq / cfg["mos_data_params"].get("mos_label_folder", "mos_labels") / f"{scan.stem}.npy")
                    if label.shape != pred_hw.shape or not np.isin(label, [-1, 0, 1]).all():
                        raise ValueError(f"Invalid pixel GT: {seq}/{scan.stem}")
                    if e2e:
                        label = label.copy()
                        label[~inputs[1][0, -1].numpy()] = -1
                    counts = compute_counts_from_pred_target(torch.from_numpy(pred_hw), torch.from_numpy(label))
                    for key in seq_pixel:
                        seq_pixel[key] += counts[key]
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
    if pixel_rows:
        pixel_rows.insert(0, {"sequence": "ALL", "n_scans": total_scans, **pixel_total,
            **{k if k.startswith("pixel_") else f"pixel_{k}": v
               for k, v in metrics_from_counts(pixel_total).items()}})
        save_csv(args.output_dir / "pixel_metrics.csv", pixel_rows)
    save_csv(args.output_dir / "projection_coverage.csv", coverage)
    save_csv(args.output_dir / "sequence_summary.csv", seq_summary)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_train_sequences": cfg["mos_data_params"].get("train_sequences"),
        "eval_sequences": ids, "sequence_07_role": "internal_holdout_official_train",
        "dataset_root": str(root), "model": name, "input_mode": input_mode,
        "channel_names": (["x", "y", "z", "range", "intensity"] if "intensity" in input_mode else None),
        "normalization": normalization(cfg, input_mode),
        "projection": vars(projection), "history_init": args.history_init if temporal else None,
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
