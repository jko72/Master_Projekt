#!/usr/bin/env python3
"""Evaluate a residual-only MOS heuristic.

Examples:
  python tools/evaluate_residual_heuristic.py \
    --cfg_path src/configs/mos_semanticKitti_default.yaml \
    --split val \
    --residual_offsets 1 \
    --thresholds "0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3" \
    --combine first \
    --output_dir mos_logs/residual_heuristic_val \
    --save_visuals true \
    --save_per_frame_csv true

  python tools/evaluate_residual_heuristic.py \
    --cfg_path src/configs/mos_semanticKitti_default.yaml \
    --split val \
    --residual_offsets 1 \
    --combine first \
    --checkpoint /path/to/MOS_range/checkpoints/best_moving_iou.pt \
    --output_dir mos_logs/residual_vs_models_val
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from helper.dataloader_helper import make_sequences
from helper.mos_eval_utils import (
    add_counts,
    compute_counts_from_pred_target,
    counts_and_metrics_row,
    empty_counts,
    metrics_from_counts,
    normalize_seq_id,
    parse_residual_offsets,
    select_sequences,
)
from mos_dataset import MOSFrameDataset
from mos_models import build_mos_model


DEFAULT_THRESHOLDS = "0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3"
METRIC_FIELDS = [
    "moving_iou",
    "moving_precision",
    "moving_recall",
    "moving_f1",
    "static_iou",
    "mean_iou",
    "pixel_accuracy",
    "tp_moving",
    "fp_moving",
    "fn_moving",
    "tn_static",
    "valid_pixels",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_cfg_path(cfg_path: str) -> str:
    candidates = [
        cfg_path,
        os.path.join(PROJECT_ROOT, cfg_path),
        os.path.join(SRC_DIR, cfg_path),
        os.path.join(PROJECT_ROOT, "src", "configs", cfg_path),
        os.path.join(SRC_DIR, "configs", os.path.basename(cfg_path)),
        os.path.join(PROJECT_ROOT, "src", "configs", os.path.basename(cfg_path)),
        os.path.join(PROJECT_ROOT, "src", cfg_path),
    ]
    seen = set()
    for c in candidates:
        ap = os.path.abspath(c)
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            return ap
    raise FileNotFoundError(f"Config not found: '{cfg_path}'. Checked: {sorted(seen)}")


def resolve_output_dir(output_dir: str, cfg: Dict) -> str:
    if os.path.isabs(output_dir):
        return os.path.abspath(output_dir)

    log_root = cfg.get("mos_log_params", {}).get("log_root", None)
    if not log_root:
        log_root = os.path.join(PROJECT_ROOT, "LidarGaussianVideoView", "mos_logs")
    log_root = os.path.abspath(str(log_root))

    parts = [p for p in str(output_dir).replace("\\", "/").split("/") if p not in {"", "."}]
    if parts and parts[0] == "mos_logs":
        parts = parts[1:]
    return os.path.join(log_root, *parts) if parts else log_root


def parse_thresholds(value: str | None) -> List[float]:
    text = DEFAULT_THRESHOLDS if value is None or str(value).strip() == "" else str(value)
    vals = [float(t) for t in text.replace(";", ",").replace(" ", ",").split(",") if t.strip() != ""]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("At least one threshold is required.")
    return vals


def parse_checkpoint_args(values: Sequence[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(os.path.abspath(token))
    return out


def format_threshold(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def extract_scan_path(path_entry) -> str | None:
    if isinstance(path_entry, (list, tuple)):
        if len(path_entry) == 0:
            return None
        return str(path_entry[0])
    if isinstance(path_entry, dict):
        for key in ("scan_path", "pc_path", "velodyne_path", "bin_path", "path"):
            if key in path_entry:
                return str(path_entry[key])
        return None
    if isinstance(path_entry, (str, os.PathLike)):
        return str(path_entry)
    return None


def build_frame_index(selected_sequences: Sequence[Dict], mos_label_folder: str) -> List[Dict]:
    frames: List[Dict] = []
    for seq in selected_sequences:
        seq_id = normalize_seq_id(seq.get("seq_id", "unknown"))
        for frame_index, path_entry in enumerate(seq.get("paths", [])):
            scan_path = extract_scan_path(path_entry)
            if scan_path is None:
                continue
            frame_stem = os.path.splitext(os.path.basename(scan_path))[0]
            seq_dir = os.path.abspath(os.path.join(os.path.dirname(scan_path), ".."))
            frames.append(
                {
                    "seq_id": seq_id,
                    "frame_index": int(frame_index),
                    "frame_stem": frame_stem,
                    "seq_dir": seq_dir,
                    "scan_path": scan_path,
                    "mos_label_path": os.path.join(seq_dir, mos_label_folder, f"{frame_stem}.npy"),
                }
            )
    return frames


def residual_path_for_frame(frame: Dict, offset: int, folder_template: str) -> str:
    folder = str(folder_template).format(offset=int(offset))
    return os.path.join(frame["seq_dir"], folder, f"{frame['frame_stem']}.npy")


def validate_label(label: np.ndarray, path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    if label.shape != expected_shape:
        raise ValueError(f"MOS label shape mismatch at {path}: got {label.shape}, expected {expected_shape}.")
    label = label.astype(np.int64, copy=False)
    bad_vals = sorted(int(v) for v in np.unique(label) if int(v) not in {-1, 0, 1})
    if bad_vals:
        warnings.warn(
            f"MOS label contains unexpected values at {path}: {bad_vals}. Expected only [-1, 0, 1].",
            RuntimeWarning,
        )
    return label


def load_label(frame: Dict, expected_shape: tuple[int, int]) -> np.ndarray:
    path = frame["mos_label_path"]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing MOS label file: {path}")
    return validate_label(np.load(path), path, expected_shape)


def load_residual(
    path: str,
    expected_shape: tuple[int, int],
    allow_missing_residuals: bool,
    missing_warned: set[str],
) -> np.ndarray:
    if not os.path.isfile(path):
        if not allow_missing_residuals:
            raise FileNotFoundError(
                f"Missing residual file: {path}. "
                "Use --allow_missing_residuals true only if zero-filled fallback is intended."
            )
        if path not in missing_warned:
            warnings.warn(f"Missing residual file: {path}. Using zeros.", RuntimeWarning)
            missing_warned.add(path)
        return np.zeros(expected_shape, dtype=np.float32)

    residual = np.load(path).astype(np.float32, copy=False)
    if residual.shape != expected_shape:
        raise ValueError(f"Residual shape mismatch at {path}: got {residual.shape}, expected {expected_shape}.")
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return residual


def load_residual_score(
    frame: Dict,
    residual_offsets: Sequence[int],
    combine: str,
    folder_template: str,
    expected_shape: tuple[int, int],
    allow_missing_residuals: bool,
    missing_warned: set[str],
) -> np.ndarray:
    offsets_to_load = [int(residual_offsets[0])] if combine == "first" else [int(o) for o in residual_offsets]
    residuals = [
        load_residual(
            residual_path_for_frame(frame, offset, folder_template),
            expected_shape,
            allow_missing_residuals,
            missing_warned,
        )
        for offset in offsets_to_load
    ]
    if combine == "first":
        return residuals[0].astype(np.float32, copy=False)
    stack = np.stack(residuals, axis=0).astype(np.float32, copy=False)
    if combine == "max":
        return np.max(stack, axis=0).astype(np.float32, copy=False)
    if combine == "mean":
        return np.mean(stack, axis=0).astype(np.float32, copy=False)
    if combine == "sum":
        return np.sum(stack, axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported combine mode: {combine}")


def remove_small_components(pred: np.ndarray, min_component_size: int, warned: set[str]) -> np.ndarray:
    min_size = int(min_component_size)
    if min_size <= 0:
        return pred.astype(np.int64, copy=False)
    mask = pred.astype(bool, copy=False)
    if not np.any(mask):
        return pred.astype(np.int64, copy=False)

    try:
        import cv2  # type: ignore

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros_like(mask, dtype=bool)
        for label_id in range(1, int(num_labels)):
            if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_size:
                keep[labels == label_id] = True
        return keep.astype(np.int64)
    except Exception as exc_cv2:
        try:
            from scipy import ndimage  # type: ignore

            labels, num_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(labels.ravel())
            keep_ids = np.where(sizes >= min_size)[0]
            keep_ids = keep_ids[keep_ids != 0]
            return np.isin(labels, keep_ids).astype(np.int64)
        except Exception as exc_scipy:
            key = "component_filter"
            if key not in warned:
                warnings.warn(
                    "Skipping --min_component_size because neither OpenCV nor scipy connected components "
                    f"could be used. cv2 error={exc_cv2}; scipy error={exc_scipy}",
                    RuntimeWarning,
                )
                warned.add(key)
            return pred.astype(np.int64, copy=False)


def make_prediction(residual_score: np.ndarray, threshold: float, min_component_size: int, pp_warned: set[str]) -> np.ndarray:
    pred = (residual_score >= float(threshold)).astype(np.int64)
    return remove_small_components(pred, min_component_size=min_component_size, warned=pp_warned)


def frame_pixel_stats(label: np.ndarray) -> Dict[str, int]:
    return {
        "moving_pixels": int(np.sum(label == 1)),
        "static_pixels": int(np.sum(label == 0)),
        "ignore_pixels": int(np.sum(label == -1)),
    }


def save_residual_visualization(
    out_path: str,
    residual_score: np.ndarray,
    y_hw: np.ndarray,
    pred_hw: np.ndarray,
    ignore_index: int,
    title: str,
):
    valid = y_hw != ignore_index
    error_map = np.zeros_like(y_hw, dtype=np.int8)
    error_map[~valid] = 0
    error_map[(pred_hw == y_hw) & valid] = 1
    error_map[(pred_hw == 1) & (y_hw == 0) & valid] = 2
    error_map[(pred_hw == 0) & (y_hw == 1) & valid] = 3

    panels = [
        ("Residual", residual_score, "residual"),
        ("GT MOS", y_hw, "gt"),
        ("Pred MOS", pred_hw, "pred"),
        ("Error", error_map, "err"),
    ]

    fig, axs = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2), constrained_layout=True)
    for ax, (name, arr, kind) in zip(axs, panels):
        if kind == "residual":
            vmax = float(max(1e-6, np.percentile(arr, 99)))
            im = ax.imshow(arr, cmap="turbo", vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)
        elif kind == "gt":
            cmap = mcolors.ListedColormap(["#3b3b3b", "#0f3473", "#f29e4c"])
            norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
            im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels(["ignore", "static", "moving"])
        elif kind == "pred":
            cmap = mcolors.ListedColormap(["#0f3473", "#f29e4c"])
            norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1])
            cbar.ax.set_yticklabels(["static", "moving"])
        else:
            cmap = mcolors.ListedColormap(["#222222", "#3cb44b", "#e41a1c", "#f6d55c"])
            norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
            im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(["ignore", "correct", "FP moving", "FN moving"])

        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_threshold_sweep(
    frames: Sequence[Dict],
    residual_offsets: Sequence[int],
    thresholds: Sequence[float],
    combine: str,
    folder_template: str,
    expected_shape: tuple[int, int],
    ignore_index: int,
    allow_missing_residuals: bool,
    min_component_size: int,
) -> tuple[List[Dict], int]:
    counts_by_threshold = {float(t): empty_counts() for t in thresholds}
    missing_warned: set[str] = set()
    pp_warned: set[str] = set()

    for frame in frames:
        label = load_label(frame, expected_shape)
        residual_score = load_residual_score(
            frame,
            residual_offsets,
            combine,
            folder_template,
            expected_shape,
            allow_missing_residuals,
            missing_warned,
        )
        target_t = torch.from_numpy(label.astype(np.int64, copy=False))
        for threshold in thresholds:
            pred = make_prediction(residual_score, threshold, min_component_size, pp_warned)
            counts = compute_counts_from_pred_target(
                torch.from_numpy(pred.astype(np.int64, copy=False)),
                target_t,
                ignore_index=ignore_index,
            )
            add_counts(counts_by_threshold[float(threshold)], counts)

    rows: List[Dict] = []
    for threshold in thresholds:
        counts = counts_by_threshold[float(threshold)]
        row = {
            "threshold": float(threshold),
            "split": "",
            "combine": combine,
            "residual_offsets": str(list(residual_offsets)),
            "num_frames": int(len(frames)),
            "min_component_size": int(min_component_size),
        }
        row.update(counts_and_metrics_row(counts))
        rows.append(row)
    return rows, len(frames)


def evaluate_single_threshold(
    frames: Sequence[Dict],
    residual_offsets: Sequence[int],
    threshold: float,
    combine: str,
    folder_template: str,
    expected_shape: tuple[int, int],
    ignore_index: int,
    allow_missing_residuals: bool,
    min_component_size: int,
    save_per_frame_csv: bool,
    save_visuals: bool,
    visuals_dir: str,
    max_visuals: int,
    visual_stride: int,
) -> tuple[Dict[str, int], List[Dict], int]:
    agg_counts = empty_counts()
    per_frame_rows: List[Dict] = []
    missing_warned: set[str] = set()
    pp_warned: set[str] = set()
    visual_saved = 0

    if save_visuals:
        os.makedirs(visuals_dir, exist_ok=True)

    for global_idx, frame in enumerate(frames):
        label = load_label(frame, expected_shape)
        residual_score = load_residual_score(
            frame,
            residual_offsets,
            combine,
            folder_template,
            expected_shape,
            allow_missing_residuals,
            missing_warned,
        )
        pred = make_prediction(residual_score, threshold, min_component_size, pp_warned)
        counts = compute_counts_from_pred_target(
            torch.from_numpy(pred.astype(np.int64, copy=False)),
            torch.from_numpy(label.astype(np.int64, copy=False)),
            ignore_index=ignore_index,
        )
        add_counts(agg_counts, counts)

        metrics_i = metrics_from_counts(counts)
        stats = frame_pixel_stats(label)
        if save_per_frame_csv:
            row = {
                "seq_id": str(frame["seq_id"]),
                "frame_stem": str(frame["frame_stem"]),
                "frame_index": int(frame["frame_index"]),
                "threshold": float(threshold),
                "moving_pixels": int(stats["moving_pixels"]),
                "static_pixels": int(stats["static_pixels"]),
                "ignore_pixels": int(stats["ignore_pixels"]),
                "moving_iou": float(metrics_i["moving_iou"]),
                "moving_precision": float(metrics_i["moving_precision"]),
                "moving_recall": float(metrics_i["moving_recall"]),
                "moving_f1": float(metrics_i["moving_f1"]),
                "static_iou": float(metrics_i["static_iou"]),
                "mean_iou": float(metrics_i["mean_iou"]),
                "pixel_accuracy": float(metrics_i["pixel_accuracy"]),
                "tp_moving": int(counts["tp_moving"]),
                "fp_moving": int(counts["fp_moving"]),
                "fn_moving": int(counts["fn_moving"]),
                "tn_static": int(counts["tn_static"]),
                "valid_pixels": int(counts["valid_pixels"]),
            }
            per_frame_rows.append(row)

        if save_visuals and visual_saved < int(max_visuals):
            if int(visual_stride) <= 0 or global_idx % int(visual_stride) == 0:
                out_name = f"{visual_saved:04d}_seq{frame['seq_id']}_{frame['frame_stem']}.png"
                title = (
                    f"seq={frame['seq_id']} frame={frame['frame_stem']} | "
                    f"threshold={threshold:.6g} moving_iou={metrics_i['moving_iou']:.4f}"
                )
                save_residual_visualization(
                    os.path.join(visuals_dir, out_name),
                    residual_score,
                    label,
                    pred,
                    ignore_index,
                    title,
                )
                visual_saved += 1

    return agg_counts, per_frame_rows, visual_saved


def write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str] | None = None):
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_meta_value(meta: Dict, key: str, idx: int):
    val = meta.get(key, None)
    if isinstance(val, list):
        return val[idx]
    if isinstance(val, tuple):
        return val[idx]
    if torch.is_tensor(val):
        return val[idx].item()
    if isinstance(val, np.ndarray):
        return val[idx].item()
    return val


def checkpoint_method_name(checkpoint_path: str) -> str:
    ckpt_parent = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(ckpt_parent) == "checkpoints":
        run_dir = os.path.dirname(ckpt_parent)
        return os.path.basename(run_dir) or os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.splitext(os.path.basename(checkpoint_path))[0]


def evaluate_checkpoint_for_comparison(
    checkpoint_path: str,
    split: str,
    device_arg: str | None,
    batch_size_arg: int | None,
    num_workers_arg: int | None,
) -> Dict:
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "cfg" not in ckpt:
        raise KeyError(
            f"Checkpoint has no 'cfg' key: {checkpoint_path}. "
            "Please evaluate a checkpoint saved by tools/train_mos.py with cfg included."
        )
    cfg = ckpt["cfg"]
    if not isinstance(cfg, dict):
        raise TypeError(f"Checkpoint 'cfg' must be a dict, got {type(cfg)}")

    cfg.setdefault("mos_data_params", {})
    cfg.setdefault("mos_model_params", {})
    cfg.setdefault("mos_train_params", {})
    mdata = cfg["mos_data_params"]
    mmodel = cfg["mos_model_params"]
    mtrain = cfg["mos_train_params"]

    dataset_path = cfg.get("dataset_path", None)
    if not dataset_path:
        raise ValueError(f"Checkpoint config missing required key dataset_path: {checkpoint_path}")

    split_key = f"{split}_sequences"
    requested_sequences = mdata.get(split_key, None)
    if requested_sequences is None:
        raise ValueError(
            f"Checkpoint config missing required key for split '{split}': mos_data_params.{split_key}"
        )

    input_mode = str(mdata.get("input_mode", ckpt.get("input_mode", "range_residual")))
    residual_offsets = parse_residual_offsets(mdata.get("residual_offsets", ckpt.get("residual_offsets", [1])))
    mos_label_folder = str(mdata.get("mos_label_folder", "mos_labels"))
    ignore_index = int(mdata.get("ignore_index", -1))

    all_sequences = make_sequences(dataset_path)
    selected_sequences = select_sequences(all_sequences, requested_sequences, split)
    eval_dataset = MOSFrameDataset(
        sequences=selected_sequences,
        cfg=cfg,
        split=split,
        input_mode=input_mode,
        residual_offsets=residual_offsets,
        mos_label_folder=mos_label_folder,
        require_moving=False,
        min_moving_pixels=int(mdata.get("min_moving_pixels", 1)),
    )
    if len(eval_dataset) == 0:
        raise RuntimeError(f"Evaluation dataset is empty for checkpoint={checkpoint_path} split={split}")

    batch_size = int(batch_size_arg) if batch_size_arg is not None else int(mtrain.get("batch_size", 8))
    num_workers = int(num_workers_arg) if num_workers_arg is not None else int(mtrain.get("num_workers", 4))
    device = str(device_arg) if device_arg is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.startswith("cuda")

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    model = build_mos_model(cfg)
    state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        raise KeyError(f"Checkpoint has no 'model_state_dict': {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    agg_counts = empty_counts()
    total_inference_time_sec = 0.0
    num_batches = 0
    num_frames = 0

    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory).long()

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_inference_time_sec += float(t1 - t0)
            num_batches += 1
            num_frames += int(x.shape[0])

            pred = torch.argmax(logits, dim=1)
            counts = compute_counts_from_pred_target(pred, y, ignore_index=ignore_index)
            add_counts(agg_counts, counts)

    mean_batch_time = float(total_inference_time_sec / max(num_batches, 1))
    mean_frame_time = float(total_inference_time_sec / max(num_frames, 1))

    row = {
        "method": checkpoint_method_name(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "split": str(split),
        "input_mode": str(input_mode),
        "residual_offsets": str(list(residual_offsets)),
        "threshold": "",
        "combine": "",
        "min_component_size": "",
        "num_frames": int(num_frames),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "model_name": str(mmodel.get("name", "unknown")),
        "total_inference_time_sec": float(total_inference_time_sec),
        "mean_inference_time_per_batch_sec": float(mean_batch_time),
        "mean_inference_time_per_frame_sec": float(mean_frame_time),
    }
    row.update(counts_and_metrics_row(agg_counts))
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate a residual-only heuristic MOS baseline.")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to MOS yaml config.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--residual_offsets", type=str, default=None)
    parser.add_argument("--combine", type=str, default="first", choices=["first", "max", "mean", "sum"])
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--mos_label_folder", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_per_frame_csv", type=str2bool, default=False)
    parser.add_argument("--save_visuals", type=str2bool, default=False)
    parser.add_argument("--max_visuals", type=int, default=50)
    parser.add_argument("--visual_stride", type=int, default=20)
    parser.add_argument("--min_component_size", type=int, default=0)
    parser.add_argument("--allow_missing_residuals", type=str2bool, default=False)
    parser.add_argument("--checkpoint", action="append", default=None, help="Checkpoint path. Repeatable or comma-separated.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    cfg_path = resolve_cfg_path(args.cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must load as a dict: {cfg_path}")

    cfg.setdefault("mos_data_params", {})
    mdata = cfg["mos_data_params"]
    dataset_path = cfg.get("dataset_path", None)
    if not dataset_path:
        raise ValueError("Config missing required key: dataset_path")

    split_key = f"{args.split}_sequences"
    requested_sequences = mdata.get(split_key, None)
    if requested_sequences is None:
        raise ValueError(f"Config missing required key for split '{args.split}': mos_data_params.{split_key}")

    residual_offsets = (
        parse_residual_offsets(args.residual_offsets)
        if args.residual_offsets is not None
        else parse_residual_offsets(mdata.get("residual_offsets", [1]))
    )
    thresholds = parse_thresholds(args.thresholds)
    mos_label_folder = str(args.mos_label_folder or mdata.get("mos_label_folder", "mos_labels"))
    folder_template = str(mdata.get("residual_folder_template", "residual_images_{offset}"))
    ignore_index = int(mdata.get("ignore_index", -1))
    expected_shape = (int(cfg["model_params"]["grid_height"]), int(cfg["model_params"]["grid_width"]))

    all_sequences = make_sequences(dataset_path)
    selected_sequences = select_sequences(all_sequences, requested_sequences, args.split)
    selected_seq_ids = [normalize_seq_id(s.get("seq_id", "")) for s in selected_sequences]
    frames = build_frame_index(selected_sequences, mos_label_folder)
    if not frames:
        raise RuntimeError(f"No frames found for split={args.split}, sequences={selected_seq_ids}")

    output_dir = resolve_output_dir(args.output_dir, cfg)
    os.makedirs(output_dir, exist_ok=True)

    sweep_rows, num_frames = evaluate_threshold_sweep(
        frames=frames,
        residual_offsets=residual_offsets,
        thresholds=thresholds,
        combine=args.combine,
        folder_template=folder_template,
        expected_shape=expected_shape,
        ignore_index=ignore_index,
        allow_missing_residuals=bool(args.allow_missing_residuals),
        min_component_size=int(args.min_component_size),
    )
    for row in sweep_rows:
        row["split"] = str(args.split)

    best_row = max(sweep_rows, key=lambda r: (float(r["moving_iou"]), float(r["moving_f1"]), -float(r["threshold"])))
    best_threshold = float(best_row["threshold"])

    sweep_csv = os.path.join(output_dir, "residual_heuristic_sweep.csv")
    best_csv = os.path.join(output_dir, "residual_heuristic_best.csv")
    sweep_fields = [
        "threshold",
        "split",
        "combine",
        "residual_offsets",
        "num_frames",
        "min_component_size",
    ] + METRIC_FIELDS
    write_csv(sweep_csv, sweep_rows, sweep_fields)
    write_csv(best_csv, [best_row], sweep_fields)

    per_frame_csv_path = ""
    visual_saved = 0
    if args.save_per_frame_csv or args.save_visuals:
        agg_best, per_frame_rows, visual_saved = evaluate_single_threshold(
            frames=frames,
            residual_offsets=residual_offsets,
            threshold=best_threshold,
            combine=args.combine,
            folder_template=folder_template,
            expected_shape=expected_shape,
            ignore_index=ignore_index,
            allow_missing_residuals=bool(args.allow_missing_residuals),
            min_component_size=int(args.min_component_size),
            save_per_frame_csv=bool(args.save_per_frame_csv),
            save_visuals=bool(args.save_visuals),
            visuals_dir=os.path.join(output_dir, "visuals"),
            max_visuals=int(args.max_visuals),
            visual_stride=int(args.visual_stride),
        )
        if args.save_per_frame_csv:
            per_frame_csv_path = os.path.join(
                output_dir, f"per_frame_metrics_threshold_{format_threshold(best_threshold)}.csv"
            )
            per_frame_fields = [
                "seq_id",
                "frame_stem",
                "frame_index",
                "threshold",
                "moving_pixels",
                "static_pixels",
                "ignore_pixels",
            ] + METRIC_FIELDS
            write_csv(per_frame_csv_path, per_frame_rows, per_frame_fields)
        best_check = counts_and_metrics_row(agg_best)
        if abs(float(best_check["moving_iou"]) - float(best_row["moving_iou"])) > 1e-12:
            warnings.warn("Best-threshold recomputation differs from sweep metrics.", RuntimeWarning)

    checkpoint_paths = parse_checkpoint_args(args.checkpoint)
    if checkpoint_paths:
        comparison_rows = [
            {
                "method": "residual_heuristic",
                "checkpoint_path": "",
                "split": str(args.split),
                "input_mode": "",
                "residual_offsets": str(list(residual_offsets)),
                "threshold": float(best_threshold),
                "combine": str(args.combine),
                "min_component_size": int(args.min_component_size),
                "num_frames": int(num_frames),
                "checkpoint_epoch": "",
                "model_name": "",
                "total_inference_time_sec": "",
                "mean_inference_time_per_batch_sec": "",
                "mean_inference_time_per_frame_sec": "",
                **{k: best_row[k] for k in METRIC_FIELDS},
            }
        ]
        for checkpoint_path in checkpoint_paths:
            comparison_rows.append(
                evaluate_checkpoint_for_comparison(
                    checkpoint_path=checkpoint_path,
                    split=args.split,
                    device_arg=args.device,
                    batch_size_arg=args.batch_size,
                    num_workers_arg=args.num_workers,
                )
            )
        comparison_csv = os.path.join(output_dir, "comparison_residual_vs_checkpoints.csv")
        comparison_fields = [
            "method",
            "checkpoint_path",
            "split",
            "input_mode",
            "residual_offsets",
            "threshold",
            "combine",
            "min_component_size",
            "num_frames",
            "checkpoint_epoch",
            "model_name",
        ] + METRIC_FIELDS + [
            "total_inference_time_sec",
            "mean_inference_time_per_batch_sec",
            "mean_inference_time_per_frame_sec",
        ]
        write_csv(comparison_csv, comparison_rows, comparison_fields)
    else:
        comparison_csv = ""

    eval_config = {
        "evaluation": {
            "cfg_path": cfg_path,
            "split": str(args.split),
            "selected_sequences": selected_seq_ids,
            "residual_offsets": list(residual_offsets),
            "combine": str(args.combine),
            "thresholds": [float(t) for t in thresholds],
            "best_threshold": float(best_threshold),
            "mos_label_folder": mos_label_folder,
            "residual_folder_template": folder_template,
            "min_component_size": int(args.min_component_size),
            "allow_missing_residuals": bool(args.allow_missing_residuals),
            "num_frames": int(num_frames),
            "output_dir": output_dir,
            "save_per_frame_csv": bool(args.save_per_frame_csv),
            "save_visuals": bool(args.save_visuals),
            "checkpoints": checkpoint_paths,
        },
        "cfg": cfg,
    }
    eval_config_path = os.path.join(output_dir, "residual_heuristic_eval_config.yaml")
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_config, f, sort_keys=False, default_flow_style=False)

    print("=== Residual Heuristic MOS Evaluation ===")
    print(f"config_path        : {cfg_path}")
    print(f"split              : {args.split}")
    print(f"selected_sequences : {selected_seq_ids}")
    print(f"residual_offsets   : {list(residual_offsets)}")
    print(f"combine            : {args.combine}")
    print(f"num_frames         : {num_frames}")
    print(f"best_threshold     : {best_threshold:.8g}")
    print(f"best_moving_iou    : {float(best_row['moving_iou']):.6f}")
    print(f"best_precision     : {float(best_row['moving_precision']):.6f}")
    print(f"best_recall        : {float(best_row['moving_recall']):.6f}")
    print(f"best_f1            : {float(best_row['moving_f1']):.6f}")
    print(f"best_mean_iou      : {float(best_row['mean_iou']):.6f}")
    print(f"output_dir         : {output_dir}")
    print(f"sweep_csv          : {sweep_csv}")
    print(f"best_csv           : {best_csv}")
    if per_frame_csv_path:
        print(f"per_frame_csv      : {per_frame_csv_path}")
    if args.save_visuals:
        print(f"visuals_saved      : {visual_saved}")
    if comparison_csv:
        print(f"comparison_csv     : {comparison_csv}")


if __name__ == "__main__":
    main()
