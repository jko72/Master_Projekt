#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, Iterable, List, Sequence

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
from mos_dataset import MOSFrameDataset
from mos_models import build_mos_model


def str2bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_seq_id(seq_id) -> str:
    s = str(seq_id)
    return s.zfill(2) if s.isdigit() else s


def parse_residual_offsets(value) -> List[int]:
    if isinstance(value, (int, np.integer)):
        vals = [int(value)]
    elif isinstance(value, str):
        toks = value.replace(";", ",").replace(" ", ",").split(",")
        vals = [int(t) for t in toks if t.strip() != ""]
    elif isinstance(value, Iterable):
        vals = [int(v) for v in value]
    else:
        raise TypeError("residual_offsets must be int, str or iterable of ints.")

    vals = sorted(set(v for v in vals if v >= 1))
    if not vals:
        raise ValueError("residual_offsets must include at least one positive integer.")
    return vals


def compute_in_channels(input_mode: str, residual_offsets: Sequence[int]) -> int:
    if input_mode == "range":
        return 1
    if input_mode == "residual":
        return len(residual_offsets)
    if input_mode == "range_residual":
        return 1 + len(residual_offsets)
    if input_mode == "range_xyz":
        return 4
    if input_mode == "range_xyz_residual":
        return 4 + len(residual_offsets)
    if input_mode == "range_xyz_normal":
        return 7
    if input_mode == "range_xyz_normal_residual":
        return 7 + len(residual_offsets)
    raise ValueError(
        "Invalid input_mode='{0}'. Supported: "
        "['range', 'residual', 'range_residual', 'range_xyz', 'range_xyz_residual', "
        "'range_xyz_normal', 'range_xyz_normal_residual']".format(input_mode)
    )


def select_sequences(all_sequences: List[Dict], requested_ids: Sequence[str], split_name: str) -> List[Dict]:
    requested_norm = [normalize_seq_id(s) for s in requested_ids]
    wanted = set(requested_norm)

    available = sorted(normalize_seq_id(s.get("seq_id", "")) for s in all_sequences)
    available_set = set(available)
    missing = sorted(s for s in wanted if s not in available_set)
    if missing:
        raise ValueError(
            f"Requested {split_name} sequences not found: {missing}. "
            f"Available sequences: {available}"
        )

    selected = [s for s in all_sequences if normalize_seq_id(s.get("seq_id", "")) in wanted]
    if not selected:
        raise ValueError(
            f"No sequences selected for {split_name}. Requested: {requested_norm}, available: {available}"
        )
    return selected


def compute_mos_counts(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> Dict[str, int]:
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        return compute_counts_from_pred_target(pred, target, ignore_index=ignore_index)


def compute_counts_from_pred_target(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> Dict[str, int]:
    with torch.no_grad():
        valid = target != ignore_index
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            return {
                "tp_moving": 0,
                "fp_moving": 0,
                "fn_moving": 0,
                "tn_static": 0,
                "correct": 0,
                "valid_pixels": 0,
            }

        tp_m = int(((pred == 1) & (target == 1) & valid).sum().item())
        fp_m = int(((pred == 1) & (target == 0) & valid).sum().item())
        fn_m = int(((pred == 0) & (target == 1) & valid).sum().item())
        tn_s = int(((pred == 0) & (target == 0) & valid).sum().item())
        correct = int(((pred == target) & valid).sum().item())

        return {
            "tp_moving": tp_m,
            "fp_moving": fp_m,
            "fn_moving": fn_m,
            "tn_static": tn_s,
            "correct": correct,
            "valid_pixels": valid_count,
        }


def metrics_from_counts(counts: Dict[str, int], eps: float = 1e-8) -> Dict[str, float]:
    tp_m = float(counts["tp_moving"])
    fp_m = float(counts["fp_moving"])
    fn_m = float(counts["fn_moving"])
    tn_s = float(counts["tn_static"])
    correct = float(counts["correct"])
    valid = float(max(counts["valid_pixels"], 1))

    moving_iou = tp_m / (tp_m + fp_m + fn_m + eps)
    moving_precision = tp_m / (tp_m + fp_m + eps)
    moving_recall = tp_m / (tp_m + fn_m + eps)
    moving_f1 = 2.0 * moving_precision * moving_recall / (moving_precision + moving_recall + eps)

    static_iou = tn_s / (tn_s + fn_m + fp_m + eps)
    mean_iou = 0.5 * (static_iou + moving_iou)
    pixel_accuracy = correct / valid

    return {
        "moving_iou": moving_iou,
        "moving_precision": moving_precision,
        "moving_recall": moving_recall,
        "moving_f1": moving_f1,
        "static_iou": static_iou,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
    }


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


def get_channel_names_from_meta(meta: Dict, idx: int):
    val = meta.get("channel_names", None)
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and all(isinstance(v, str) for v in val):
        return list(val)
    if isinstance(val, (list, tuple)) and all(isinstance(v, (list, tuple)) for v in val):
        if len(val) > 0 and all(len(v) > idx for v in val):
            return [str(v[idx]) for v in val]
    return None


def select_visual_channels(x_chw: np.ndarray, input_mode: str, channel_names=None):
    range_idx = None
    residual_idx = None

    if channel_names:
        names = [str(name) for name in channel_names]
        if "range" in names:
            range_idx = names.index("range")
        residual_indices = [i for i, name in enumerate(names) if name.startswith("residual_")]
        if residual_indices:
            residual_idx = residual_indices[0]
    else:
        if input_mode in {"range", "range_residual"}:
            range_idx = 0
        elif input_mode in {"range_xyz", "range_xyz_residual", "range_xyz_normal", "range_xyz_normal_residual"}:
            range_idx = 3

        if input_mode == "residual":
            residual_idx = 0
        elif input_mode == "range_residual":
            residual_idx = 1
        elif input_mode == "range_xyz_residual":
            residual_idx = 4
        elif input_mode == "range_xyz_normal_residual":
            residual_idx = 7

    range_img = x_chw[range_idx] if range_idx is not None and range_idx < x_chw.shape[0] else None
    residual_img = x_chw[residual_idx] if residual_idx is not None and residual_idx < x_chw.shape[0] else None
    return range_img, residual_img


def save_visualization(
    out_path: str,
    x_chw: np.ndarray,
    y_hw: np.ndarray,
    pred_hw: np.ndarray,
    input_mode: str,
    channel_names,
    ignore_index: int,
    title: str,
):
    range_img, residual_img = select_visual_channels(
        x_chw=x_chw,
        input_mode=input_mode,
        channel_names=channel_names,
    )

    valid = y_hw != ignore_index
    error_map = np.zeros_like(y_hw, dtype=np.int8)
    error_map[~valid] = 0
    error_map[(pred_hw == y_hw) & valid] = 1
    error_map[(pred_hw == 1) & (y_hw == 0) & valid] = 2
    error_map[(pred_hw == 0) & (y_hw == 1) & valid] = 3

    panels = []
    if range_img is not None:
        panels.append(("Range", range_img, "range"))
    if residual_img is not None:
        panels.append(("Residual", residual_img, "residual"))
    panels.append(("GT MOS", y_hw, "gt"))
    panels.append(("Pred MOS", pred_hw, "pred"))
    panels.append(("Error", error_map, "err"))

    ncols = len(panels)
    fig, axs = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.2), constrained_layout=True)
    if ncols == 1:
        axs = [axs]

    for ax, (name, arr, kind) in zip(axs, panels):
        if kind == "range":
            vmax = float(max(1e-6, np.percentile(arr, 99)))
            im = ax.imshow(arr, cmap="turbo", vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)
        elif kind == "residual":
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MOS checkpoint on range-view labels.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--input_mode",
        type=str,
        default=None,
        choices=[
            "range",
            "residual",
            "range_residual",
            "range_xyz",
            "range_xyz_residual",
            "range_xyz_normal",
            "range_xyz_normal_residual",
        ],
        help="Optional input-mode override; defaults to the mode stored in the checkpoint cfg.",
    )
    parser.add_argument("--residual_offsets", type=str, default=None, help="Optional override, e.g. '1' or '1,2,3'.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_visuals", type=str2bool, default=False)
    parser.add_argument("--max_visuals", type=int, default=50)
    parser.add_argument("--visual_stride", type=int, default=20)
    parser.add_argument("--save_per_frame_csv", type=str2bool, default=False)
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    ckpt_parent = os.path.dirname(checkpoint_path)
    if os.path.basename(ckpt_parent) == "checkpoints":
        run_dir = os.path.dirname(ckpt_parent)
    else:
        run_dir = ckpt_parent

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir is not None
        else os.path.join(run_dir, "eval", f"{ckpt_name}_{args.split}")
    )
    os.makedirs(output_dir, exist_ok=True)

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
        raise ValueError("Config missing required key: dataset_path")

    split_key = f"{args.split}_sequences"
    if args.split == "test" and split_key not in mdata:
        raise ValueError(
            "Requested split='test' but cfg['mos_data_params']['test_sequences'] is missing."
        )

    requested_sequences = mdata.get(split_key, None)
    if requested_sequences is None:
        raise ValueError(
            f"Config missing required key for split '{args.split}': mos_data_params.{split_key}"
        )

    input_mode = str(args.input_mode or mdata.get("input_mode", ckpt.get("input_mode", "range_residual")))
    if args.residual_offsets is not None:
        residual_offsets = parse_residual_offsets(args.residual_offsets)
    else:
        residual_offsets = parse_residual_offsets(mdata.get("residual_offsets", ckpt.get("residual_offsets", [1])))
    mdata["input_mode"] = input_mode
    mdata["residual_offsets"] = list(residual_offsets)
    mmodel["in_channels"] = compute_in_channels(input_mode, residual_offsets)
    mos_label_folder = str(mdata.get("mos_label_folder", "mos_labels"))
    ignore_index = int(mdata.get("ignore_index", -1))

    all_sequences = make_sequences(dataset_path)
    selected_sequences = select_sequences(all_sequences, requested_sequences, args.split)
    selected_seq_ids = [normalize_seq_id(s.get("seq_id", "")) for s in selected_sequences]

    eval_dataset = MOSFrameDataset(
        sequences=selected_sequences,
        cfg=cfg,
        split=args.split,
        input_mode=input_mode,
        residual_offsets=residual_offsets,
        mos_label_folder=mos_label_folder,
        require_moving=False,
        min_moving_pixels=int(mdata.get("min_moving_pixels", 1)),
    )
    if len(eval_dataset) == 0:
        raise RuntimeError(f"Evaluation dataset is empty for split={args.split}")

    batch_size = int(args.batch_size) if args.batch_size is not None else int(mtrain.get("batch_size", 8))
    num_workers = int(args.num_workers) if args.num_workers is not None else int(mtrain.get("num_workers", 4))

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = str(device)
    pin_memory = device.startswith("cuda")

    eval_loader = DataLoader(
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

    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    missing = sorted(model_keys - state_keys)
    unexpected = sorted(state_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "State-dict key mismatch before strict load. "
            f"missing_keys={missing[:20]} unexpected_keys={unexpected[:20]}"
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"strict=True model.load_state_dict failed for checkpoint {checkpoint_path}: {e}"
        ) from e

    model.to(device)
    model.eval()

    visuals_dir = os.path.join(output_dir, "visuals")
    if args.save_visuals:
        os.makedirs(visuals_dir, exist_ok=True)

    print("=== MOS Checkpoint Evaluation ===")
    print(f"checkpoint_path    : {checkpoint_path}")
    print(f"split              : {args.split}")
    print(f"selected_sequences : {selected_seq_ids}")
    print(f"input_mode         : {input_mode}")
    print(f"model_name         : {mmodel.get('name', 'unknown')}")
    print(f"num_frames         : {len(eval_dataset)}")
    print(f"output_dir         : {output_dir}")

    agg_counts = {
        "tp_moving": 0,
        "fp_moving": 0,
        "fn_moving": 0,
        "tn_static": 0,
        "correct": 0,
        "valid_pixels": 0,
    }

    per_frame_rows = []
    total_inference_time_sec = 0.0
    num_batches = 0
    num_frames = 0
    visual_saved = 0
    global_frame_idx = 0

    with torch.no_grad():
        for x, y, meta in eval_loader:
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory).long()

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_dt = float(t1 - t0)
            total_inference_time_sec += batch_dt
            num_batches += 1
            num_frames += int(x.shape[0])

            pred = torch.argmax(logits, dim=1)
            batch_counts = compute_counts_from_pred_target(pred, y, ignore_index=ignore_index)
            for k in agg_counts:
                agg_counts[k] += int(batch_counts[k])

            need_per_frame = bool(args.save_per_frame_csv)
            need_visuals = bool(args.save_visuals)
            if need_per_frame or need_visuals:
                x_cpu = x.detach().cpu().numpy()
                y_cpu = y.detach().cpu().numpy()
                pred_cpu = pred.detach().cpu().numpy()

                bs = int(x.shape[0])
                for i in range(bs):
                    counts_i = compute_counts_from_pred_target(
                        torch.from_numpy(pred_cpu[i]),
                        torch.from_numpy(y_cpu[i]),
                        ignore_index=ignore_index,
                    )
                    metrics_i = metrics_from_counts(counts_i)

                    if args.save_per_frame_csv:
                        row = {
                            "seq_id": str(get_meta_value(meta, "seq_id", i)),
                            "frame_stem": str(get_meta_value(meta, "frame_stem", i)),
                            "frame_index": int(get_meta_value(meta, "frame_index", i)),
                            "moving_pixels": int(get_meta_value(meta, "moving_pixels", i)),
                            "static_pixels": int(get_meta_value(meta, "static_pixels", i)),
                            "ignore_pixels": int(get_meta_value(meta, "ignore_pixels", i)),
                            "moving_iou": float(metrics_i["moving_iou"]),
                            "moving_precision": float(metrics_i["moving_precision"]),
                            "moving_recall": float(metrics_i["moving_recall"]),
                            "moving_f1": float(metrics_i["moving_f1"]),
                            "static_iou": float(metrics_i["static_iou"]),
                            "mean_iou": float(metrics_i["mean_iou"]),
                            "pixel_accuracy": float(metrics_i["pixel_accuracy"]),
                            "tp_moving": int(counts_i["tp_moving"]),
                            "fp_moving": int(counts_i["fp_moving"]),
                            "fn_moving": int(counts_i["fn_moving"]),
                            "tn_static": int(counts_i["tn_static"]),
                            "valid_pixels": int(counts_i["valid_pixels"]),
                        }
                        per_frame_rows.append(row)

                    if need_visuals and visual_saved < int(args.max_visuals):
                        if int(args.visual_stride) <= 0 or (global_frame_idx % int(args.visual_stride) == 0):
                            seq_id = str(get_meta_value(meta, "seq_id", i))
                            frame_stem = str(get_meta_value(meta, "frame_stem", i))
                            fig_name = f"{visual_saved:04d}_seq{seq_id}_{frame_stem}.png"
                            out_path = os.path.join(visuals_dir, fig_name)
                            title = (
                                f"seq={seq_id} frame={frame_stem} | "
                                f"moving_iou={metrics_i['moving_iou']:.4f}"
                            )
                            save_visualization(
                                out_path=out_path,
                                x_chw=x_cpu[i],
                                y_hw=y_cpu[i],
                                pred_hw=pred_cpu[i],
                                input_mode=input_mode,
                                channel_names=get_channel_names_from_meta(meta, i),
                                ignore_index=ignore_index,
                                title=title,
                            )
                            visual_saved += 1

                    global_frame_idx += 1
            else:
                global_frame_idx += int(x.shape[0])

    final_metrics = metrics_from_counts(agg_counts)
    mean_batch_time = float(total_inference_time_sec / max(num_batches, 1))
    mean_frame_time = float(total_inference_time_sec / max(num_frames, 1))

    eval_row = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "split": str(args.split),
        "model_name": str(mmodel.get("name", "unknown")),
        "input_mode": str(input_mode),
        "residual_offsets": str(list(residual_offsets)),
        "num_frames": int(num_frames),
        "moving_iou": float(final_metrics["moving_iou"]),
        "moving_precision": float(final_metrics["moving_precision"]),
        "moving_recall": float(final_metrics["moving_recall"]),
        "moving_f1": float(final_metrics["moving_f1"]),
        "static_iou": float(final_metrics["static_iou"]),
        "mean_iou": float(final_metrics["mean_iou"]),
        "pixel_accuracy": float(final_metrics["pixel_accuracy"]),
        "tp_moving": int(agg_counts["tp_moving"]),
        "fp_moving": int(agg_counts["fp_moving"]),
        "fn_moving": int(agg_counts["fn_moving"]),
        "tn_static": int(agg_counts["tn_static"]),
        "valid_pixels": int(agg_counts["valid_pixels"]),
        "total_inference_time_sec": float(total_inference_time_sec),
        "mean_inference_time_per_batch_sec": float(mean_batch_time),
        "mean_inference_time_per_frame_sec": float(mean_frame_time),
    }

    eval_metrics_csv_path = os.path.join(output_dir, "eval_metrics.csv")
    with open(eval_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_row.keys()))
        writer.writeheader()
        writer.writerow(eval_row)

    per_frame_csv_path = os.path.join(output_dir, "per_frame_metrics.csv")
    if args.save_per_frame_csv:
        per_frame_fields = [
            "seq_id",
            "frame_stem",
            "frame_index",
            "moving_pixels",
            "static_pixels",
            "ignore_pixels",
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
        with open(per_frame_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=per_frame_fields)
            writer.writeheader()
            for row in per_frame_rows:
                writer.writerow(row)

    eval_config = {
        "evaluation": {
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "split": str(args.split),
            "device": device,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "output_dir": output_dir,
            "save_visuals": bool(args.save_visuals),
            "max_visuals": int(args.max_visuals),
            "visual_stride": int(args.visual_stride),
            "save_per_frame_csv": bool(args.save_per_frame_csv),
            "num_frames": int(num_frames),
        },
        "cfg": cfg,
    }
    eval_config_path = os.path.join(output_dir, "eval_config.yaml")
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_config, f, sort_keys=False, default_flow_style=False)

    print("=== Evaluation Finished ===")
    print(f"moving_iou        : {final_metrics['moving_iou']:.6f}")
    print(f"moving_f1         : {final_metrics['moving_f1']:.6f}")
    print(f"moving_precision  : {final_metrics['moving_precision']:.6f}")
    print(f"moving_recall     : {final_metrics['moving_recall']:.6f}")
    print(f"mean_iou          : {final_metrics['mean_iou']:.6f}")
    print(f"output_dir        : {output_dir}")
    print(f"eval_metrics_csv  : {eval_metrics_csv_path}")
    print(f"eval_config_yaml  : {eval_config_path}")
    print(f"visuals_saved     : {'yes' if args.save_visuals else 'no'} ({visual_saved})")
    if args.save_per_frame_csv:
        print(f"per_frame_csv     : {per_frame_csv_path}")


if __name__ == "__main__":
    main()
