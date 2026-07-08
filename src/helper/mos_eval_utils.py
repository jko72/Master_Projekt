from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


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


def empty_counts() -> Dict[str, int]:
    return {
        "tp_moving": 0,
        "fp_moving": 0,
        "fn_moving": 0,
        "tn_static": 0,
        "correct": 0,
        "valid_pixels": 0,
    }


def compute_counts_from_pred_target(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> Dict[str, int]:
    with torch.no_grad():
        valid = target != ignore_index
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            return empty_counts()

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


def add_counts(dst: Dict[str, int], src: Dict[str, int]) -> Dict[str, int]:
    for k in dst:
        dst[k] += int(src[k])
    return dst


def counts_and_metrics_row(counts: Dict[str, int]) -> Dict[str, float | int]:
    metrics = metrics_from_counts(counts)
    row: Dict[str, float | int] = {
        "moving_iou": float(metrics["moving_iou"]),
        "moving_precision": float(metrics["moving_precision"]),
        "moving_recall": float(metrics["moving_recall"]),
        "moving_f1": float(metrics["moving_f1"]),
        "static_iou": float(metrics["static_iou"]),
        "mean_iou": float(metrics["mean_iou"]),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
        "tp_moving": int(counts["tp_moving"]),
        "fp_moving": int(counts["fp_moving"]),
        "fn_moving": int(counts["fn_moving"]),
        "tn_static": int(counts["tn_static"]),
        "valid_pixels": int(counts["valid_pixels"]),
    }
    return row
