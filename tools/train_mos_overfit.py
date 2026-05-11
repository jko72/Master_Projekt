#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from helper.dataloader_helper import make_sequences
from mos_dataset import MOSFrameDataset
from mos_models import build_mos_model


def parse_residual_offsets(text: str) -> List[int]:
    toks = text.replace(";", ",").replace(" ", ",").split(",")
    vals = [int(t) for t in toks if t.strip() != ""]
    vals = sorted(set(v for v in vals if v >= 1))
    if not vals:
        raise ValueError("residual_offsets must include at least one positive integer.")
    return vals


def normalize_seq_id(seq_id: str) -> str:
    s = str(seq_id)
    return s.zfill(2) if s.isdigit() else s


def resolve_cfg_path(cfg_path: str) -> str:
    candidates = [
        cfg_path,
        os.path.join(PROJECT_ROOT, cfg_path),
        os.path.join(SRC_DIR, cfg_path),
        os.path.join(SRC_DIR, "configs", os.path.basename(cfg_path)),
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def compute_expected_in_channels(input_mode: str, residual_offsets: List[int]) -> int:
    if input_mode == "range":
        return 1
    if input_mode == "residual":
        return len(residual_offsets)
    if input_mode == "range_residual":
        return 1 + len(residual_offsets)
    raise ValueError(f"Unknown input_mode: {input_mode}")


def compute_mos_counts(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -1) -> Dict[str, int]:
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
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

        p = pred[valid]
        t = target[valid]

        tp_m = int(((p == 1) & (t == 1)).sum().item())
        fp_m = int(((p == 1) & (t == 0)).sum().item())
        fn_m = int(((p == 0) & (t == 1)).sum().item())
        tn_s = int(((p == 0) & (t == 0)).sum().item())
        correct = int((p == t).sum().item())
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

    # For static class (0):
    # TP_static = TN_static (pred=0,target=0)
    # FP_static = FN_moving (pred=0,target=1)
    # FN_static = FP_moving (pred=1,target=0)
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


def main():
    parser = argparse.ArgumentParser(description="Overfit test for MOS baseline model.")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--seq_id", type=str, default="07")
    parser.add_argument(
        "--input_mode",
        type=str,
        default="range_residual",
        choices=["range", "residual", "range_residual"],
    )
    parser.add_argument("--residual_offsets", type=str, default="1")
    parser.add_argument("--mos_label_folder", type=str, default="mos_labels")
    parser.add_argument("--require_moving", action="store_true")
    parser.add_argument("--min_moving_pixels", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--moving_weight", type=float, default=20.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(int(args.seed))
    cfg_path = resolve_cfg_path(args.cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    residual_offsets = parse_residual_offsets(args.residual_offsets)
    seq_id = normalize_seq_id(args.seq_id)
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = str(device)

    log_root = os.path.join(PROJECT_ROOT, "LidarGaussianVideoView", "mos_logs")
    if args.log_dir is None:
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root, f"overfit_{ts}")
    else:
        log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    seqs = make_sequences(cfg["dataset_path"])
    seqs = [s for s in seqs if normalize_seq_id(s.get("seq_id", "")) == seq_id]
    if not seqs:
        raise ValueError(f"No sequence matched seq_id={seq_id} in dataset_path={cfg['dataset_path']}.")

    if not args.require_moving:
        print("WARNING: require_moving is disabled. Overfit test may include many frames without moving pixels.")

    ds = MOSFrameDataset(
        sequences=seqs,
        cfg=cfg,
        split="train",
        input_mode=args.input_mode,
        residual_offsets=residual_offsets,
        mos_label_folder=args.mos_label_folder,
        device="cpu",
        require_moving=args.require_moving,
        min_moving_pixels=int(args.min_moving_pixels),
    )
    if len(ds) == 0:
        raise RuntimeError("MOS dataset is empty after filtering.")

    n_use = min(len(ds), int(args.max_samples))
    subset = Subset(ds, list(range(n_use)))
    if n_use > 50:
        print("WARNING: Overfit test is intended for small datasets; consider max_samples <= 50.")

    stats_selected = ds.get_class_stats(max_samples=n_use)
    if int(stats_selected["moving_pixels"]) <= 0 or int(stats_selected["frames_with_moving"]) <= 0:
        raise RuntimeError(
            "No samples with moving pixels found in selected subset. "
            "Try --require_moving and/or lower --min_moving_pixels."
        )

    computed_in_channels = compute_expected_in_channels(args.input_mode, residual_offsets)
    cfg.setdefault("mos_model_params", {})
    previous_in_channels = cfg["mos_model_params"].get("in_channels", None)
    if previous_in_channels is not None and int(previous_in_channels) != int(computed_in_channels):
        print(
            f"WARNING: cfg.mos_model_params.in_channels={previous_in_channels} "
            f"does not match input_mode={args.input_mode}. Overriding to {computed_in_channels}."
        )
    cfg["mos_model_params"]["in_channels"] = int(computed_in_channels)
    cfg["mos_model_params"]["num_classes"] = 2
    cfg["mos_model_params"].setdefault("name", "unet_small")
    cfg["mos_model_params"].setdefault("base_channels", 32)
    cfg["mos_model_params"].setdefault("dropout", 0.1)
    cfg["mos_model_params"].setdefault("norm", "batch")

    model = build_mos_model(cfg).to(device)
    class_weights = torch.tensor([1.0, float(args.moving_weight)], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.startswith("cuda"),
    )

    print("=== MOS Overfit Training ===")
    print(f"log_root         : {log_root}")
    print(f"log_dir          : {log_dir}")
    print(f"seq_id           : {seq_id}")
    print(f"num_samples      : {n_use}")
    print(f"input_mode       : {args.input_mode}")
    print(f"residual_offsets : {residual_offsets}")
    print(f"batch_size       : {int(args.batch_size)}")
    print(f"epochs           : {int(args.epochs)}")
    print(f"device           : {device}")
    print(f"moving_weight    : {float(args.moving_weight)}")

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar("Data/num_samples", n_use, 0)
    writer.add_scalar("Data/batch_size", int(args.batch_size), 0)
    writer.add_scalar("Data/moving_weight", float(args.moving_weight), 0)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        agg = {
            "tp_moving": 0,
            "fp_moving": 0,
            "fn_moving": 0,
            "tn_static": 0,
            "correct": 0,
            "valid_pixels": 0,
        }

        for x, y, _meta in loader:
            x = x.to(device, non_blocking=device.startswith("cuda"))
            y = y.to(device, non_blocking=device.startswith("cuda")).long()

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_counts = compute_mos_counts(logits.detach(), y.detach(), ignore_index=-1)
            for k in agg:
                agg[k] += int(batch_counts[k])

            loss_sum += float(loss.item())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No batches were processed. Check dataset and DataLoader settings.")

        epoch_loss = loss_sum / n_batches
        m = metrics_from_counts(agg)
        lr_cur = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d}/{int(args.epochs)} | "
            f"loss={epoch_loss:.6f} | "
            f"mIoU={m['mean_iou']:.4f} | "
            f"moving_iou={m['moving_iou']:.4f} | "
            f"moving_f1={m['moving_f1']:.4f} | "
            f"moving_recall={m['moving_recall']:.4f}"
        )

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Metrics/moving_iou", m["moving_iou"], epoch)
        writer.add_scalar("Metrics/moving_precision", m["moving_precision"], epoch)
        writer.add_scalar("Metrics/moving_recall", m["moving_recall"], epoch)
        writer.add_scalar("Metrics/moving_f1", m["moving_f1"], epoch)
        writer.add_scalar("Metrics/static_iou", m["static_iou"], epoch)
        writer.add_scalar("Metrics/mean_iou", m["mean_iou"], epoch)
        writer.add_scalar("Metrics/pixel_accuracy", m["pixel_accuracy"], epoch)
        writer.add_scalar("Confusion/tp_moving", agg["tp_moving"], epoch)
        writer.add_scalar("Confusion/fp_moving", agg["fp_moving"], epoch)
        writer.add_scalar("Confusion/fn_moving", agg["fn_moving"], epoch)
        writer.add_scalar("Confusion/tn_static", agg["tn_static"], epoch)
        writer.add_scalar("LearningRate/lr", lr_cur, epoch)

    writer.flush()
    writer.close()
    print("Training finished. TensorBoard events written.")


if __name__ == "__main__":
    main()

