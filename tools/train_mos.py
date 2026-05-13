#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import random
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from helper.dataloader_helper import make_sequences
from mos_dataset import MOSFrameDataset
from mos_models import build_mos_model


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


def normalize_seq_id(seq_id) -> str:
    s = str(seq_id)
    return s.zfill(2) if s.isdigit() else s


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


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def compute_in_channels(input_mode: str, residual_offsets: Sequence[int]) -> int:
    if input_mode == "range":
        return 1
    if input_mode == "residual":
        return len(residual_offsets)
    if input_mode == "range_residual":
        return 1 + len(residual_offsets)
    raise ValueError(
        f"Invalid input_mode='{input_mode}'. Supported: ['range', 'residual', 'range_residual']"
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


def build_moving_frame_sampler(dataset: MOSFrameDataset, moving_weight: float, static_weight: float):
    if len(dataset) == 0:
        raise ValueError("Cannot build sampler for empty dataset.")

    weights = []
    for sample in dataset.samples:
        moving_pixels = sample.get("moving_pixels_cached", None)
        if moving_pixels is None:
            stats = dataset._get_cached_or_load_label_stats(sample["mos_label_path"])
            moving_pixels = int(stats["moving_pixels"])
            sample["moving_pixels_cached"] = moving_pixels
        weight = float(moving_weight) if int(moving_pixels) > 0 else float(static_weight)
        weights.append(weight)

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights_t, num_samples=len(weights), replacement=True)
    return sampler


def train_one_epoch(model, loader, criterion, optimizer, device: str, cfg: Dict):
    model.train()

    clip_grad_norm = float(cfg.get("mos_train_params", {}).get("clip_grad_norm", 0.0))
    ignore_index = int(cfg.get("mos_data_params", {}).get("ignore_index", -1))

    loss_sum = 0.0
    sample_count = 0
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
        if clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        bs = int(x.shape[0])
        loss_sum += float(loss.item()) * bs
        sample_count += bs

        counts = compute_mos_counts(logits.detach(), y.detach(), ignore_index=ignore_index)
        for k in agg:
            agg[k] += int(counts[k])

    if sample_count == 0:
        raise RuntimeError("Train loader produced zero samples.")

    mean_loss = loss_sum / float(sample_count)
    metrics = metrics_from_counts(agg)
    return mean_loss, agg, metrics


def validate_one_epoch(model, loader, criterion, device: str, cfg: Dict):
    model.eval()

    ignore_index = int(cfg.get("mos_data_params", {}).get("ignore_index", -1))

    loss_sum = 0.0
    sample_count = 0
    agg = {
        "tp_moving": 0,
        "fp_moving": 0,
        "fn_moving": 0,
        "tn_static": 0,
        "correct": 0,
        "valid_pixels": 0,
    }

    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device, non_blocking=device.startswith("cuda"))
            y = y.to(device, non_blocking=device.startswith("cuda")).long()

            logits = model(x)
            loss = criterion(logits, y)

            bs = int(x.shape[0])
            loss_sum += float(loss.item()) * bs
            sample_count += bs

            counts = compute_mos_counts(logits, y, ignore_index=ignore_index)
            for k in agg:
                agg[k] += int(counts[k])

    if sample_count == 0:
        raise RuntimeError("Validation loader produced zero samples.")

    mean_loss = loss_sum / float(sample_count)
    metrics = metrics_from_counts(agg)
    return mean_loss, agg, metrics


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_dataset_class_stats(name: str, dataset: MOSFrameDataset):
    stats = dataset.get_class_stats()
    print(
        f"{name}_label_stats  : "
        f"ignore_pixels={int(stats['ignore_pixels'])}, "
        f"static_pixels={int(stats['static_pixels'])}, "
        f"moving_pixels={int(stats['moving_pixels'])}, "
        f"frames_with_moving={int(stats['frames_with_moving'])}"
    )


def main():
    parser = argparse.ArgumentParser(description="Full MOS baseline training with train/val split.")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to MOS YAML config")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: cuda if available else cpu")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--input_mode", type=str, default=None, choices=["range", "residual", "range_residual"])
    parser.add_argument("--residual_offsets", type=str, default=None, help="e.g. '1' or '1,2,3'")
    parser.add_argument("--moving_weight", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    cfg_path = resolve_cfg_path(args.cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)

    cfg.setdefault("mos_data_params", {})
    cfg.setdefault("mos_model_params", {})
    cfg.setdefault("mos_train_params", {})
    cfg.setdefault("mos_log_params", {})

    mdata = cfg["mos_data_params"]
    mmodel = cfg["mos_model_params"]
    mtrain = cfg["mos_train_params"]
    mlog = cfg["mos_log_params"]

    mdata.setdefault("input_mode", "range_residual")
    mdata.setdefault("residual_offsets", [1])
    mdata.setdefault("mos_label_folder", "mos_labels")
    mdata.setdefault("ignore_index", -1)
    mdata.setdefault("train_sequences", ["07"])
    mdata.setdefault("val_sequences", ["08"])
    mdata.setdefault("require_moving_train", False)
    mdata.setdefault("require_moving_val", False)
    mdata.setdefault("min_moving_pixels", 1)
    mdata.setdefault("use_moving_frame_sampler", False)
    mdata.setdefault("moving_frame_weight", 5.0)
    mdata.setdefault("static_frame_weight", 1.0)

    mmodel.setdefault("name", "unet_small")
    mmodel.setdefault("base_channels", 32)
    mmodel.setdefault("dropout", 0.1)
    mmodel.setdefault("norm", "batch")

    mtrain.setdefault("batch_size", 8)
    mtrain.setdefault("num_workers", 4)
    mtrain.setdefault("epochs", 20)
    mtrain.setdefault("learning_rate", 5e-4)
    mtrain.setdefault("weight_decay", 1e-4)
    mtrain.setdefault("clip_grad_norm", 5.0)
    mtrain.setdefault("static_class_weight", 1.0)
    mtrain.setdefault("moving_class_weight", 10.0)
    mtrain.setdefault("optimizer", "adamw")
    mtrain.setdefault("scheduler", "none")
    mtrain.setdefault("seed", 42)
    mtrain.setdefault("deterministic", False)

    mlog.setdefault("log_root", "/home/devuser/workspace/LidarGaussianVideoView/mos_logs")
    mlog.setdefault("run_name", "mos_baseline")
    mlog.setdefault("use_tensorboard", True)

    if args.run_name is not None:
        mlog["run_name"] = str(args.run_name)
    if args.epochs is not None:
        mtrain["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        mtrain["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        mtrain["learning_rate"] = float(args.lr)
    if args.input_mode is not None:
        mdata["input_mode"] = str(args.input_mode)
    if args.residual_offsets is not None:
        mdata["residual_offsets"] = parse_residual_offsets(args.residual_offsets)
    else:
        mdata["residual_offsets"] = parse_residual_offsets(mdata.get("residual_offsets", [1]))
    if args.moving_weight is not None:
        mtrain["moving_class_weight"] = float(args.moving_weight)
    if args.num_workers is not None:
        mtrain["num_workers"] = int(args.num_workers)

    in_channels = compute_in_channels(str(mdata["input_mode"]), mdata["residual_offsets"])
    mmodel["in_channels"] = int(in_channels)
    mmodel["num_classes"] = 2

    dataset_path = cfg.get("dataset_path", None)
    if not dataset_path:
        raise ValueError("Config missing required key: dataset_path")

    seed = int(mtrain["seed"])
    deterministic = bool(mtrain["deterministic"])
    set_seed(seed, deterministic)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = str(device)
    pin_memory = device.startswith("cuda")

    all_sequences = make_sequences(dataset_path)
    train_sequences = select_sequences(all_sequences, mdata["train_sequences"], "train")
    val_sequences = select_sequences(all_sequences, mdata["val_sequences"], "val")

    train_dataset = MOSFrameDataset(
        sequences=train_sequences,
        cfg=cfg,
        split="train",
        input_mode=mdata["input_mode"],
        residual_offsets=mdata["residual_offsets"],
        mos_label_folder=mdata["mos_label_folder"],
        require_moving=bool(mdata["require_moving_train"]),
        min_moving_pixels=int(mdata["min_moving_pixels"]),
    )
    val_dataset = MOSFrameDataset(
        sequences=val_sequences,
        cfg=cfg,
        split="val",
        input_mode=mdata["input_mode"],
        residual_offsets=mdata["residual_offsets"],
        mos_label_folder=mdata["mos_label_folder"],
        require_moving=bool(mdata["require_moving_val"]),
        min_moving_pixels=int(mdata["min_moving_pixels"]),
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset is empty after filtering.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after filtering.")

    # Print real label-derived class stats (loaded from mos_labels/*.npy), not only cached meta.
    print_dataset_class_stats("train", train_dataset)
    print_dataset_class_stats("val", val_dataset)

    batch_size = int(mtrain["batch_size"])
    num_workers = int(mtrain["num_workers"])

    train_sampler = None
    train_shuffle = True
    if bool(mdata.get("use_moving_frame_sampler", False)):
        train_sampler = build_moving_frame_sampler(
            train_dataset,
            moving_weight=float(mdata.get("moving_frame_weight", 5.0)),
            static_weight=float(mdata.get("static_frame_weight", 1.0)),
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    model = build_mos_model(cfg).to(device)
    n_params = count_trainable_params(model)

    class_weights = torch.tensor(
        [float(mtrain["static_class_weight"]), float(mtrain["moving_class_weight"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=int(mdata["ignore_index"]),
        weight=class_weights,
    )

    optimizer_name = str(mtrain.get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(mtrain["learning_rate"]),
            weight_decay=float(mtrain["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer='{optimizer_name}'. Currently supported: ['adamw']")

    scheduler_name = str(mtrain.get("scheduler", "none")).lower()
    if scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler='{scheduler_name}'. Currently supported: ['none']")

    log_root = os.path.abspath(str(mlog.get("log_root", "/home/devuser/workspace/LidarGaussianVideoView/mos_logs")))
    run_name = str(mlog.get("run_name", "mos_baseline"))
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    writer = None
    if bool(mlog.get("use_tensorboard", True)):
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar("Data/train_num_samples", len(train_dataset), 0)
        writer.add_scalar("Data/val_num_samples", len(val_dataset), 0)
        writer.add_scalar("Data/batch_size", batch_size, 0)
        writer.add_scalar("Data/moving_class_weight", float(mtrain["moving_class_weight"]), 0)

    print("=== MOS Training ===")
    print(f"cfg_path          : {cfg_path}")
    print(f"dataset_path      : {dataset_path}")
    print(f"train_sequences   : {[normalize_seq_id(s) for s in mdata['train_sequences']]}")
    print(f"val_sequences     : {[normalize_seq_id(s) for s in mdata['val_sequences']]}")
    print(f"train_samples     : {len(train_dataset)}")
    print(f"val_samples       : {len(val_dataset)}")
    print(f"input_mode        : {mdata['input_mode']}")
    print(f"residual_offsets  : {mdata['residual_offsets']}")
    print(f"model             : {mmodel.get('name', 'unet_small')}")
    print(f"in_channels       : {int(mmodel['in_channels'])}")
    print(f"num_classes       : {int(mmodel['num_classes'])}")
    print(f"base_channels     : {int(mmodel.get('base_channels', 32))}")
    print(f"trainable_params  : {n_params}")
    print(f"batch_size        : {batch_size}")
    print(f"epochs            : {int(mtrain['epochs'])}")
    print(f"lr                : {float(mtrain['learning_rate'])}")
    print(f"moving_weight     : {float(mtrain['moving_class_weight'])}")
    print(f"device            : {device}")
    print(f"log_dir           : {log_dir}")

    epochs = int(mtrain["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss, train_counts, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
        )
        val_loss, val_counts, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        if scheduler is not None:
            scheduler.step()

        lr_cur = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mIoU={val_metrics['mean_iou']:.4f} | "
            f"val_moving_iou={val_metrics['moving_iou']:.4f} | "
            f"val_moving_f1={val_metrics['moving_f1']:.4f} | "
            f"val_moving_recall={val_metrics['moving_recall']:.4f} | "
            f"val_moving_precision={val_metrics['moving_precision']:.4f}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

            writer.add_scalar("Train/moving_iou", train_metrics["moving_iou"], epoch)
            writer.add_scalar("Train/moving_precision", train_metrics["moving_precision"], epoch)
            writer.add_scalar("Train/moving_recall", train_metrics["moving_recall"], epoch)
            writer.add_scalar("Train/moving_f1", train_metrics["moving_f1"], epoch)
            writer.add_scalar("Train/static_iou", train_metrics["static_iou"], epoch)
            writer.add_scalar("Train/mean_iou", train_metrics["mean_iou"], epoch)
            writer.add_scalar("Train/pixel_accuracy", train_metrics["pixel_accuracy"], epoch)

            writer.add_scalar("Val/moving_iou", val_metrics["moving_iou"], epoch)
            writer.add_scalar("Val/moving_precision", val_metrics["moving_precision"], epoch)
            writer.add_scalar("Val/moving_recall", val_metrics["moving_recall"], epoch)
            writer.add_scalar("Val/moving_f1", val_metrics["moving_f1"], epoch)
            writer.add_scalar("Val/static_iou", val_metrics["static_iou"], epoch)
            writer.add_scalar("Val/mean_iou", val_metrics["mean_iou"], epoch)
            writer.add_scalar("Val/pixel_accuracy", val_metrics["pixel_accuracy"], epoch)

            writer.add_scalar("ConfusionTrain/tp_moving", train_counts["tp_moving"], epoch)
            writer.add_scalar("ConfusionTrain/fp_moving", train_counts["fp_moving"], epoch)
            writer.add_scalar("ConfusionTrain/fn_moving", train_counts["fn_moving"], epoch)
            writer.add_scalar("ConfusionTrain/tn_static", train_counts["tn_static"], epoch)

            writer.add_scalar("ConfusionVal/tp_moving", val_counts["tp_moving"], epoch)
            writer.add_scalar("ConfusionVal/fp_moving", val_counts["fp_moving"], epoch)
            writer.add_scalar("ConfusionVal/fn_moving", val_counts["fn_moving"], epoch)
            writer.add_scalar("ConfusionVal/tn_static", val_counts["tn_static"], epoch)

            writer.add_scalar("LearningRate/lr", lr_cur, epoch)

    if writer is not None:
        writer.flush()
        writer.close()

    print(f"Training finished. TensorBoard events written to: {log_dir}")


if __name__ == "__main__":
    main()
