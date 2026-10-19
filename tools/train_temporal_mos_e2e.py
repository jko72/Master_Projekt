#!/usr/bin/env python3
"""Independent supervised train/validation entry for temporal SalsaNext MOS."""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helper.dataloader_helper import make_sequences
from helper.mos_eval_utils import add_counts, compute_counts_from_pred_target, empty_counts, metrics_from_counts, select_sequences
from mos_models import build_mos_model
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from helper.mos_input import channel_count, normalization


def read_config(path):
    path = Path(path)
    if not path.exists():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["mos_model_params"]["name"] != "salsanext_temporal_e2e":
        raise ValueError("This entry requires mos_model_params.name=salsanext_temporal_e2e")
    if cfg["mos_train_params"].get("pretrained_path") is not None:
        raise ValueError("This scratch entry does not load a pretrained checkpoint")
    if cfg["mos_model_params"].get("fusion", "bottleneck_only") != "bottleneck_only":
        raise ValueError("Only bottleneck_only fusion is implemented")
    mode = cfg["mos_data_params"].get("input_mode")
    if mode not in {"range_xyz", "range_xyz_intensity"} or int(cfg["mos_data_params"].get("ignore_index", -1)) != -1:
        raise ValueError("This path requires range_xyz or range_xyz_intensity input and ignore_index=-1")
    expected = channel_count(mode)
    if int(cfg["mos_model_params"].get("in_channels", expected)) != expected:
        raise ValueError(f"mos_model_params.in_channels must be {expected} for {mode}")
    cfg["mos_model_params"]["in_channels"] = expected
    normalization(cfg, mode)
    if str(cfg["mos_train_params"].get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("This path supports AdamW")
    if str(cfg["mos_train_params"].get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("This path supports cosine scheduling")
    return path.resolve(), cfg


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def check_splits(data):
    splits = [{str(v).zfill(2) for v in data[f"{name}_sequences"]} for name in ("train", "val", "test")]
    if any(splits[i] & splits[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train, val and test sequence IDs must be disjoint")


def make_loader(cfg, split, batch_size, workers, device):
    data = cfg["mos_data_params"]
    sequences = select_sequences(make_sequences(cfg["dataset_path"]), data[f"{split}_sequences"], split)
    dataset = TemporalMOSE2EDataset(sequences, cfg, split=split)
    if len(dataset) == 0:
        raise ValueError(f"No usable {split} samples after previous_offsets filtering")
    generator = torch.Generator().manual_seed(int(cfg["mos_train_params"]["seed"]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=workers,
                      pin_memory=device.type == "cuda", generator=generator)


def make_criterion(cfg, device):
    train = cfg["mos_train_params"]
    weights = torch.tensor([train["static_class_weight"], train["moving_class_weight"]],
                           dtype=torch.float32, device=device)
    return torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1)


def amp_settings(cfg, device):
    params = cfg["mos_train_params"]
    enabled = bool(params.get("amp", True)) and device.type == "cuda"
    name = str(params.get("amp_dtype", "bfloat16")).lower()
    if name not in {"bfloat16", "float16"}:
        raise ValueError("mos_train_params.amp_dtype must be bfloat16 or float16")
    if enabled and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support bfloat16; set amp_dtype: float16 or amp: false")
    return enabled, (torch.bfloat16 if name == "bfloat16" else torch.float16)


def run_phase(model, loader, criterion, device, optimizer=None, scaler=None,
              accumulation=1, amp=False, amp_dtype=torch.bfloat16, clip=5.0, max_batches=None):
    training = optimizer is not None
    model.train(training)
    counts = empty_counts()
    total_loss = 0.0
    samples = 0
    total_seconds = 0.0
    limit = min(len(loader), max_batches) if max_batches else len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, (x, valid, dt, y, _meta) in enumerate(loader):
        if index >= limit:
            break
        x = x.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        dt = dt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if not (y != -1).any():
            raise ValueError("Batch contains no supervised current pixels")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp and device.type == "cuda"):
                logits = model(x, valid, dt)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                bad_parameters = [name for name, param in model.named_parameters()
                                  if not torch.isfinite(param).all()]
                sample_ids = list(zip(_meta["seq_id"], _meta["frame_stem"]))
                raise FloatingPointError(
                    f"Non-finite loss in batch {index}: samples={sample_ids}, "
                    f"amp={amp}, amp_dtype={amp_dtype}, "
                    f"input_finite={bool(torch.isfinite(x).all())}, "
                    f"logits_finite={bool(torch.isfinite(logits).all())}, "
                    f"nonfinite_parameters={bad_parameters[:10]}"
                )
            if training:
                group_size = min(accumulation, limit - (index // accumulation) * accumulation)
                scaler.scale(loss / group_size).backward()
                if (index + 1) % accumulation == 0 or index + 1 == limit:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip if clip > 0 else float("inf"),
                        error_if_nonfinite=not scaler.is_enabled(),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_seconds += time.perf_counter() - start
        n = x.shape[0]
        total_loss += float(loss.detach()) * n
        samples += n
        add_counts(counts, compute_counts_from_pred_target(logits.detach().argmax(1), y))
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == limit:
            print(f"{'train' if training else 'val'} {index + 1}/{limit} loss={total_loss / samples:.5f}", flush=True)
    if samples == 0:
        raise ValueError("DataLoader produced no batches")
    result = {"loss": total_loss / samples, **metrics_from_counts(counts),
              "samples": samples,
              ("step_seconds_per_sample" if training else "inference_seconds_per_sample"): total_seconds / samples,
              "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
    return result


def synthetic_dry_run(cfg, device):
    model = build_mos_model(cfg).to(device)
    height, width = 32, 64
    t = len(cfg["mos_data_params"]["previous_offsets"]) + 1
    x = torch.randn(1, t, int(cfg["mos_model_params"]["in_channels"]), height, width, device=device)
    valid = torch.ones(1, t, height, width, dtype=torch.bool, device=device)
    valid[:, 0, :4] = False
    offsets = sorted(cfg["mos_data_params"]["previous_offsets"], reverse=True)
    dt = torch.tensor([[-float(v) / cfg["mos_data_params"].get("scan_rate_hz", 10.0)
                        for v in offsets] + [0.0]], device=device)
    y = torch.randint(0, 2, (1, height, width), device=device)
    y[:, :4] = -1
    criterion = make_criterion(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    amp_enabled, amp_dtype = amp_settings(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    model.train()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = model(x, valid, dt)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0,
                                   error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    if logits.shape != (1, 2, height, width) or not torch.isfinite(logits).all():
        raise AssertionError("Synthetic forward failed")
    for name, module in (("encoder", model.encoder), ("temporal", model.temporal), ("decoder", model.decoder)):
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters()):
            raise AssertionError(f"Missing finite gradient in {name}")
    print(f"Dry run passed: T={t}, logits={tuple(logits.shape)}, loss={loss.item():.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", default="src/configs/mos_semanticKitti_temporal_e2e.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_batches", type=int, help="Limit train and val batches for a smoke run")
    parser.add_argument("--dry-run", action="store_true", help="One synthetic forward/backward, no dataset or logs")
    args = parser.parse_args()
    cfg_path, cfg = read_config(args.cfg_path)
    check_splits(cfg["mos_data_params"])
    train_cfg = cfg["mos_train_params"]
    seed_all(int(train_cfg["seed"]), bool(train_cfg.get("deterministic", False)))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.dry_run:
        synthetic_dry_run(cfg, device)
        return
    epochs = args.epochs or int(train_cfg["epochs"])
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    workers = args.num_workers if args.num_workers is not None else int(train_cfg["num_workers"])
    accumulation = int(train_cfg.get("gradient_accumulation", 1))
    if min(epochs, batch_size, accumulation) <= 0:
        raise ValueError("epochs, batch_size and gradient_accumulation must be positive")
    train_loader = make_loader(cfg, "train", batch_size, workers, device)
    val_loader = make_loader(cfg, "val", batch_size, workers, device)
    model = build_mos_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    if sum(p.numel() for p in params) != sum(p.numel() for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("Trainable parameters missing from optimizer")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_cfg["learning_rate_min"]))
    criterion = make_criterion(cfg, device)
    amp_enabled, amp_dtype = amp_settings(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["mos_log_params"]["log_root"]) / f"{cfg['mos_log_params']['run_name']}_{stamp}"
    checkpoint_dir, csv_dir, tb_dir = (run_dir / name for name in ("checkpoints", "csv", "tensorboard"))
    for path in (checkpoint_dir, csv_dir, tb_dir):
        path.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cfg_path, run_dir / "config.yaml")
    writer = None
    if cfg["mos_log_params"].get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(tb_dir))
        except ImportError:
            print("TensorBoard unavailable; continuing with CSV")
    print(f"run={run_dir} params={sum(p.numel() for p in model.parameters())} trainable={sum(p.numel() for p in params)}")
    best_iou = -1.0
    csv_path = csv_dir / "epochs.csv"
    for epoch in range(1, epochs + 1):
        train = run_phase(model, train_loader, criterion, device, optimizer, scaler,
                          accumulation, amp_enabled, amp_dtype,
                          float(train_cfg.get("clip_grad_norm", 5.0)), args.max_batches)
        val = run_phase(model, val_loader, criterion, device, max_batches=args.max_batches,
                        amp=amp_enabled, amp_dtype=amp_dtype)
        trained_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch, "lr": trained_lr}
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"val_{k}": v for k, v in val.items()})
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=row.keys())
            if epoch == 1:
                csv_writer.writeheader()
            csv_writer.writerow(row)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(), "config": cfg,
                   "input_normalization": normalization(cfg, cfg["mos_data_params"]["input_mode"]),
                   "val_moving_iou": val["moving_iou"]}
        torch.save(payload, checkpoint_dir / "last.pt")
        if val["moving_iou"] > best_iou:
            best_iou = val["moving_iou"]
            torch.save(payload, checkpoint_dir / "best_moving_iou.pt")
        print(f"epoch={epoch} train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} "
              f"val_moving_iou={val['moving_iou']:.5f} best={best_iou:.5f}")
    if writer:
        writer.close()
    print(f"Best checkpoint: {checkpoint_dir / 'best_moving_iou.pt'}")


if __name__ == "__main__":
    main()
