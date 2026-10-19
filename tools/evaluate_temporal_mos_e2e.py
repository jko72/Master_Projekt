#!/usr/bin/env python3
"""Strictly reload and evaluate temporal E2E MOS checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from helper.mos_eval_utils import add_counts, compute_counts_from_pred_target, empty_counts, metrics_from_counts
from mos_models import build_mos_model
from helper.mos_input import normalization
from train_temporal_mos_e2e import check_splits, make_criterion, make_loader


def apply_history_mode(x, valid, mode):
    if mode == "normal":
        return x, valid
    x, valid = x.clone(), valid.clone()
    if mode == "zero":
        x[:, :-1].zero_()
        valid[:, :-1].zero_()
    elif mode == "copy_current":
        x[:, :-1] = x[:, -1:].expand_as(x[:, :-1])
        valid[:, :-1] = valid[:, -1:].expand_as(valid[:, :-1])
    else:
        raise ValueError(f"Unknown history mode: {mode}")
    return x, valid


def evaluate(model, loader, criterion, device, history_mode="normal", max_batches=None):
    model.eval()
    counts, total_loss, samples, seconds = empty_counts(), 0.0, 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    limit = min(len(loader), max_batches) if max_batches else len(loader)
    with torch.inference_mode():
        for index, (x, valid, dt, y, _meta) in enumerate(loader):
            if index >= limit:
                break
            x, valid = apply_history_mode(x, valid, history_mode)
            x, valid, dt, y = (v.to(device) for v in (x, valid, dt, y))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(x, valid, dt)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds += time.perf_counter() - start
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite evaluation loss in batch {index}")
            n = x.shape[0]
            samples += n
            total_loss += float(loss) * n
            add_counts(counts, compute_counts_from_pred_target(logits.argmax(1), y))
            if index == 0 or (index + 1) % 20 == 0 or index + 1 == limit:
                print(f"eval {index + 1}/{limit}", flush=True)
    if not samples:
        raise ValueError("Evaluation loader produced no samples")
    return {"loss": total_loss / samples, **metrics_from_counts(counts), **counts,
            "samples": samples, "inference_seconds_per_sample": seconds / samples,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--history-mode", choices=("normal", "zero", "copy_current"), default="normal")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    if cfg["mos_model_params"]["name"] != "salsanext_temporal_e2e":
        raise ValueError("Checkpoint is not a temporal E2E MOS run")
    check_splits(cfg["mos_data_params"])
    model = build_mos_model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    loader = make_loader(cfg, args.split, args.batch_size, args.num_workers, device)
    result = evaluate(model, loader, make_criterion(cfg, device), device, args.history_mode, args.max_batches)
    result = {"split": args.split, "history_mode": args.history_mode,
              "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": payload["epoch"],
              "normalization": normalization(cfg, cfg["mos_data_params"]["input_mode"])["mode"], **result}
    output = checkpoint.parent.parent / "eval"
    output.mkdir(exist_ok=True)
    csv_path = output / f"{args.split}_{args.history_mode}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(result)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
