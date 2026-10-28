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
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from helper.dataloader_helper import make_sequences
from helper.mos_eval_utils import add_counts, compute_counts_from_pred_target, empty_counts, metrics_from_counts, select_sequences
from mos_models import build_mos_model
from temporal_mos_e2e_dataset import TemporalMOSE2EDataset
from helper.mos_input import channel_count, normalization
from helper.mos_augmentation import augmentation_settings
from helper.mos_lovasz_loss import MOSCrossEntropyLovasz, loss_components, loss_settings


def distributed():
    return dist.is_available() and dist.is_initialized()


def primary():
    return not distributed() or dist.get_rank() == 0


def setup_device(requested):
    device = torch.device(requested)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device.type == "cuda":
        print(f"[startup] local_rank={os.environ.get('LOCAL_RANK', '0')} checking CUDA", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if world_size > 1:
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError("torchrun requested more workers than visible CUDA devices")
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        # Fail before loading the dataset if the container's CUDA build cannot run on this GPU.
        try:
            torch.ones(1, device=device).add_(1)
            torch.cuda.synchronize(device)
        except RuntimeError as exc:
            raise RuntimeError("Container PyTorch/CUDA cannot execute on this GPU. "
                               "For Tesla P100 use a build supporting Pascal (sm_60).") from exc
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        print(f"[startup] rank={os.environ.get('RANK')} initializing {backend}, "
              f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}", flush=True)
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=120))
        print(f"[startup] rank={dist.get_rank()} process group ready", flush=True)
    return device


def wrap_distributed(model, device):
    if not distributed():
        return model
    return DDP(model, device_ids=[device.index] if device.type == "cuda" else None)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def sync_buffers(model):
    # Validation bypasses DDP to allow unequal, non-padded validation shards.
    # All ranks must start from the same BatchNorm running statistics.
    if distributed():
        for buffer in unwrap_model(model).buffers():
            dist.broadcast(buffer, src=0)


def nonfinite_model_state(model):
    """Return non-finite parameter and buffer names for phase diagnostics."""
    raw_model = unwrap_model(model)
    bad_parameters = [name for name, param in raw_model.named_parameters()
                      if not torch.isfinite(param).all()]
    bad_buffers = [name for name, buffer in raw_model.named_buffers()
                   if not torch.isfinite(buffer).all()]
    return bad_parameters, bad_buffers


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
    cfg["mos_augmentation"] = augmentation_settings(cfg.get("mos_augmentation"), mode)
    if str(cfg["mos_train_params"].get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("This path supports AdamW")
    if str(cfg["mos_train_params"].get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("This path supports cosine scheduling")
    cfg["mos_train_params"]["loss"] = loss_settings(cfg["mos_train_params"].get("loss"))
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
    seed = int(cfg["mos_train_params"]["seed"])
    sampler = None
    if distributed():
        if split == "train":
            # Equal numbers of training batches on all ranks; reshuffled each epoch.
            sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        else:
            # Each validation sample is evaluated exactly once, without padding duplicates.
            dataset = Subset(dataset, range(dist.get_rank(), len(dataset), dist.get_world_size()))
    generator = torch.Generator().manual_seed(seed + (dist.get_rank() if distributed() else 0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train" and sampler is None,
                      sampler=sampler, num_workers=workers,
                      pin_memory=device.type == "cuda", generator=generator)


def make_criterion(cfg, device):
    train = cfg["mos_train_params"]
    weights = torch.tensor([train["static_class_weight"], train["moving_class_weight"]],
                           dtype=torch.float32, device=device)
    settings = loss_settings(train.get("loss"))
    if settings["type"] == "ce_lovasz":
        return MOSCrossEntropyLovasz(weights, settings["lovasz_weight"],
                                    settings["lovasz_classes"], settings["lovasz_per_image"])
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
              accumulation=1, amp=False, amp_dtype=torch.bfloat16, clip=5.0, max_batches=None,
              diagnose_validation=False):
    training = optimizer is not None
    if diagnose_validation and not training:
        print(f"[val-diagnostic] rank={dist.get_rank() if distributed() else 0} "
              "entering model.eval", flush=True)
    model.train(training)
    if not training:
        if diagnose_validation:
            print(f"[val-diagnostic] rank={dist.get_rank() if distributed() else 0} "
                  "synchronizing buffers", flush=True)
        sync_buffers(model)
        if diagnose_validation:
            bad_parameters, bad_buffers = nonfinite_model_state(model)
            print(f"[val-diagnostic] rank={dist.get_rank() if distributed() else 0} "
                  f"buffers synchronized; nonfinite_parameters={bad_parameters[:10]} "
                  f"nonfinite_buffers={bad_buffers[:10]}", flush=True)
            if bad_parameters or bad_buffers:
                raise FloatingPointError("Non-finite model state before validation")
    forward_model = model if training else unwrap_model(model)
    counts = empty_counts()
    total_loss = 0.0
    total_loss_weight = 0.0
    total_lovasz = 0.0
    loss_batches = 0
    samples = 0
    total_seconds = 0.0
    limit = min(len(loader), max_batches) if max_batches else len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    for index in range(limit):
        if diagnose_validation and not training and index == 0:
            print(f"[val-diagnostic] rank={dist.get_rank() if distributed() else 0} "
                  "fetching first validation batch", flush=True)
        x, valid, dt, y, _meta = next(iterator)
        if diagnose_validation and not training and index == 0:
            sample_ids = list(zip(_meta["seq_id"], _meta["frame_stem"]))
            print(f"[val-diagnostic] rank={dist.get_rank() if distributed() else 0} "
                  f"first validation batch loaded: samples={sample_ids}", flush=True)
        x = x.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        dt = dt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if not (y != -1).any():
            raise ValueError("Batch contains no supervised current pixels")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        sync_step = (index + 1) % accumulation == 0 or index + 1 == limit
        sync_context = model.no_sync() if training and isinstance(model, DDP) and not sync_step else nullcontext()
        with torch.set_grad_enabled(training), sync_context:
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp and device.type == "cuda"):
                logits = forward_model(x, valid, dt)
                components = loss_components(criterion, logits, y)
                loss_ce, loss_lovasz = components["loss_ce"], components["loss_lovasz"]
                loss = components["loss_total"]
            if diagnose_validation and not training and index == 0:
                print(f"[val-diagnostic] rank={dist.get_rank() if distributed() else 0} "
                      f"first validation forward complete: logits_finite={bool(torch.isfinite(logits).all())} "
                    f"loss_finite={bool(torch.isfinite(loss))}", flush=True)
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
                # CrossEntropyLoss(weight=...) divides by the local sum of class weights.
                # Correct DDP's rank average to match a single global weighted batch.
                loss_weight = criterion.weight[y[y != -1]].sum() if criterion.weight is not None else (y != -1).sum()
                backward_loss = loss_ce
                if distributed():
                    global_weight = loss_weight.detach().clone()
                    dist.all_reduce(global_weight)
                    backward_loss = loss_ce * loss_weight * dist.get_world_size() / global_weight
                # DDP already averages rank gradients. Lovasz is local and
                # nonadditive: never apply CE's weight correction to it.
                if isinstance(criterion, MOSCrossEntropyLovasz):
                    backward_loss = backward_loss + criterion.lovasz_weight * loss_lovasz
                group_size = min(accumulation, limit - (index // accumulation) * accumulation)
                scaler.scale(backward_loss / group_size).backward()
                if sync_step:
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
        weight = float(criterion.weight[y[y != -1]].sum()) if criterion.weight is not None else int((y != -1).sum())
        total_loss += float(loss_ce.detach()) * weight
        total_loss_weight += weight
        total_lovasz += float(loss_lovasz.detach())
        loss_batches += 1
        samples += n
        add_counts(counts, compute_counts_from_pred_target(logits.detach().argmax(1), y))
        if primary() and (index == 0 or (index + 1) % 20 == 0 or index + 1 == limit):
            running_ce = total_loss / total_loss_weight
            running_lovasz = total_lovasz / loss_batches
            running_total = running_ce + getattr(criterion, "lovasz_weight", 0.0) * running_lovasz
            print(f"{'train' if training else 'val'} {index + 1}/{limit} "
                  f"loss={running_total:.5f} ce={running_ce:.5f} lovasz={running_lovasz:.5f}", flush=True)
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if distributed():
        keys = list(counts)
        totals = torch.tensor([total_loss, total_loss_weight, samples, total_lovasz, loss_batches, *[counts[k] for k in keys]],
                              dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        total_loss, total_loss_weight, samples = totals[:3].tolist()
        total_lovasz, loss_batches = totals[3:5].tolist()
        counts = {key: int(value) for key, value in zip(keys, totals[5:].tolist())}
        maxima = torch.tensor([total_seconds, peak_vram], dtype=torch.float64, device=device)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        total_seconds, peak_vram = maxima.tolist()
        samples = int(samples)
    if samples == 0:
        raise ValueError("DataLoader produced no batches")
    mean_ce = total_loss / total_loss_weight
    mean_lovasz = total_lovasz / loss_batches
    mean_total = mean_ce + getattr(criterion, "lovasz_weight", 0.0) * mean_lovasz
    result = {"loss": mean_total, "loss_ce": mean_ce, "loss_lovasz": mean_lovasz,
              "loss_total": mean_total, **metrics_from_counts(counts),
              "samples": samples,
              ("step_seconds_per_sample" if training else "inference_seconds_per_sample"): total_seconds / samples,
              "peak_vram_bytes": int(peak_vram)}
    return result


def gradient_health(model):
    """Inspect unscaled gradients before clipping can turn inf gradients into NaNs."""
    missing, nonfinite = [], []
    active = set()
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        elif (parameter.grad != 0).any():
            active.add(name.split(".")[0])
    inactive = sorted({"encoder", "temporal", "decoder"} - active)
    return {"missing": missing, "nonfinite": nonfinite, "inactive_modules": inactive}


def dry_run_step(model, x, valid, dt, y, criterion, optimizer, scaler, device,
                 amp_enabled, amp_dtype, max_attempts=16):
    for attempt in range(1, max_attempts + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x, valid, dt)
            loss = criterion(logits, y)
        if logits.shape != y.shape[:1] + (2,) + y.shape[1:] or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise AssertionError("Synthetic forward produced an invalid shape or non-finite logits/loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        health = gradient_health(model)
        if health["missing"]:
            raise AssertionError(f"Synthetic backward has missing gradients: {health['missing'][:8]}")
        overflow = torch.tensor(int(bool(health["nonfinite"])), device=device)
        if distributed():
            dist.all_reduce(overflow, op=dist.ReduceOp.MAX)
        if overflow.item():
            if not scaler.is_enabled():
                raise AssertionError(f"Non-finite gradients without FP16 scaling: {health['nonfinite'][:8]}")
            # unscale_ recorded inf/NaN: step skips the update; update lowers the scale.
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if primary():
                print(f"[dry-run] FP16 overflow attempt={attempt}/{max_attempts}; "
                      f"scale={old_scale:g}->{scaler.get_scale():g}; "
                      f"parameters={health['nonfinite'][:4]}; retrying", flush=True)
            continue
        if health["inactive_modules"]:
            raise AssertionError(f"No nonzero gradients in modules: {health['inactive_modules']}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise AssertionError("Synthetic optimizer update produced non-finite parameters")
        return logits, loss, attempt
    raise AssertionError(f"FP16 gradients remained non-finite after {max_attempts} attempts; "
                         f"last_scale={scaler.get_scale():g}; parameters={health['nonfinite'][:8]}. "
                         "Check the model numerics or compare with amp: false.")


def synthetic_dry_run(cfg, device):
    if primary():
        print("[dry-run] building temporal model", flush=True)
    model = wrap_distributed(build_mos_model(cfg).to(device), device)
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
    if primary():
        print(f"[dry-run] forward/backward: T={t}, input={tuple(x.shape)}, "
              f"amp={amp_enabled}, dtype={amp_dtype}", flush=True)
    logits, loss, attempts = dry_run_step(model, x, valid, dt, y, criterion, optimizer, scaler,
                                         device, amp_enabled, amp_dtype)
    if primary():
        print(f"Dry run passed: T={t}, logits={tuple(logits.shape)}, loss={loss.item():.5f}, "
              f"attempts={attempts}, scale={scaler.get_scale():g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", default="src/configs/mos_semanticKitti_temporal_e2e.yaml")
    parser.add_argument("--log_root", help="Override mos_log_params.log_root for this run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU/process")
    parser.add_argument("--num_workers", type=int, help="DataLoader workers per GPU/process")
    parser.add_argument("--max_batches", type=int, help="Limit train and val batches for a smoke run")
    parser.add_argument("--dry-run", action="store_true", help="One synthetic forward/backward, no dataset or logs")
    parser.add_argument("--diagnose-validation", action="store_true",
                        help="Log the transition to validation and the first validation forward pass")
    args = parser.parse_args()
    print(f"[startup] torch={torch.__version__} rank={os.environ.get('RANK', '0')} "
          f"world_size={os.environ.get('WORLD_SIZE', '1')} device={args.device}", flush=True)
    device = setup_device(args.device)
    try:
        train(args, device)
    finally:
        if distributed():
            dist.destroy_process_group()


def train(args, device):
    cfg_path, cfg = read_config(args.cfg_path)
    if args.log_root is not None:
        cfg["mos_log_params"]["log_root"] = args.log_root
    check_splits(cfg["mos_data_params"])
    train_cfg = cfg["mos_train_params"]
    rank = dist.get_rank() if distributed() else 0
    world_size = dist.get_world_size() if distributed() else 1
    seed_all(int(train_cfg["seed"]) + rank, bool(train_cfg.get("deterministic", False)))
    if args.dry_run:
        synthetic_dry_run(cfg, device)
        return
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    workers = args.num_workers if args.num_workers is not None else int(train_cfg["num_workers"])
    accumulation = int(train_cfg.get("gradient_accumulation", 1))
    if min(epochs, batch_size, accumulation) <= 0:
        raise ValueError("epochs, batch_size and gradient_accumulation must be positive")
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    train_cfg.update(epochs=epochs, batch_size=batch_size, num_workers=workers)
    cfg["run_metadata"] = {"world_size": world_size, "batch_size_per_gpu": batch_size,
                           "effective_global_batch_size": batch_size * world_size * accumulation,
                           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cli_args": vars(args)}
    amp_enabled, amp_dtype = amp_settings(cfg, device)
    if device.type == "cuda":
        print(f"rank={rank} device={device} gpu={torch.cuda.get_device_name(device)}", flush=True)
    if primary():
        print(f"world_size={world_size} batch_size_per_gpu={batch_size} "
              f"effective_global_batch_size={batch_size * world_size * accumulation} "
              f"amp={amp_enabled} amp_dtype={amp_dtype}", flush=True)
    train_loader = make_loader(cfg, "train", batch_size, workers, device)
    val_loader = make_loader(cfg, "val", batch_size, workers, device)
    model = wrap_distributed(build_mos_model(cfg).to(device), device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(train_cfg["learning_rate"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    if sum(p.numel() for p in params) != sum(p.numel() for group in optimizer.param_groups for p in group["params"]):
        raise RuntimeError("Trainable parameters missing from optimizer")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_cfg["learning_rate_min"]))
    criterion = make_criterion(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    writer = None
    if primary():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["mos_log_params"]["log_root"]) / f"{cfg['mos_log_params']['run_name']}_{stamp}"
        checkpoint_dir, csv_dir, tb_dir = (run_dir / name for name in ("checkpoints", "csv", "tensorboard"))
        for path in (checkpoint_dir, csv_dir, tb_dir):
            path.mkdir(parents=True, exist_ok=False)
        shutil.copy2(cfg_path, run_dir / "config_source.yaml")
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        if cfg["mos_log_params"].get("use_tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(tb_dir))
            except ImportError:
                print("TensorBoard unavailable; continuing with CSV")
        print(f"run={run_dir} params={sum(p.numel() for p in model.parameters())} trainable={sum(p.numel() for p in params)}")
        csv_path = csv_dir / "epochs.csv"
    best_iou = -1.0
    for epoch in range(1, epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train = run_phase(model, train_loader, criterion, device, optimizer, scaler,
                          accumulation, amp_enabled, amp_dtype,
                          float(train_cfg.get("clip_grad_norm", 5.0)), args.max_batches)
        val = run_phase(model, val_loader, criterion, device, max_batches=args.max_batches,
                        amp=amp_enabled, amp_dtype=amp_dtype,
                        diagnose_validation=getattr(args, "diagnose_validation", False))
        trained_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if not primary():
            continue
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
        payload = {"epoch": epoch, "model_state_dict": unwrap_model(model).state_dict(),
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
    if primary():
        print(f"Best checkpoint: {checkpoint_dir / 'best_moving_iou.pt'}")


if __name__ == "__main__":
    main()
