#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import datetime as dt
from datetime import timedelta
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from torch.utils.data import DataLoader, DistributedSampler, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from helper.dataloader_helper import make_sequences
from mos_dataset import MOSFrameDataset, TemporalMOSDataset
from mos_models import build_mos_model
from helper.mos_input import channel_count, normalization
from helper.mos_lovasz_loss import MOSCrossEntropyLovasz, loss_components, loss_settings


def distributed():
    return dist.is_available() and dist.is_initialized()


def primary():
    return not distributed() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def setup_device(requested):
    device = torch.device(requested)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", device.index or 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.ones(1, device=device).add_(1)
        torch.cuda.synchronize(device)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=timedelta(seconds=120))
    return str(device)


def make_criterion(cfg, device):
    train = cfg["mos_train_params"]
    weights = torch.tensor([train["static_class_weight"], train["moving_class_weight"]],
                           dtype=torch.float32, device=device)
    settings = loss_settings(train.get("loss"))
    ignore_index = int(cfg.get("mos_data_params", {}).get("ignore_index", -1))
    if settings["type"] == "ce_lovasz":
        if ignore_index != -1:
            raise ValueError("CE + Lovasz requires ignore_index=-1")
        return MOSCrossEntropyLovasz(weights, settings["lovasz_weight"],
                                    settings["lovasz_classes"], settings["lovasz_per_image"])
    return torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


def amp_settings(cfg, device):
    params = cfg.get("mos_train_params", {})
    enabled = bool(params.get("amp", False)) and torch.device(device).type == "cuda"
    name = str(params.get("amp_dtype", "float16")).lower()
    if name not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    if enabled and name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bfloat16; use amp_dtype: float16")
    return enabled, torch.float16 if name == "float16" else torch.bfloat16


def make_grad_scaler(cfg, device):
    enabled, dtype = amp_settings(cfg, device)
    # A modest initial FP16 scale reduces startup overflows with weighted MOS.
    return torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16,
                                init_scale=128.0)


def shard_validation(dataset):
    # No padding: every validation frame contributes exactly once.
    if distributed():
        return Subset(dataset, range(dist.get_rank(), len(dataset), dist.get_world_size()))
    return dataset


def _smoke_sequences(sequences, limit):
    result = []
    for sequence in sequences:
        if limit <= 0:
            break
        count = min(limit, len(sequence["paths"]))
        result.append({**sequence, "paths": sequence["paths"][:count],
                       "poses": sequence["poses"][:count]})
        limit -= count
    return result


def parse_residual_offsets(value, *, allow_empty: bool = False) -> List[int]:
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
    if not vals and not allow_empty:
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
    return channel_count(input_mode, residual_offsets)


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



def _print_batch_progress(phase: str, epoch: int, total_epochs: int,
                          completed: int, total_batches: int, running_loss: float,
                          last_bucket: int) -> int:
    if not primary():
        return last_bucket
    percent = 100 * completed // total_batches
    filled = 20 * completed // total_batches
    bar = "#" * filled + "." * (20 - filled)
    line = (
        f"Epoche {epoch}/{total_epochs} | {phase:11s} "
        f"[{bar}] {percent:3d}% ({completed}/{total_batches} Batches) "
        f"Loss {running_loss:.5f}"
    )
    if sys.stdout.isatty():
        print(f"\r{line}", end="", flush=True)
    else:
        bucket = percent // 10
        if bucket != last_bucket or completed == total_batches:
            print(line, flush=True)
            last_bucket = bucket
    return last_bucket


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    cfg: Dict,
    encoder_frozen: bool = False,
    freeze_encoder_bn_eval: bool = True,
    epoch_index: int = 1,
    total_epochs: int = 1,
    scaler=None,
):
    return _run_epoch(model, loader, criterion, device, cfg, optimizer=optimizer,
                      encoder_frozen=encoder_frozen, freeze_encoder_bn_eval=freeze_encoder_bn_eval,
                      epoch_index=epoch_index, total_epochs=total_epochs, scaler=scaler)


def validate_one_epoch(model, loader, criterion, device: str, cfg: Dict,
                       epoch_index: int = 1, total_epochs: int = 1):
    return _run_epoch(model, loader, criterion, device, cfg,
                      epoch_index=epoch_index, total_epochs=total_epochs)


def _run_epoch(model, loader, criterion, device, cfg, optimizer=None,
               encoder_frozen=False, freeze_encoder_bn_eval=True,
               epoch_index=1, total_epochs=1, scaler=None):
    training = optimizer is not None
    amp_enabled, amp_dtype = amp_settings(cfg, device)
    if scaler is None:
        scaler = make_grad_scaler(cfg, device)
    model.train(training)
    raw_model = unwrap_model(model)
    if training and encoder_frozen and freeze_encoder_bn_eval:
        if not hasattr(raw_model, "encoder"):
            raise RuntimeError("Cannot freeze encoder BatchNorm: model has no encoder")
        raw_model.encoder.eval()
    # Unequal validation shards must not enter DDP forward collectives. Broadcast
    # running BN statistics once, then evaluate each shard on the unwrapped model.
    if not training and distributed():
        for buffer in raw_model.buffers():
            dist.broadcast(buffer, src=0)
    forward_model = model if training else raw_model
    ignore = int(cfg.get("mos_data_params", {}).get("ignore_index", -1))
    clip = float(cfg.get("mos_train_params", {}).get("clip_grad_norm", 0.0))
    agg = dict(tp_moving=0, fp_moving=0, fn_moving=0, tn_static=0, correct=0, valid_pixels=0)
    ce_numerator = ce_denominator = lovasz_sum = legacy_sum = 0.0
    sample_count = batch_count = optimizer_steps = skipped_steps = 0
    peak_vram = 0
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_bucket = -1
    for index, (x, y, _meta) in enumerate(loader, 1):
        x = x.to(device, non_blocking=str(device).startswith("cuda"))
        y = y.to(device, non_blocking=str(device).startswith("cuda")).long()
        valid = y != ignore
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                                enabled=amp_enabled):
                logits = forward_model(x)
                # Native weighted CE autocasts to FP32; Lovasz explicitly uses FP32.
                if valid.any():
                    parts = loss_components(criterion, logits, y)
                else:
                    zero = logits.float().reshape(-1)[:0].sum()
                    parts = dict(loss_ce=zero, loss_lovasz=zero, loss_total=zero)
            # Never cast the CE denominator to FP16: class-weight sums at full
            # resolution readily exceed FP16's maximum finite value (65504).
            denominator = (criterion.weight[y[valid]].float().sum() if criterion.weight is not None
                           else valid.sum().float())
            if training:
                global_denominator = denominator.detach().clone()
                finite = torch.isfinite(parts["loss_total"]).to(dtype=torch.int32)
                if distributed():
                    dist.all_reduce(global_denominator)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item() or global_denominator.item() <= 0:
                    raise FloatingPointError("Training batch has non-finite loss or no supervised pixels")
                # DDP averages rank gradients; correct weighted CE to the global
                # denominator. Lovasz remains the mean of local batch objectives.
                world = dist.get_world_size() if distributed() else 1
                objective = parts["loss_ce"]
                if distributed():
                    objective = objective * denominator * world / global_denominator
                objective = objective + getattr(criterion, "lovasz_weight", 0.0) * parts["loss_lovasz"]
                optimizer.zero_grad(set_to_none=True)
                previous_scale = scaler.get_scale()
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip,
                                                  error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped_steps += 1
                    if primary():
                        print(f"[AMP] Gradient overflow: skipped update, scale={previous_scale:g}->{scaler.get_scale():g}", flush=True)
                else:
                    optimizer_steps += 1
            elif not torch.isfinite(parts["loss_total"]):
                raise FloatingPointError("Validation produced non-finite loss")
        bs = int(x.shape[0])
        ce_numerator += float(parts["loss_ce"].detach()) * float(denominator)
        ce_denominator += float(denominator)
        lovasz_sum += float(parts["loss_lovasz"].detach())
        legacy_sum += float(parts["loss_total"].detach()) * bs
        sample_count += bs
        batch_count += 1
        counts = compute_mos_counts(logits.detach(), y, ignore_index=ignore)
        for key in agg:
            agg[key] += counts[key]
        last_bucket = _print_batch_progress(
            "Training" if training else "Validierung", epoch_index, total_epochs,
            index, len(loader), legacy_sum / sample_count, last_bucket)
    if torch.device(device).type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device)
    totals = torch.tensor([ce_numerator, ce_denominator, lovasz_sum, batch_count,
                           legacy_sum, sample_count, *agg.values()], dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(totals)
    if distributed():
        maximum = torch.tensor(peak_vram, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        peak_vram = int(maximum.item())
    values = totals.tolist()
    ce_numerator, ce_denominator, lovasz_sum, batch_count, legacy_sum, sample_count = values[:6]
    agg = {key: int(value) for key, value in zip(agg, values[6:])}
    if not sample_count or not ce_denominator:
        raise RuntimeError("Loader produced no supervised samples")
    ce = ce_numerator / ce_denominator
    lovasz = lovasz_sum / batch_count
    total = ce + getattr(criterion, "lovasz_weight", 0.0) * lovasz
    # Preserve legacy single-process CE-only epoch logging.
    reported = total if isinstance(criterion, MOSCrossEntropyLovasz) or distributed() else legacy_sum / sample_count
    metrics = {**metrics_from_counts(agg), "loss_ce": ce, "loss_lovasz": lovasz, "loss_total": total,
               "optimizer_steps": optimizer_steps, "amp_skipped_steps": skipped_steps,
               "amp_scale": scaler.get_scale(), "peak_vram_bytes": peak_vram}
    if training and cfg.get("run_metadata", {}).get("smoke_test") and optimizer_steps == 0:
        raise RuntimeError("GPU smoke test completed without a successful optimizer update")
    if primary() and sys.stdout.isatty():
        print()
    return reported, agg, metrics


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _non_encoder_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    if not hasattr(model, "encoder") or not isinstance(model.encoder, torch.nn.Module):
        return list(model.parameters())
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    return [p for p in model.parameters() if id(p) not in encoder_param_ids]


def print_trainable_summary(model: torch.nn.Module):
    total = count_params(model)
    trainable = count_trainable_params(model)
    print(f"[PARAMS] total={total}, trainable={trainable}")

    if hasattr(model, "encoder") and isinstance(model.encoder, torch.nn.Module):
        encoder_total = count_params(model.encoder)
        encoder_trainable = count_trainable_params(model.encoder)
        non_encoder_params = _non_encoder_parameters(model)
        non_encoder_total = sum(p.numel() for p in non_encoder_params)
        non_encoder_trainable = sum(p.numel() for p in non_encoder_params if p.requires_grad)
        print(f"[PARAMS] encoder total={encoder_total}, trainable={encoder_trainable}")
        print(f"[PARAMS] non_encoder total={non_encoder_total}, trainable={non_encoder_trainable}")


def _make_optimizer(optimizer_name: str, params, lr: float, weight_decay: float):
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer='{optimizer_name}'. Currently supported: ['adamw', 'adam', 'sgd']")


def get_group_name(group, idx):
    """
    Reads group.get("name") if available, otherwise returns f"group_{idx}".
    """
    name = group.get("name", None)
    if name is None or str(name).strip() == "":
        return f"group_{idx}"
    return str(name)


def get_optimizer_group_lrs(optimizer):
    """
    Returns dictionary with group names and current learning rates.
    Example:
      {"encoder": 5e-5, "decoder_head": 5e-4}
    If group has no name, use "group_0", "group_1", ...
    """
    return {
        get_group_name(group, idx): float(group["lr"])
        for idx, group in enumerate(optimizer.param_groups)
    }


def resolve_min_lrs_for_groups(optimizer, cfg):
    """
    Returns list of min_lrs in the same order as optimizer.param_groups.

    Rules:
    - If group name == "encoder":
        use encoder_learning_rate_min if not None
        else use 0.01 * current encoder lr as fallback
    - If group name == "decoder_head":
        use learning_rate_min if not None
        else use 0.01 * current decoder/head lr as fallback
    - For single/default group:
        use learning_rate_min if not None
        else use 0.01 * current lr as fallback
    """
    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    learning_rate_min = mtrain.get("learning_rate_min", None)
    encoder_learning_rate_min = mtrain.get("encoder_learning_rate_min", None)
    single_group = len(optimizer.param_groups) == 1

    min_lrs = []
    for idx, group in enumerate(optimizer.param_groups):
        group_name = get_group_name(group, idx)
        current_lr = float(group["lr"])

        if group_name == "encoder":
            configured_min_lr = encoder_learning_rate_min
        elif group_name == "decoder_head" or single_group:
            configured_min_lr = learning_rate_min
        else:
            configured_min_lr = learning_rate_min

        if configured_min_lr is None:
            min_lrs.append(0.01 * current_lr)
        else:
            min_lrs.append(float(configured_min_lr))
    return min_lrs


class GroupDecayLRScheduler:
    def __init__(self, optimizer, min_lrs, epochs: int, start_epoch: int = 1, schedule_type: str = "linear"):
        if len(min_lrs) != len(optimizer.param_groups):
            raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lrs)}.")
        if schedule_type not in {"linear", "cosine"}:
            raise ValueError(f"Unsupported GroupDecayLRScheduler schedule_type='{schedule_type}'.")

        self.optimizer = optimizer
        self.min_lrs = [float(v) for v in min_lrs]
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.epochs = int(epochs)
        self.start_epoch = max(1, int(start_epoch))
        self.schedule_type = str(schedule_type)
        self.last_epoch = 0
        self._last_lr = list(self.base_lrs)

    def _progress_for_epoch(self, epoch: int) -> float:
        if int(epoch) < self.start_epoch:
            return 0.0
        effective_epochs = max(1, self.epochs - self.start_epoch + 1)
        current_step = int(epoch) - self.start_epoch + 1
        return min(1.0, max(0.0, float(current_step) / float(effective_epochs)))

    def _lr_for_group(self, base_lr: float, min_lr: float, progress: float) -> float:
        if base_lr == 0.0:
            return 0.0
        if self.schedule_type == "linear":
            return min_lr + (base_lr - min_lr) * (1.0 - progress)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor

    def step(self, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        if self.last_epoch < self.start_epoch:
            self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]
            return self._last_lr

        progress = self._progress_for_epoch(self.last_epoch)
        new_lrs = []
        for group, base_lr, min_lr in zip(self.optimizer.param_groups, self.base_lrs, self.min_lrs):
            lr = self._lr_for_group(float(base_lr), float(min_lr), progress)
            group["lr"] = float(lr)
            new_lrs.append(float(lr))
        self._last_lr = new_lrs
        return new_lrs

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return {
            "min_lrs": list(self.min_lrs),
            "base_lrs": list(self.base_lrs),
            "epochs": int(self.epochs),
            "start_epoch": int(self.start_epoch),
            "schedule_type": str(self.schedule_type),
            "last_epoch": int(self.last_epoch),
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict):
        self.min_lrs = [float(v) for v in state_dict["min_lrs"]]
        self.base_lrs = [float(v) for v in state_dict["base_lrs"]]
        self.epochs = int(state_dict["epochs"])
        self.start_epoch = int(state_dict["start_epoch"])
        self.schedule_type = str(state_dict["schedule_type"])
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        self._last_lr = [float(v) for v in state_dict.get("_last_lr", self.base_lrs)]


def build_scheduler(optimizer, cfg, epochs: int):
    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    scheduler_name = str(mtrain.get("scheduler", "none")).lower()
    supported = "none, cosine, linear, plateau"

    if scheduler_name == "none":
        return None, scheduler_name

    if scheduler_name not in {"cosine", "linear", "plateau"}:
        raise ValueError(f"Unsupported scheduler='{scheduler_name}'. Supported: {supported}.")

    start_epoch = int(mtrain.get("scheduler_start_epoch", 1) or 1)
    min_lrs = resolve_min_lrs_for_groups(optimizer, cfg)

    print(f"[SCHEDULER] type={scheduler_name} start_epoch={start_epoch} epochs={int(epochs)}")
    for idx, (group, min_lr) in enumerate(zip(optimizer.param_groups, min_lrs)):
        print(
            "[SCHEDULER] group={0} base_lr={1:g} min_lr={2:g}".format(
                get_group_name(group, idx),
                float(group["lr"]),
                float(min_lr),
            )
        )

    if scheduler_name in {"cosine", "linear"}:
        return (
            GroupDecayLRScheduler(
                optimizer=optimizer,
                min_lrs=min_lrs,
                epochs=int(epochs),
                start_epoch=start_epoch,
                schedule_type=scheduler_name,
            ),
            scheduler_name,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(mtrain.get("scheduler_mode", "max")),
        factor=float(mtrain.get("scheduler_factor", 0.5)),
        patience=int(mtrain.get("scheduler_patience", 5)),
        threshold=float(mtrain.get("scheduler_threshold", 0.0001)),
        cooldown=int(mtrain.get("scheduler_cooldown", 0)),
        min_lr=min_lrs,
    )
    return scheduler, scheduler_name


def _get_plateau_monitor_value(cfg, val_loss, val_metrics):
    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    monitor = str(mtrain.get("scheduler_monitor", "val_moving_iou"))
    supported = {
        "val_moving_iou": float(val_metrics["moving_iou"]),
        "val_mean_iou": float(val_metrics["mean_iou"]),
        "val_loss": float(val_loss),
        "val_moving_f1": float(val_metrics["moving_f1"]),
        "val_moving_recall": float(val_metrics["moving_recall"]),
        "val_moving_precision": float(val_metrics["moving_precision"]),
    }
    if monitor not in supported:
        raise ValueError(
            "Unsupported scheduler_monitor='{0}'. Supported: {1}.".format(
                monitor,
                ", ".join(sorted(supported.keys())),
            )
        )
    return supported[monitor]


def step_scheduler_if_needed(scheduler, scheduler_name, cfg, epoch, val_loss, val_metrics):
    if scheduler is None or scheduler_name == "none":
        return

    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    start_epoch = int(mtrain.get("scheduler_start_epoch", 1) or 1)
    if int(epoch) < start_epoch:
        return

    if scheduler_name in {"cosine", "linear"}:
        scheduler.step(int(epoch))
        return

    if scheduler_name == "plateau":
        metric_value = _get_plateau_monitor_value(cfg, val_loss, val_metrics)
        scheduler.step(metric_value)
        return

    raise ValueError(f"Unsupported scheduler='{scheduler_name}'. Supported: none, cosine, linear, plateau.")


def lr_value_for_logging(group_lrs: Dict[str, float], optimizer):
    lr_main = float(optimizer.param_groups[0]["lr"])
    lr_encoder = group_lrs.get("encoder", None)
    lr_decoder_head = group_lrs.get("decoder_head", None)
    if lr_decoder_head is None and len(optimizer.param_groups) == 1:
        lr_decoder_head = lr_main
    return lr_main, lr_encoder, lr_decoder_head


def format_lr_for_print(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def build_optimizer_with_param_groups(model: torch.nn.Module, cfg: Dict):
    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    base_lr = float(mtrain["learning_rate"])
    encoder_lr = mtrain.get("encoder_learning_rate", None)
    weight_decay = float(mtrain.get("weight_decay", 0.0))
    optimizer_name = str(mtrain.get("optimizer", "adamw")).lower()

    if encoder_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = _make_optimizer(
            optimizer_name=optimizer_name,
            params=params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
        print("[OPTIMIZER] using single parameter group lr={0:g} params={1}".format(
            base_lr,
            sum(p.numel() for p in params),
        ))
        return optimizer

    if not hasattr(model, "encoder") or not isinstance(model.encoder, torch.nn.Module):
        raise RuntimeError(
            "mos_train_params.encoder_learning_rate is set, but the MOS model has no "
            "self.encoder module for separate finetuning."
        )

    encoder_lr = float(encoder_lr)
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_head_params = [
        p for p in model.parameters()
        if id(p) not in encoder_param_ids and p.requires_grad
    ]

    overlap_ids = {id(p) for p in encoder_params}.intersection(id(p) for p in decoder_head_params)
    if overlap_ids:
        raise RuntimeError("Optimizer parameter group construction produced duplicate parameters.")
    if not encoder_params:
        raise RuntimeError("encoder_learning_rate is set, but no trainable encoder parameters were found.")
    if not decoder_head_params:
        raise RuntimeError("encoder_learning_rate is set, but no trainable decoder/head parameters were found.")

    param_groups = [
        {"params": encoder_params, "lr": encoder_lr, "name": "encoder"},
        {"params": decoder_head_params, "lr": base_lr, "name": "decoder_head"},
    ]
    optimizer = _make_optimizer(
        optimizer_name=optimizer_name,
        params=param_groups,
        lr=base_lr,
        weight_decay=weight_decay,
    )
    print("[OPTIMIZER] using separate parameter groups:")
    for group in optimizer.param_groups:
        print(
            "  group={0} lr={1:g} params={2}".format(
                group.get("name", "unnamed"),
                float(group["lr"]),
                sum(p.numel() for p in group["params"]),
            )
        )
    return optimizer


def set_encoder_trainable(model: torch.nn.Module, trainable: bool, bn_eval: bool = True):
    if not hasattr(model, "encoder") or not isinstance(model.encoder, torch.nn.Module):
        if trainable:
            return
        raise RuntimeError("Cannot freeze encoder because model has no self.encoder module.")

    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)

    if trainable:
        model.encoder.train()
    elif bn_eval:
        model.encoder.eval()


def is_encoder_frozen_for_epoch(epoch: int, freeze_encoder_epochs: int) -> bool:
    return int(freeze_encoder_epochs) > 0 and int(epoch) <= int(freeze_encoder_epochs)


def print_dataset_class_stats(name: str, dataset: MOSFrameDataset):
    stats = dataset.get_class_stats()
    print(
        f"{name}_label_stats  : "
        f"ignore_pixels={int(stats['ignore_pixels'])}, "
        f"static_pixels={int(stats['static_pixels'])}, "
        f"moving_pixels={int(stats['moving_pixels'])}, "
        f"frames_with_moving={int(stats['frames_with_moving'])}"
    )


def print_dataset_channel_debug(name: str, dataset: MOSFrameDataset):
    print(f"{name}_channel_count: {int(dataset.channel_count)}")
    if len(dataset) > 0:
        _, _, meta = dataset[0]
        channel_names = meta.get("channel_names", None)
        if channel_names:
            print(f"{name}_channel_names: {channel_names}")

def _extract_prefixed_state(state_dict, prefix: str) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict to be a dict, got {type(state_dict)}")
    prefix = str(prefix)
    out = {}
    for k, v in state_dict.items():
        ks = str(k)
        if ks.startswith(prefix):
            out[ks[len(prefix) :]] = v
    return out


def _extract_encoder_state_from_checkpoint(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        state = ckpt["encoder_state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint['encoder_state_dict'] is not a dict.")
        return state

    if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
        backbone = ckpt["backbone_state_dict"]
        if isinstance(backbone, dict) and "encoder" in backbone:
            state = backbone["encoder"]
            if not isinstance(state, dict):
                raise TypeError("checkpoint['backbone_state_dict']['encoder'] is not a dict.")
            return state

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ms = ckpt["model_state_dict"]
        enc = _extract_prefixed_state(ms, "encoder.")
        if enc:
            return enc

    if isinstance(ckpt, dict):
        # Plain encoder-only state_dict or full model state_dict with "encoder." prefix.
        enc = _extract_prefixed_state(ckpt, "encoder.")
        if enc:
            return enc
        return ckpt

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _extract_decoder_state_from_checkpoint(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "decoder_state_dict" in ckpt:
        state = ckpt["decoder_state_dict"]
        if not isinstance(state, dict):
            raise TypeError("checkpoint['decoder_state_dict'] is not a dict.")
        return state

    if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
        backbone = ckpt["backbone_state_dict"]
        if isinstance(backbone, dict) and "decoder" in backbone:
            state = backbone["decoder"]
            if not isinstance(state, dict):
                raise TypeError("checkpoint['backbone_state_dict']['decoder'] is not a dict.")
            return state

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ms = ckpt["model_state_dict"]
        dec = _extract_prefixed_state(ms, "decoder.")
        if dec:
            return dec

    if isinstance(ckpt, dict):
        dec = _extract_prefixed_state(ckpt, "decoder.")
        if dec:
            return dec

    raise KeyError("Could not find decoder weights in checkpoint.")


def _check_pretrain_channels(ckpt, target_cfg, adapt):
    """Reject ambiguous transfers before an encoder tensor is changed or loaded."""
    state = _extract_encoder_state_from_checkpoint(ckpt)
    key = next((k for k in state if k.endswith("down_cntx.conv1.weight")), None)
    if key is None:
        raise ValueError("Pretraining checkpoint lacks down_cntx.conv1.weight; cannot verify input channels")
    source_channels = int(state[key].shape[1])
    target_channels = int(target_cfg["mos_model_params"]["in_channels"])
    source_cfg = ckpt.get("cfg", ckpt.get("config", {})) if isinstance(ckpt, dict) else {}
    source_names = ckpt.get("channel_names") if isinstance(ckpt, dict) else None
    if source_names is None and isinstance(source_cfg, dict):
        source_names = source_cfg.get("channel_names")
        if source_names is None and source_channels == 5:
            # Only the repository's temporal MAE dataset has exactly five XYZR(I) inputs.
            source_model = source_cfg.get("model_params", {})
            source_pretrain = source_cfg.get("pretrain_params", {})
            if (int(source_model.get("grid_channels", -1)) == 5
                    and source_model.get("name") == "salsanext_temporal_mae"
                    and source_pretrain.get("task") == "tmae_rangeview"):
                source_names = ["x", "y", "z", "range", "intensity"]
    target_mode = target_cfg["mos_data_params"]["input_mode"]
    target_names = ["x", "y", "z", "range", "intensity"] if "intensity" in target_mode else None
    if source_channels != target_channels and not adapt:
        raise ValueError(f"Pretraining encoder has {source_channels} channels, MOS expects {target_channels}. "
                         "Set pretrained_encoder_adapt_input_channels: true explicitly to adapt it.")
    if target_names and source_channels == 5 and list(source_names or []) != target_names:
        raise ValueError("Five-channel pretraining order is unverified or differs from "
                         "[x,y,z,range,intensity]; refusing transfer.")
    if target_names and source_channels == 4 and adapt:
        if target_mode != "range_xyz_intensity":
            raise ValueError("Four-to-five pretraining adaptation requires range_xyz_intensity without residuals")
        print("[PRETRAIN MOS] Explicit four-to-five transfer: first four filters retained; "
              "intensity filter initialized by pretrained_encoder_init_new_channels")


def maybe_load_pretrained_backbone(model: torch.nn.Module, cfg: Dict, path_override: str | None = None):
    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    ckpt_path = path_override if path_override is not None else mtrain.get("pretrained_backbone_path", None)
    if not ckpt_path:
        return False

    ckpt_path = os.path.abspath(str(ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained backbone checkpoint not found: {ckpt_path}")
    if not hasattr(model, "load_pretrained_backbone"):
        raise RuntimeError(
            "pretrained_backbone_path is set, but the selected MOS model has no "
            "load_pretrained_backbone(...) method."
        )

    load_encoder = bool(mtrain.get("pretrained_load_encoder", True))
    load_decoder = bool(mtrain.get("pretrained_load_decoder", False))
    skip_decoder_head = bool(mtrain.get("pretrained_skip_decoder_head", True))
    strict_encoder = bool(mtrain.get("pretrained_encoder_strict", True))
    adapt_input_channels = bool(mtrain.get("pretrained_encoder_adapt_input_channels", False))
    init_new_channels = str(mtrain.get("pretrained_encoder_init_new_channels", "zero"))

    print(f"[PRETRAIN MOS] Loading pretrained backbone from: {ckpt_path}")
    print(
        f"[PRETRAIN MOS] load_encoder={load_encoder} "
        f"load_decoder={load_decoder} skip_decoder_head={skip_decoder_head}"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if load_encoder:
        _check_pretrain_channels(ckpt, cfg, adapt_input_channels)

    encoder_state = None
    decoder_state = None
    if load_encoder:
        encoder_state = _extract_encoder_state_from_checkpoint(ckpt)
    if load_decoder:
        decoder_state = _extract_decoder_state_from_checkpoint(ckpt)

    cross_attention_state = None
    if hasattr(model, "cross_attention"):
        backbone = ckpt.get("backbone_state_dict", {}) if isinstance(ckpt, dict) else {}
        cross_attention_state = ckpt.get("cross_attention_state_dict", backbone.get("cross_attention"))
        if cross_attention_state is None:
            raise KeyError("T-MAE checkpoint lacks cross_attention_state_dict")

    load_kwargs = {}
    if cross_attention_state is not None:
        load_kwargs["cross_attention_state_dict"] = cross_attention_state
    model.load_pretrained_backbone(
        encoder_state_dict=encoder_state,
        decoder_state_dict=decoder_state,
        strict_encoder=strict_encoder,
        strict_decoder=False,
        adapt_input_channels=adapt_input_channels,
        init_new_channels=init_new_channels,
        skip_decoder_head=skip_decoder_head,
        **load_kwargs,
    )
    return True


def maybe_load_pretrained_encoder(model: torch.nn.Module, cfg: Dict, path_override: str | None = None):
    mtrain = (cfg or {}).get("mos_train_params", {}) or {}
    ckpt_path = path_override if path_override is not None else mtrain.get("pretrained_encoder_path", None)
    if not ckpt_path:
        print("[PRETRAIN MOS] No pretrained encoder path configured. Training from random init.")
        return

    ckpt_path = os.path.abspath(str(ckpt_path))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {ckpt_path}")

    strict = bool(mtrain.get("pretrained_encoder_strict", True))
    adapt_input_channels = bool(mtrain.get("pretrained_encoder_adapt_input_channels", False))
    init_new_channels = str(mtrain.get("pretrained_encoder_init_new_channels", "mean"))

    print(f"[PRETRAIN MOS] Loading pretrained encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _check_pretrain_channels(ckpt, cfg, adapt_input_channels)
    encoder_state = _extract_encoder_state_from_checkpoint(ckpt)

    if hasattr(model, "load_pretrained_encoder"):
        model.load_pretrained_encoder(
            encoder_state_dict=encoder_state,
            strict=strict,
            adapt_input_channels=adapt_input_channels,
            init_new_channels=init_new_channels,
        )
        return

    if hasattr(model, "encoder") and isinstance(model.encoder, torch.nn.Module):
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=strict)
        print(
            f"[PRETRAIN MOS] Loaded into model.encoder via fallback. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        return

    raise RuntimeError(
        "Model has no encoder loading interface. Expected load_pretrained_encoder(...) "
        "or model.encoder to exist."
    )


def main():
    parser = argparse.ArgumentParser(description="Full MOS baseline training with train/val split.")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to MOS YAML config")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: cuda if available else cpu")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_root", default=None)
    parser.add_argument("--smoke_test", action="store_true", help="One epoch on a small subset; separate smoke logs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=None)
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
    )
    parser.add_argument("--residual_offsets", type=str, default=None, help="e.g. '1' or '1,2,3'")
    parser.add_argument("--moving_weight", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        default=None,
        help="Optional path to forecasting checkpoint (.pt) with encoder_state_dict for MOS encoder init.",
    )
    parser.add_argument(
        "--pretrained_backbone",
        type=str,
        default=None,
        help="Optional path to forecasting checkpoint (.pt) with encoder+decoder backbone weights.",
    )
    parser.add_argument(
        "--pretrained_load_decoder",
        action="store_true",
        help="Load decoder upsampling weights from --pretrained_backbone and skip the forecasting head by default.",
    )
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
    mdata.setdefault("previous_offset", 3)
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
    mmodel.setdefault("cross_attention_heads", 8)
    mmodel.setdefault("dropout", 0.1)
    mmodel.setdefault("norm", "batch")

    mtrain.setdefault("batch_size", 8)
    mtrain.setdefault("num_workers", 4)
    mtrain.setdefault("epochs", 20)
    mtrain.setdefault("learning_rate", 5e-4)
    mtrain.setdefault("encoder_learning_rate", None)
    mtrain.setdefault("freeze_encoder_epochs", 0)
    mtrain.setdefault("freeze_encoder_bn_eval", True)
    mtrain.setdefault("learning_rate_min", None)
    mtrain.setdefault("encoder_learning_rate_min", None)
    mtrain.setdefault("weight_decay", 1e-4)
    mtrain.setdefault("clip_grad_norm", 5.0)
    mtrain.setdefault("static_class_weight", 1.0)
    mtrain.setdefault("moving_class_weight", 10.0)
    mtrain["loss"] = loss_settings(mtrain.get("loss"))
    mtrain.setdefault("optimizer", "adamw")
    mtrain.setdefault("scheduler", "none")
    mtrain.setdefault("scheduler_monitor", "val_moving_iou")
    mtrain.setdefault("scheduler_mode", "max")
    mtrain.setdefault("scheduler_factor", 0.5)
    mtrain.setdefault("scheduler_patience", 5)
    mtrain.setdefault("scheduler_threshold", 0.0001)
    mtrain.setdefault("scheduler_cooldown", 0)
    mtrain.setdefault("scheduler_start_epoch", 1)
    mtrain.setdefault("seed", 42)
    mtrain.setdefault("deterministic", False)
    mtrain.setdefault("pretrained_encoder_path", None)
    mtrain.setdefault("pretrained_backbone_path", None)
    mtrain.setdefault("pretrained_load_encoder", True)
    mtrain.setdefault("pretrained_load_decoder", False)
    mtrain.setdefault("pretrained_skip_decoder_head", True)
    mtrain.setdefault("pretrained_encoder_strict", True)
    mtrain.setdefault("pretrained_encoder_adapt_input_channels", False)
    mtrain.setdefault("pretrained_encoder_init_new_channels", "mean")

    mlog.setdefault("log_root", "/home/devuser/workspace/LidarGaussianVideoView/mos_logs")
    mlog.setdefault("run_name", "mos_baseline")
    mlog.setdefault("use_tensorboard", True)
    mlog.setdefault("save_checkpoints", True)
    mlog.setdefault("save_csv", True)
    mlog.setdefault("save_config_copy", True)

    if args.run_name is not None:
        mlog["run_name"] = str(args.run_name)
    if args.epochs is not None:
        mtrain["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        mtrain["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        mtrain["learning_rate"] = float(args.lr)
    if args.encoder_lr is not None:
        mtrain["encoder_learning_rate"] = float(args.encoder_lr)
    if args.freeze_encoder_epochs is not None:
        mtrain["freeze_encoder_epochs"] = int(args.freeze_encoder_epochs)
    if args.input_mode is not None:
        mdata["input_mode"] = str(args.input_mode)
    offsets = args.residual_offsets if args.residual_offsets is not None else mdata.get("residual_offsets", [1])
    mdata["residual_offsets"] = parse_residual_offsets(
        offsets, allow_empty="residual" not in str(mdata["input_mode"])
    )
    if args.moving_weight is not None:
        mtrain["moving_class_weight"] = float(args.moving_weight)
    if args.num_workers is not None:
        mtrain["num_workers"] = int(args.num_workers)
    if args.pretrained_encoder is not None:
        mtrain["pretrained_encoder_path"] = str(args.pretrained_encoder)
    if args.pretrained_backbone is not None:
        mtrain["pretrained_backbone_path"] = str(args.pretrained_backbone)
    if args.pretrained_load_decoder:
        mtrain["pretrained_load_decoder"] = True

    if mtrain.get("pretrained_backbone_path") and mtrain.get("pretrained_encoder_path"):
        raise ValueError("Set either pretrained_backbone_path or pretrained_encoder_path, not both.")

    if str(mmodel["name"]).lower() == "salsanext_temporal_mos":
        if mdata["input_mode"] not in {"range_xyz", "range_xyz_intensity"}:
            raise ValueError("Temporal MOS requires range_xyz or range_xyz_intensity")
        if mtrain.get("pretrained_backbone_path") and not bool(mtrain.get("pretrained_load_encoder", True)):
            raise ValueError("Temporal MOS checkpoint transfer needs pretrained_load_encoder=true")

    in_channels = compute_in_channels(str(mdata["input_mode"]), mdata["residual_offsets"])
    normalization(cfg, str(mdata["input_mode"]))
    configured_channels = mmodel.get("in_channels")
    if configured_channels is not None and int(configured_channels) != in_channels:
        raise ValueError(f"mos_model_params.in_channels={configured_channels} conflicts with "
                         f"input_mode={mdata['input_mode']} ({in_channels} channels)")
    mmodel["in_channels"] = int(in_channels)
    mmodel["num_classes"] = 2

    dataset_path = cfg.get("dataset_path", None)
    if not dataset_path:
        raise ValueError("Config missing required key: dataset_path")

    seed = int(mtrain["seed"])
    deterministic = bool(mtrain["deterministic"])
    set_seed(seed, deterministic)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    if int(mtrain.get("gradient_accumulation", 1)) != 1:
        raise ValueError("Baseline trainer requires gradient_accumulation=1")
    device = setup_device(device)
    if distributed() and (int(mtrain.get("freeze_encoder_epochs", 0)) > 0
                          or bool(mdata.get("use_moving_frame_sampler", False))):
        raise ValueError("DDP baseline does not support encoder freezing or moving-frame sampling")
    if args.log_root:
        mlog["log_root"] = args.log_root
    if args.smoke_test:
        mtrain["epochs"] = 1
        mlog["run_name"] = str(mlog["run_name"]) + "_SMOKE"
    cfg.setdefault("run_metadata", {})["smoke_test"] = args.smoke_test
    pin_memory = device.startswith("cuda")

    all_sequences = make_sequences(dataset_path)
    train_sequences = select_sequences(all_sequences, mdata["train_sequences"], "train")
    val_sequences = select_sequences(all_sequences, mdata["val_sequences"], "val")

    if args.smoke_test:
        world = dist.get_world_size() if distributed() else 1
        limit = int(mtrain["batch_size"]) * world
        train_sequences = _smoke_sequences(train_sequences, 2 * limit)
        val_sequences = _smoke_sequences(val_sequences, limit + 1)

    dataset_class = TemporalMOSDataset if str(mmodel["name"]).lower() == "salsanext_temporal_mos" else MOSFrameDataset
    train_dataset = dataset_class(
        sequences=train_sequences,
        cfg=cfg,
        split="train",
        input_mode=mdata["input_mode"],
        residual_offsets=mdata["residual_offsets"],
        mos_label_folder=mdata["mos_label_folder"],
        require_moving=bool(mdata["require_moving_train"]),
        min_moving_pixels=int(mdata["min_moving_pixels"]),
    )
    val_dataset = dataset_class(
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
    print_dataset_channel_debug("train", train_dataset)
    print_dataset_channel_debug("val", val_dataset)

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

    if distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
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
        shard_validation(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    model = build_mos_model(cfg).to(device)
    if mtrain.get("pretrained_backbone_path"):
        maybe_load_pretrained_backbone(
            model=model,
            cfg=cfg,
            path_override=args.pretrained_backbone,
        )
    else:
        maybe_load_pretrained_encoder(
            model=model,
            cfg=cfg,
            path_override=args.pretrained_encoder,
        )
    n_params = count_trainable_params(model)

    criterion = make_criterion(cfg, device)
    scaler = make_grad_scaler(cfg, device)
    amp_enabled, amp_dtype = amp_settings(cfg, device)

    optimizer = build_optimizer_with_param_groups(model, cfg)
    epochs = int(mtrain["epochs"])
    scheduler, scheduler_name = build_scheduler(optimizer, cfg, epochs)

    log_root = os.path.abspath(str(mlog.get("log_root", "/home/devuser/workspace/LidarGaussianVideoView/mos_logs")))
    run_name = str(mlog.get("run_name", "mos_baseline"))
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if distributed():
        shared_timestamp = [timestamp if primary() else None]
        dist.broadcast_object_list(shared_timestamp, src=0)
        timestamp = shared_timestamp[0]
    log_dir = os.path.join(log_root, f"{run_name}_{timestamp}")
    if primary():
        os.makedirs(log_dir, exist_ok=True)

    save_checkpoints = primary() and bool(mlog.get("save_checkpoints", True))
    save_csv = primary() and bool(mlog.get("save_csv", True))
    save_config_copy = primary() and bool(mlog.get("save_config_copy", True))

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_moving_iou.pt")
    last_ckpt_path = os.path.join(checkpoint_dir, "last.pt")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    metrics_csv_path = os.path.join(log_dir, "metrics.csv")
    config_copy_path = os.path.join(log_dir, "config.yaml")

    cfg.setdefault("run_metadata", {})
    cfg["run_metadata"]["cfg_path"] = cfg_path
    cfg["run_metadata"]["log_dir"] = log_dir
    cfg["run_metadata"]["timestamp"] = timestamp
    cfg["run_metadata"]["cli_args"] = vars(args)
    world = dist.get_world_size() if distributed() else 1
    cfg["run_metadata"].update(world_size=world, per_gpu_batch_size=batch_size,
                               global_batch_size=batch_size * world,
                               lovasz_reduction="mean_of_local_batches",
                               amp_enabled=amp_enabled, amp_dtype=str(amp_dtype))

    if save_config_copy:
        with open(config_copy_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    metrics_file = None
    metrics_writer = None
    if save_csv:
        metrics_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
        metrics_columns = [
            "epoch",
            "lr",
            "lr_encoder",
            "lr_decoder_head",
            "train_loss",
            "val_loss",
            "train_optimizer_steps", "train_amp_skipped_steps", "train_amp_scale",
            "train_peak_vram_bytes", "val_peak_vram_bytes",
            "train_loss_ce", "train_loss_lovasz", "train_loss_total",
            "val_loss_ce", "val_loss_lovasz", "val_loss_total",
            "train_moving_iou",
            "train_moving_precision",
            "train_moving_recall",
            "train_moving_f1",
            "train_static_iou",
            "train_mean_iou",
            "train_pixel_accuracy",
            "val_moving_iou",
            "val_moving_precision",
            "val_moving_recall",
            "val_moving_f1",
            "val_static_iou",
            "val_mean_iou",
            "val_pixel_accuracy",
            "train_tp_moving",
            "train_fp_moving",
            "train_fn_moving",
            "train_tn_static",
            "train_valid_pixels",
            "val_tp_moving",
            "val_fp_moving",
            "val_fn_moving",
            "val_tn_static",
            "val_valid_pixels",
            "is_best_moving_iou",
            "best_moving_iou_so_far",
            "best_epoch_so_far",
        ]
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metrics_columns)
        metrics_writer.writeheader()
        metrics_file.flush()

    writer = None
    if primary() and bool(mlog.get("use_tensorboard", True)):
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
    print(f"channel_count     : {int(train_dataset.channel_count)}")
    print(f"model             : {mmodel.get('name', 'unet_small')}")
    print(f"in_channels       : {int(mmodel['in_channels'])}")
    print(f"num_classes       : {int(mmodel['num_classes'])}")
    print(f"base_channels     : {int(mmodel.get('base_channels', 32))}")
    print(f"trainable_params  : {n_params}")
    print(f"batch_size/GPU    : {batch_size}")
    print(f"world_size        : {world}; global_batch_size: {batch_size * world}")
    print(f"loss_settings     : {mtrain['loss']}")
    print(f"AMP               : {amp_enabled}; dtype={amp_dtype}; scale={scaler.get_scale():g}")
    print(f"epochs            : {int(mtrain['epochs'])}")
    print(f"lr                : {float(mtrain['learning_rate'])}")
    print(f"encoder_lr        : {mtrain.get('encoder_learning_rate', None)}")
    print(f"freeze_encoder_ep : {int(mtrain.get('freeze_encoder_epochs', 0))}")
    print(f"moving_weight     : {float(mtrain['moving_class_weight'])}")
    print(f"device            : {device}")
    print(f"log_dir           : {log_dir}")
    print(f"pretrained_encoder: {mtrain.get('pretrained_encoder_path', None)}")
    print(f"pretrained_backbone: {mtrain.get('pretrained_backbone_path', None)}")
    print_trainable_summary(model)

    epochs = int(mtrain["epochs"])
    freeze_encoder_epochs = int(mtrain.get("freeze_encoder_epochs", 0) or 0)
    freeze_encoder_bn_eval = bool(mtrain.get("freeze_encoder_bn_eval", True))
    if freeze_encoder_epochs > 0 and (
        not hasattr(model, "encoder") or not isinstance(model.encoder, torch.nn.Module)
    ):
        raise RuntimeError(
            "mos_train_params.freeze_encoder_epochs is greater than 0, but the MOS model "
            "has no self.encoder module to freeze."
        )

    best_moving_iou = -1.0
    best_epoch = -1
    if distributed():
        model = DDP(model, device_ids=[torch.device(device).index] if pin_memory else None)
    prev_encoder_frozen = None
    for epoch in range(1, epochs + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        encoder_frozen = is_encoder_frozen_for_epoch(epoch, freeze_encoder_epochs)
        set_encoder_trainable(
            unwrap_model(model),
            trainable=not encoder_frozen,
            bn_eval=freeze_encoder_bn_eval,
        )
        if freeze_encoder_epochs > 0 and (prev_encoder_frozen is None or encoder_frozen != prev_encoder_frozen):
            state = "frozen" if encoder_frozen else "unfrozen"
            print(f"[FREEZE] Epoch {epoch}/{epochs}: encoder {state}")
            print_trainable_summary(model)
        prev_encoder_frozen = encoder_frozen

        train_loss, train_counts, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            encoder_frozen=encoder_frozen,
            freeze_encoder_bn_eval=freeze_encoder_bn_eval,
            epoch_index=epoch,
            total_epochs=epochs,
        )
        val_loss, val_counts, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            epoch_index=epoch,
            total_epochs=epochs,
        )

        step_scheduler_if_needed(
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            cfg=cfg,
            epoch=epoch,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )

        group_lrs = get_optimizer_group_lrs(optimizer)
        lr_cur, lr_encoder, lr_decoder_head = lr_value_for_logging(group_lrs, optimizer)
        current_moving_iou = float(val_metrics["moving_iou"])
        is_best_moving_iou = current_moving_iou > best_moving_iou
        if is_best_moving_iou:
            best_moving_iou = current_moving_iou
            best_epoch = int(epoch)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_mIoU={val_metrics['mean_iou']:.4f} | "
            f"val_moving_iou={val_metrics['moving_iou']:.4f} | "
            f"val_moving_f1={val_metrics['moving_f1']:.4f} | "
            f"val_moving_recall={val_metrics['moving_recall']:.4f} | "
            f"val_moving_precision={val_metrics['moving_precision']:.4f} | "
            f"decoder_lr={format_lr_for_print(lr_decoder_head)} | "
            f"encoder_lr={format_lr_for_print(lr_encoder)}"
        )

        print(f"[MEMORY/AMP] peak_train={train_metrics['peak_vram_bytes'] / 2**30:.2f} GiB "
              f"peak_val={val_metrics['peak_vram_bytes'] / 2**30:.2f} GiB "
              f"optimizer_steps={train_metrics['optimizer_steps']} "
              f"skipped_steps={train_metrics['amp_skipped_steps']} scale={train_metrics['amp_scale']:g}")
        if writer is not None:
            for key in ("optimizer_steps", "amp_skipped_steps", "amp_scale", "peak_vram_bytes"):
                writer.add_scalar(f"Train/{key}", train_metrics[key], epoch)
            writer.add_scalar("Val/peak_vram_bytes", val_metrics["peak_vram_bytes"], epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
                for component in ("loss_ce", "loss_lovasz", "loss_total"):
                    writer.add_scalar(f"Loss/{phase}_{component}", metrics[component], epoch)

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
            writer.add_scalar("LR/main", lr_cur, epoch)
            if lr_encoder is not None:
                writer.add_scalar("LR/encoder", float(lr_encoder), epoch)
            if lr_decoder_head is not None:
                writer.add_scalar("LR/decoder_head", float(lr_decoder_head), epoch)
            for group_name, group_lr in group_lrs.items():
                writer.add_scalar(f"LearningRate/{group_name}", float(group_lr), epoch)

        if metrics_writer is not None:
            row = {
                "epoch": int(epoch),
                "lr": float(lr_cur),
                "lr_encoder": lr_encoder,
                "lr_decoder_head": lr_decoder_head,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_moving_iou": float(train_metrics["moving_iou"]),
                "train_moving_precision": float(train_metrics["moving_precision"]),
                "train_moving_recall": float(train_metrics["moving_recall"]),
                "train_moving_f1": float(train_metrics["moving_f1"]),
                "train_static_iou": float(train_metrics["static_iou"]),
                "train_mean_iou": float(train_metrics["mean_iou"]),
                "train_pixel_accuracy": float(train_metrics["pixel_accuracy"]),
                "val_moving_iou": float(val_metrics["moving_iou"]),
                "val_moving_precision": float(val_metrics["moving_precision"]),
                "val_moving_recall": float(val_metrics["moving_recall"]),
                "val_moving_f1": float(val_metrics["moving_f1"]),
                "val_static_iou": float(val_metrics["static_iou"]),
                "val_mean_iou": float(val_metrics["mean_iou"]),
                "val_pixel_accuracy": float(val_metrics["pixel_accuracy"]),
                "train_tp_moving": int(train_counts["tp_moving"]),
                "train_fp_moving": int(train_counts["fp_moving"]),
                "train_fn_moving": int(train_counts["fn_moving"]),
                "train_tn_static": int(train_counts["tn_static"]),
                "train_valid_pixels": int(train_counts["valid_pixels"]),
                "val_tp_moving": int(val_counts["tp_moving"]),
                "val_fp_moving": int(val_counts["fp_moving"]),
                "val_fn_moving": int(val_counts["fn_moving"]),
                "val_tn_static": int(val_counts["tn_static"]),
                "val_valid_pixels": int(val_counts["valid_pixels"]),
                "is_best_moving_iou": int(is_best_moving_iou),
                "best_moving_iou_so_far": float(best_moving_iou),
                "best_epoch_so_far": int(best_epoch),
            }
            for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
                for component in ("loss_ce", "loss_lovasz", "loss_total"):
                    row[f"{phase}_{component}"] = float(metrics[component])
            for key in ("optimizer_steps", "amp_skipped_steps", "amp_scale", "peak_vram_bytes"):
                row[f"train_{key}"] = train_metrics[key]
            row["val_peak_vram_bytes"] = val_metrics["peak_vram_bytes"]
            metrics_writer.writerow(row)
            metrics_file.flush()

        if save_checkpoints:
            checkpoint_payload = {
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_moving_iou": float(best_moving_iou),
                "model_state_dict": unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "cfg": cfg,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "train_counts": train_counts,
                "val_counts": val_counts,
                "model_name": mmodel.get("name"),
                "input_mode": mdata.get("input_mode"),
                "input_normalization": normalization(cfg, mdata["input_mode"]),
                "residual_offsets": mdata.get("residual_offsets"),
            }
            if is_best_moving_iou:
                torch.save(checkpoint_payload, best_ckpt_path)
                print(
                    f"[CKPT] New best moving_iou={best_moving_iou:.6f} "
                    f"at epoch {best_epoch} -> {best_ckpt_path}"
                )

            if (epoch % 15 == 0) or (epoch == epochs):
                torch.save(checkpoint_payload, last_ckpt_path)

    if writer is not None:
        writer.flush()
        writer.close()

    if metrics_file is not None:
        metrics_file.flush()
        metrics_file.close()

    was_primary = primary()
    if distributed():
        dist.barrier()
        dist.destroy_process_group()
    if not was_primary:
        return

    print("Training finished.")
    print(f"Best moving_iou: {best_moving_iou:.6f} at epoch {best_epoch}")
    print(f"Last checkpoint: {last_ckpt_path if save_checkpoints else 'disabled by config'}")
    print(f"Best checkpoint: {best_ckpt_path if save_checkpoints else 'disabled by config'}")
    print(f"Metrics CSV: {metrics_csv_path if save_csv else 'disabled by config'}")
    print(f"Config copy: {config_copy_path if save_config_copy else 'disabled by config'}")
    print(f"TensorBoard log_dir: {log_dir}")


if __name__ == "__main__":
    main()
