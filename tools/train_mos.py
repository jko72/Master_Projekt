#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import datetime as dt
import math
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
    if input_mode == "range_xyz":
        return 4
    if input_mode == "range_xyz_residual":
        return 4 + len(residual_offsets)
    raise ValueError(
        "Invalid input_mode='{0}'. Supported: "
        "['range', 'residual', 'range_residual', 'range_xyz', 'range_xyz_residual']".format(input_mode)
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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    cfg: Dict,
    encoder_frozen: bool = False,
    freeze_encoder_bn_eval: bool = True,
):
    model.train()
    if encoder_frozen and freeze_encoder_bn_eval:
        if not hasattr(model, "encoder") or not isinstance(model.encoder, torch.nn.Module):
            raise RuntimeError("Cannot keep frozen encoder in eval mode because model has no encoder module.")
        model.encoder.eval()

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
    adapt_input_channels = bool(mtrain.get("pretrained_encoder_adapt_input_channels", True))
    init_new_channels = str(mtrain.get("pretrained_encoder_init_new_channels", "zero"))

    print(f"[PRETRAIN MOS] Loading pretrained backbone from: {ckpt_path}")
    print(
        f"[PRETRAIN MOS] load_encoder={load_encoder} "
        f"load_decoder={load_decoder} skip_decoder_head={skip_decoder_head}"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoder_state = None
    decoder_state = None
    if load_encoder:
        encoder_state = _extract_encoder_state_from_checkpoint(ckpt)
    if load_decoder:
        decoder_state = _extract_decoder_state_from_checkpoint(ckpt)

    model.load_pretrained_backbone(
        encoder_state_dict=encoder_state,
        decoder_state_dict=decoder_state,
        strict_encoder=strict_encoder,
        strict_decoder=False,
        adapt_input_channels=adapt_input_channels,
        init_new_channels=init_new_channels,
        skip_decoder_head=skip_decoder_head,
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
    adapt_input_channels = bool(mtrain.get("pretrained_encoder_adapt_input_channels", True))
    init_new_channels = str(mtrain.get("pretrained_encoder_init_new_channels", "mean"))

    print(f"[PRETRAIN MOS] Loading pretrained encoder from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--encoder_lr", type=float, default=None)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=None)
    parser.add_argument(
        "--input_mode",
        type=str,
        default=None,
        choices=["range", "residual", "range_residual", "range_xyz", "range_xyz_residual"],
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
    mtrain.setdefault("encoder_learning_rate", None)
    mtrain.setdefault("freeze_encoder_epochs", 0)
    mtrain.setdefault("freeze_encoder_bn_eval", True)
    mtrain.setdefault("learning_rate_min", None)
    mtrain.setdefault("encoder_learning_rate_min", None)
    mtrain.setdefault("weight_decay", 1e-4)
    mtrain.setdefault("clip_grad_norm", 5.0)
    mtrain.setdefault("static_class_weight", 1.0)
    mtrain.setdefault("moving_class_weight", 10.0)
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
    mtrain.setdefault("pretrained_encoder_adapt_input_channels", True)
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
    if args.residual_offsets is not None:
        mdata["residual_offsets"] = parse_residual_offsets(args.residual_offsets)
    else:
        mdata["residual_offsets"] = parse_residual_offsets(mdata.get("residual_offsets", [1]))
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

    class_weights = torch.tensor(
        [float(mtrain["static_class_weight"]), float(mtrain["moving_class_weight"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=int(mdata["ignore_index"]),
        weight=class_weights,
    )

    optimizer = build_optimizer_with_param_groups(model, cfg)
    epochs = int(mtrain["epochs"])
    scheduler, scheduler_name = build_scheduler(optimizer, cfg, epochs)

    log_root = os.path.abspath(str(mlog.get("log_root", "/home/devuser/workspace/LidarGaussianVideoView/mos_logs")))
    run_name = str(mlog.get("run_name", "mos_baseline"))
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    save_checkpoints = bool(mlog.get("save_checkpoints", True))
    save_csv = bool(mlog.get("save_csv", True))
    save_config_copy = bool(mlog.get("save_config_copy", True))

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
    print(f"channel_count     : {int(train_dataset.channel_count)}")
    print(f"model             : {mmodel.get('name', 'unet_small')}")
    print(f"in_channels       : {int(mmodel['in_channels'])}")
    print(f"num_classes       : {int(mmodel['num_classes'])}")
    print(f"base_channels     : {int(mmodel.get('base_channels', 32))}")
    print(f"trainable_params  : {n_params}")
    print(f"batch_size        : {batch_size}")
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
    prev_encoder_frozen = None
    for epoch in range(1, epochs + 1):
        encoder_frozen = is_encoder_frozen_for_epoch(epoch, freeze_encoder_epochs)
        set_encoder_trainable(
            model,
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
            device=device,
            cfg=cfg,
            encoder_frozen=encoder_frozen,
            freeze_encoder_bn_eval=freeze_encoder_bn_eval,
        )
        val_loss, val_counts, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
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
            metrics_writer.writerow(row)
            metrics_file.flush()

        if save_checkpoints:
            checkpoint_payload = {
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_moving_iou": float(best_moving_iou),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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

    print("Training finished.")
    print(f"Best moving_iou: {best_moving_iou:.6f} at epoch {best_epoch}")
    print(f"Last checkpoint: {last_ckpt_path if save_checkpoints else 'disabled by config'}")
    print(f"Best checkpoint: {best_ckpt_path if save_checkpoints else 'disabled by config'}")
    print(f"Metrics CSV: {metrics_csv_path if save_csv else 'disabled by config'}")
    print(f"Config copy: {config_copy_path if save_config_copy else 'disabled by config'}")
    print(f"TensorBoard log_dir: {log_dir}")


if __name__ == "__main__":
    main()
