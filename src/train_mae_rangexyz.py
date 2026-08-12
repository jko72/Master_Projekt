#!/usr/bin/env python3
"""Independent MAE-RangeXYZ pretraining entry point."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from helper.dataloader_helper import make_sequences
from loss_mae_pretrain import mae_rangexyz_loss
from mae_dataset import RangeXYZMAEDataset, select_sequences
from models import build_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


METRIC_KEYS = (
    "loss_total",
    "loss_xyz",
    "loss_range",
    "loss_normals",
    "loss_residual",
    "residual_pos_ratio",
    "masked_valid_ratio",
    "valid_ratio",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised MAE pretraining on LiDAR RangeXYZ views.")
    parser.add_argument("--cfg_path", required=True, type=str)
    parser.add_argument("--device", default=None, type=str, help="cuda/cpu; auto-detected if omitted")
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    return parser.parse_args()


def resolve_cfg_path(path: str) -> str:
    candidates = [Path(path), Path.cwd() / path, Path(__file__).resolve().parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Config not found: {path}")


def apply_defaults_and_cli(cfg: dict, args) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model_params", {})
    cfg.setdefault("pretrain_params", {})
    cfg["pretrain_params"].setdefault("mask", {})
    cfg["pretrain_params"].setdefault("loss", {})
    cfg["pretrain_params"].setdefault("residual_inputs", {})
    cfg["pretrain_params"].setdefault("auxiliary_tasks", {})
    cfg.setdefault("data_params", {})
    cfg.setdefault("train_params", {})

    model_cfg = cfg["model_params"]
    model_cfg.setdefault("name", "salsanext_mae_rangexyz")
    model_cfg.setdefault("grid_channels", 4)
    model_cfg.setdefault("grid_height", 64)
    model_cfg.setdefault("grid_width", 512)
    model_cfg.setdefault("dropout_prob", 0.2)
    grid_channels = int(model_cfg["grid_channels"])

    mask_cfg = cfg["pretrain_params"]["mask"]
    mask_cfg.setdefault("type", "patch")
    mask_cfg.setdefault("patch_h", 4)
    mask_cfg.setdefault("patch_w", 16)
    mask_cfg.setdefault("mask_ratio", 0.5)
    mask_cfg.setdefault("mask_only_valid", True)

    loss_cfg = cfg["pretrain_params"]["loss"]
    loss_cfg.setdefault("name", "smooth_l1")
    loss_cfg.setdefault("xyz_weight", 0.5)
    loss_cfg.setdefault("range_weight", 1.0)
    loss_cfg.setdefault("loss_on_mask_only", True)
    loss_cfg.setdefault("min_range", 0.1)
    residual_cfg = cfg["pretrain_params"]["residual_inputs"]
    residual_cfg.setdefault("enabled", False)
    residual_cfg.setdefault("offsets", [1])
    residual_cfg.setdefault("folder_template", "residual_images_{offset}")
    residual_cfg.setdefault("allow_missing", False)
    residual_offsets = [int(v) for v in residual_cfg.get("offsets", [1])] if bool(residual_cfg.get("enabled", False)) else []
    if bool(residual_cfg.get("enabled", False)) and not residual_offsets:
        raise ValueError("pretrain_params.residual_inputs.offsets must contain at least one value")
    if any(offset <= 0 for offset in residual_offsets):
        raise ValueError(f"pretrain_params.residual_inputs.offsets must be positive, got {residual_offsets}")

    normals_cfg = cfg["pretrain_params"]["auxiliary_tasks"].setdefault("surface_normals", {})
    normals_cfg.setdefault("enabled", False)
    normals_cfg.setdefault("weight", 0.1)
    normals_cfg.setdefault("loss", "cosine")
    residual_aux_cfg = cfg["pretrain_params"]["auxiliary_tasks"].setdefault("residual_reconstruction", {})
    residual_aux_cfg.setdefault("enabled", bool(residual_offsets))
    residual_aux_cfg.setdefault("weight", 0.2)
    residual_aux_cfg.setdefault("loss", "smooth_l1")
    residual_aux_cfg.setdefault("loss_on_mask_only", True)
    residual_aux_cfg.setdefault("positive_threshold", 0.02)
    residual_aux_cfg.setdefault("positive_weight", 1.0)
    normals_enabled = bool(normals_cfg.get("enabled", False))
    expected_channels = (7 if normals_enabled else 4) + len(residual_offsets)
    if grid_channels != expected_channels:
        raise ValueError(
            "model_params.grid_channels must equal base MAE channels plus residual inputs: "
            f"got {grid_channels}, expected {expected_channels}."
        )

    data_cfg = cfg["data_params"]
    data_cfg.setdefault("split_type", "predefined")
    data_cfg.setdefault("train_sequences", [])
    data_cfg.setdefault("val_sequences", [])
    if str(data_cfg["split_type"]).lower() != "predefined":
        raise ValueError("train_mae_rangexyz.py currently requires data_params.split_type: predefined")

    train_cfg = cfg["train_params"]
    train_cfg.setdefault("batch_size", 16)
    train_cfg.setdefault("num_workers", 4)
    train_cfg.setdefault("num_total_epochs", 50)
    train_cfg.setdefault("learning_rate", 5e-4)
    train_cfg.setdefault("learning_rate_min", 5e-6)
    train_cfg.setdefault("num_warmup_epochs", 2)
    train_cfg.setdefault("weight_decay", 1e-4)
    train_cfg.setdefault("clip_grad_norm", 5.0)
    train_cfg.setdefault("optimizer", "adamw")
    train_cfg.setdefault("scheduler", "warmup_cosine")
    train_cfg.setdefault("seed", 42)
    train_cfg.setdefault("deterministic", False)
    train_cfg.setdefault("with_save", True)
    train_cfg.setdefault("logs_save_dir", str(Path.cwd() / "pretrain_logs"))
    train_cfg.setdefault("save_visualizations", True)
    train_cfg.setdefault("visualization_interval_epochs", 5)
    train_cfg.setdefault("use_tensorboard", True)
    train_cfg.setdefault("plot_examples", True)
    train_cfg.setdefault("plot_batch_step", 5)

    if args.epochs is not None:
        train_cfg["num_total_epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        train_cfg["learning_rate"] = float(args.lr)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    return cfg


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loaders(cfg: dict, device: str):
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("Config requires dataset_path")
    dataset_path = os.path.abspath(os.path.expanduser(str(dataset_path)))
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")

    all_sequences = make_sequences(dataset_path)
    data_cfg = cfg["data_params"]
    train_sequences = select_sequences(all_sequences, data_cfg["train_sequences"])
    val_sequences = select_sequences(all_sequences, data_cfg["val_sequences"])
    train_dataset = RangeXYZMAEDataset(train_sequences, cfg, split="train")
    val_dataset = RangeXYZMAEDataset(val_sequences, cfg, split="val")
    if not train_dataset or not val_dataset:
        raise RuntimeError(f"Empty dataset: train={len(train_dataset)}, val={len(val_dataset)}")

    train_cfg = cfg["train_params"]
    seed = int(train_cfg["seed"])
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg["num_workers"]),
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_dataset, val_dataset, train_loader, val_loader


def build_scheduler(optimizer, cfg: dict):
    train_cfg = cfg["train_params"]
    name = str(train_cfg.get("scheduler", "warmup_cosine")).lower()
    if name == "none":
        return None
    if name not in {"warmup_cosine", "cosine"}:
        raise ValueError("scheduler must be 'warmup_cosine', 'cosine', or 'none'")

    epochs = int(train_cfg["num_total_epochs"])
    warmup = int(train_cfg.get("num_warmup_epochs", 0)) if name == "warmup_cosine" else 0
    base_lr = float(train_cfg["learning_rate"])
    min_lr = float(train_cfg.get("learning_rate_min", 0.0))
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        cosine_epochs = max(epochs - warmup, 1)
        progress = min(max((epoch - warmup + 1) / cosine_epochs, 0.0), 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch(batch: dict, device: str):
    non_blocking = device.startswith("cuda")
    return (
        batch["target_xyzd"].to(device, non_blocking=non_blocking),
        batch["masked_xyzd"].to(device, non_blocking=non_blocking),
        batch["mask"].to(device, non_blocking=non_blocking),
        batch["valid_mask"].to(device, non_blocking=non_blocking),
    )


def run_epoch(
    model,
    loader,
    cfg,
    device,
    optimizer=None,
    debug_first_batch=False,
    live_visualizer=None,
    epoch=0,
):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in METRIC_KEYS}
    sample_count = 0
    last_visual = None
    clip_norm = float(cfg["train_params"].get("clip_grad_norm", 0.0))

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            target, masked, mask, valid = move_batch(batch, device)
            pred = model(masked, mask)
            loss, loss_dict = mae_rangexyz_loss(pred, target, mask, valid, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite MAE loss at batch {batch_index}: {loss.item()}")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            if debug_first_batch and batch_index == 0:
                masked_valid_ratio = ((mask > 0.5) & (valid > 0.5)).float().sum() / (valid > 0.5).float().sum().clamp_min(1)
                print(f"target_xyzd shape: {tuple(target.shape)}")
                print(f"masked_xyzd shape: {tuple(masked.shape)}")
                print(f"mask shape: {tuple(mask.shape)}")
                print(f"valid_mask shape: {tuple(valid.shape)}")
                print(f"pred shape: {tuple(pred.shape)}")
                metas = batch.get("meta", {})
                if isinstance(metas, dict) and "channel_names" in metas:
                    print(f"channels: {metas['channel_names']}")
                print(f"valid ratio: {(valid > 0.5).float().mean().item():.6f}")
                print(f"masked valid ratio: {masked_valid_ratio.item():.6f}")
                print(f"finite loss: {torch.isfinite(loss).item()} ({loss.item():.6f})")

            if live_visualizer is not None:
                plot_step = max(int(cfg["train_params"].get("plot_batch_step", 5)), 1)
                if batch_index % plot_step == 0:
                    live_visualizer.update(
                        target=target,
                        masked=masked,
                        pred=pred,
                        valid=valid,
                        epoch=epoch,
                        batch_index=batch_index,
                    )

            batch_size = int(target.shape[0])
            for key in METRIC_KEYS:
                totals[key] += float(loss_dict[key].item()) * batch_size
            sample_count += batch_size
            last_visual = (target[:1].detach().cpu(), masked[:1].detach().cpu(), pred[:1].detach().cpu(), valid[:1].detach().cpu())

    if sample_count == 0:
        raise RuntimeError("DataLoader produced zero samples")
    return {key: value / sample_count for key, value in totals.items()}, last_visual


class LiveRangeVisualizer:
    """Interactive training view modeled after the plotting loop in videoNet.py."""

    def __init__(self, height: int, width: int):
        self.enabled = False
        self.plt = None
        try:
            import matplotlib.pyplot as plt

            backend = str(plt.get_backend()).lower()
            non_interactive_backends = {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}
            if backend in non_interactive_backends:
                print(
                    f"[WARN] Matplotlib backend '{plt.get_backend()}' is non-interactive; "
                    "live visualization is disabled. Saved visualizations remain enabled."
                )
                return

            plt.ion()
            aspect = float(width) / float(height)
            height_per = 2.0
            width_per = height_per * aspect
            self.fig, axes = plt.subplots(
                1,
                3,
                figsize=(width_per * 3.0, height_per),
                constrained_layout=True,
            )
            self.axes = np.asarray(axes).reshape(3)
            self.images = []
            self.colorbars = []
            cmap = copy.copy(plt.get_cmap("turbo"))
            cmap.set_bad(color="black")
            for axis, title in zip(
                self.axes,
                ("Original Range", "Masked Range", "Reconstructed Range"),
            ):
                image = axis.imshow(
                    np.full((height, width), np.nan, dtype=np.float32),
                    aspect="equal",
                    vmin=0.0,
                    vmax=50.0,
                    cmap=cmap,
                    interpolation="nearest",
                )
                axis.set_title(title)
                axis.axis("off")
                colorbar = self.fig.colorbar(
                    image,
                    ax=axis,
                    fraction=0.046,
                    pad=0.04,
                    label="Distance (m)",
                )
                self.images.append(image)
                self.colorbars.append(colorbar)
            self.fig.canvas.manager.set_window_title("MAE-RangeXYZ Training")
            self.plt = plt
            self.enabled = True
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:
            print(f"[WARN] Live visualization could not be initialized: {exc}")
            self.enabled = False

    def update(self, target, masked, pred, valid, epoch: int, batch_index: int) -> None:
        if not self.enabled:
            return
        valid_np = valid[0, 0].detach().cpu().numpy() > 0.5
        panels = [
            target[0, 3].detach().cpu().numpy(),
            masked[0, 3].detach().cpu().numpy(),
            pred[0, 3].detach().cpu().numpy(),
        ]
        valid_values = panels[0][valid_np]
        vmax = float(np.percentile(valid_values, 99)) if valid_values.size else 50.0
        vmax = max(5.0, min(vmax, 120.0))

        for image_handle, colorbar, panel in zip(self.images, self.colorbars, panels):
            shown = panel.astype(np.float32).copy()
            shown[~valid_np] = np.nan
            image_handle.set_data(shown)
            image_handle.set_clim(0.0, vmax)
            colorbar.update_normal(image_handle)
        self.fig.suptitle(f"MAE-RangeXYZ — Epoch {epoch}, Batch {batch_index}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        if self.enabled and self.plt is not None:
            self.plt.ioff()
            self.plt.close(self.fig)
        self.enabled = False


def save_visualization(tensors, output_path: str) -> bool:
    if tensors is None:
        return False
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib unavailable; skipping visualizations.")
        return False

    target, masked, pred, valid = tensors
    valid_np = valid[0, 0].numpy() > 0.5
    original = target[0, 3].numpy()
    masked_range = masked[0, 3].numpy()
    reconstructed = pred[0, 3].numpy()
    error = np.abs(reconstructed - original)
    panels = [original, masked_range, reconstructed, error]
    titles = ["Original Range", "Masked Range", "Reconstructed Range", "Absolute Error"]
    finite_values = original[valid_np]
    vmax = float(np.percentile(finite_values, 99)) if finite_values.size else 1.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("black")
    for axis, image, title in zip(axes, panels, titles):
        shown = image.astype(np.float32).copy()
        shown[~valid_np] = np.nan
        local_vmax = vmax if title != "Absolute Error" else None
        axis.imshow(shown, cmap=cmap, vmin=0.0, vmax=local_vmax, aspect="auto")
        axis.set_title(title)
        axis.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def checkpoint_payload(model, cfg, epoch: int, val_loss: float) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.get_encoder_state_dict(),
        "decoder_state_dict": model.get_decoder_state_dict(),
        "backbone_state_dict": model.get_backbone_state_dict(),
        "epoch": int(epoch),
        "cfg": copy.deepcopy(cfg),
        "val_loss": float(val_loss),
    }


def main():
    args = parse_args()
    cfg_path = resolve_cfg_path(args.cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = apply_defaults_and_cli(yaml.safe_load(handle) or {}, args)

    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    seed = int(cfg["train_params"]["seed"])
    set_seed(seed, bool(cfg["train_params"]["deterministic"]))

    run_name = str(args.run_name or cfg["train_params"].get("run_name", "mae_rangexyz"))
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(os.path.abspath(os.path.expanduser(str(cfg["train_params"]["logs_save_dir"]))), f"{run_name}_{timestamp}")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    visual_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if bool(cfg["train_params"]["save_visualizations"]):
        os.makedirs(visual_dir, exist_ok=True)
    shutil.copy2(cfg_path, os.path.join(log_dir, "config_source.yaml"))
    with open(os.path.join(log_dir, "config_resolved.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(cfg, device)
    model = build_model(cfg["model_params"]["name"], cfg).to(device)
    if str(cfg["train_params"].get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("MAE pretraining currently supports optimizer: adamw")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train_params"]["learning_rate"]),
        weight_decay=float(cfg["train_params"]["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, cfg)

    writer = None
    if bool(cfg["train_params"].get("use_tensorboard", True)):
        if SummaryWriter is None:
            print("[WARN] TensorBoard is unavailable; console and CSV logging remain enabled.")
        else:
            writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    csv_path = os.path.join(log_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(
            ["epoch", "lr"] + [f"train/{k}" for k in METRIC_KEYS] + [f"val/{k}" for k in METRIC_KEYS]
        )

    epochs = int(cfg["train_params"]["num_total_epochs"])
    best_val = float("inf")
    last_val = float("inf")
    print(f"[MAE] device={device} train_frames={len(train_dataset)} val_frames={len(val_dataset)}")
    print(f"[MAE] log_dir={log_dir}")
    live_visualizer = None
    if bool(cfg["train_params"].get("plot_examples", True)):
        live_visualizer = LiveRangeVisualizer(
            height=int(cfg["model_params"]["grid_height"]),
            width=int(cfg["model_params"]["grid_width"]),
        )
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(0)
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            cfg,
            device,
            optimizer=optimizer,
            debug_first_batch=(epoch == 0),
            live_visualizer=live_visualizer,
            epoch=epoch + 1,
        )
        val_metrics, visual = run_epoch(model, val_loader, cfg, device, optimizer=None)
        last_val = float(val_metrics["loss_total"])
        lr = float(optimizer.param_groups[0]["lr"])

        if writer is not None:
            for key in METRIC_KEYS:
                writer.add_scalar(f"train/{key}", train_metrics[key], epoch + 1)
                writer.add_scalar(f"val/{key}", val_metrics[key], epoch + 1)
            writer.add_scalar("train/learning_rate", lr, epoch + 1)

        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch + 1, lr] + [train_metrics[k] for k in METRIC_KEYS] + [val_metrics[k] for k in METRIC_KEYS]
            )
        print(
            f"[Epoch {epoch + 1:03d}/{epochs:03d}] lr={lr:.3e} "
            f"train_total={train_metrics['loss_total']:.6f} "
            f"val_total={val_metrics['loss_total']:.6f} "
            f"val_xyz={val_metrics['loss_xyz']:.6f} "
            f"val_range={val_metrics['loss_range']:.6f} "
            f"val_normals={val_metrics['loss_normals']:.6f} "
            f"val_residual={val_metrics['loss_residual']:.6f}"
        )

        payload = checkpoint_payload(model, cfg, epoch + 1, val_metrics["loss_total"])
        if bool(cfg["train_params"].get("with_save", True)):
            torch.save(payload, os.path.join(checkpoint_dir, "last.pt"))
            if val_metrics["loss_total"] < best_val:
                best_val = val_metrics["loss_total"]
                torch.save(payload, os.path.join(checkpoint_dir, "best_val_loss.pt"))

        interval = max(int(cfg["train_params"].get("visualization_interval_epochs", 5)), 1)
        if bool(cfg["train_params"]["save_visualizations"]) and ((epoch + 1) % interval == 0 or epoch == 0):
            image_path = os.path.join(visual_dir, f"epoch_{epoch + 1:04d}.png")
            if save_visualization(visual, image_path) and writer is not None:
                try:
                    from PIL import Image
                    image = np.asarray(Image.open(image_path).convert("RGB"))
                    writer.add_image("val/range_reconstruction", image, epoch + 1, dataformats="HWC")
                except ImportError:
                    pass

        if scheduler is not None:
            scheduler.step()

    final_payload = checkpoint_payload(model, cfg, epochs, last_val)
    if bool(cfg["train_params"].get("with_save", True)):
        torch.save(final_payload, os.path.join(checkpoint_dir, "model_final.pt"))
    if writer is not None:
        writer.close()
    if live_visualizer is not None:
        live_visualizer.close()
    print(f"[MAE] Finished. best_val_loss={best_val:.6f}")
    print(f"[MAE] checkpoints={checkpoint_dir}")


if __name__ == "__main__":
    main()
