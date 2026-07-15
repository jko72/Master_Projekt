#!/usr/bin/env python3
"""Evaluate and visualize a trained MAE-RangeXYZ checkpoint."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from helper.dataloader_helper import make_sequences
from loss_mae_pretrain import mae_rangexyz_loss
from mae_dataset import RangeXYZMAEDataset, select_sequences
from models import build_model
from train_mae_rangexyz import seed_worker, set_seed


METRIC_KEYS = (
    "loss_total",
    "loss_xyz",
    "loss_range",
    "loss_normals",
    "masked_valid_ratio",
    "valid_ratio",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test MAE-RangeXYZ reconstruction on configured test sequences."
    )
    parser.add_argument("--cfg_path", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--test_sequences", default=None, type=str, help="Override, e.g. '08' or '0006,0009'")
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument(
        "--max_visualizations",
        default=None,
        type=int,
        help="Maximum PNGs to save/show; 0 disables visualizations, negative saves all.",
    )
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", dest="show", action="store_true", help="Show figures interactively.")
    show_group.add_argument("--no_show", dest="show", action="store_false", help="Only save figures.")
    parser.set_defaults(show=None)
    return parser.parse_args()


def resolve_file(path: str, description: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        candidate = Path.cwd() / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return str(candidate.resolve())


def parse_sequence_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]


def extract_model_state(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint model_state_dict is not a dictionary.")
    return state


class TestVisualizer:
    """Save complete test plots and optionally display them interactively."""

    def __init__(self, output_dir: str, show: bool, pause_seconds: float):
        self.output_dir = output_dir
        self.show = bool(show)
        self.pause_seconds = float(pause_seconds)
        self._headless_warned = False
        os.makedirs(self.output_dir, exist_ok=True)

    def render(
        self,
        target: torch.Tensor,
        masked: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
        title: str,
        filename: str,
    ) -> str:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for MAE test visualization.") from exc

        target_range = target[3].numpy()
        masked_range = masked[3].numpy()
        reconstructed_range = pred[3].numpy()
        mask_np = mask[0].numpy() > 0.5
        valid_np = valid[0].numpy() > 0.5
        completed_range = np.where(mask_np, reconstructed_range, target_range)
        absolute_error = np.abs(reconstructed_range - target_range)

        panels = [
            target_range,
            masked_range,
            reconstructed_range,
            completed_range,
            absolute_error,
        ]
        titles = [
            "Original Range",
            "Masked Range",
            "Raw Reconstructed Range",
            "Completed Range",
            "Masked Absolute Error",
        ]
        valid_values = target_range[valid_np]
        vmax = float(np.percentile(valid_values, 99)) if valid_values.size else 50.0
        vmax = max(5.0, min(vmax, 120.0))
        error_values = absolute_error[valid_np & mask_np]
        error_vmax = float(np.percentile(error_values, 99)) if error_values.size else 1.0
        error_vmax = max(error_vmax, 1e-3)

        fig, axes = plt.subplots(5, 1, figsize=(18, 12), constrained_layout=True)
        range_cmap = plt.get_cmap("turbo").copy()
        range_cmap.set_bad("black")
        error_cmap = plt.get_cmap("magma").copy()
        error_cmap.set_bad("black")

        for index, (axis, panel, panel_title) in enumerate(zip(axes, panels, titles)):
            shown = panel.astype(np.float32).copy()
            if index == 4:
                shown[~(valid_np & mask_np)] = np.nan
                image = axis.imshow(
                    shown,
                    aspect="equal",
                    interpolation="nearest",
                    cmap=error_cmap,
                    vmin=0.0,
                    vmax=error_vmax,
                )
                label = "Absolute error (m)"
            else:
                shown[~valid_np] = np.nan
                image = axis.imshow(
                    shown,
                    aspect="equal",
                    interpolation="nearest",
                    cmap=range_cmap,
                    vmin=0.0,
                    vmax=vmax,
                )
                label = "Distance (m)"
            axis.set_title(panel_title)
            axis.axis("off")
            fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01, label=label)

        fig.suptitle(title)
        output_path = os.path.join(self.output_dir, filename)
        fig.savefig(output_path, dpi=140)

        if self.show:
            backend = str(plt.get_backend()).lower()
            non_interactive = {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}
            if backend in non_interactive:
                if not self._headless_warned:
                    print(
                        f"[WARN] Matplotlib backend '{plt.get_backend()}' is non-interactive; "
                        "test figures are saved but cannot be shown."
                    )
                    self._headless_warned = True
            elif self.pause_seconds <= 0.0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(self.pause_seconds)
        plt.close(fig)
        return output_path


def main():
    args = parse_args()
    cfg_path = resolve_file(args.cfg_path, "Config")
    checkpoint_path = resolve_file(args.checkpoint, "Checkpoint")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    cfg.setdefault("test_params", {})
    test_cfg = cfg["test_params"]
    train_cfg = cfg.get("train_params", {}) or {}
    data_cfg = cfg.get("data_params", {}) or {}

    sequence_ids = parse_sequence_ids(args.test_sequences)
    if sequence_ids is None:
        sequence_ids = [str(value) for value in data_cfg.get("test_sequences", [])]
    if not sequence_ids:
        raise ValueError(
            "No test sequences configured. Set data_params.test_sequences in the YAML "
            "or pass --test_sequences, for example --test_sequences 08."
        )

    seed = int(args.seed if args.seed is not None else test_cfg.get("seed", train_cfg.get("seed", 42)))
    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    set_seed(seed, bool(train_cfg.get("deterministic", False)))

    dataset_path = os.path.abspath(os.path.expanduser(str(cfg.get("dataset_path", ""))))
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    all_sequences = make_sequences(dataset_path)
    test_sequences = select_sequences(all_sequences, sequence_ids)
    dataset = RangeXYZMAEDataset(test_sequences, cfg, split="test", seed=seed)
    dataset.set_epoch(int(test_cfg.get("mask_epoch", 0)))
    if len(dataset) == 0:
        raise RuntimeError("The configured MAE test dataset is empty.")

    batch_size = int(args.batch_size if args.batch_size is not None else test_cfg.get("batch_size", 1))
    num_workers = int(
        args.num_workers if args.num_workers is not None else test_cfg.get("num_workers", train_cfg.get("num_workers", 0))
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=False,
        worker_init_fn=seed_worker,
    )

    model = build_model(cfg["model_params"]["name"], cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(extract_model_state(checkpoint), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    max_samples = int(args.max_samples if args.max_samples is not None else test_cfg.get("max_samples", 20))
    if max_samples <= 0:
        max_samples = len(dataset)
    max_visualizations = int(
        args.max_visualizations
        if args.max_visualizations is not None
        else test_cfg.get("max_visualizations", 100)
    )
    default_output = Path(checkpoint_path).parent.parent / "test_results" / Path(checkpoint_path).stem
    output_dir = os.path.abspath(
        os.path.expanduser(str(args.output_dir or test_cfg.get("output_dir") or default_output))
    )
    show = bool(args.show if args.show is not None else test_cfg.get("show", True))
    visualizer = TestVisualizer(
        output_dir=output_dir,
        show=show,
        pause_seconds=float(test_cfg.get("show_pause_seconds", 0.5)),
    )

    csv_path = os.path.join(output_dir, "metrics.csv")
    os.makedirs(output_dir, exist_ok=True)
    totals = {key: 0.0 for key in METRIC_KEYS}
    sample_count = 0
    first_batch = True
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_handle:
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(["seq_id", "frame_stem", *METRIC_KEYS, "visualization"])

        with torch.no_grad():
            for batch in loader:
                remaining = max_samples - sample_count
                if remaining <= 0:
                    break
                target = batch["target_xyzd"][:remaining].to(device, non_blocking=device.startswith("cuda"))
                masked = batch["masked_xyzd"][:remaining].to(device, non_blocking=device.startswith("cuda"))
                mask = batch["mask"][:remaining].to(device, non_blocking=device.startswith("cuda"))
                valid = batch["valid_mask"][:remaining].to(device, non_blocking=device.startswith("cuda"))
                pred = model(masked, mask)

                if first_batch:
                    print(f"target_xyzd shape: {tuple(target.shape)}")
                    print(f"masked_xyzd shape: {tuple(masked.shape)}")
                    print(f"mask shape: {tuple(mask.shape)}")
                    print(f"valid_mask shape: {tuple(valid.shape)}")
                    print(f"pred shape: {tuple(pred.shape)}")
                    first_batch = False

                for local_index in range(int(target.shape[0])):
                    sample_loss, metrics = mae_rangexyz_loss(
                        pred[local_index : local_index + 1],
                        target[local_index : local_index + 1],
                        mask[local_index : local_index + 1],
                        valid[local_index : local_index + 1],
                        cfg,
                    )
                    if not torch.isfinite(sample_loss):
                        raise FloatingPointError(f"Non-finite test loss at sample {sample_count}.")

                    meta = batch["meta"]
                    seq_id = str(meta["seq_id"][local_index])
                    frame_stem = str(meta["frame_stem"][local_index])
                    filename = f"{sample_count:05d}_seq-{seq_id}_frame-{frame_stem}.png"
                    should_visualize = max_visualizations < 0 or sample_count < max_visualizations
                    visual_path = ""
                    if should_visualize:
                        visual_path = visualizer.render(
                            target[local_index].detach().cpu(),
                            masked[local_index].detach().cpu(),
                            pred[local_index].detach().cpu(),
                            mask[local_index].detach().cpu(),
                            valid[local_index].detach().cpu(),
                            title=f"MAE-RangeXYZ | Sequence {seq_id} | Frame {frame_stem}",
                            filename=filename,
                        )
                    values = {key: float(metrics[key].item()) for key in METRIC_KEYS}
                    for key in METRIC_KEYS:
                        totals[key] += values[key]
                    csv_writer.writerow(
                        [seq_id, frame_stem, *[values[key] for key in METRIC_KEYS], visual_path]
                    )
                    sample_count += 1
                    print(
                        f"[TEST {sample_count:04d}/{min(max_samples, len(dataset)):04d}] "
                        f"seq={seq_id} frame={frame_stem} "
                        f"loss={values['loss_total']:.6f} "
                        f"visualization={visual_path if visual_path else 'skipped'}"
                    )
                    if sample_count >= max_samples:
                        break

    if sample_count == 0:
        raise RuntimeError("No test samples were evaluated.")
    means = {key: value / sample_count for key, value in totals.items()}
    summary_path = os.path.join(output_dir, "summary.yaml")
    with open(summary_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "checkpoint": checkpoint_path,
                "test_sequences": sequence_ids,
                "num_samples": sample_count,
                "num_visualizations": min(
                    sample_count,
                    sample_count if max_visualizations < 0 else max_visualizations,
                ),
                **means,
            },
            handle,
            sort_keys=False,
        )
    print(f"[TEST] samples={sample_count} loss_total={means['loss_total']:.6f}")
    print(f"[TEST] output_dir={output_dir}")


if __name__ == "__main__":
    main()
