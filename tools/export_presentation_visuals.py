#!/usr/bin/env python3
"""Export reproducible, inference-only figures for the Phase-2 presentation.

The default paths are the three MOS runs documented in the final protocol and
the spatial MAE run used to initialise the best MOS encoder.  Every MOS input
is constructed by :class:`MOSFrameDataset`; this deliberately preserves the
training-time projection, channel order and residual loading implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helper.dataloader_helper import make_sequences
from mae_dataset import RangeXYZMAEDataset, select_sequences as select_mae_sequences
from models import build_model
from mos_dataset import MOSFrameDataset
from mos_models import build_mos_model


DEFAULT_BASELINE = ROOT / "LidarGaussianVideoView/mos_logs/MOS_semanticKitti_Baseline_range_2026/checkpoints/best_moving_iou.pt"
DEFAULT_RESIDUAL = ROOT / "LidarGaussianVideoView/mos_logs/MOS_semanticKitti_Range_residual_of5_2026_2026-07-13_18-50-03/checkpoints/best_moving_iou.pt"
DEFAULT_MAE_MOS = ROOT / "LidarGaussianVideoView/mos_logs/MOS_MAE_residual_12345_encoder_2026-08-12_19-22-56/checkpoints/best_moving_iou.pt"
DEFAULT_MAE = ROOT / "LidarGaussianVideoView/pretrain_logs/mae_rangexyz_2026_Baseline/checkpoints/best_val_loss.pt"
DEFAULT_MAE_CFG = ROOT / "LidarGaussianVideoView/pretrain_logs/mae_rangexyz_2026_Baseline/config_resolved.yaml"

STATIC = "#103e8a"
MOVING = "#f28e2b"
INVALID = "#808080"
MASK_CMAP = mcolors.ListedColormap([INVALID, STATIC, MOVING])
MASK_NORM = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], MASK_CMAP.N)
# Prediction panels retain the GT ignore mask: -1=grey, 0=static, 1=moving.
PRED_CMAP = MASK_CMAP
PRED_NORM = MASK_NORM
ERROR_CMAP = mcolors.ListedColormap([INVALID, "#f7f7f7", "#d73027", "#fee08b"])
ERROR_NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ERROR_CMAP.N)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "presentation_assets")
    p.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--residual-checkpoint", type=Path, default=DEFAULT_RESIDUAL)
    p.add_argument("--mae-mos-checkpoint", type=Path, default=DEFAULT_MAE_MOS)
    p.add_argument("--mae-checkpoint", type=Path, default=DEFAULT_MAE)
    p.add_argument("--mae-config", type=Path, default=DEFAULT_MAE_CFG)
    p.add_argument("--seq-id", default="07")
    p.add_argument("--num-overview-frames", type=int, default=8)
    p.add_argument("--mask-seed", type=int, default=42)
    p.add_argument("--device", default=None, help="Default: CUDA if available, otherwise CPU.")
    return p.parse_args()


def ensure_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} fehlt: {path}")
    return path


def load_checkpoint(path: Path):
    # Explicitly retain configuration metadata stored in the project checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def normalise_seq(value) -> str:
    value = str(value)
    return value.zfill(2) if value.isdigit() else value


def iou_moving(pred: np.ndarray, target: np.ndarray) -> float:
    valid = target != -1
    tp = np.count_nonzero((pred == 1) & (target == 1) & valid)
    fp = np.count_nonzero((pred == 1) & (target == 0) & valid)
    fn = np.count_nonzero((pred == 0) & (target == 1) & valid)
    return float(tp / max(tp + fp + fn, 1))


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """0 ignore, 1 correct valid, 2 false moving, 3 missed moving."""
    result = np.zeros_like(target, dtype=np.int8)
    valid = target != -1
    result[valid & (pred == target)] = 1
    result[valid & (pred == 1) & (target == 0)] = 2
    result[valid & (pred == 0) & (target == 1)] = 3
    return result


def load_mos_run(checkpoint_path: Path, seq_id: str, device: str):
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("cfg")
    if not isinstance(cfg, dict):
        raise KeyError(f"Checkpoint ohne gespeicherte cfg: {checkpoint_path}")
    data_cfg = cfg.setdefault("mos_data_params", {})
    cfg.setdefault("mos_model_params", {})
    input_mode = str(data_cfg["input_mode"])
    offsets = [int(v) for v in data_cfg.get("residual_offsets", [])]
    all_sequences = make_sequences(cfg["dataset_path"])
    sequences = [s for s in all_sequences if normalise_seq(s.get("seq_id")) == normalise_seq(seq_id)]
    if not sequences:
        raise ValueError(f"Sequenz {seq_id} nicht unter {cfg['dataset_path']} gefunden.")
    dataset = MOSFrameDataset(
        sequences, cfg, split="test", input_mode=input_mode, residual_offsets=offsets,
        mos_label_folder=str(data_cfg.get("mos_label_folder", "mos_labels")),
        require_moving=False, min_moving_pixels=int(data_cfg.get("min_moving_pixels", 1)),
    )
    model = build_mos_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    by_stem = {sample["frame_stem"]: index for index, sample in enumerate(dataset.samples)}
    return {"checkpoint": checkpoint_path, "ckpt": ckpt, "cfg": cfg, "dataset": dataset,
            "model": model, "by_stem": by_stem, "input_mode": input_mode, "offsets": offsets}


@torch.no_grad()
def predict_selected(run: dict, frame_stems: list[str], device: str) -> dict[str, np.ndarray]:
    output = {}
    for stem in frame_stems:
        idx = run["by_stem"].get(stem)
        if idx is None:
            raise KeyError(f"Frame {stem} fehlt im Dataset für {run['checkpoint']}")
        x, _, _ = run["dataset"][idx]
        logits = run["model"](x.unsqueeze(0).to(device))
        output[stem] = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)
    return output


def choose_overview_samples(dataset: MOSFrameDataset, n: int) -> list[dict]:
    """Highest moving-pixel frame per equally sized temporal sequence interval."""
    candidates = [s for s in dataset.samples if s["moving_pixels_cached"] > 0]
    if len(candidates) < n:
        raise RuntimeError(f"Nur {len(candidates)} Frames mit bewegten Punkten vorhanden, erwartet mindestens {n}.")
    selected = []
    edges = np.linspace(0, len(dataset.samples), n + 1, dtype=int)
    for start, end in zip(edges[:-1], edges[1:]):
        interval = [s for s in candidates if start <= s["frame_index"] < end]
        if not interval:
            # This keeps the stated temporal coverage even in an empty interval.
            centre = (start + end) / 2.0
            interval = candidates
            selected.append(min(interval, key=lambda s: abs(s["frame_index"] - centre)))
        else:
            selected.append(max(interval, key=lambda s: s["moving_pixels_cached"]))
    return selected


def largest_moving_crop(target: np.ndarray, min_h: int = 14, min_w: int = 72, pad: int = 5):
    """Bounding box around largest 8-connected moving component, with safe padding."""
    mask = target == 1
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack, coords = [(int(y), int(x))], []
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            coords.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append(coords)
    if not components:
        return slice(0, h), slice(0, w)
    comp = max(components, key=len)
    ys, xs = zip(*comp)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad + 1)
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad + 1)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, y1 = max(0, cy - min_h // 2), min(h, cy - min_h // 2 + min_h)
    x0, x1 = max(0, cx - min_w // 2), min(w, cx - min_w // 2 + min_w)
    # Shift boxes touching a boundary so the requested minimum size is retained.
    y0, y1 = max(0, min(y0, h - min_h)), min(h, max(y1, min_h))
    x0, x1 = max(0, min(x0, w - min_w)), min(w, max(x1, min_w))
    return slice(y0, y1), slice(x0, x1)


def style_axis(ax, title: str):
    ax.set_title(title, fontsize=17, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])


def draw(ax, arr, kind: str, title: str, range_vmax: float = 80.0, residual_vmax: float = 1.0):
    if kind == "range":
        im = ax.imshow(arr, cmap="turbo", vmin=0, vmax=range_vmax, interpolation="nearest", aspect="equal")
    elif kind == "residual":
        im = ax.imshow(arr, cmap="magma", vmin=0, vmax=residual_vmax, interpolation="nearest", aspect="equal")
    elif kind == "gt":
        im = ax.imshow(arr, cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest", aspect="equal")
    elif kind == "pred":
        im = ax.imshow(arr, cmap=PRED_CMAP, norm=PRED_NORM, interpolation="nearest", aspect="equal")
    elif kind == "error":
        im = ax.imshow(arr, cmap=ERROR_CMAP, norm=ERROR_NORM, interpolation="nearest", aspect="equal")
    elif kind == "mae_error":
        cmap = plt.get_cmap("magma").copy(); cmap.set_bad(INVALID)
        im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=residual_vmax, interpolation="nearest", aspect="equal")
    else:
        raise ValueError(kind)
    style_axis(ax, title)
    return im


def save_single(out: Path, arr, kind, title, range_vmax=80.0, residual_vmax=1.0):
    fig, ax = plt.subplots(figsize=(12, 2.4), constrained_layout=True)
    im = draw(ax, arr, kind, title, range_vmax, residual_vmax)
    if kind in {"range", "residual", "mae_error"}:
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_intro(out: Path, range_img, target, stem: str, range_vmax: float):
    fig, axs = plt.subplots(2, 1, figsize=(12, 4.4), constrained_layout=True)
    draw(axs[0], range_img, "range", "Range-Image", range_vmax)
    draw(axs[1], target, "gt", "Ground Truth: statisch / bewegt", range_vmax)
    fig.suptitle(f"Testsequenz 07 – Frame {stem}", fontsize=20)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_residual_figure(out: Path, arrays: list, titles: list[str], crop, range_vmax, residual_vmax, stem, cropped=False):
    fig, axs = plt.subplots(4, 1, figsize=(12, 8.0 if cropped else 5.7), constrained_layout=True)
    kinds = ["range", "residual", "residual", "gt"]
    for ax, arr, kind, title in zip(axs, arrays, kinds, titles):
        sub = arr[crop] if cropped else arr
        draw(ax, sub, kind, title, range_vmax, residual_vmax)
    suffix = " – Objekt-Ausschnitt" if cropped else ""
    fig.suptitle(f"Testsequenz 07 – Frame {stem}{suffix}", fontsize=20)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_mae_figure(out: Path, original, masked, reconstruction, error, valid, masked_pixels, crop, stem, range_vmax, cropped=False):
    panels = [(original, "range", "Originales Range-Image"), (masked, "range", "Tatsächlich maskierte Eingabe"),
              (reconstruction, "range", "Rohe Modellrekonstruktion")]
    if error is not None:
        panels.append((error, "mae_error", "Absoluter Fehler (nur gültig & maskiert)"))
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 8.2 if len(panels) == 4 else 6.3), constrained_layout=True)
    for ax, (arr, kind, title) in zip(np.atleast_1d(axs), panels):
        sub = arr[crop] if cropped else arr
        draw(ax, sub, kind, title, range_vmax, max(1e-3, np.nanpercentile(error[valid & masked_pixels], 99)) if error is not None and np.any(valid & masked_pixels) else 1.0)
    suffix = " – Objekt-Ausschnitt" if cropped else ""
    fig.suptitle(f"Räumliches MAE, Testsequenz 07 – Frame {stem}{suffix}", fontsize=20)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_contact_sheet(out: Path, records, predictions, range_vmax):
    labels = [("Range", "range"), ("GT", "gt"), ("Range+XYZ\nScratch", "pred"),
              ("+ Residuen 1–5\nScratch", "pred"), ("+ MAE-Encoder\nResiduen 1–5", "pred")]
    fig, axs = plt.subplots(len(records), 5, figsize=(15, max(8.5, 1.35 * len(records))), constrained_layout=True)
    for row, rec in enumerate(records):
        valid = rec["target"] != -1
        values = [rec["range"], rec["target"],
                  np.where(valid, predictions["baseline"][rec["stem"]], -1),
                  np.where(valid, predictions["residual"][rec["stem"]], -1),
                  np.where(valid, predictions["mae"][rec["stem"]], -1)]
        for col, ((label, kind), value) in enumerate(zip(labels, values)):
            title = label if row == 0 else ""
            draw(axs[row, col], value, kind, title, range_vmax)
        axs[row, 0].set_ylabel(f"{rec['stem']}\n({rec['moving']} bewegt)", fontsize=13)
    fig.suptitle("Kontaktübersicht: zeitlich gleichmäßig verteilte Frames mit bewegten Punkten", fontsize=20)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_detail(out: Path, rec, predictions, range_vmax, cropped=False):
    crop = rec["crop"]
    valid = rec["target"] != -1
    panels = [(rec["range"], "range", "Range-Image"), (rec["target"], "gt", "Ground Truth"),
              (np.where(valid, predictions["baseline"][rec["stem"]], -1), "pred", "Range+XYZ Scratch"),
              (np.where(valid, predictions["residual"][rec["stem"]], -1), "pred", "Scratch + Residuen 1–5"),
              (np.where(valid, predictions["mae"][rec["stem"]], -1), "pred", "MAE-Encoder + Residuen 1–5")]
    fig, axs = plt.subplots(len(panels), 1, figsize=(12, 9.8 if cropped else 7.2), constrained_layout=True)
    for ax, (arr, kind, title) in zip(axs, panels):
        draw(ax, arr[crop] if cropped else arr, kind, title, range_vmax)
    suffix = " – identischer Objekt-Ausschnitt" if cropped else ""
    fig.suptitle(f"MOS-Vergleich, Frame {rec['stem']}{suffix}", fontsize=20)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_error_detail(out: Path, rec, predictions, cropped=False):
    crop = rec["crop"]
    names = ["Range+XYZ Scratch", "Scratch + Residuen 1–5", "MAE-Encoder + Residuen 1–5"]
    keys = ["baseline", "residual", "mae"]
    fig, axs = plt.subplots(3, 1, figsize=(12, 6.2 if cropped else 4.8), constrained_layout=True)
    for ax, name, key in zip(axs, names, keys):
        arr = error_map(predictions[key][rec["stem"]], rec["target"])
        draw(ax, arr[crop] if cropped else arr, "error", f"Fehlerkarte: {name}")
    suffix = " – identischer Objekt-Ausschnitt" if cropped else ""
    fig.suptitle("Weiß: korrekt, Rot: falsch bewegt, Gelb: bewegte Punkte übersehen, Grau: ungültig" + suffix, fontsize=16)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def load_residual(rec, offset: int) -> np.ndarray:
    path = Path(rec["seq_dir"]) / f"residual_images_{offset}" / f"{rec['stem']}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Residualbild fehlt: {path}")
    return np.load(path).astype(np.float32)


def run_mae(args, focus_stem: str, crop, device: str, range_vmax: float):
    cfg_path = ensure_file(args.mae_config, "MAE-Konfiguration")
    checkpoint_path = ensure_file(args.mae_checkpoint, "MAE-Checkpoint")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    all_sequences = make_sequences(cfg["dataset_path"])
    sequences = select_mae_sequences(all_sequences, [args.seq_id])
    dataset = RangeXYZMAEDataset(sequences, cfg, split="test", seed=args.mask_seed)
    dataset.set_epoch(int(cfg.get("test_params", {}).get("mask_epoch", 0)))
    index = next((i for i, sample in enumerate(dataset.samples) if sample["frame_stem"] == focus_stem), None)
    if index is None:
        raise KeyError(f"Frame {focus_stem} nicht im MAE-Dataset.")
    batch = dataset[index]
    model = build_model(cfg["model_params"]["name"], cfg).to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        pred = model(batch["masked_xyzd"].unsqueeze(0).to(device), batch["mask"].unsqueeze(0).to(device))[0].cpu().numpy()
    target = batch["target_xyzd"].numpy()
    masked = batch["masked_xyzd"].numpy()
    mask = batch["mask"].numpy()[0] > 0.5
    valid = batch["valid_mask"].numpy()[0] > 0.5
    error = np.abs(pred[3] - target[3]).astype(np.float32)
    error[~(valid & mask)] = np.nan
    return {"cfg": cfg, "checkpoint": checkpoint_path, "original": target[3], "masked": masked[3],
            "reconstruction": pred[3], "error": error, "valid": valid, "mask": mask, "crop": crop,
            "range_vmax": range_vmax}


def main():
    args = parse_args()
    device = str(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA angefordert, aber nicht verfügbar.")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for path, label in [(args.baseline_checkpoint, "Baseline-Checkpoint"), (args.residual_checkpoint, "Residual-Checkpoint"),
                        (args.mae_mos_checkpoint, "MAE-MOS-Checkpoint")]:
        ensure_file(path, label)

    baseline = load_mos_run(args.baseline_checkpoint.resolve(), args.seq_id, device)
    residual = load_mos_run(args.residual_checkpoint.resolve(), args.seq_id, device)
    mae_run = load_mos_run(args.mae_mos_checkpoint.resolve(), args.seq_id, device)
    overview_samples = choose_overview_samples(baseline["dataset"], args.num_overview_frames)
    stems = [s["frame_stem"] for s in overview_samples]
    predictions = {"baseline": predict_selected(baseline, stems, device),
                   "residual": predict_selected(residual, stems, device),
                   "mae": predict_selected(mae_run, stems, device)}

    records = []
    for sample in overview_samples:
        idx = baseline["by_stem"][sample["frame_stem"]]
        x, target, meta = baseline["dataset"][idx]
        range_img = x[3].numpy()  # documented [x,y,z,range] convention for range_xyz.
        record = {"stem": sample["frame_stem"], "frame_index": sample["frame_index"], "moving": sample["moving_pixels_cached"],
                  "range": range_img, "target": target.numpy().astype(np.int8), "seq_dir": Path(meta["scan_path"]).parent.parent}
        record["crop"] = largest_moving_crop(record["target"])
        records.append(record)
    range_vmax = float(baseline["cfg"]["model_params"].get("max_range", 80.0))
    residual_vmax = 1.0

    # Candidate selection is explicitly frame-level, not a sequence metric.
    for rec in records:
        rec["baseline_iou"] = iou_moving(predictions["baseline"][rec["stem"]], rec["target"])
        rec["residual_iou"] = iou_moving(predictions["residual"][rec["stem"]], rec["target"])
        rec["mae_iou"] = iou_moving(predictions["mae"][rec["stem"]], rec["target"])
    improved = max(records, key=lambda r: r["mae_iou"] - r["baseline_iou"])
    remaining = min(records, key=lambda r: r["mae_iou"])
    focus = improved

    save_intro(out / "01_range_ground_truth.png", focus["range"], focus["target"], focus["stem"], range_vmax)
    save_single(out / "01_range.png", focus["range"], "range", "Range-Image", range_vmax)
    save_single(out / "01_ground_truth.png", focus["target"], "gt", "Ground Truth: statisch / bewegt", range_vmax)
    save_intro(out / "01_range_ground_truth_ausschnitt.png", focus["range"][focus["crop"]], focus["target"][focus["crop"]], focus["stem"], range_vmax)

    residual1, residual5 = load_residual(focus, 1), load_residual(focus, 5)
    residual_arrays = [focus["range"], residual1, residual5, focus["target"]]
    residual_titles = ["Aktuelles Range-Image", "Residual Offset 1 (0,1 s)", "Residual Offset 5 (0,5 s)", "MOS Ground Truth"]
    save_residual_figure(out / "02_residual_offsets.png", residual_arrays, residual_titles, focus["crop"], range_vmax, residual_vmax, focus["stem"])
    save_residual_figure(out / "02_residual_offsets_ausschnitt.png", residual_arrays, residual_titles, focus["crop"], range_vmax, residual_vmax, focus["stem"], cropped=True)
    for name, arr, kind, title in zip(["range", "residual_offset1", "residual_offset5", "ground_truth"], residual_arrays,
                                      ["range", "residual", "residual", "gt"], residual_titles):
        save_single(out / f"02_{name}.png", arr, kind, title, range_vmax, residual_vmax)
        save_single(out / f"02_{name}_ausschnitt.png", arr[focus["crop"]], kind, title + " – Ausschnitt", range_vmax, residual_vmax)

    mae = run_mae(args, focus["stem"], focus["crop"], device, range_vmax)
    save_mae_figure(out / "03_mae_maskierung_rekonstruktion.png", mae["original"], mae["masked"], mae["reconstruction"], mae["error"], mae["valid"], mae["mask"], mae["crop"], focus["stem"], range_vmax)
    save_mae_figure(out / "03_mae_maskierung_rekonstruktion_ausschnitt.png", mae["original"], mae["masked"], mae["reconstruction"], mae["error"], mae["valid"], mae["mask"], mae["crop"], focus["stem"], range_vmax, cropped=True)
    for name, arr, kind, title in [("original", mae["original"], "range", "Originales Range-Image"),
                                   ("maskiert", mae["masked"], "range", "Tatsächlich maskierte Eingabe"),
                                   ("rohe_rekonstruktion", mae["reconstruction"], "range", "Rohe Modellrekonstruktion"),
                                   ("absoluter_fehler", mae["error"], "mae_error", "Absoluter Fehler (nur gültig & maskiert)")]:
        save_single(out / f"03_mae_{name}.png", arr, kind, title, range_vmax, 10.0)

    save_contact_sheet(out / "04_mos_kontaktuebersicht.png", records, predictions, range_vmax)
    for prefix, rec, description in [("05_verbesserung", improved, "Beispiel mit größter MAE-gegen-Baseline-Verbesserung innerhalb der Kontaktübersicht"),
                                     ("06_restfehler", remaining, "Beispiel mit niedrigster MAE-Frame-IoU innerhalb der Kontaktübersicht")]:
        save_detail(out / f"{prefix}_vergleich.png", rec, predictions, range_vmax)
        save_detail(out / f"{prefix}_vergleich_ausschnitt.png", rec, predictions, range_vmax, cropped=True)
        save_error_detail(out / f"{prefix}_fehlerkarten.png", rec, predictions)
        save_error_detail(out / f"{prefix}_fehlerkarten_ausschnitt.png", rec, predictions, cropped=True)
        rec["description"] = description

    with (out / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["frame_stem", "frame_index", "moving_pixels", "baseline_frame_moving_iou", "residual_frame_moving_iou", "mae_frame_moving_iou"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for rec in records:
            writer.writerow({"frame_stem": rec["stem"], "frame_index": rec["frame_index"], "moving_pixels": rec["moving"],
                             "baseline_frame_moving_iou": rec["baseline_iou"], "residual_frame_moving_iou": rec["residual_iou"], "mae_frame_moving_iou": rec["mae_iou"]})
    manifest = {"sequence": args.seq_id, "focus_frame": focus["stem"], "improvement_frame": improved["stem"], "remaining_error_frame": remaining["stem"],
                "device": device, "range_color_limits_m": [0.0, range_vmax], "residual_color_limits": [0.0, residual_vmax],
                "mae_mask_seed": args.mask_seed, "mae_mask_epoch": int(mae["cfg"].get("test_params", {}).get("mask_epoch", 0)),
                "mae_patch": mae["cfg"]["pretrain_params"]["mask"],
                "checkpoints": {"baseline": str(args.baseline_checkpoint.resolve()), "residual": str(args.residual_checkpoint.resolve()),
                                "mae_mos": str(args.mae_mos_checkpoint.resolve()), "spatial_mae": str(args.mae_checkpoint.resolve())}}
    with (out / "export_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    readme = f"""# Präsentationsgrafiken – Projektphase 2

Erzeugt am lokalen Projektstand durch reine Inferenz. Exportbefehl:

```sh
python tools/export_presentation_visuals.py --device cuda
```

## Daten und Auswahl

- Testsequenz: `{args.seq_id}`.
- Kontaktübersicht: acht gleich große Zeitintervalle der Sequenz; pro Intervall der Frame mit den meisten Ground-Truth-Punkten der Klasse *bewegt*. Die konkreten Frames und **Frame-Metriken** stehen in `frame_metrics.csv`; diese Werte sind keine Sequenzmetriken.
- Detail *Verbesserung*: größte Differenz MAE-Frame-IoU minus Baseline-Frame-IoU unter den acht Kontakt-Frames (`{improved['stem']}`). Detail *Restfehler*: niedrigste MAE-Frame-IoU unter denselben Frames (`{remaining['stem']}`).
- Der Objekt-Ausschnitt umschließt die größte 8-zusammenhängende bewegte GT-Komponente, mit derselben gepolsterten Pixelbox in allen zugehörigen Ansichten.

## Modelle und Konfigurationen

- Range+XYZ Scratch, Best-Validation-Checkpoint: `{args.baseline_checkpoint.resolve()}`; gespeicherte Test-Moving-IoU: **17,13 %**.
- Range+XYZ + Residual-Offsets 1–5 Scratch, Best-Validation-Checkpoint: `{args.residual_checkpoint.resolve()}`; gespeicherte Test-Moving-IoU: **35,69 %**.
- MAE-Encoder + Range+XYZ + Residual-Offsets 1–5, Best-Validation-Checkpoint: `{args.mae_mos_checkpoint.resolve()}`; gespeicherte Test-Moving-IoU: **56,76 %**.
- Räumliches MAE: `{args.mae_checkpoint.resolve()}` mit `{args.mae_config.resolve()}`. Kanalreihenfolge: `[x, y, z, range]`; Patchgröße 4×16, Maskierungsgrad 0,5, `mask_only_valid=true`, Seed `{args.mask_seed}`, Mask-Epoche 0.

MOS-Eingaben werden direkt mit `MOSFrameDataset` aufgebaut: Range+XYZ ist `[x,y,z,range]`; bei Residualvarianten werden die Kanäle `residual_1` bis `residual_5` angehängt. Dadurch stimmen Projektion, Ego-Motion-kompensierte, vorab berechnete Residualbilder und Datenaufbereitung mit dem jeweiligen Trainingslauf überein.

## Gestaltung und Dateien

- `01_*`: Einstieg mit Range und GT, einschließlich Objekt-Ausschnitt.
- `02_*`: Range, Offset 1 (0,1 s), Offset 5 (0,5 s) und GT; beide Residuen nutzen die gemeinsame Skala 0–1.
- `03_*`: originales Bild, tatsächlich maskierte Eingabe, **rohe Modellrekonstruktion** und Fehler nur auf gültigen maskierten Pixeln. Es wurde keine Mischung aus Original- und Rekonstruktionswerten als Rekonstruktion exportiert.
- `04_*`: Kontaktübersicht.
- `05_*`, `06_*`: volle und identisch beschnittene Modellvergleiche sowie separate Fehlerkarten. Fehlerkarten: rot = falsch bewegt, gelb = bewegte Punkte übersehen, grau = ungültig; ungültige Pixel gehen nicht als korrekt ein.

Masken verwenden durchgehend statisch dunkelblau, bewegt orange und ungültig grau; alle diskreten Ansichten werden ohne Interpolation gezeichnet. Range-Skala: 0–{range_vmax:g} m. Zusammengesetzte Grafiken sind 2400 Pixel breit (12 Zoll bei 200 dpi).
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    archive = out / "presentation_assets.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(out.iterdir()):
            if file.is_file() and file.name != archive.name:
                zf.write(file, arcname=file.name)
    print(f"Export fertig: {out}")
    print(f"ZIP: {archive}")


if __name__ == "__main__":
    main()
