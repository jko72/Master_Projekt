#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from helper.dataloader_helper import make_sequences
from helper.mos_label_helper import (
    compute_mos_label_stats,
    load_semantickitti_labels,
    map_semantickitti_to_mos,
    mos_label_path_for_frame,
    project_mos_labels_to_range,
    split_semantic_instance,
)
from helper.residual_helper import project_range_for_residual


def _ensure_mos_defaults(cfg: Dict) -> Dict:
    mp = cfg.setdefault("mos_label_params", {})
    mp.setdefault("enabled", True)
    mp.setdefault("mode", "precompute")
    mp.setdefault("folder_name", "mos_labels")
    mp.setdefault("ignore_index", -1)
    mp.setdefault("static_index", 0)
    mp.setdefault("moving_index", 1)
    mp.setdefault("overwrite", False)
    mp.setdefault("min_range", 0.0)
    mp.setdefault("max_range", 80.0)
    mp.setdefault("label_mapping", "semantickitti_mos")
    return cfg


def _print_seq_header(seq: Dict, cfg: Dict):
    mosp = cfg["mos_label_params"]
    mp = cfg["model_params"]
    seq_dir = os.path.abspath(os.path.join(os.path.dirname(seq["paths"][0][0]), ".."))
    target_dir = os.path.join(seq_dir, mosp["folder_name"])
    print(f"\n=== Sequence {seq['seq_id']} ===")
    print(f"frames           : {len(seq['paths'])}")
    print(f"target_folder    : {target_dir}")
    print(f"grid(H,W)        : ({mp['grid_height']}, {mp['grid_width']})")
    print(f"FOV_UP/FOV_DOWN  : ({mp.get('FOV_UP', 3.0)}, {mp.get('FOV_DOWN', -25.0)})")
    print(f"min/max_range    : ({mosp['min_range']}, {mosp['max_range']})")
    print(f"mapping          : {mosp.get('label_mapping', 'semantickitti_mos')}")


def _debug_plot(
    cfg: Dict,
    mos_img: np.ndarray,
    points_xyzi: np.ndarray,
    seq_id: str,
    frame_stem: str,
    residual: np.ndarray | None = None,
):
    mosp = cfg["mos_label_params"]
    ignore_index = int(mosp["ignore_index"])
    static_index = int(mosp["static_index"])
    moving_index = int(mosp["moving_index"])

    range_img = project_range_for_residual(points_xyzi, cfg)

    # Discrete MOS colormap: ignore -> gray, static -> blue, moving -> yellow/red.
    colors = ["#4a4a4a", "#103e8a", "#f8d548"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [ignore_index - 0.5, static_index + 0.5, moving_index + 0.5, moving_index + 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    n_rows = 3 if residual is not None else 2
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, 7 if residual is not None else 5), constrained_layout=True)
    if n_rows == 2:
        ax_range, ax_mos = axs
    else:
        ax_range, ax_res, ax_mos = axs

    im0 = ax_range.imshow(range_img, cmap="turbo")
    ax_range.set_title("Range")
    ax_range.set_xticks([])
    ax_range.set_yticks([])
    fig.colorbar(im0, ax=ax_range, shrink=0.8, label="Range")

    if residual is not None:
        vmax = 1.0 if cfg.get("residual_params", {}).get("normalize", True) else float(np.nanmax(residual))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im1 = ax_res.imshow(residual, cmap="turbo", vmin=0.0, vmax=vmax)
        ax_res.set_title("Residual")
        ax_res.set_xticks([])
        ax_res.set_yticks([])
        fig.colorbar(im1, ax=ax_res, shrink=0.8, label="Residual")

    im2 = ax_mos.imshow(mos_img, cmap=cmap, norm=norm, interpolation="nearest")
    ax_mos.set_title("MOS GT")
    ax_mos.set_xticks([])
    ax_mos.set_yticks([])
    cbar = fig.colorbar(im2, ax=ax_mos, shrink=0.8, ticks=[ignore_index, static_index, moving_index])
    cbar.ax.set_yticklabels(["ignore", "static", "moving"])

    fig.suptitle(f"MOS Debug | seq={seq_id} | frame={frame_stem}")
    plt.show()
    plt.close(fig)


def _load_residual_if_available(seq_dir: str, frame_stem: str, cfg: Dict):
    rp = cfg.get("residual_params", {})
    offsets = rp.get("offsets", [1])
    if not offsets:
        return None
    off = int(offsets[0])
    folder_template = rp.get("folder_template", "residual_images_{offset}")
    folder = folder_template.format(offset=off)
    path = os.path.join(seq_dir, folder, f"{frame_stem}.npy")
    if not os.path.isfile(path):
        return None
    arr = np.load(path)
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Offline MOS ground-truth label generation on range-view grid.")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--seq_id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--debug_frame", type=int, default=None)
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _ensure_mos_defaults(cfg)

    mosp = cfg["mos_label_params"]
    if not mosp.get("enabled", True):
        print("MOS label generation disabled in config (mos_label_params.enabled=false). Nothing to do.")
        return
    if mosp.get("mode", "precompute") != "precompute":
        print("mos_label_params.mode is not 'precompute'. This tool only supports offline precompute mode.")
        return
    if mosp.get("label_mapping", "semantickitti_mos") != "semantickitti_mos":
        raise ValueError("Only mos_label_params.label_mapping='semantickitti_mos' is supported right now.")

    ignore_index = int(mosp["ignore_index"])
    static_index = int(mosp["static_index"])
    moving_index = int(mosp["moving_index"])

    seqs = make_sequences(cfg["dataset_path"])
    if args.seq_id is not None:
        seqs = [s for s in seqs if s["seq_id"] == args.seq_id]
        if not seqs:
            raise ValueError(f"No sequence with seq_id={args.seq_id} in dataset_path={cfg['dataset_path']}")

    total_written = 0
    total_skipped = 0
    total_missing_labels = 0
    all_stats = []

    for seq in seqs:
        _print_seq_header(seq, cfg)
        seq_paths = seq["paths"]
        seq_dir = os.path.abspath(os.path.join(os.path.dirname(seq_paths[0][0]), ".."))
        out_dir = os.path.join(seq_dir, mosp["folder_name"])
        os.makedirs(out_dir, exist_ok=True)

        n_frames = len(seq_paths)
        if args.max_frames is not None:
            n_frames = min(n_frames, int(args.max_frames))

        seq_written = 0
        seq_skipped = 0
        seq_missing = 0

        for idx in range(n_frames):
            scan_path, label_path = seq_paths[idx]
            frame_stem = os.path.splitext(os.path.basename(scan_path))[0]
            out_path = mos_label_path_for_frame(seq_dir, frame_stem, cfg)
            do_overwrite = bool(args.overwrite or mosp.get("overwrite", False))

            if (not do_overwrite) and os.path.isfile(out_path):
                seq_skipped += 1
                total_skipped += 1
                continue

            if not os.path.isfile(label_path):
                print(f"[WARN] Missing label file, skipping frame: {label_path}")
                seq_missing += 1
                total_missing_labels += 1
                continue

            points_xyzi = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            raw_labels = load_semantickitti_labels(label_path)
            semantic_labels, _ = split_semantic_instance(raw_labels)

            if points_xyzi.shape[0] != semantic_labels.shape[0]:
                print(
                    f"[WARN] Point/label length mismatch in seq={seq['seq_id']} frame={frame_stem}: "
                    f"points={points_xyzi.shape[0]} labels={semantic_labels.shape[0]}. Truncating to min length."
                )
                n = min(points_xyzi.shape[0], semantic_labels.shape[0])
                points_xyzi = points_xyzi[:n]
                semantic_labels = semantic_labels[:n]

            mos_point_labels = map_semantickitti_to_mos(
                semantic_labels,
                ignore_index=ignore_index,
                static_index=static_index,
                moving_index=moving_index,
            )
            mos_img = project_mos_labels_to_range(points_xyzi, mos_point_labels, cfg).astype(np.int16)
            np.save(out_path, mos_img)

            seq_written += 1
            total_written += 1

            stats = compute_mos_label_stats(mos_img, ignore_index, static_index, moving_index)
            all_stats.append(stats)

            if args.debug_frame is not None and idx == args.debug_frame:
                print(f"[DEBUG] seq={seq['seq_id']} frame={frame_stem}")
                print(
                    "mos_label_stats:\n"
                    f"  ignore_ratio={stats['ignore_ratio']:.4f}\n"
                    f"  static_ratio={stats['static_ratio']:.4f}\n"
                    f"  moving_ratio={stats['moving_ratio']:.4f}\n"
                    f"  unique_values={stats['unique_values']}"
                )
                residual = _load_residual_if_available(seq_dir, frame_stem, cfg)
                _debug_plot(cfg, mos_img, points_xyzi, seq["seq_id"], frame_stem, residual=residual)

        print(f"generated_files       : {seq_written}")
        print(f"skipped_existing      : {seq_skipped}")
        print(f"missing_label_files   : {seq_missing}")

        if all_stats:
            last = all_stats[-1]
            print(
                "mos_label_stats:\n"
                f"  ignore_ratio={last['ignore_ratio']:.4f}\n"
                f"  static_ratio={last['static_ratio']:.4f}\n"
                f"  moving_ratio={last['moving_ratio']:.4f}\n"
                f"  moving_pixels={last['moving_count']}\n"
                f"  static_pixels={last['static_count']}\n"
                f"  ignore_pixels={last['ignore_count']}\n"
                f"  unique_values={last['unique_values']}"
            )

    print("\n=== MOS Label Generation Summary ===")
    print(f"sequences_done         : {len(seqs)}")
    print(f"written_files          : {total_written}")
    print(f"skipped_existing_files : {total_skipped}")
    print(f"missing_label_files    : {total_missing_labels}")
    if all_stats:
        ignore_mean = float(np.mean([s["ignore_ratio"] for s in all_stats]))
        static_mean = float(np.mean([s["static_ratio"] for s in all_stats]))
        moving_mean = float(np.mean([s["moving_ratio"] for s in all_stats]))
        print(f"ignore_ratio_mean      : {ignore_mean:.4f}")
        print(f"static_ratio_mean      : {static_mean:.4f}")
        print(f"moving_ratio_mean      : {moving_mean:.4f}")


if __name__ == "__main__":
    main()
