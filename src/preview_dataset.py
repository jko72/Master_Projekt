import argparse
import os
import random
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dataloader import RandomWindowSeqDataset
from helper.dataloader_helper import make_sequences
from helper.mos_label_helper import compute_mos_label_stats


def _filter_sequences_by_split(seqs: List[dict], cfg: dict, split: str) -> List[dict]:
    if split == "all":
        return seqs

    splits = cfg.get("data_params", {}).get("predefined_splits", None)
    if not splits:
        raise ValueError("Config has no data_params.predefined_splits. Use --split all.")

    if split not in splits:
        raise ValueError(f"Split '{split}' not found in config predefined_splits.")

    wanted = set(splits[split])
    filtered = [s for s in seqs if s["seq_id"] in wanted]

    if not filtered:
        raise ValueError(f"No sequences matched split '{split}'.")

    return filtered


def _pick_index(ds_len: int, sample_index: int | None, random_sample: bool, seed: int) -> int:
    if ds_len <= 0:
        raise ValueError("Dataset has no windows. Check split/filter and horizon config.")

    if sample_index is not None:
        if sample_index < 0 or sample_index >= ds_len:
            raise IndexError(f"sample_index {sample_index} outside [0, {ds_len - 1}].")
        return sample_index

    if random_sample:
        rng = random.Random(seed)
        return rng.randrange(ds_len)

    return 0


def _compute_vmax(cfg: dict, hist_ranges: np.ndarray, fut_ranges: np.ndarray) -> float:
    stats = cfg.get("data_params", {}).get("stats", {})
    configured_max = stats.get("max_range", None)
    if configured_max is not None:
        return float(configured_max)
    return float(np.percentile(np.concatenate([hist_ranges.ravel(), fut_ranges.ravel()]), 99))


def _ensure_mos_defaults(cfg: dict) -> dict:
    mp = cfg.setdefault("mos_label_params", {})
    mp.setdefault("enabled", True)
    mp.setdefault("mode", "precompute")
    mp.setdefault("folder_name", "mos_labels")
    mp.setdefault("ignore_index", -1)
    mp.setdefault("static_index", 0)
    mp.setdefault("moving_index", 1)
    return cfg


def _load_residuals_for_history(ds, idx: int, cfg: dict, residual_offset: int):
    hist_len = int(cfg["model_params"]["input_horizon"])
    H = int(cfg["model_params"]["grid_height"])
    W = int(cfg["model_params"]["grid_width"])
    rp = cfg.get("residual_params", {})
    folder_template = rp.get("folder_template", "residual_images_{offset}")
    folder = folder_template.format(offset=residual_offset)
    invalid_value = np.float32(rp.get("invalid_value", 0.0))

    s_id, start = ds.windows[idx]
    seq = ds.seqs[s_id]

    residuals = []
    residual_paths = []
    missing_paths = []

    for j in range(start, start + hist_len):
        frame_path = seq["paths"][j][0]
        seq_dir = os.path.abspath(os.path.join(os.path.dirname(frame_path), ".."))
        frame_stem = os.path.splitext(os.path.basename(frame_path))[0]
        res_path = os.path.join(seq_dir, folder, f"{frame_stem}.npy")
        residual_paths.append(res_path)

        if os.path.isfile(res_path):
            arr = np.load(res_path)
            if arr.shape != (H, W):
                print(f"[WARN] Residual shape mismatch at {res_path}: {arr.shape} != {(H, W)}. Using zeros.")
                arr = np.full((H, W), invalid_value, dtype=np.float32)
            else:
                arr = arr.astype(np.float32)
        else:
            print(f"[WARN] Missing residual file: {res_path}")
            missing_paths.append(res_path)
            arr = np.full((H, W), invalid_value, dtype=np.float32)

        residuals.append(arr)

    return np.stack(residuals, axis=0), residual_paths, missing_paths


def _load_mos_labels_for_history(ds, idx: int, cfg: dict, mos_label_folder: str):
    hist_len = int(cfg["model_params"]["input_horizon"])
    H = int(cfg["model_params"]["grid_height"])
    W = int(cfg["model_params"]["grid_width"])
    mosp = cfg.get("mos_label_params", {})
    ignore_index = int(mosp.get("ignore_index", -1))

    s_id, start = ds.windows[idx]
    seq = ds.seqs[s_id]

    mos_imgs = []
    mos_paths = []
    missing_paths = []
    stats = []

    for j in range(start, start + hist_len):
        frame_path = seq["paths"][j][0]
        seq_dir = os.path.abspath(os.path.join(os.path.dirname(frame_path), ".."))
        frame_stem = os.path.splitext(os.path.basename(frame_path))[0]
        mos_path = os.path.join(seq_dir, mos_label_folder, f"{frame_stem}.npy")
        mos_paths.append(mos_path)

        if os.path.isfile(mos_path):
            arr = np.load(mos_path)
            if arr.shape != (H, W):
                print(f"[WARN] MOS label shape mismatch at {mos_path}: {arr.shape} != {(H, W)}. Using ignore image.")
                arr = np.full((H, W), ignore_index, dtype=np.int16)
            else:
                arr = arr.astype(np.int16)
        else:
            print(f"[WARN] Missing MOS label file: {mos_path}")
            missing_paths.append(mos_path)
            arr = np.full((H, W), ignore_index, dtype=np.int16)

        st = compute_mos_label_stats(
            arr,
            ignore_index=ignore_index,
            static_index=int(mosp.get("static_index", 0)),
            moving_index=int(mosp.get("moving_index", 1)),
        )
        stats.append(st)
        mos_imgs.append(arr)

    return np.stack(mos_imgs, axis=0), mos_paths, missing_paths, stats


def _residual_stats(residuals: np.ndarray, invalid_value: float):
    valid_mask = residuals > float(invalid_value)
    valid_ratio = float(valid_mask.mean())
    if np.any(valid_mask):
        vals = residuals[valid_mask].astype(np.float32)
        mean_valid = float(np.mean(vals))
        max_valid = float(np.max(vals))
        p95_valid = float(np.percentile(vals, 95))
    else:
        mean_valid = 0.0
        max_valid = 0.0
        p95_valid = 0.0
    return {
        "valid_ratio": valid_ratio,
        "mean_valid": mean_valid,
        "max_valid": max_valid,
        "p95_valid": p95_valid,
    }


def _mos_plot_style(cfg: dict):
    mosp = cfg.get("mos_label_params", {})
    ignore_index = int(mosp.get("ignore_index", -1))
    static_index = int(mosp.get("static_index", 0))
    moving_index = int(mosp.get("moving_index", 1))

    cmap = mcolors.ListedColormap(["#4a4a4a", "#103e8a", "#f8d548"])
    levels = [ignore_index - 0.5, static_index + 0.5, moving_index + 0.5, moving_index + 1.5]
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    ticks = [ignore_index, static_index, moving_index]
    labels = ["ignore", "static", "moving"]
    return cmap, norm, ticks, labels


def _draw_frame(
    fig,
    axes,
    ds,
    cfg: dict,
    idx: int,
    split: str,
    vmin: float,
    vmax: float,
    show_residuals: bool,
    residual_offset: int,
    residuals_only: bool = False,
    show_mos_labels: bool = False,
    mos_label_folder: str = "mos_labels",
):
    hist_xyzd, future_xyz, future_ranges = ds[idx]

    s_id, start = ds.windows[idx]
    seq_id = ds.seqs[s_id]["seq_id"]

    hist_ranges = hist_xyzd[:, 3, :, :].numpy()
    fut_ranges = future_ranges.numpy()

    residuals = None
    residual_paths = []
    missing_residual_paths = []
    residual_stats = None
    if show_residuals:
        residuals, residual_paths, missing_residual_paths = _load_residuals_for_history(ds, idx, cfg, residual_offset)
        invalid_value = float(cfg.get("residual_params", {}).get("invalid_value", 0.0))
        residual_stats = _residual_stats(residuals, invalid_value)

    mos_labels = None
    mos_paths = []
    missing_mos_paths = []
    mos_stats = []
    if show_mos_labels:
        mos_labels, mos_paths, missing_mos_paths, mos_stats = _load_mos_labels_for_history(ds, idx, cfg, mos_label_folder)

    t_in = hist_ranges.shape[0]
    t_out = fut_ranges.shape[0]
    n_slots = t_in if residuals_only else max(t_in, t_out)

    ims = []
    range_im = None
    mos_cbar_meta = None
    mos_cmap, mos_norm, mos_ticks, mos_ticklabels = _mos_plot_style(cfg)

    for i in range(n_slots):
        if not residuals_only:
            ax_in = axes[0, i]
            ax_out = axes[1, i]
            ax_in.clear()
            ax_out.clear()

            if i < t_in:
                im = ax_in.imshow(hist_ranges[i], cmap="turbo", vmin=vmin, vmax=vmax)
                ims.append(im)
                if range_im is None:
                    range_im = im
                ax_in.set_title(f"Input t-{t_in - i}")
                ax_in.set_xticks([])
                ax_in.set_yticks([])
            else:
                ax_in.axis("off")

            if i < t_out:
                im = ax_out.imshow(fut_ranges[i], cmap="turbo", vmin=vmin, vmax=vmax)
                ims.append(im)
                ax_out.set_title(f"Target +{i + 1}")
                ax_out.set_xticks([])
                ax_out.set_yticks([])
            else:
                ax_out.axis("off")

        if show_residuals and not residuals_only:
            row_idx = 2
            ax_res = axes[row_idx, i]
            ax_res.clear()
            if i < t_in:
                if cfg.get("residual_params", {}).get("normalize", True):
                    residual_vmax = 1.0
                else:
                    residual_vmax = float(np.nanmax(residuals[i]) if np.isfinite(np.nanmax(residuals[i])) else 1.0)
                if residual_vmax <= 0:
                    residual_vmax = 1.0
                im_res = ax_res.imshow(residuals[i], cmap="turbo", vmin=0.0, vmax=residual_vmax)
                ims.append(im_res)
                ax_res.set_title(f"Residual t-{t_in - i}")
                ax_res.set_xticks([])
                ax_res.set_yticks([])
            else:
                ax_res.axis("off")
        elif show_residuals and residuals_only:
            ax_res = axes[i, 0]
            ax_res.clear()
            if i < t_in:
                if cfg.get("residual_params", {}).get("normalize", True):
                    residual_vmax = 1.0
                else:
                    residual_vmax = float(np.nanmax(residuals[i]) if np.isfinite(np.nanmax(residuals[i])) else 1.0)
                if residual_vmax <= 0:
                    residual_vmax = 1.0
                im_res = ax_res.imshow(residuals[i], cmap="turbo", vmin=0.0, vmax=residual_vmax)
                ims.append(im_res)
                ax_res.set_title(f"Residual t-{t_in - i}")
                ax_res.set_xticks([])
                ax_res.set_yticks([])
            else:
                ax_res.axis("off")

        if show_mos_labels and not residuals_only:
            row_idx = 2 + (1 if show_residuals else 0)
            ax_mos = axes[row_idx, i]
            ax_mos.clear()
            if i < t_in:
                im_mos = ax_mos.imshow(mos_labels[i], cmap=mos_cmap, norm=mos_norm, interpolation="nearest")
                if mos_cbar_meta is None:
                    mos_cbar_meta = {
                        "im": im_mos,
                        "ticks": mos_ticks,
                        "ticklabels": mos_ticklabels,
                    }
                ax_mos.set_title(f"MOS Label t-{t_in - i}")
                ax_mos.set_xticks([])
                ax_mos.set_yticks([])
            else:
                ax_mos.axis("off")

    fig.suptitle(
        f"Dataset Preview | split={split} | seq={seq_id} | window_start={start} | idx={idx}\n"
        f"hist_xyzd={tuple(hist_xyzd.shape)} future_xyz={tuple(future_xyz.shape)} future_ranges={tuple(future_ranges.shape)}"
        + (f" residuals={tuple(residuals.shape)} offset={residual_offset}" if show_residuals else "")
        + (f" mos_labels={tuple(mos_labels.shape)} folder={mos_label_folder}" if show_mos_labels else ""),
        fontsize=11,
    )

    return {
        "seq_id": seq_id,
        "window_start": start,
        "hist_shape": tuple(hist_xyzd.shape),
        "future_xyz_shape": tuple(future_xyz.shape),
        "future_ranges_shape": tuple(future_ranges.shape),
        "residuals_shape": tuple(residuals.shape) if residuals is not None else None,
        "residual_stats": residual_stats,
        "residual_paths": residual_paths,
        "missing_residual_paths": missing_residual_paths,
        "mos_shape": tuple(mos_labels.shape) if mos_labels is not None else None,
        "mos_stats": mos_stats,
        "mos_paths": mos_paths,
        "missing_mos_paths": missing_mos_paths,
        "ims": ims,
        "range_im": range_im,
        "mos_cbar_meta": mos_cbar_meta,
    }


def _print_mos_stats(mos_stats: list):
    for i, st in enumerate(mos_stats):
        print(
            f"mos_stats[t-{len(mos_stats) - i}] : "
            f"shape={st['shape']} unique={st['unique_values']} "
            f"ignore_ratio={st['ignore_ratio']:.4f} "
            f"static_ratio={st['static_ratio']:.4f} "
            f"moving_ratio={st['moving_ratio']:.4f}"
        )


def _frame_label_in_history(frame_in_window: int, hist_len: int) -> str:
    lag = (hist_len - 1) - frame_in_window
    return "t0" if lag == 0 else f"t-{lag}"


def _collect_single_mos_payload(ds, cfg: dict, idx: int, frame_in_window_arg, residual_offset: int, mos_label_folder: str):
    hist_xyzd, _, _ = ds[idx]
    hist_ranges = hist_xyzd[:, 3, :, :].numpy()
    hist_len = hist_ranges.shape[0]

    if frame_in_window_arg is None:
        frame_in_window = hist_len - 1
    else:
        frame_in_window = int(frame_in_window_arg)
    if frame_in_window < 0 or frame_in_window >= hist_len:
        raise IndexError(f"--frame_in_window {frame_in_window} outside [0, {hist_len - 1}].")

    s_id, window_start = ds.windows[idx]
    seq = ds.seqs[s_id]
    abs_frame_idx = window_start + frame_in_window
    frame_path = seq["paths"][abs_frame_idx][0]
    seq_dir = os.path.abspath(os.path.join(os.path.dirname(frame_path), ".."))
    frame_stem = os.path.splitext(os.path.basename(frame_path))[0]
    seq_id = seq["seq_id"]

    H = int(cfg["model_params"]["grid_height"])
    W = int(cfg["model_params"]["grid_width"])
    rp = cfg.get("residual_params", {})
    mosp = cfg.get("mos_label_params", {})
    invalid_residual = np.float32(rp.get("invalid_value", 0.0))
    ignore_index = int(mosp.get("ignore_index", -1))
    static_index = int(mosp.get("static_index", 0))
    moving_index = int(mosp.get("moving_index", 1))

    folder_template = rp.get("folder_template", "residual_images_{offset}")
    residual_folder = folder_template.format(offset=int(residual_offset))
    residual_path = os.path.join(seq_dir, residual_folder, f"{frame_stem}.npy")
    if os.path.isfile(residual_path):
        residual_img = np.load(residual_path).astype(np.float32)
        if residual_img.shape != (H, W):
            print(f"[WARN] Residual shape mismatch at {residual_path}: {residual_img.shape} != {(H, W)}. Using zeros.")
            residual_img = np.full((H, W), invalid_residual, dtype=np.float32)
    else:
        print(f"[WARN] Missing residual file: {residual_path}")
        residual_img = np.full((H, W), invalid_residual, dtype=np.float32)

    mos_path = os.path.join(seq_dir, mos_label_folder, f"{frame_stem}.npy")
    if os.path.isfile(mos_path):
        mos_img = np.load(mos_path).astype(np.int16)
        if mos_img.shape != (H, W):
            print(f"[WARN] MOS label shape mismatch at {mos_path}: {mos_img.shape} != {(H, W)}. Using ignore image.")
            mos_img = np.full((H, W), ignore_index, dtype=np.int16)
    else:
        print(f"[WARN] Missing MOS label file: {mos_path}")
        mos_img = np.full((H, W), ignore_index, dtype=np.int16)

    range_img = hist_ranges[frame_in_window]
    range_vmin = 0.0
    range_vmax = float(cfg.get("data_params", {}).get("stats", {}).get("max_range", np.percentile(range_img, 99)))
    if not np.isfinite(range_vmax) or range_vmax <= range_vmin:
        range_vmax = 80.0

    if cfg.get("residual_params", {}).get("normalize", True):
        residual_vmax = 1.0
    else:
        residual_vmax = float(np.nanmax(residual_img) if np.isfinite(np.nanmax(residual_img)) else 1.0)
        if residual_vmax <= 0:
            residual_vmax = 1.0

    mos_stats = compute_mos_label_stats(
        mos_img,
        ignore_index=ignore_index,
        static_index=static_index,
        moving_index=moving_index,
    )

    return {
        "seq_id": seq_id,
        "sample_index": idx,
        "window_start": window_start,
        "frame_in_window": frame_in_window,
        "abs_frame_idx": abs_frame_idx,
        "frame_stem": frame_stem,
        "hist_len": hist_len,
        "range_img": range_img,
        "range_vmin": range_vmin,
        "range_vmax": range_vmax,
        "residual_img": residual_img,
        "residual_vmax": residual_vmax,
        "residual_path": residual_path,
        "mos_img": mos_img,
        "mos_path": mos_path,
        "mos_stats": mos_stats,
    }


def _draw_single_mos_figure(fig, axes, payload: dict, cfg: dict):
    mos_cmap, mos_norm, mos_ticks, mos_ticklabels = _mos_plot_style(cfg)
    frame_label = _frame_label_in_history(payload["frame_in_window"], payload["hist_len"])

    for ax in axes:
        ax.clear()

    im0 = axes[0].imshow(payload["range_img"], cmap="turbo", vmin=payload["range_vmin"], vmax=payload["range_vmax"])
    axes[0].set_title(f"Range {frame_label}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(payload["residual_img"], cmap="turbo", vmin=0.0, vmax=payload["residual_vmax"])
    axes[1].set_title(f"Residual {frame_label}")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    im2 = axes[2].imshow(payload["mos_img"], cmap=mos_cmap, norm=mos_norm, interpolation="nearest")
    axes[2].set_title(f"MOS Label {frame_label}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(
        "Dataset Preview MOS Check\n"
        f"seq_id={payload['seq_id']} sample_index={payload['sample_index']} window_start={payload['window_start']} "
        f"frame_in_window={payload['frame_in_window']} abs_frame_idx={payload['abs_frame_idx']} frame_stem={payload['frame_stem']}",
        fontsize=11,
    )
    return im0, im1, im2, mos_ticks, mos_ticklabels


def _print_single_mos_payload(payload: dict):
    mos_stats = payload["mos_stats"]

    print("=== Single MOS Preview ===")
    print(f"seq_id           : {payload['seq_id']}")
    print(f"sample_index     : {payload['sample_index']}")
    print(f"window_start     : {payload['window_start']}")
    print(f"frame_in_window  : {payload['frame_in_window']}")
    print(f"abs_frame_idx    : {payload['abs_frame_idx']}")
    print(f"frame_stem       : {payload['frame_stem']}")
    print(f"range_shape      : {payload['range_img'].shape}")
    print(f"residual_path    : {payload['residual_path']}")
    print(f"mos_label_path   : {payload['mos_path']}")
    print(f"mos_unique       : {mos_stats['unique_values']}")
    print(f"moving_pixels    : {mos_stats['moving_count']}")
    print(f"static_pixels    : {mos_stats['static_count']}")
    print(f"ignore_pixels    : {mos_stats['ignore_count']}")


def _run_single_mos_preview(ds, cfg: dict, idx: int, args):
    payload = _collect_single_mos_payload(
        ds,
        cfg,
        idx,
        args.frame_in_window,
        args.residual_offset,
        args.mos_label_folder,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0, im1, im2, mos_ticks, mos_ticklabels = _draw_single_mos_figure(fig, axes, payload, cfg)
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    cbar = fig.colorbar(im2, ax=axes[2], shrink=0.85, ticks=mos_ticks)
    cbar.ax.set_yticklabels(mos_ticklabels)
    _print_single_mos_payload(payload)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"saved_figure     : {args.save}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


def _run_single_mos_play(ds, cfg: dict, args):
    indices = list(range(0, len(ds), args.stride))
    if args.max_windows is not None:
        indices = indices[:args.max_windows]
    if not indices:
        raise ValueError("No windows to display for --single_mos_play.")
    if args.no_show:
        raise ValueError("--single_mos_play requires a display. Remove --no_show.")

    print("single_mos_play  : ON")
    print(f"num_shown        : {len(indices)}")
    print("controls         : close window to stop")

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    cbar0 = None
    cbar1 = None
    cbar2 = None
    for idx in indices:
        payload = _collect_single_mos_payload(
            ds,
            cfg,
            idx,
            args.frame_in_window,
            args.residual_offset,
            args.mos_label_folder,
        )
        im0, im1, im2, mos_ticks, mos_ticklabels = _draw_single_mos_figure(fig, axes, payload, cfg)
        if cbar0 is None:
            cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85)
            cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85)
            cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.85, ticks=mos_ticks)
            cbar2.ax.set_yticklabels(mos_ticklabels)

        _print_single_mos_payload(payload)
        fig.canvas.draw_idle()
        plt.pause(max(0.001, args.interval_ms / 1000.0))
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visual preview of dataset samples exactly as used by training.")
    parser.set_defaults(play=True)
    parser.add_argument("--cfg_path", type=str, default="/home/devuser/workspace/src/configs/semanticKitti_default.yaml", help="Path to training config YAML.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"], help="Which predefined split to preview.")
    parser.add_argument("--seq_id", type=str, default=None, help="Optional sequence id filter (e.g. 00 or 0006).")
    parser.add_argument("--sample_index", type=int, default=None, help="Dataset window index. If omitted, uses 0 or random with --random_sample.")
    parser.add_argument("--random_sample", action="store_true", help="Pick a random window instead of 0.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --random_sample.")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path (single frame mode only).")
    parser.add_argument("--no_show", action="store_true", help="Do not open a plot window (useful on headless systems).")
    parser.add_argument("--play", action="store_true", help="Run through multiple windows in order (sequence walkthrough).")
    parser.add_argument("--interval_ms", type=float, default=300.0, help="Delay between windows in play mode.")
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth window in play mode.")
    parser.add_argument("--max_windows", type=int, default=None, help="Optional limit for number of windows in play mode.")
    parser.add_argument("--show_residuals", action="store_true", help="Load and show precomputed residual images for history frames.")
    parser.add_argument("--residual_offset", type=int, default=1, help="Residual offset folder index (e.g., 1 -> residual_images_1).")
    parser.add_argument("--residuals_only", action="store_true", help="Show only residual row (requires --show_residuals).")
    parser.add_argument("--show_mos_labels", action="store_true", help="Load and show precomputed MOS labels for history frames.")
    parser.add_argument("--mos_label_folder", type=str, default="mos_labels", help="Folder name for MOS labels inside each sequence directory.")
    parser.add_argument("--single_mos_preview", action="store_true", help="Show one selected history frame as [Range, Residual, MOS].")
    parser.add_argument("--single_mos_play", action="store_true", help="Play multiple windows in compact [Range, Residual, MOS] layout.")
    parser.add_argument("--frame_in_window", type=int, default=None, help="History frame index inside the loaded window. Default: last history frame (t0).")
    parser.add_argument("--fig_w_per_col", type=float, default=3.2, help="Figure width per column.")
    parser.add_argument("--fig_h_per_row", type=float, default=3.2, help="Figure height per row.")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _ensure_mos_defaults(cfg)

    all_seqs = make_sequences(cfg["dataset_path"])
    seqs = _filter_sequences_by_split(all_seqs, cfg, args.split)

    single_mos_mode = args.single_mos_preview or args.single_mos_play

    if single_mos_mode and args.seq_id is None:
        seq07 = [s for s in seqs if s["seq_id"] == "07"]
        if seq07:
            seqs = seq07

    if args.seq_id is not None:
        seqs = [s for s in seqs if s["seq_id"] == args.seq_id]
        if not seqs:
            if single_mos_mode:
                fallback = [s for s in all_seqs if s["seq_id"] == args.seq_id]
                if fallback:
                    print(
                        f"[WARN] seq_id={args.seq_id} not found in split='{args.split}'. "
                        "Using the sequence from dataset without split filter for single MOS mode."
                    )
                    seqs = fallback
                else:
                    raise ValueError(f"No sequence with id '{args.seq_id}' in dataset.")
            else:
                raise ValueError(f"No sequence with id '{args.seq_id}' in selected split '{args.split}'.")

    ds = RandomWindowSeqDataset(seqs, cfg, device="cpu")
    if len(ds) == 0:
        raise ValueError("Dataset has no windows after filtering.")

    if args.single_mos_play:
        _run_single_mos_play(ds, cfg, args)
        return

    if args.single_mos_preview:
        idx = _pick_index(len(ds), args.sample_index, args.random_sample, args.seed)
        _run_single_mos_preview(ds, cfg, idx, args)
        return

    if args.residuals_only and not args.show_residuals:
        raise ValueError("--residuals_only requires --show_residuals.")
    if args.residuals_only and args.show_mos_labels:
        raise ValueError("--residuals_only cannot be combined with --show_mos_labels.")

    first_hist, _, first_fut = ds[0]
    n_cols = max(first_hist.shape[0], first_fut.shape[0])

    if args.residuals_only:
        n_rows = int(first_hist.shape[0])
        n_cols = 1
    else:
        n_rows = 2 + (1 if args.show_residuals else 0) + (1 if args.show_mos_labels else 0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(args.fig_w_per_col * n_cols, args.fig_h_per_row * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array(axes).reshape(1, n_cols)
    elif n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    hist_ranges0 = first_hist[:, 3, :, :].numpy()
    fut_ranges0 = first_fut.numpy()
    vmin = 0.0
    vmax = _compute_vmax(cfg, hist_ranges0, fut_ranges0)

    print("=== Dataset Preview ===")
    print(f"config          : {args.cfg_path}")
    print(f"dataset_path    : {cfg['dataset_path']}")
    print(f"split           : {args.split}")
    print(f"num_sequences   : {len(seqs)}")
    print(f"dataset_windows : {len(ds)}")
    print(f"show_residuals  : {args.show_residuals}")
    if args.show_residuals:
        print(f"residual_offset : {args.residual_offset}")
    print(f"show_mos_labels : {args.show_mos_labels}")
    if args.show_mos_labels:
        print(f"mos_label_folder: {args.mos_label_folder}")
    print(f"residuals_only  : {args.residuals_only}")
    print(f"fig_w_per_col   : {args.fig_w_per_col}")
    print(f"fig_h_per_row   : {args.fig_h_per_row}")

    if args.play:
        if args.no_show:
            raise ValueError("--play requires a display. Remove --no_show.")

        plt.ion()
        cbar_created = False
        mos_cbar_created = False
        indices = list(range(0, len(ds), args.stride))
        if args.max_windows is not None:
            indices = indices[:args.max_windows]

        print("play_mode       : ON")
        print(f"num_shown       : {len(indices)}")
        print("controls        : close window to stop")

        for idx in indices:
            meta = _draw_frame(
                fig,
                axes,
                ds,
                cfg,
                idx,
                args.split,
                vmin,
                vmax,
                args.show_residuals,
                args.residual_offset,
                args.residuals_only,
                args.show_mos_labels,
                args.mos_label_folder,
            )
            if (not cbar_created) and (meta["range_im"] is not None):
                cbar = fig.colorbar(meta["range_im"], ax=axes.ravel().tolist(), shrink=0.85)
                cbar.set_label("Range")
                cbar_created = True
            if (not mos_cbar_created) and (meta["mos_cbar_meta"] is not None):
                mc = meta["mos_cbar_meta"]
                cbar_m = fig.colorbar(mc["im"], ax=axes.ravel().tolist(), shrink=0.85, ticks=mc["ticks"])
                cbar_m.ax.set_yticklabels(mc["ticklabels"])
                cbar_m.set_label("MOS")
                mos_cbar_created = True

            print(f"idx={idx} seq={meta['seq_id']} window_start={meta['window_start']}")
            if args.show_residuals:
                print(f"residuals_shape : {meta['residuals_shape']}")
                print(
                    "residual_stats  : "
                    f"valid_ratio={meta['residual_stats']['valid_ratio']:.4f} "
                    f"mean_valid={meta['residual_stats']['mean_valid']:.6f} "
                    f"max_valid={meta['residual_stats']['max_valid']:.6f} "
                    f"p95_valid={meta['residual_stats']['p95_valid']:.6f}"
                )
                for p in meta["residual_paths"]:
                    print(f"residual_path   : {p}")
            if args.show_mos_labels:
                print(f"mos_shape       : {meta['mos_shape']}")
                for p in meta["mos_paths"]:
                    print(f"mos_label_path  : {p}")
                _print_mos_stats(meta["mos_stats"])

            fig.canvas.draw_idle()
            plt.pause(max(0.001, args.interval_ms / 1000.0))

            if not plt.fignum_exists(fig.number):
                break

        plt.ioff()
        if plt.fignum_exists(fig.number):
            plt.show()
        return

    idx = _pick_index(len(ds), args.sample_index, args.random_sample, args.seed)
    meta = _draw_frame(
        fig,
        axes,
        ds,
        cfg,
        idx,
        args.split,
        vmin,
        vmax,
        args.show_residuals,
        args.residual_offset,
        args.residuals_only,
        args.show_mos_labels,
        args.mos_label_folder,
    )

    if meta["range_im"] is not None:
        cbar = fig.colorbar(meta["range_im"], ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("Range")
    if meta["mos_cbar_meta"] is not None:
        mc = meta["mos_cbar_meta"]
        cbar_m = fig.colorbar(mc["im"], ax=axes.ravel().tolist(), shrink=0.85, ticks=mc["ticks"])
        cbar_m.ax.set_yticklabels(mc["ticklabels"])
        cbar_m.set_label("MOS")

    print(f"picked_index    : {idx}")
    print(f"sequence_id     : {meta['seq_id']}")
    print(f"window_start    : {meta['window_start']}")
    print(f"hist_xyzd       : {meta['hist_shape']}")
    print(f"future_xyz      : {meta['future_xyz_shape']}")
    print(f"future_ranges   : {meta['future_ranges_shape']}")
    if args.show_residuals:
        print(f"residuals       : {meta['residuals_shape']}")
        print(
            "residual_stats  : "
            f"valid_ratio={meta['residual_stats']['valid_ratio']:.4f} "
            f"mean_valid={meta['residual_stats']['mean_valid']:.6f} "
            f"max_valid={meta['residual_stats']['max_valid']:.6f} "
            f"p95_valid={meta['residual_stats']['p95_valid']:.6f}"
        )
        for p in meta["residual_paths"]:
            print(f"residual_path   : {p}")
    if args.show_mos_labels:
        print(f"mos_labels      : {meta['mos_shape']}")
        for p in meta["mos_paths"]:
            print(f"mos_label_path  : {p}")
        _print_mos_stats(meta["mos_stats"])

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"saved_figure    : {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
