import argparse
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dataloader import RandomWindowSeqDataset
from helper.dataloader_helper import make_sequences


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

    # Residuals are frame-based deterministic offline artifacts.
    # Windows simply load per-frame files for their history frame ids.
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


def _draw_frame(fig, axes, ds, cfg: dict, idx: int, split: str, vmin: float, vmax: float, show_residuals: bool, residual_offset: int):
    hist_xyzd, future_xyz, future_ranges = ds[idx]

    s_id, start = ds.windows[idx]
    seq_id = ds.seqs[s_id]["seq_id"]

    hist_ranges = hist_xyzd[:, 3, :, :].numpy()
    fut_ranges = future_ranges.numpy()

    residuals = None
    residual_paths = []
    missing_residual_paths = []
    if show_residuals:
        residuals, residual_paths, missing_residual_paths = _load_residuals_for_history(ds, idx, cfg, residual_offset)

    t_in = hist_ranges.shape[0]
    t_out = fut_ranges.shape[0]
    n_cols = max(t_in, t_out)

    ims = []
    for i in range(n_cols):
        ax_in = axes[0, i]
        ax_out = axes[1, i]
        ax_in.clear()
        ax_out.clear()

        if i < t_in:
            im = ax_in.imshow(hist_ranges[i], cmap="turbo", vmin=vmin, vmax=vmax)
            ims.append(im)
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

        if show_residuals:
            ax_res = axes[2, i]
            ax_res.clear()
            if i < t_in:
                residual_vmax = 1.0 if cfg.get("residual_params", {}).get("normalize", True) else float(np.nanmax(residuals[i]) if np.isfinite(np.nanmax(residuals[i])) else 1.0)
                if residual_vmax <= 0:
                    residual_vmax = 1.0
                im_res = ax_res.imshow(residuals[i], cmap="turbo", vmin=0.0, vmax=residual_vmax)
                ims.append(im_res)
                ax_res.set_title(f"Residual t-{t_in - i}")
                ax_res.set_xticks([])
                ax_res.set_yticks([])
            else:
                ax_res.axis("off")

    fig.suptitle(
        f"Dataset Preview | split={split} | seq={seq_id} | window_start={start} | idx={idx}\n"
        f"hist_xyzd={tuple(hist_xyzd.shape)} future_xyz={tuple(future_xyz.shape)} future_ranges={tuple(future_ranges.shape)}"
        + (f" residuals={tuple(residuals.shape)} offset={residual_offset}" if show_residuals else ""),
        fontsize=11,
    )

    return {
        "seq_id": seq_id,
        "window_start": start,
        "hist_shape": tuple(hist_xyzd.shape),
        "future_xyz_shape": tuple(future_xyz.shape),
        "future_ranges_shape": tuple(future_ranges.shape),
        "residuals_shape": tuple(residuals.shape) if residuals is not None else None,
        "residual_paths": residual_paths,
        "missing_residual_paths": missing_residual_paths,
        "ims": ims,
    }


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
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    all_seqs = make_sequences(cfg["dataset_path"])
    seqs = _filter_sequences_by_split(all_seqs, cfg, args.split)

    if args.seq_id is not None:
        seqs = [s for s in seqs if s["seq_id"] == args.seq_id]
        if not seqs:
            raise ValueError(f"No sequence with id '{args.seq_id}' in selected split '{args.split}'.")

    ds = RandomWindowSeqDataset(seqs, cfg, device="cpu")
    if len(ds) == 0:
        raise ValueError("Dataset has no windows after filtering.")

    first_hist, _, first_fut = ds[0]
    n_cols = max(first_hist.shape[0], first_fut.shape[0])
    n_rows = 3 if args.show_residuals else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), constrained_layout=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    hist_ranges0 = first_hist[:, 3, :, :].numpy()
    fut_ranges0 = first_fut.numpy()
    vmin = 0.0
    vmax = _compute_vmax(cfg, hist_ranges0, fut_ranges0)

    print("=== Dataset Preview ===")
    print(f"config         : {args.cfg_path}")
    print(f"dataset_path   : {cfg['dataset_path']}")
    print(f"split          : {args.split}")
    print(f"num_sequences  : {len(seqs)}")
    print(f"dataset_windows: {len(ds)}")
    print(f"show_residuals : {args.show_residuals}")
    if args.show_residuals:
        print(f"residual_offset: {args.residual_offset}")

    if args.play:
        if args.no_show:
            raise ValueError("--play requires a display. Remove --no_show.")

        plt.ion()
        cbar_created = False
        indices = list(range(0, len(ds), args.stride))
        if args.max_windows is not None:
            indices = indices[:args.max_windows]

        print("play_mode      : ON")
        print(f"num_shown      : {len(indices)}")
        print("controls       : close window to stop")

        for idx in indices:
            meta = _draw_frame(fig, axes, ds, cfg, idx, args.split, vmin, vmax, args.show_residuals, args.residual_offset)
            if (not cbar_created) and meta["ims"]:
                cbar = fig.colorbar(meta["ims"][-1], ax=axes.ravel().tolist(), shrink=0.85)
                cbar.set_label("Range/Residual")
                cbar_created = True

            print(f"idx={idx} seq={meta['seq_id']} window_start={meta['window_start']}")
            if args.show_residuals:
                print(f"residuals_shape : {meta['residuals_shape']}")
                for p in meta["residual_paths"]:
                    print(f"residual_path   : {p}")
            fig.canvas.draw_idle()
            plt.pause(max(0.001, args.interval_ms / 1000.0))

            if not plt.fignum_exists(fig.number):
                break

        plt.ioff()
        if plt.fignum_exists(fig.number):
            plt.show()
        return

    idx = _pick_index(len(ds), args.sample_index, args.random_sample, args.seed)
    meta = _draw_frame(fig, axes, ds, cfg, idx, args.split, vmin, vmax, args.show_residuals, args.residual_offset)

    if meta["ims"]:
        cbar = fig.colorbar(meta["ims"][-1], ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("Range/Residual")

    print(f"picked_index   : {idx}")
    print(f"sequence_id    : {meta['seq_id']}")
    print(f"window_start   : {meta['window_start']}")
    print(f"hist_xyzd      : {meta['hist_shape']}")
    print(f"future_xyz     : {meta['future_xyz_shape']}")
    print(f"future_ranges  : {meta['future_ranges_shape']}")
    if args.show_residuals:
        print(f"residuals      : {meta['residuals_shape']}")
        for p in meta["residual_paths"]:
            print(f"residual_path  : {p}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"saved_figure   : {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
