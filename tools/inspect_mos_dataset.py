#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from helper.dataloader_helper import make_sequences
from mos_dataset import MOSFrameDataset


def _parse_residual_offsets(text: str) -> List[int]:
    toks = text.replace(";", ",").replace(" ", ",").split(",")
    vals = [int(t) for t in toks if t.strip() != ""]
    vals = sorted(set(v for v in vals if v >= 1))
    if not vals:
        raise ValueError("residual_offsets must include at least one positive integer.")
    return vals


def _normalize_seq_id(seq_id: str) -> str:
    s = str(seq_id)
    return s.zfill(2) if s.isdigit() else s


def _resolve_cfg_path(cfg_path: str) -> str:
    candidates = [
        cfg_path,
        os.path.join(ROOT, cfg_path),
        os.path.join(SRC, cfg_path),
        os.path.join(ROOT, "src", cfg_path),
    ]
    seen = set()
    for p in candidates:
        p_abs = os.path.abspath(p)
        if p_abs in seen:
            continue
        seen.add(p_abs)
        if os.path.isfile(p_abs):
            return p_abs
    raise FileNotFoundError(
        f"Config not found: '{cfg_path}'. Checked: " + ", ".join(sorted(seen))
    )


def _ensure_defaults(cfg: Dict) -> Dict:
    mosp = cfg.setdefault("mos_label_params", {})
    mosp.setdefault("folder_name", "mos_labels")
    mosp.setdefault("ignore_index", -1)
    mosp.setdefault("static_index", 0)
    mosp.setdefault("moving_index", 1)
    return cfg


def _filter_sequences_by_split(seqs: List[Dict], cfg: Dict, split: str) -> List[Dict]:
    if split == "all":
        return seqs
    splits = cfg.get("data_params", {}).get("predefined_splits", None)
    if not isinstance(splits, dict):
        raise ValueError("Config has no data_params.predefined_splits. Use --split all.")
    if split not in splits:
        raise ValueError(f"Split '{split}' not in config predefined_splits.")
    wanted = set(_normalize_seq_id(s) for s in splits[split])
    filtered = [s for s in seqs if _normalize_seq_id(s.get("seq_id", "")) in wanted]
    if not filtered:
        raise ValueError(f"No sequences matched split '{split}'.")
    return filtered


def _print_header(cfg: Dict, seqs: List[Dict], ds: MOSFrameDataset, residual_offsets: List[int]):
    mp = cfg["model_params"]
    seq_ids = [_normalize_seq_id(s.get("seq_id", "")) for s in seqs]
    print("=== MOS Dataset Inspect ===")
    print(f"dataset_path       : {cfg['dataset_path']}")
    print(f"seq_ids            : {seq_ids}")
    print(f"num_samples        : {len(ds)}")
    print(f"input_mode         : {ds.input_mode}")
    print(f"residual_offsets   : {residual_offsets}")
    print(f"grid(H,W)          : ({int(mp['grid_height'])},{int(mp['grid_width'])})")
    print(f"channels           : {ds.channel_count}")
    print("")


def _print_class_stats(stats: Dict):
    print("Class stats:")
    print(f"ignore_pixels      : {stats['ignore_pixels']}")
    print(f"static_pixels      : {stats['static_pixels']}")
    print(f"moving_pixels      : {stats['moving_pixels']}")
    print(f"ignore_ratio       : {stats['ignore_ratio']:.6f}")
    print(f"static_ratio       : {stats['static_ratio']:.6f}")
    print(f"moving_ratio       : {stats['moving_ratio']:.6f}")
    print(f"frames_with_moving : {stats['frames_with_moving']}")
    print(f"moving_pixels_min  : {stats['moving_pixels_min']}")
    print(f"moving_pixels_max  : {stats['moving_pixels_max']}")
    print(f"moving_pixels_mean : {stats['moving_pixels_mean']:.3f}")
    print("")


def _sample_indices(n_total: int, n_check: int) -> List[int]:
    if n_total <= 0:
        return []
    n = min(n_total, max(1, int(n_check)))
    idx = np.linspace(0, n_total - 1, num=n, dtype=np.int64).tolist()
    out = []
    seen = set()
    for i in idx:
        ii = int(i)
        if ii not in seen:
            seen.add(ii)
            out.append(ii)
    return out


def _summarize_sample(ds: MOSFrameDataset, idx: int):
    x, y, meta = ds[idx]
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    y_unique = sorted(int(v) for v in np.unique(y_np))

    print("Sample check:")
    print(f"idx                : {idx}")
    print(f"x_shape            : {list(x.shape)}")
    print(f"y_shape            : {list(y.shape)}")
    print(f"x_dtype            : {x.dtype}")
    print(f"y_dtype            : {y.dtype}")
    print(f"y_unique           : {y_unique}")

    if ds.input_mode in {"range", "range_residual"}:
        range_img = x_np[0]
        print(f"range_min/max      : {float(np.min(range_img)):.6f}/{float(np.max(range_img)):.6f}")
    else:
        print("range_min/max      : n/a")

    if ds.input_mode == "range_residual":
        res_stack = x_np[1:]
    elif ds.input_mode == "residual":
        res_stack = x_np
    else:
        res_stack = None

    if res_stack is not None and res_stack.size > 0:
        print(f"residual_min/max   : {float(np.min(res_stack)):.6f}/{float(np.max(res_stack)):.6f}")
        print(f"residual_p95       : {float(np.percentile(res_stack, 95)):.6f}")
    else:
        print("residual_min/max   : n/a")
        print("residual_p95       : n/a")

    print(f"moving_pixels      : {meta['moving_pixels']}")
    print(f"static_pixels      : {meta['static_pixels']}")
    print(f"ignore_pixels      : {meta['ignore_pixels']}")
    print(f"meta               : {meta['seq_id']}/{meta['frame_stem']}")
    print("")


def _plot_preview(ds: MOSFrameDataset, idx: int, no_show: bool):
    x, y, meta = ds[idx]
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    panels = []
    if ds.input_mode in {"range", "range_residual"}:
        panels.append(("Range", x_np[0], "range"))
    if ds.input_mode == "range_residual":
        for i, off in enumerate(ds.residual_offsets):
            panels.append((f"Residual_{off}", x_np[1 + i], "residual"))
    elif ds.input_mode == "residual":
        for i, off in enumerate(ds.residual_offsets):
            panels.append((f"Residual_{off}", x_np[i], "residual"))
    panels.append(("MOS Label", y_np, "mos"))

    ncols = len(panels)
    fig_w = max(4.0 * ncols, 10.0)
    fig, axs = plt.subplots(1, ncols, figsize=(fig_w, 4.2), constrained_layout=True)
    if ncols == 1:
        axs = [axs]

    for ax, (title, arr, kind) in zip(axs, panels):
        if kind == "range":
            p99 = float(np.percentile(arr, 99))
            vmax = min(80.0, max(1e-6, p99))
            im = ax.imshow(arr, cmap="turbo", vmin=0.0, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.85)
        elif kind == "residual":
            im = ax.imshow(arr, cmap="turbo", vmin=0.0, vmax=0.1)
            fig.colorbar(im, ax=ax, shrink=0.85)
        else:
            cmap = mcolors.ListedColormap(["#3b3b3b", "#0f3473", "#f29e4c"])
            norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
            im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels(["ignore", "static", "moving"])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"seq={meta['seq_id']} frame={meta['frame_stem']} moving_pixels={meta['moving_pixels']} "
        f"input_mode={ds.input_mode}"
    )

    if not no_show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect MOSFrameDataset before MOS training.")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--seq_id", type=str, default=None)
    parser.add_argument(
        "--input_mode",
        type=str,
        default="range_residual",
        choices=["range", "residual", "range_residual"],
    )
    parser.add_argument("--residual_offsets", type=str, default="1")
    parser.add_argument("--mos_label_folder", type=str, default="mos_labels")
    parser.add_argument("--num_samples_to_check", type=int, default=10)
    parser.add_argument("--require_moving", action="store_true")
    parser.add_argument("--min_moving_pixels", type=int, default=1)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    cfg_path = _resolve_cfg_path(args.cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _ensure_defaults(cfg)

    residual_offsets = _parse_residual_offsets(args.residual_offsets)
    seqs = make_sequences(cfg["dataset_path"])
    seqs = _filter_sequences_by_split(seqs, cfg, args.split)

    if args.seq_id is not None:
        wanted = _normalize_seq_id(args.seq_id)
        seqs = [s for s in seqs if _normalize_seq_id(s.get("seq_id", "")) == wanted]
        if not seqs:
            raise ValueError(f"No sequence matched seq_id={wanted} after split filtering.")

    ds = MOSFrameDataset(
        sequences=seqs,
        cfg=cfg,
        split=args.split,
        input_mode=args.input_mode,
        residual_offsets=residual_offsets,
        mos_label_folder=args.mos_label_folder,
        device="cpu",
        require_moving=args.require_moving,
        min_moving_pixels=args.min_moving_pixels,
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset has 0 samples after filtering.")

    _print_header(cfg, seqs, ds, residual_offsets)
    class_stats = ds.get_class_stats()
    _print_class_stats(class_stats)

    idxs = _sample_indices(len(ds), args.num_samples_to_check)
    for i in idxs:
        _summarize_sample(ds, i)

    if not args.no_show:
        _plot_preview(ds, idxs[0], no_show=args.no_show)


if __name__ == "__main__":
    main()
