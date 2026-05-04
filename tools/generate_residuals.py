#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from helper.dataloader_helper import make_sequences
from helper.residual_helper import (
    compute_residual_image,
    load_scan_xyzi,
    residual_path_for_frame,
)


def _ensure_residual_defaults(cfg: Dict) -> Dict:
    rp = cfg.setdefault("residual_params", {})
    rp.setdefault("enabled", True)
    rp.setdefault("mode", "precompute")
    rp.setdefault("offsets", [1])
    rp.setdefault("normalize", True)
    rp.setdefault("min_range", 2.0)
    rp.setdefault("max_range", 50.0)
    rp.setdefault("invalid_value", 0.0)
    rp.setdefault("folder_template", "residual_images_{offset}")
    rp.setdefault("overwrite", False)
    rp.setdefault("visualize_debug", False)
    return cfg


def _print_seq_header(seq: Dict, cfg: Dict, offsets: List[int]):
    rp = cfg["residual_params"]
    mp = cfg["model_params"]
    print(f"\n=== Sequence {seq['seq_id']} ===")
    print(f"frames           : {len(seq['paths'])}")
    print(f"offsets          : {offsets}")
    for off in offsets:
        folder = rp["folder_template"].format(offset=off)
        print(f"target_folder[{off}] : {os.path.join(os.path.dirname(seq['paths'][0][0]), '..', folder)}")
    print(f"grid(H,W)        : ({mp['grid_height']}, {mp['grid_width']})")
    print(f"FOV_UP/FOV_DOWN  : ({mp.get('FOV_UP', 3.0)}, {mp.get('FOV_DOWN', -25.0)})")
    print(f"min/max_range    : ({rp['min_range']}, {rp['max_range']})")


def main():
    parser = argparse.ArgumentParser(description="Offline residual image generation (MOS-style transform, pipeline projection).")
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--seq_id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--debug_frame", type=int, default=None)
    parser.add_argument("--save_debug", type=str, default=None)
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _ensure_residual_defaults(cfg)

    rp = cfg["residual_params"]
    if not rp.get("enabled", True):
        print("Residual generation disabled in config (residual_params.enabled=false). Nothing to do.")
        return
    if rp.get("mode", "precompute") != "precompute":
        print("residual_params.mode is not 'precompute'. This tool only supports offline precompute mode.")
        return

    seqs = make_sequences(cfg["dataset_path"])
    if args.seq_id is not None:
        seqs = [s for s in seqs if s["seq_id"] == args.seq_id]
        if not seqs:
            raise ValueError(f"No sequence with seq_id={args.seq_id} in dataset_path={cfg['dataset_path']}")

    offsets = [int(o) for o in rp.get("offsets", [1])]
    offsets = sorted(set(o for o in offsets if o >= 1))
    if not offsets:
        raise ValueError("residual_params.offsets must contain positive integers.")

    total_skipped = 0
    total_written = 0
    all_valid_ratios = []
    all_values = []

    for seq in seqs:
        _print_seq_header(seq, cfg, offsets)
        seq_paths = seq["paths"]
        seq_poses = seq["poses"]
        seq_dir = os.path.dirname(seq_paths[0][0])
        seq_dir = os.path.abspath(os.path.join(seq_dir, ".."))
        n_frames = len(seq_paths)
        if args.max_frames is not None:
            n_frames = min(n_frames, int(args.max_frames))

        for off in offsets:
            folder = rp["folder_template"].format(offset=off)
            os.makedirs(os.path.join(seq_dir, folder), exist_ok=True)

        for idx in range(n_frames):
            frame_stem = os.path.splitext(os.path.basename(seq_paths[idx][0]))[0]
            current_scan = load_scan_xyzi(seq_paths[idx][0])

            for off in offsets:
                out_path = residual_path_for_frame(seq_dir, frame_stem, off, cfg)
                do_overwrite = bool(args.overwrite or rp.get("overwrite", False))
                if (not do_overwrite) and os.path.isfile(out_path):
                    total_skipped += 1
                    continue

                if idx < off:
                    residual = np.full(
                        (int(cfg["model_params"]["grid_height"]), int(cfg["model_params"]["grid_width"])),
                        np.float32(rp.get("invalid_value", 0.0)),
                        dtype=np.float32,
                    )
                else:
                    past_scan = load_scan_xyzi(seq_paths[idx - off][0])
                    residual = compute_residual_image(
                        current_scan=current_scan,
                        past_scan=past_scan,
                        current_pose=np.asarray(seq_poses[idx], dtype=np.float64),
                        past_pose=np.asarray(seq_poses[idx - off], dtype=np.float64),
                        cfg=cfg,
                        offset=off,
                    )

                    valid = residual > float(rp.get("invalid_value", 0.0))
                    valid_ratio = float(valid.mean())
                    all_valid_ratios.append(valid_ratio)
                    if np.any(valid):
                        vals = residual[valid]
                        all_values.append(vals)

                    if args.debug_frame is not None and idx == args.debug_frame:
                        from helper.residual_helper import project_range_for_residual, transform_points

                        T = np.linalg.inv(np.asarray(seq_poses[idx], dtype=np.float64)) @ np.asarray(seq_poses[idx - off], dtype=np.float64)
                        past_scan_in_current = transform_points(past_scan, T)
                        current_range = project_range_for_residual(current_scan, cfg)
                        past_range_warped = project_range_for_residual(past_scan_in_current, cfg)

                        fig, axs = plt.subplots(3, 1, figsize=(12, 6), constrained_layout=True)
                        axs[0].imshow(current_range, cmap="turbo")
                        axs[0].set_title("current_range")
                        axs[1].imshow(past_range_warped, cmap="turbo")
                        axs[1].set_title(f"past_range_warped (offset={off})")
                        axs[2].imshow(residual, cmap="turbo", vmin=0.0, vmax=1.0 if rp.get("normalize", True) else None)
                        axs[2].set_title("residual")
                        for ax in axs:
                            ax.set_xticks([])
                            ax.set_yticks([])
                        if args.save_debug:
                            os.makedirs(os.path.dirname(args.save_debug) or ".", exist_ok=True)
                            fig.savefig(args.save_debug, dpi=150)
                            print(f"saved_debug_figure: {args.save_debug}")
                        else:
                            plt.show()
                        plt.close(fig)

                np.save(out_path, residual.astype(np.float32))
                total_written += 1

        print(f"completed seq={seq['seq_id']} written={total_written} skipped={total_skipped}")

    print("\n=== Residual Generation Summary ===")
    print(f"sequences_done            : {len(seqs)}")
    print(f"written_files             : {total_written}")
    print(f"skipped_existing_files    : {total_skipped}")

    if all_valid_ratios:
        print(f"valid_ratio_mean          : {np.mean(all_valid_ratios):.4f}")
        print(f"valid_ratio_std           : {np.std(all_valid_ratios):.4f}")
    else:
        print("valid_ratio_mean          : n/a")

    if all_values:
        vals = np.concatenate(all_values, axis=0)
        print(f"residual_mean_valid       : {float(np.mean(vals)):.6f}")
        print(f"residual_std_valid        : {float(np.std(vals)):.6f}")
        print(f"residual_max_valid        : {float(np.max(vals)):.6f}")
    else:
        print("residual_stats_valid      : n/a")


if __name__ == "__main__":
    main()
