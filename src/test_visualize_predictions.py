import argparse
import os
import random
import time
from typing import Optional, Tuple

import matplotlib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper.dataloader_helper import build_dataloaders, make_sequences
from models import build_model
from utils_torch import make_angle_grids

# Optional: set a fixed weights path directly in this script.
# Used as fallback if --weights and cfg train_params.pre_train_weights are not set.
DEFAULT_WEIGHTS_PATH = "/home/devuser/workspace/LidarGaussianVideoView/logs/SemanticKITTI ohne Ray/weights/model_final.pt"

# Debug switch: set to "2d", "3d" or "both" for IDE/debugger runs.
# If None, CLI --viz_mode is used.
DEBUG_VIZ_MODE_OVERRIDE = "2d"
# 3D camera start zoom for debugger runs (smaller = farther away).
DEBUG_3D_CAMERA_ZOOM = 0.19
# Smooth visualization overrides for debugger runs.
# If None, CLI args (--viz_batch_size / --viz_num_workers) are used.
DEBUG_VIZ_BATCH_SIZE_OVERRIDE = 4
DEBUG_VIZ_NUM_WORKERS_OVERRIDE = 2


def print_runtime_info(cfg: dict, device: str) -> None:
    tp = cfg.get("train_params", {})
    seed = tp.get("random_seed", None)
    deterministic = tp.get("deterministic", False)
    print("\n===== INFERENCE ENVIRONMENT INFO =====")
    print(f"Device:               {device}")
    print(f"Random Seed:          {seed if seed is not None else 'None (random each run)'}")
    print(f"Deterministic Mode:   {'ON' if deterministic else 'OFF'}")
    print(f"CuDNN benchmark:      {torch.backends.cudnn.benchmark}")
    print(f"CuDNN deterministic:  {torch.backends.cudnn.deterministic}")
    print("======================================\n")


def apply_seed_and_determinism(cfg: dict) -> None:
    tp = cfg.get("train_params", {})
    seed = tp.get("random_seed", None)
    deterministic = tp.get("deterministic", False)
    cudnn_benchmark = tp.get("cudnn_benchmark", False)

    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.deterministic = False


def load_pretrained_weights(model: torch.nn.Module, cfg: dict, weights_override: Optional[str]) -> str:
    pre_path = (
        weights_override
        or cfg.get("train_params", {}).get("pre_train_weights", None)
        or DEFAULT_WEIGHTS_PATH
    )
    if not pre_path:
        raise ValueError(
            "No weights path configured. Set DEFAULT_WEIGHTS_PATH in script, "
            "or train_params.pre_train_weights in config, or pass --weights."
        )
    if not os.path.isfile(pre_path):
        raise FileNotFoundError(f"Weights file not found: {pre_path}")

    print(f"[CKPT] Loading pretrained weights from: {pre_path}")
    ckpt = torch.load(pre_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        print(f"[CKPT] Detected Lightning checkpoint with {len(state)} tensors.")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print(f"[CKPT] Detected checkpoint['model_state_dict'] with {len(state)} tensors.")
    elif isinstance(ckpt, dict):
        state = ckpt
        print(f"[CKPT] Detected plain state_dict with {len(state)} tensors.")
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    clean_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("module."):
            k = k[len("module."):]

        if k.startswith(("enc.", "hid.", "dec.")):
            k = "acc." + k

        clean_state[k] = v

        if k == "mean":
            clean_state["acc.mean"] = v
        elif k == "std":
            clean_state["acc.std"] = v

    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in clean_state.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered_state[k] = v

    if skipped:
        print("[CKPT] Skipping keys due to shape mismatch:")
        for k, s_ckpt, s_model in skipped:
            print(f"  - {k}: ckpt{s_ckpt} -> model{s_model}")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"[CKPT] Load done  Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    if missing:
        print("[CKPT] Example missing keys:", missing[:10])
    if unexpected:
        print("[CKPT] Example unexpected keys:", unexpected[:10])

    return pre_path


def select_eval_loader(cfg: dict, dataloader_device: str, split: str):
    all_seqs = make_sequences(cfg["dataset_path"])
    data_cfg = cfg.get("data_params", {})
    split_type = data_cfg.get("split_type", "rotary")
    predefined_splits = data_cfg.get("predefined_splits", None)

    if split_type == "predefined":
        if predefined_splits is None:
            raise ValueError("split_type='predefined' but data_params.predefined_splits is missing.")

        loaders = build_dataloaders(
            all_seqs,
            cfg,
            dataloader_device,
            split_type="predefined",
            predefined_splits=predefined_splits,
        )

        if len(loaders) == 3:
            train_loader, val_loader, test_loader = loaders
        else:
            train_loader, val_loader = loaders
            test_loader = None

        split = split.lower()
        if split == "train":
            return train_loader, "train"
        if split == "val":
            return val_loader, "val"
        if split == "test":
            if test_loader is None:
                print("[WARN] No dedicated test split in config. Falling back to validation split.")
                return val_loader, "val(fallback_no_test)"
            return test_loader, "test"
        raise ValueError(f"Unknown split '{split}'. Use train|val|test.")

    if split_type == "rotary":
        rotary_loaders = build_dataloaders(
            all_seqs,
            cfg,
            dataloader_device,
            split_type="rotary",
        )
        holdout_id, _, val_loader = rotary_loaders[0]
        print(f"[WARN] split_type='rotary' has no dedicated test split. Using holdout val loader of seq {holdout_id}.")
        return val_loader, f"rotary_val_holdout_{holdout_id}"

    raise ValueError(f"Unknown split_type: {split_type}")


def get_horizon_labels(cfg: dict, t_out: int) -> list:
    horizons = cfg.get("train_params", {}).get("output_horizons", None)
    if isinstance(horizons, (list, tuple)) and len(horizons) == t_out:
        return [f"t+{h}" for h in horizons]
    return [f"t+{t}" for t in range(t_out)]


def resolve_sample_meta(dataset, dataset_idx: int) -> Tuple[str, int]:
    if dataset is None or not hasattr(dataset, "windows") or not hasattr(dataset, "seqs"):
        return "unknown", -1
    if dataset_idx < 0 or dataset_idx >= len(dataset.windows):
        return "unknown", -1

    s_id, start = dataset.windows[dataset_idx]
    seq_id = dataset.seqs[s_id]["seq_id"]
    return str(seq_id), int(start)


def init_figure(plt, t_out: int, h: int, w: int, horizon_labels: list):
    n_rows, n_cols = t_out, 3
    height_per = 2.0
    aspect = w / h
    width_per = height_per * aspect
    fig_width = width_per * n_cols
    fig_height = height_per * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    col_titles = [
        "Ground Truth Range [m]",
        "Predicted Range [m]",
        "Absoluter Fehler |GT-Pred| [m]",
    ]

    im_handles = [[None] * n_cols for _ in range(n_rows)]
    cb_handles = [[None] * n_cols for _ in range(n_rows)]

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            cmap = matplotlib.colormaps["turbo"].copy()
            cmap.set_bad(color="black")
            im = ax.imshow(np.full((h, w), np.nan, dtype=np.float32), aspect="equal", vmin=0, vmax=50, cmap=cmap)
            ax.set_title(f"{horizon_labels[i]} | {col_titles[j]}")
            ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Distanz [m]")
            im_handles[i][j] = im
            cb_handles[i][j] = cb

    return fig, im_handles, cb_handles


def compute_vmax(cfg: dict, gt_sample: np.ndarray, percentile: float) -> float:
    valid_gt = gt_sample[gt_sample > 0.0]
    vmax = float(np.percentile(valid_gt, percentile)) if valid_gt.size > 0 else 50.0
    max_range_cfg = cfg.get("data_params", {}).get("stats", {}).get("max_range", None)
    vmax_cap = float(max_range_cfg) if max_range_cfg is not None else 120.0
    return max(5.0, min(vmax, vmax_cap))


def make_viz_eval_loader(base_loader, batch_size: int, num_workers: int):
    dataset = getattr(base_loader, "dataset", None)
    if dataset is None:
        return base_loader

    if batch_size <= 0:
        batch_size = int(getattr(base_loader, "batch_size", 1))
    if num_workers < 0:
        num_workers = 0

    pin_memory = bool(getattr(base_loader, "pin_memory", False))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def get_theta_range(cfg: dict) -> Tuple[float, float]:
    fov_up = float(cfg["model_params"].get("FOV_UP", 3.0))
    fov_down = float(cfg["model_params"].get("FOV_DOWN", -25.0))
    return (np.deg2rad(fov_down), np.deg2rad(fov_up))


def range_image_to_points(
    range_img: np.ndarray,
    phi_grid: np.ndarray,
    theta_grid: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
) -> np.ndarray:
    if valid_mask is None:
        valid_mask = range_img > 0.0
    else:
        valid_mask = valid_mask & (range_img > 0.0)

    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32)

    r = range_img[valid_mask]
    phi = phi_grid[valid_mask]
    theta = theta_grid[valid_mask]

    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    if max_points is not None and max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    return pts


def configure_open3d_camera(vis, gt_pts: np.ndarray, pred_pts: np.ndarray) -> None:
    all_pts = []
    if gt_pts.size > 0:
        all_pts.append(gt_pts)
    if pred_pts.size > 0:
        all_pts.append(pred_pts)
    if not all_pts:
        return

    pts = np.concatenate(all_pts, axis=0)
    center = pts.mean(axis=0)

    # Start with Open3D's internal fit so near/far clipping is valid,
    # then enforce bird's-eye orientation.
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_lookat(center.tolist())
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(float(DEBUG_3D_CAMERA_ZOOM))


def main(args):
    mode_source = "CLI"
    if DEBUG_VIZ_MODE_OVERRIDE is not None:
        mode = str(DEBUG_VIZ_MODE_OVERRIDE).lower()
        mode_source = "DEBUG_VIZ_MODE_OVERRIDE"
    else:
        mode = args.viz_mode.lower()

    if mode not in {"2d", "3d", "both"}:
        raise ValueError(
            f"Invalid visualization mode '{mode}' from {mode_source}. "
            "Use one of: 2d, 3d, both."
        )

    show_2d = mode in ("2d", "both")
    show_3d = mode in ("3d", "both")

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    apply_seed_and_determinism(cfg)
    print_runtime_info(cfg, args.device)

    model_name = cfg["model_params"].get("name", "swin")
    model = build_model(model_name, cfg)

    eval_loader_raw, split_used = select_eval_loader(cfg, args.dataloader_device, args.split)
    viz_batch_size = (
        int(DEBUG_VIZ_BATCH_SIZE_OVERRIDE)
        if DEBUG_VIZ_BATCH_SIZE_OVERRIDE is not None
        else int(args.viz_batch_size)
    )
    viz_num_workers = (
        int(DEBUG_VIZ_NUM_WORKERS_OVERRIDE)
        if DEBUG_VIZ_NUM_WORKERS_OVERRIDE is not None
        else int(args.viz_num_workers)
    )
    eval_loader = make_viz_eval_loader(
        eval_loader_raw,
        batch_size=viz_batch_size,
        num_workers=viz_num_workers,
    )
    weight_path = load_pretrained_weights(model, cfg, args.weights)

    model = model.to(args.device)
    model.eval()

    use_mdn = bool(cfg["model_params"].get("use_mdn", True))
    if use_mdn and not hasattr(model, "build_mixture"):
        raise ValueError("use_mdn=True but selected model has no build_mixture() method.")

    wants_2d_render = show_2d and (not args.no_show)
    wants_2d_save = show_2d and (not args.no_save)
    wants_3d_render = show_3d and (not args.no_show)

    if show_3d and args.no_show:
        print("[WARN] --viz_mode includes 3d but --no_show is set. 3D visualization is disabled.")
    if (not show_2d) and (not args.no_save):
        print("[WARN] PNG saving is only available for 2D mode. Disabling save output.")
        wants_2d_save = False

    if not (wants_2d_render or wants_2d_save or wants_3d_render):
        raise ValueError(
            "No active output. Enable 2D rendering/saving or 3D rendering "
            "(do not combine all with --no_show and --no_save)."
        )

    save_dir = None
    if wants_2d_save:
        if args.save_dir:
            save_dir = args.save_dir
        else:
            ts = time.strftime("%y-%m-%d_%H-%M-%S", time.gmtime())
            logs_root = cfg.get("train_params", {}).get("logs_save_dir", "./logs")
            save_dir = os.path.join(logs_root, f"test_visualization_{ts}")
        os.makedirs(save_dir, exist_ok=True)

    t_out = int(cfg["model_params"]["forecast_horizon"])
    h = int(cfg["model_params"]["grid_height"])
    w = int(cfg["model_params"]["grid_width"])
    horizon_labels = get_horizon_labels(cfg, t_out)

    fig = None
    im_handles = None
    cb_handles = None
    if show_2d and (wants_2d_render or wants_2d_save):
        fig, im_handles, cb_handles = init_figure(plt, t_out, h, w, horizon_labels)

    o3d = None
    vis = None
    pcd_gt = None
    pcd_pred = None
    phi_grid = None
    theta_grid = None
    if wants_3d_render:
        try:
            import open3d as o3d  # type: ignore
        except ImportError as exc:
            raise RuntimeError("3D visualization requires open3d. Install it or switch to --viz_mode 2d.") from exc

        if args.pc_horizon_idx < 0 or args.pc_horizon_idx >= t_out:
            raise ValueError(f"--pc_horizon_idx must be in [0, {t_out - 1}], got {args.pc_horizon_idx}.")

        theta_range = get_theta_range(cfg)
        phi_grid_t, theta_grid_t = make_angle_grids(h, w, theta_range, device="cpu")
        phi_grid = phi_grid_t.cpu().numpy()
        theta_grid = theta_grid_t.cpu().numpy()

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="LiDAR 3D | GT (gray) vs Pred (blue)", width=1280, height=720)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_pred = o3d.geometry.PointCloud()
        vis.add_geometry(pcd_gt)
        vis.add_geometry(pcd_pred)

        pcd_gt.paint_uniform_color([0.65, 0.65, 0.65])
        pcd_pred.paint_uniform_color([0.10, 0.55, 0.95])

        render_opt = vis.get_render_option()
        if render_opt is not None:
            render_opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            render_opt.point_size = 2.5
    first_3d_fit_done = False
    empty_3d_frames = 0

    print("=== Test Visualization ===")
    print(f"Config           : {args.cfg_path}")
    print(f"Split used       : {split_used}")
    print(f"Weights          : {weight_path}")
    print(f"use_mdn          : {use_mdn}")
    print(f"Visualization    : {mode}")
    print(f"Viz mode source  : {mode_source}")
    print(
        f"Viz loader       : batch_size={getattr(eval_loader, 'batch_size', '?')} "
        f"num_workers={getattr(eval_loader, 'num_workers', '?')}"
    )
    print("Run mode         : full split inference")
    print(f"Max saved visuals: {args.max_save_samples}")
    print(f"Save dir         : {save_dir if save_dir else '(disabled)'}")

    dataset = getattr(eval_loader, "dataset", None)
    batch_size = int(getattr(eval_loader, "batch_size", cfg.get("train_params", {}).get("batch_size", 1)))

    processed = 0
    saved = 0
    visualized_2d = 0
    visualized_3d = 0
    mae_sum = 0.0
    mae_count = 0
    infer_times_ms = []

    try:
        with torch.no_grad():
            for batch_idx, (hist_xyzd, _, future_ranges) in enumerate(tqdm(eval_loader, total=len(eval_loader))):
                hist_xyzd = hist_xyzd.to(args.device)
                future_ranges = future_ranges.to(args.device)

                if hist_xyzd.shape[2] == 4:
                    hist_in = hist_xyzd[:, :, 3:4, :, :]
                elif hist_xyzd.shape[2] == 1:
                    hist_in = hist_xyzd
                else:
                    raise ValueError(f"Unexpected input channels from dataloader: {hist_xyzd.shape}")

                t0 = time.perf_counter()
                output = model(hist_in)
                infer_times_ms.append((time.perf_counter() - t0) * 1000.0)

                if use_mdn:
                    mixture, ok = model.build_mixture(cfg, output)
                    if not ok:
                        print(f"[WARN] Skipping batch {batch_idx}: invalid mixture output.")
                        continue
                    bsz, tout, hh, ww = future_ranges.shape
                    pred_ranges = mixture.mean.view(bsz, tout, hh, ww)
                else:
                    pred_ranges = output
                    if pred_ranges.ndim == 5 and pred_ranges.shape[2] == 1:
                        pred_ranges = pred_ranges[:, :, 0]
                    elif pred_ranges.ndim == 5 and pred_ranges.shape[-1] == 1:
                        pred_ranges = pred_ranges[..., 0]
                    elif pred_ranges.ndim == 5:
                        pred_ranges = pred_ranges[..., 0]
                    if pred_ranges.ndim != 4:
                        raise ValueError(f"Unexpected non-MDN output shape: {tuple(output.shape)}")

                gt_all = future_ranges.detach().cpu().numpy()
                pred_all = pred_ranges.detach().cpu().numpy()

                for b in range(gt_all.shape[0]):
                    processed += 1

                    gt_sample = gt_all[b].astype(np.float32)
                    pred_sample = pred_all[b].astype(np.float32)
                    valid = gt_sample > 0.0
                    err_sample = np.abs(pred_sample - gt_sample)

                    if valid.any():
                        mae_sum += float(err_sample[valid].sum())
                        mae_count += int(valid.sum())

                    dataset_idx = batch_idx * batch_size + b
                    seq_id, window_start = resolve_sample_meta(dataset, dataset_idx)

                    if show_2d:
                        vmax = compute_vmax(cfg, gt_sample, args.vmax_percentile)
                        for t in range(t_out):
                            gt_img = np.where(valid[t], gt_sample[t], np.nan)
                            pred_img = np.where(valid[t], pred_sample[t], np.nan)
                            err_img = np.where(valid[t], err_sample[t], np.nan)

                            im_handles[t][0].set_data(gt_img)
                            im_handles[t][1].set_data(pred_img)
                            im_handles[t][2].set_data(err_img)

                            for c in range(3):
                                im_handles[t][c].set_clim(0.0, vmax)
                                cb_handles[t][c].update_normal(im_handles[t][c])

                        fig.suptitle(
                            f"Split={split_used} | seq={seq_id} | window_start={window_start} | "
                            f"sample={processed - 1} | color_scale=[0,{vmax:.2f}] m",
                            fontsize=11,
                        )

                        if save_dir is not None and saved < args.max_save_samples:
                            safe_seq = str(seq_id).replace("/", "_")
                            out_path = os.path.join(
                                save_dir,
                                f"sample_{processed - 1:05d}_seq_{safe_seq}_start_{window_start}.png",
                            )
                            fig.savefig(out_path, dpi=args.dpi)
                            saved += 1

                        if wants_2d_render:
                            fig.canvas.draw_idle()
                            plt.pause(max(0.001, args.interval_ms / 1000.0))
                            visualized_2d += 1

                    if wants_3d_render:
                        t_idx = args.pc_horizon_idx
                        gt_pts = range_image_to_points(
                            gt_sample[t_idx], phi_grid, theta_grid, valid_mask=valid[t_idx], max_points=args.pc_max_points
                        )
                        pred_pts = range_image_to_points(
                            pred_sample[t_idx], phi_grid, theta_grid, valid_mask=None, max_points=args.pc_max_points
                        )

                        if gt_pts.shape[0] == 0 and pred_pts.shape[0] == 0:
                            empty_3d_frames += 1
                            if empty_3d_frames <= 5:
                                print(
                                    f"[WARN][3D] Empty clouds at sample={processed - 1} "
                                    f"(seq={seq_id}, start={window_start}, horizon={t_idx})."
                                )
                            continue

                        pcd_gt.points = o3d.utility.Vector3dVector(gt_pts.astype(np.float64))
                        pcd_pred.points = o3d.utility.Vector3dVector(pred_pts.astype(np.float64))
                        pcd_gt.paint_uniform_color([0.65, 0.65, 0.65])
                        pcd_pred.paint_uniform_color([0.10, 0.55, 0.95])

                        vis.update_geometry(pcd_gt)
                        vis.update_geometry(pcd_pred)
                        vis.poll_events()
                        vis.update_renderer()
                        if not first_3d_fit_done:
                            configure_open3d_camera(vis, gt_pts, pred_pts)
                            first_3d_fit_done = True
                        visualized_3d += 1
    finally:
        if vis is not None:
            vis.destroy_window()

    mean_mae = (mae_sum / max(mae_count, 1))
    mean_infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else float("nan")

    print("\n=== Done ===")
    print(f"Processed samples   : {processed}")
    print(f"Saved count         : {saved}")
    print(f"Visualized 2D       : {visualized_2d}")
    print(f"Visualized 3D       : {visualized_3d}")
    print(f"Mean absolute error : {mean_mae:.4f} m (on valid GT pixels)")
    print(f"Mean inference time : {mean_infer_ms:.3f} ms per batch")
    if save_dir is not None:
        print(f"Saved directory     : {save_dir}")

    if wants_2d_render:
        plt.ioff()
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run trained weights on eval split and visualize GT, prediction and absolute error."
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/home/devuser/workspace/src/configs/semanticKitti_default.yaml", #thab_default, semanticKitti_default
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional checkpoint path override. Falls back to train_params.pre_train_weights.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataloader_device", type=str, default="cpu")
    parser.add_argument("--max_save_samples", type=int, default=15)
    parser.add_argument("--vmax_percentile", type=float, default=99.0)
    parser.add_argument("--interval_ms", type=float, default=300.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--viz_mode", type=str, default="2d", choices=["2d", "3d", "both"])
    parser.add_argument(
        "--viz_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size for visualization stream (smaller = smoother).",
    )
    parser.add_argument(
        "--viz_num_workers",
        type=int,
        default=0,
        help="Dataloader workers for visualization stream.",
    )
    parser.add_argument("--pc_horizon_idx", type=int, default=0, help="Forecast horizon index to render in 3D.")
    parser.add_argument(
        "--pc_max_points",
        type=int,
        default=120000,
        help="Max points per cloud in 3D mode (0 or negative disables downsampling).",
    )
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
