import argparse
import os
import random
import time
from typing import Optional, Tuple

import matplotlib
import numpy as np
import torch
import yaml
from tqdm import tqdm

from helper.dataloader_helper import build_dataloaders, make_sequences
from models import build_model

# Optional: set a fixed weights path directly in this script.
# Used as fallback if --weights and cfg train_params.pre_train_weights are not set.
DEFAULT_WEIGHTS_PATH = "/home/devuser/workspace/LidarGaussianVideoView/logs/SemanticTHAB ohne Ray/weights/model_final.pt"


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


def main(args):
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.cfg_path, "r") as file:
        cfg = yaml.safe_load(file)

    apply_seed_and_determinism(cfg)
    print_runtime_info(cfg, args.device)

    model_name = cfg["model_params"].get("name", "swin")
    model = build_model(model_name, cfg)

    eval_loader, split_used = select_eval_loader(cfg, args.dataloader_device, args.split)
    weight_path = load_pretrained_weights(model, cfg, args.weights)

    model = model.to(args.device)
    model.eval()

    use_mdn = bool(cfg["model_params"].get("use_mdn", True))
    if use_mdn and not hasattr(model, "build_mixture"):
        raise ValueError("use_mdn=True but selected model has no build_mixture() method.")

    if args.no_show and args.no_save:
        raise ValueError("At least one output mode is required. Disable either --no_show or --no_save.")

    save_dir = None
    if not args.no_save:
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
    if (not args.no_show) or (not args.no_save):
        fig, im_handles, cb_handles = init_figure(plt, t_out, h, w, horizon_labels)

    print("=== Test Visualization ===")
    print(f"Config           : {args.cfg_path}")
    print(f"Split used       : {split_used}")
    print(f"Weights          : {weight_path}")
    print(f"use_mdn          : {use_mdn}")
    print("Run mode         : full split inference")
    print(f"Max saved visuals: {args.max_save_samples}")
    print(f"Save dir         : {save_dir if save_dir else '(disabled)'}")

    dataset = getattr(eval_loader, "dataset", None)
    batch_size = int(getattr(eval_loader, "batch_size", cfg.get("train_params", {}).get("batch_size", 1)))

    processed = 0
    saved = 0
    visualized = 0
    mae_sum = 0.0
    mae_count = 0
    infer_times_ms = []

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

                vmax = compute_vmax(cfg, gt_sample, args.vmax_percentile)
                dataset_idx = batch_idx * batch_size + b
                seq_id, window_start = resolve_sample_meta(dataset, dataset_idx)

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

                if not args.no_show:
                    fig.canvas.draw_idle()
                    plt.pause(max(0.001, args.interval_ms / 1000.0))
                    visualized += 1

    mean_mae = (mae_sum / max(mae_count, 1))
    mean_infer_ms = float(np.mean(infer_times_ms)) if infer_times_ms else float("nan")

    print("\n=== Done ===")
    print(f"Processed samples   : {processed}")
    print(f"Saved count         : {saved}")
    print(f"Visualized samples  : {visualized}")
    print(f"Mean absolute error : {mean_mae:.4f} m (on valid GT pixels)")
    print(f"Mean inference time : {mean_infer_ms:.3f} ms per batch")
    if save_dir is not None:
        print(f"Saved directory     : {save_dir}")

    if not args.no_show:
        plt.ioff()
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Run trained weights on eval split and visualize GT, prediction and absolute error."
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/home/devuser/workspace/src/configs/thab_default.yaml", #thab_default, semanticKitti_default
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
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
