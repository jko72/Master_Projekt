import os
# solves issue if wrong display is attached "FigureCanvasAgg is non-interactive, and thus cannot be shown"
os.environ['DISPLAY'] = ':0'

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Normal, MixtureSameFamily

#from dataloader import AlignedSeqDataset, RandomWindowSeqDataset #AlignedProjDataset
from utils_torch import make_angle_grids
from prob import build_range_mixture_distribution, compute_nll_range_loss
from prob import generate_point_clouds_from_mixture, visualize_mixture_pdfs
from models import build_model
from models.loss import Loss
from models.chamfer import cham_dist  # NEU: Chamfer-Metrik


# Helper functions
from helper.pointcloud_visualization import pointcloud_from_expected_range
from helper.dataloader_helper import make_sequences, build_dataloaders

import torch.optim as optim
import cv2
import copy
import open3d as o3d
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter #Visualisierug
import time

import threading
import multiprocessing as mp

import matplotlib
import math
#matplotlib.use('TkAgg')   # or 'Qt5Agg' if you have Qt
import matplotlib.pyplot as plt
import open3d as o3d

# ——— Global flags & state ———
show_pdf_flag = False
show_pc_flag  = False
show_ray_flag = False

pdf_figs   = []  # Matplotlib Figure handles
pcl_procs  = []  # Open3D subprocess handles
ray_procs  = []  # Ray‐viz subprocess handles


# Key‐press handler
def on_key_press(event):
    global show_pdf_flag, show_pc_flag, show_ray_flag
    key = event.key.lower()

    if key == 'p':
        show_pdf_flag = True
        show_pc_flag  = True
    elif key == 'r':
        show_ray_flag = True
    elif key == 'c':  # c key for clear subprocesses
        # Close all PDF figures
        for f in pdf_figs:
            try: plt.close(f)
            except: pass
        pdf_figs.clear()
        # Terminate all PCL subprocesses
        for p in pcl_procs:
            if p.is_alive(): p.terminate()
        pcl_procs.clear()
        # Terminate all Ray subprocesses
        for p in ray_procs:
            if p.is_alive(): p.terminate()
        ray_procs.clear()
        
    
def mixture_to_cpu(mixture: torch.distributions.MixtureSameFamily):
    cat = mixture.mixture_distribution
    comp = mixture.component_distribution

    cat_cpu = Categorical(probs=cat.probs.detach().cpu())
    comp_cpu= Normal(
        loc   = comp.loc.detach().cpu(),
        scale = comp.scale.detach().cpu()
    )
    return MixtureSameFamily(cat_cpu, comp_cpu)
    

def generate_ray_samples_from_mixture(
    cfg,
    mixture_cpu: torch.distributions.MixtureSameFamily,
    b: int, t: int, j: int,
    phi_grid: torch.Tensor,
    theta_grid: torch.Tensor,
    N_r: int = 50,
    device: str = "cpu"
):
    """
    Draws N_r samples for each vertical pixel i=0..H-1 along column j,
    according to the mixture distribution at (b,t,i,j).
        
    Returns:
        pts_np  : [H*N_r, 3] sampled points in 3D
        vals_np : [H*N_r]   corresponding PDF values
    """
    from torch.distributions import Categorical, Normal, MixtureSameFamily

    B = cfg["train_params"]["batch_size"]
    T = cfg["model_params"]["forecast_horizon"]
    H = cfg["model_params"]["grid_height"]
    W = cfg["model_params"]["grid_width"]
    K = cfg["model_params"]["mdn_num_gaussians"]
    
    # Draw N_r samples for *every* flattened distribution [B*T*H*W]
    samples = mixture_cpu.sample((N_r,))                  # [N_r, B*T*H*W]
    logp    = mixture_cpu.log_prob(samples)               # [N_r, B*T*H*W]
    pdf_all = torch.exp(logp)                             # [N_r, B*T*H*W]

    sample_pts = []
    sample_vals= []
    # for each vertical index i, pick the N_r samples for that pixel
    for i in range(H):
        # compute flattened index
        idx = (b * T + t) * (H * W) + (i * W) + j

        r_samp   = samples[:, idx].numpy()    # [N_r]
        pdf_samp = pdf_all[:, idx].numpy()    # [N_r]

        # direction vector from phi/theta
        phi   = float(phi_grid[i, j].item())
        theta = float(theta_grid[i, j].item())
        dir_vec = np.array([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            np.sin(theta)
        ], dtype=np.float32)                 # [3]

        # build the actual 3D points
        pts_ij = r_samp[:, None] * dir_vec[None, :]  # [N_r, 3]
        sample_pts.append(pts_ij)
        sample_vals.append(pdf_samp)

    # stack them
    pts_np  = np.vstack(sample_pts)    # [H*N_r, 3]
    vals_np = np.concatenate(sample_vals)  # [H*N_r]
    return pts_np, vals_np


def _launch_ray_proc(
    gt_pts_np: np.ndarray,
    ray_pts_np: np.ndarray,
    ray_vals_np: np.ndarray,
    H: int, N_r: int,
    title: str
):
    import open3d as o3d
    from matplotlib import cm

    # 1) Ground-truth cloud in grey
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts_np)
    pcd_gt.paint_uniform_color([0.7, 0.7, 0.7])

    # 2) Ray samples colored by confidence
    vals = ray_vals_np.reshape(H, N_r)
    # normalize each row independently
    vmin = vals.min(axis=1, keepdims=True)
    vmax = vals.max(axis=1, keepdims=True)
    norm = (vals - vmin) / (vmax - vmin + 1e-8)
    # flatten back to [H*N_r]
    norm = norm.reshape(-1)

    pcd_ray = o3d.geometry.PointCloud()
    pcd_ray.points = o3d.utility.Vector3dVector(ray_pts_np)
    #norm = (ray_vals_np - ray_vals_np.min()) / (ray_vals_np.max() - ray_vals_np.min() + 1e-8)
    cols = cm.get_cmap("turbo")(norm)[:, :3]
    pcd_ray.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))

    # 3) Visualize together
    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_ray],
        window_name=title,
        width=800, height=600
    )

# # Helper: visualize a point cloud using Open3D
# def visualize_open3d(points: torch.Tensor, title: str = "Open3D Point Cloud"):
#     """
#     Display Nx3 tensor points in an Open3D window.
#     """
#     # Convert to numpy
#     pts_np = points.detach().cpu().numpy()
#     pcl = o3d.geometry.PointCloud()
#     pcl.points = o3d.utility.Vector3dVector(pts_np)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title)
#     vis.add_geometry(pcl)
#     vis.run()  # blocks until window closed
#     vis.destroy_window()


def compute_conf_intervals(mixture, levels, n_samples=1000):
    """
    Approximate quantiles of a 1D mixture by sampling.
    Args:
        mixture : MixtureSameFamily with batch_shape [B*T*H*W]
        levels  : list of floats in (0,1), e.g. [0.68, 0.95]
        n_samples: int, number of samples per mixture to draw
    Returns:
        dict level -> Tensor([B*T*H*W]) of approximate quantile values
    """
    # Draw samples: shape [n_samples, batch_flat]
    samples = mixture.sample((n_samples,))  
    intervals = {}
    for lvl in levels:
        # The .quantile call takes a q in [0,1] on the sample‐dimension
        intervals[lvl] = samples.quantile(lvl, dim=0)
    return intervals

def to_uint8(img):
    lo, hi = img.min(), img.max()
    return ((img - lo)/(hi - lo + 1e-6)*255).astype(np.uint8)

def estimate_mixture_modes(mixture, n_samples=500):
    """
    Approximate the mode of each 1D mixture by sampling.

    Args:
      mixture   : MixtureSameFamily with batch_shape [N]
      n_samples : number of samples per mixture to draw

    Returns:
      modes : Tensor of shape [N] with approximate mode for each mixture
    """
    # Draw samples: shape [n_samples, N]
    samples = mixture.sample((n_samples,))  
    # Evaluate log-prob at each sample: [n_samples, N]
    logp = mixture.log_prob(samples)
    # Find index of max log-prob per mixture
    mode_idx = torch.argmax(logp, dim=0)  # [N]
    # Gather the corresponding sample as the mode
    outputs_mode = samples[mode_idx, torch.arange(samples.shape[1])]
    #outputs_mode = torch.gather(samples, dim=0, index=mode_idx.unsqueeze(0)).squeeze(0)
    return outputs_mode

def print_training_env_info(cfg):
    import torch
    tp = cfg.get("train_params", {})
    seed = tp.get("random_seed", None)
    deterministic = tp.get("deterministic", False)
    shuffle_train = tp.get("shuffle_train", True)

    print("\n===== TRAINING ENVIRONMENT INFO =====")
    print(f"Random Seed:          {seed if seed is not None else 'None (random each run)'}")
    print(f"Deterministic Mode:   {'ON' if deterministic else 'OFF'}")
    print(f"Train Shuffle:        {'ON' if shuffle_train else 'OFF'}")
    print(f"CuDNN benchmark:      {torch.backends.cudnn.benchmark}")
    print(f"CuDNN deterministic:  {torch.backends.cudnn.deterministic}")
    print("=====================================\n")

def main(args):
    global show_pc_flag, show_pdf_flag, show_ray_flag
    with open(args.cfg_path) as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("[ERROR] Could not load config:", exc)
            return

    # --- NEU: globales Seeding & Determinismus ---
    seed = cfg.get('train_params', {}).get('random_seed', None)
    deterministic = cfg.get('train_params', {}).get('deterministic', False)
    cudnn_benchmark = cfg.get('train_params', {}).get('cudnn_benchmark', False)

    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = False

    # NEU: Ausgabe der aktiven Einstellungen
    print_training_env_info(cfg)

    from models import build_model
    name = cfg["model_params"].get("name", "swin")   # "swin" | "acc_m1" | "acc_m2"
    model = build_model(name, cfg)

    all_seqs = make_sequences(cfg["dataset_path"])

    # --- Split-Config aus YAML lesen ---
    data_cfg = cfg.get("data_params", {})
    split_type = data_cfg.get("split_type", "rotary")
    predefined_splits = data_cfg.get("predefined_splits", None)

    if split_type == "rotary":
        rotary_loaders = build_dataloaders(
            all_seqs,
            cfg,
            args.dataloader_device,
            split_type='rotary'
        )  # list of (holdout_id, train_loader, val_loader)
        n_splits = len(rotary_loaders)

        # Für Scheduler: min. Länge aller Trainingsloader
        steps_per_epoch = min(len(tl) for (_, tl, _) in rotary_loaders)

        print(f"[DATA] Using ROTARY split with {n_splits} folds.")

    elif split_type == "predefined":
        # Splits müssen in data_params.predefined_splits vorhanden sein
        if predefined_splits is None:
            raise ValueError(
                "split_type='predefined', aber data_params.predefined_splits fehlt in der Config!"
            )

        loaders = build_dataloaders(
            all_seqs,
            cfg,
            args.dataloader_device,
            split_type='predefined',
            predefined_splits=predefined_splits
        )

        # Kann (train, val) oder (train, val, test) zurückgeben
        if len(loaders) == 3:
            train_loader, val_loader, test_loader = loaders
            print(f"[DATA] Using PREDEFINED split with TRAIN/VAL/TEST.")
        else:
            train_loader, val_loader = loaders
            test_loader = None
            print(f"[DATA] Using PREDEFINED split with TRAIN/VAL.")

        steps_per_epoch = len(train_loader)

    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    # model definition
    # from torchvision baseline "Video Classification" models, see https://pytorch.org/vision/main/models.html#video-classification
    #model = RangeMixtureVideoModel(cfg)

    # load weights
    # ---- Load pretrained weights (.pt or .ckpt) --------------------------------
    pre_path = cfg["train_params"].get("pre_train_weights", None)

    if pre_path:
        try:
            if os.path.isfile(pre_path):
                print(f"[CKPT] Loading pretrained weights from: {pre_path}")

                ckpt = torch.load(pre_path, map_location="cpu")

                # 1) get raw state_dict
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    state = ckpt["state_dict"]          # Lightning .ckpt
                    print(f"[CKPT] Detected Lightning checkpoint with {len(state)} tensors.")
                elif isinstance(ckpt, dict):
                    state = ckpt                        # plain state_dict saved as dict
                    print(f"[CKPT] Detected plain state_dict with {len(state)} tensors.")
                else:
                    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

                # 2) strip common prefixes (model. / module.)
                clean_state = {}
                for k, v in state.items():
                    # strip lightning / ddp prefixes
                    if k.startswith("model."):
                        k = k[len("model."):]
                    if k.startswith("module."):
                        k = k[len("module."):]
    
                    # IMPORTANT: map ACC core weights into adapter
                    # Checkpoint: enc.enc.*  -> Model: acc.enc.enc.*
                    if k.startswith(("enc.", "hid.", "dec.")):
                        k = "acc." + k

                    clean_state[k] = v

                    if k == "mean":
                        clean_state["acc.mean"] = v
                    elif k == "std":
                        clean_state["acc.std"] = v

                # 3) drop keys with incompatible shapes (PyTorch would crash otherwise)
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

                # optional: print a few names to sanity-check
                if missing:
                    print("[CKPT] Example missing keys:", missing[:10])
                if unexpected:
                    print("[CKPT] Example unexpected keys:", unexpected[:10])
            else:
                print(f"[CKPT] pre_train_weights path not found: {pre_path}")
        except Exception as ex:
            print(f"[CKPT] Failed to load pretrained weights: {ex}")
    else:
        print("[CKPT] No pretrained weights set (pre_train_weights is None).")
    # ---------------------------------------------------------------------------

    # pre_path = cfg["train_params"].get("pre_train_weights", None)

    # if pre_path and os.path.isfile(pre_path):
    #     print(f"[CKPT] Loading pretrained weights from: {pre_path}")
    #     ckpt = torch.load(pre_path, map_location="cpu")

    #     if isinstance(ckpt, dict) and "state_dict" in ckpt:
    #         state = ckpt["state_dict"]
    #         print("[CKPT] Detected PyTorch-Lightning checkpoint (using 'state_dict').")
    #     elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    #         state = ckpt["model_state_dict"]
    #         print("[CKPT] Using 'model_state_dict'.")
    #     else:
    #         state = ckpt
    #         print("[CKPT] Using checkpoint as plain state_dict.")

    #     clean_state = {}
    #     for k, v in state.items():
    #         if k.startswith("model."):
    #             k = k[len("model."):]
    #         if k.startswith("module."):
    #             k = k[len("module."):]
    #         clean_state[k] = v

    #     missing, unexpected = model.load_state_dict(clean_state, strict=False)

    #     print(f"[CKPT] Loaded weights  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    #     if missing:
    #         print(f"[CKPT] Example missing: {missing[:10]}")
    #     if unexpected:
    #         print(f"[CKPT] Example unexpected: {unexpected[:10]}")
    # else:
    #     print("[CKPT] No pretrained weights set/found -> training from scratch")        

    # try:
    #     if os.path.isfile(cfg["train_params"]["pre_train_weights"]):    # throws TypeError if NoneType provided
    #         weights = torch.load(cfg["train_params"]["pre_train_weights"])
    #         model.load_state_dict(weights)
    # except Exception as ex:
    #         print("no custom pretrained weights found, use default vanilla")
    model.to(args.device)
    criterion = Loss(cfg)
    # NEU: Chamfer-Metrik (nur als Metrik, nicht im Trainings-Loss)
    chamfer_metric = cham_dist(cfg)

    
    # Define optimizer
    #optimizer = optim.Adam(model.parameters(), lr=cfg["train_params"]["start_learning_rate"])
    
    # Optimizer: AdamW with decoupled weight decay, 
    # see https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1

    optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg["train_params"].get("learning_rate", 
                               cfg["train_params"].get("start_learning_rate", 5e-4)),
    weight_decay=cfg["train_params"].get("weight_decay", 1e-4)
    )
    
    # ------------------------------------------------------------
    # Per-Batch Scheduler: Warmup + Cosine Decay (LambdaLR)
    # ------------------------------------------------------------
    num_epochs = cfg["train_params"].get("num_total_epochs",
                                        cfg["train_params"].get("num_epochs", 50))

    # Zahl der Training-Batches pro Epoch aus Rotary-Splits
    #steps_per_epoch = min(len(tl) for (_, tl, _) in rotary_loaders)
    total_steps = max(1, num_epochs * steps_per_epoch)

    warmup_epochs = cfg["train_params"].get("num_warmup_epochs", 2)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = cfg["train_params"].get("learning_rate_min", 5e-6)
    warmup_start = 0.3  # 30% des Start-LR

    def lr_lambda(step: int):
        # 1) Warmup linear
        if step < warmup_steps:
            return warmup_start + (1.0 - warmup_start) * (step / warmup_steps)
        # 2) Cosine Decay
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return (eta_min / base_lr) + (1 - eta_min / base_lr) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # TensorBoard
    if cfg["train_params"]["with_save"]:
        t = time.gmtime()
        time_start = time.strftime("%y-%m-%d_%H-%M-%S", t)  # Changed format to avoid colons
        save_path = os.path.join(cfg["train_params"]["logs_save_dir"], time_start)
        os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
        writer = SummaryWriter(save_path)
        #save_path ="/home/devuser/workspace/LidarGaussianVideoView/logs"
        # save current training config file
        with open(os.path.join(save_path, "config.yaml"), "w") as file:
            yaml.safe_dump(
                cfg, 
                file,
                default_flow_style=False,  # use block style (indented) rather than inline
                sort_keys=False           # preserve the order in your dict, if PyYAML ≥5.1
            )
    
    # Image dimensions
    H, W = cfg["model_params"]["grid_height"], cfg["model_params"]["grid_width"]
    if cfg["train_params"]["plot_examples"]:
        # Enable interactive mode
        plt.ion()
        
        flip = cfg["train_params"].get("plot_time_vertically", True)
        T = cfg["model_params"]["forecast_horizon"]
        horizons = cfg["train_params"]["output_horizons"]
        
        if not flip:
            # rows = [GT, Mean, Mode], cols = time
            n_rows, n_cols = 3, T
            row_titles = ["Ground-Truth", "Predicted Mean", "Predicted Mode"]
            col_titles = [f"t+{h}" for h in horizons]
        else:
            # rows = time, cols = [GT,Mean,Mode]
            n_rows, n_cols = T, 3
            row_titles = [f"t+{h}" for h in horizons]
            col_titles = ["Ground-Truth", "Predicted Mean", "Predicted Mode"]
        
        # compute a sensible "per–subplot" size in inches
        # so that height_per = 2in gives width_per = 2in * (W/H)
        height_per = 2.0
        aspect = W / H                    # e.g. 512/64 = 8
        width_per  = height_per * aspect  # with 512w, 64h -> 16 inches
        fig_width  = width_per  * n_cols  # e.g. 16 * 3 = 48
        fig_height = height_per * n_rows  # e.g. 2  * 3 =  6
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            constrained_layout=True
        )
        axes = np.array(axes).reshape(n_rows, n_cols)

        im_handles = [[None]*n_cols for _ in range(n_rows)]
        cb_handles = [[None]*n_cols for _ in range(n_rows)]

        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                cmap = plt.get_cmap('turbo').copy()
                cmap.set_bad(color='black')  # invalid pixels (NaN) shown as black
                im = ax.imshow(
                    np.full((H, W), np.nan, dtype=np.float32),
                    aspect='equal',
                    vmin=0, vmax=50,
                    cmap=cmap
                )
                # title and subtitle
                title, subtitle = row_titles[i], col_titles[j]
                ax.set_title(f"{title}  ({subtitle})")
                
                ax.axis("off")
                cb = fig.colorbar(im, ax=ax,
                                fraction=0.046, pad=0.04,
                                label="Distance (m)")
                im_handles[i][j] = im
                cb_handles[i][j] = cb
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Option 2: predefined splits
    # splits = {
    #     'train': ['0001', '0002', '0003'],
    #     'val': ['0004'],
    #     'test': ['0005']
    # }
    # result = build_dataloaders(
    #     all_seqs,
    #     cfg,
    #     args.dataloader_device,
    #     split_type='predefined',
    #     predefined_splits=splits
    # )
    # # Unpack based on presence of test split
    # if len(result) == 3:
    #     train_loader, val_loader, test_loader = result
    # else:
    #     train_loader, val_loader = result
    # print("Predefined split loaders ready.")
        global_step = 0  # falls noch nicht vorher definiert

    for epoch in range(cfg["train_params"]["num_total_epochs"]):

        if split_type == "rotary":
            # Leave-one-out pro Epoch
            split_idx = epoch % n_splits
            holdout_id, train_loader, val_loader = rotary_loaders[split_idx]
            print(
                f"Epoch {epoch+1}/{cfg['train_params']['num_total_epochs']}, "
                f"training on sequence {holdout_id} (ROTARY split)"
            )
        else:
            # PREDEFINED: immer gleiche Loader
            holdout_id = None
            
            print(
            f"\nEpoch {epoch+1}/{cfg['train_params']['num_total_epochs']} - "
            f"TRAIN on {predefined_splits['train']}  |  VAL on {predefined_splits['val']}"
            )

        total_loss = 0.0
        total_loss_val = 0.0
        t_prev = time.perf_counter()
        #batch = next(iter(train_loader))
        #hist_xyzd, future_xyz, future_ranges = batch

        #print("hist_xyzd:", hist_xyzd.shape)        # erwartet [B, T, C, H, W] torch.Size([8, 10, 4, 64, 512]
        #print("future_xyz:", future_xyz.shape)      # [B, F, 3, H, W] torch.Size([8, 5, 3, 64, 512]
        #print("future_ranges:", future_ranges.shape)# [B, F, H, W] torch.Size([8, 5, 64, 512]
        #print("C (input channels) =", hist_xyzd.shape[2]) #C (input channels) = 4

        # --- Training Loop ---
        for batch_idx, (hist_xyzd, future_xyz, future_ranges) in enumerate(tqdm(iterable=train_loader, total=len(train_loader))):
            # hist_xyz      = [B, T_in,     3,  H, W], [B, T_in,     4,  H, W]
            # future_xyz    = [B, T_out,    3,  H, W]
            # future_ranges = [B, T_out,        H, W]
            t_now = time.perf_counter()
            dt_wait = t_now - t_prev
            if batch_idx in (0, 1, 2, 5, 10, 20, 50, 100):
                print(f"[DL WAIT] batch={batch_idx} wait={dt_wait:.3f}s")
            #if batch_idx % 50 == 0:
            #    print(f"[DL WAIT] batch={batch_idx} wait={dt_wait:.3f}s")
            t_prev = time.perf_counter()
        
            model.train()
            #model.eval()
            
            # model's forward gives "output" of shape [B,T,H,W,3K]
            hist_xyzd = hist_xyzd.to(args.device)
            # Range-only input: take range channel (index 3) -> [B, T, 1, H, W]
            if hist_xyzd.shape[2] == 4:
                hist_in = hist_xyzd[:, :, 3:4, :, :]
            elif hist_xyzd.shape[2] == 1:
                hist_in = hist_xyzd
            else:
                raise ValueError(f"Unexpected input channels from dataloader: {hist_xyzd.shape}")
            # ===================== DEBUG: preserve_ray sanity checks =====================
            if batch_idx == 0:
                hx = hist_xyzd.to(args.device)
                fr = future_ranges.to(args.device)

                # hist range channel (bei dir Kanal 3)
                h_range = hx[:, :, 3, :, :]  # [B,P,H,W]
                h_invalid = (h_range <= 0.0)
                fr_invalid = (fr <= 0.0)

                print("\n[DBG PRESERVE_RAY] ---- SANITY ----")
                print("[DBG] hist_range invalid ratio:", h_invalid.float().mean().item())
                print("[DBG] future_range invalid ratio:", fr_invalid.float().mean().item())

                # per-time invalid ratio
                per_t = []
                for t in range(fr.shape[1]):
                    per_t.append(fr_invalid[:, t].float().mean().item())
                print("[DBG] future invalid ratio per t:", [f"{x:.3f}" for x in per_t])

                # check: are there any zeros? (should be ~0 if you strictly use -1 for invalid)
                print("[DBG] future ratio == 0:", (fr == 0.0).float().mean().item())

                # range stats only on valid pixels
                valid_fr = fr[~fr_invalid]
                valid_h  = h_range[~h_invalid]

                if valid_fr.numel() > 0:
                    print("[DBG] future valid mean/std/min/max:",
                        valid_fr.mean().item(), valid_fr.std().item(),
                        valid_fr.min().item(), valid_fr.max().item())
                else:
                    print("[DBG] future valid: EMPTY (all invalid!) -> pipeline broken")

                if valid_h.numel() > 0:
                    print("[DBG] hist valid mean/std/min/max:",
                        valid_h.mean().item(), valid_h.std().item(),
                        valid_h.min().item(), valid_h.max().item())
                else:
                    print("[DBG] hist valid: EMPTY (all invalid!) -> pipeline broken")

                print("[DBG PRESERVE_RAY] --------------\n")
            # =================== END DEBUG: preserve_ray sanity checks ===================
            
            start_time = time.perf_counter()    # fractional time in seconds
            output = model(hist_in)
            curr_time = (time.perf_counter() - start_time) * 1000   # elapsed time in ms
            
            # build & compute 1D‐range loss
            # mixture, ok = build_range_mixture_distribution(cfg, output)
            if cfg["model_params"].get("use_mdn", True):
                mixture, ok = model.build_mixture(cfg, output)
                if not ok:
                    continue
                target = future_xyz.to(args.device)
                loss_dict = criterion(output, target, mode="train", epoch_number=epoch)

                loss_tensor = loss_dict["loss"]
                nll = loss_dict["loss_range_view"]
                valid_ratio = loss_dict["valid_ratio"]

            else:
                # Direkte Range-Regression ohne Gaußparameter
                loss_tensor = torch.nn.functional.l1_loss(output, future_ranges.to(args.device))
                nll = loss_tensor.item()            
            # add batch's train loss to overall loss
            total_loss += nll
                
                # Nur bei MDN aktiv – mixture existiert nur dann
            if cfg["model_params"].get("use_mdn", True):
                mixture_cpu = mixture_to_cpu(mixture)
            else:
                mixture_cpu = None

            #r_exp = mixture.mean
            # loss_occ = occlusion_penalty(r_exp, future_ranges)
            # loss_tensor = loss_tensor + loss_occ
            # loss_tensor = loss_tensor
            
            print(f"inference took {curr_time:.3f} ms.\tLR: {optimizer.param_groups[0]['lr']}\tloss: {nll:.3f}\t@Epoch {epoch+1}/{cfg['train_params']['num_total_epochs']}")
            
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            scheduler.step()

            B, T, H, W = future_ranges.shape
            # if ok:
            #     phi_grid, theta_grid = make_angle_grids(H, W, theta_range=[-np.pi/8, np.pi/8])
            
            # # pcs is a list of length B*F with [M,3] 3D points per time-step
            # # assume lidar sensor noise is sigma=0.3m
            # pcs = generate_point_clouds_from_mixture(
            #     mixture, B, T, H, W, phi_grid, theta_grid,
            #     alpha_threshold=cfg['train_params']['alpha_threshold'],p
            #     use_density_threshold=True,
            #     density_threshold= -0.5 * (np.log(2*np.pi) + 2 * np.log(0.3) )
            # )
    
            # VISUALIZATION
            if cfg["train_params"]["plot_examples"] and (batch_idx % cfg["train_params"]["plot_batch_step"] == 0):
                mp = cfg['model_params']
                if 'theta_range' in mp and mp['theta_range'] is not None:
                    theta_range = mp['theta_range']
                else:
                    fov_up = float(mp.get('FOV_UP', 3.0))
                    fov_down = float(mp.get('FOV_DOWN', -25.0))
                    theta_range = [fov_down * np.pi / 180.0, fov_up * np.pi / 180.0]
                phi_grid, theta_grid = make_angle_grids(H, W, theta_range, device="cpu")
                
                # Reshape to [B, T, H, W]
                B, T, H, W = future_ranges.shape
                
                # gather gt, mode, mean
                    # get gt
                gt_all = future_ranges.detach().cpu().numpy()

                    # --- Modellvorhersagen ---
                if cfg["model_params"].get("use_mdn", True):
                     # MDN aktiv → Mittelwert & Moden aus Mixture ziehen
                    modes_flat = estimate_mixture_modes(mixture, n_samples=cfg["train_params"]["num_samples"])  # [B*T*H*W]
                    modes_all = modes_flat.view(B, T, H, W).detach().cpu().numpy()

                    mean_flat = mixture.mean  # shape [B*T*H*W]
                    mean_all = mean_flat.view(B, T, H, W).detach().cpu().numpy()
                else:
                    # Direkte Range-Regression → Output selbst ist Mean & Mode
                    modes_all = output.detach().cpu().numpy()
                    mean_all  = output.detach().cpu().numpy()

                # VISUALIZATION
                b = 0   # which batch‐element to show
                flip = cfg["train_params"].get("plot_time_vertically", True)
                
                # Use GT valid pixels to set a stable, meaningful color range.
                valid_gt = gt_all[gt_all > 0.0]
                vmax = float(np.percentile(valid_gt, 99)) if valid_gt.size > 0 else 50.0
                vmax = max(5.0, min(vmax, 120.0))

                if flip:
                    # rows = time, cols = [GT,Mean,Mode]
                    for row in range(T):
                        gt_img = gt_all[b, row].astype(np.float32)
                        gt_img[gt_img <= 0.0] = np.nan
                        im_handles[row][0].set_data(gt_img)
                        im_handles[row][0].set_clim(0.0, vmax)
                        cb_handles[row][0].update_normal(im_handles[row][0])

                        mean_img = mean_all[b, row].astype(np.float32)
                        mean_img[mean_img <= 0.0] = np.nan
                        im_handles[row][1].set_data(mean_img)
                        im_handles[row][1].set_clim(0.0, vmax)
                        cb_handles[row][1].update_normal(im_handles[row][1])

                        mode_img = modes_all[b, row].astype(np.float32)
                        mode_img[mode_img <= 0.0] = np.nan
                        im_handles[row][2].set_data(mode_img)
                        im_handles[row][2].set_clim(0.0, vmax)
                        cb_handles[row][2].update_normal(im_handles[row][2])
                else:
                    # rows = [GT,Mean,Mode], cols=time
                    for col in range(T):
                        gt_img = gt_all[b, col].astype(np.float32)
                        gt_img[gt_img <= 0.0] = np.nan
                        im_handles[0][col].set_data(gt_img)
                        im_handles[0][col].set_clim(0.0, vmax)
                        cb_handles[0][col].update_normal(im_handles[0][col])

                        mean_img = mean_all[b, col].astype(np.float32)
                        mean_img[mean_img <= 0.0] = np.nan
                        im_handles[1][col].set_data(mean_img)
                        im_handles[1][col].set_clim(0.0, vmax)
                        cb_handles[1][col].update_normal(im_handles[1][col])

                        mode_img = modes_all[b, col].astype(np.float32)
                        mode_img[mode_img <= 0.0] = np.nan
                        im_handles[2][col].set_data(mode_img)
                        im_handles[2][col].set_clim(0.0, vmax)
                        cb_handles[2][col].update_normal(im_handles[2][col])
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
                
                # density_threshold = -0.5 * (np.log(2*np.pi) + 2 * np.log(2) )
                # # --- ON‐DEMAND PDF viz (MAIN THREAD) ---
                # if show_pdf_flag:
                #     # Now capture returned figures
                #     figs = visualize_mixture_pdfs(
                #         mixture, B, T, H, W,
                #         phi_grid, theta_grid,
                #         alpha_threshold=cfg['train_params']['alpha_threshold'],
                #         use_density_threshold=cfg['train_params']['use_density_threshold'],
                #         density_threshold=density_threshold,
                #         device=args.device
                #     )
                #     pdf_figs.extend(figs)           # store handles
                #     show_pdf_flag = False

                # --- ON‐DEMAND POINT‐CLOUD (SEPERATE PROCESS) ---
                if show_pc_flag:
                    K = cfg["model_params"]["mdn_num_gaussians"]
                    # 2) Compute expected range per pixel
                    r_exp_flat = mixture_cpu.mean           # [B*T*H*W]
                    r_exp = r_exp_flat.view(B, T, H, W)                # [B,T,H,W]

                    # 3) Extract alpha for confidence masking
                    alpha = mixture_cpu.mixture_distribution.probs        # [N,K]
                    alpha = alpha.view(B, T, H, W, K)               # [B,T,H,W,K]

                    # 4) Build the point clouds
                    pcs = pointcloud_from_expected_range(
                        r_exp, phi_grid, theta_grid,
                        alpha=alpha,
                        alpha_thresh=cfg["train_params"]["alpha_threshold"]
                    )

                    # 5) Visualize the first point-cloud of the batch
                    pts0 = pcs[0].cpu().numpy()  # batch=0, time=0
                    def _launch_pcl_proc(pts, title):
                        import open3d as o3d
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        o3d.visualization.draw_geometries([pcd], window_name=title)

                    proc = mp.Process(
                        target=_launch_pcl_proc,
                        args=(pts0, f"Batch {batch_idx} Exp-Range PC"),
                        daemon=True
                    )
                    proc.start()
                    pcl_procs.append(proc)
                    show_pc_flag = False
                    # # generate the cloud once
                    # theta_range = cfg['model_params'].get('theta_range', [-np.pi/8, np.pi/8])
                    # phi_grid, theta_grid = make_angle_grids(H, W, theta_range, device=args.device)
                    # pcs = generate_point_clouds_from_mixture(
                    #     mixture, B, T, H, W,
                    #     phi_grid, theta_grid,
                    #     alpha_threshold=cfg['train_params']['alpha_threshold'],
                    #     use_density_threshold=cfg['train_params']['use_density_threshold'],
                    #     density_threshold=cfg['train_params']['density_threshold']
                    # )
                    # pts0 = pcs[0].cpu().numpy()

                    # def _launch_pcl_proc(pts, title):
                    #     import open3d as o3d
                    #     pcd = o3d.geometry.PointCloud()
                    #     pcd.points = o3d.utility.Vector3dVector(pts)
                    #     o3d.visualization.draw_geometries([pcd], window_name=title)

                    # proc = mp.Process(
                    #     target=_launch_pcl_proc,
                    #     args=(pts0, f"Batch {batch_idx} Point Cloud"),
                    #     daemon=True
                    # )
                    # proc.start()
                    # pcl_procs.append(proc)
                    # show_pc_flag = False

                #  Ray‐confidence on R
                if show_ray_flag:
                    # flatten GT point cloud for (b,t)
                    b, t, j = 0, 0, np.random.choice(range(W))
                    N_r=50
                    # future_xyz    = [B, T_out,    3,  H, W]
                    gt_pts_flat = future_xyz[b, t].permute(1,2,0).reshape(-1,3).cpu().numpy()

                    # first compute samples & confidences in the main process
                    pts_np, vals_np = generate_ray_samples_from_mixture(
                        cfg, 
                        mixture_cpu, 
                        b, t, j,
                        phi_grid, 
                        theta_grid,
                        N_r=N_r,
                        device=args.device
                    )

                    # then launch the Open3D viz in its own subprocess
                    proc = mp.Process(
                        target=_launch_ray_proc,
                        args=(gt_pts_flat, pts_np, vals_np, H, N_r,f"Ray b{b} t{t} j{j}"),
                        daemon=True
                    )
                    proc.start()
                    ray_procs.append(proc)

                    show_ray_flag = False

            
                # # MIXTURE PDF VISUALIZATION ON DEMAND
                # if show_pdf_flag:
                #     theta_range = cfg['model_params'].get('theta_range', [-np.pi/8, np.pi/8])
                #     phi_grid, theta_grid = make_angle_grids(H, W, theta_range, device=args.device)
                #     visualize_mixture_pdfs(
                #         mixture, B, T, H, W,
                #         phi_grid, theta_grid,
                #         cfg['train_params']['alpha_threshold'],
                #         device=args.device
                #     )
                #     show_pdf_flag = False
                    
                # # OPEN3D POINT-CLOUD ON DEMAND
                # if show_pc_flag:
                #     # Build angle grids
                #     theta_range = cfg['model_params'].get('theta_range', [-np.pi/8, np.pi/8])
                #     phi_grid, theta_grid = make_angle_grids(H, W, theta_range, device=args.device)
                #     # Generate point clouds
                #     pcs = generate_point_clouds_from_mixture(
                #         mixture, B, T, H, W,
                #         phi_grid, theta_grid,
                #         alpha_threshold=cfg['train_params']['alpha_threshold']
                #     )
                #     # Visualize first timestep of batch 0
                #     visualize_open3d(pcs[0], title=f"Batch {batch_idx} Point Cloud")
                #     # Reset flag until next key press
                #     show_pc_flag = False
                    
                

                
            # Log to TensorBoard
            if cfg["train_params"]["with_save"]:
                if batch_idx % cfg["train_params"]["tensorboard_log_interval"] == 0:
                    step = epoch * len(train_loader) + batch_idx

                    # ===== PAPER-METRICS: TRAINING LOSSES =====
                    # Nur loggen, wenn loss_dict existiert (use_mdn=True & build_mixture ok)
                    if "loss_dict" in locals() and isinstance(loss_dict, dict):
                        if "loss" in loss_dict:
                            writer.add_scalar("train/loss_total", loss_dict["loss"].item(), step)
                        if "loss_range_view" in loss_dict:
                            writer.add_scalar("train/range_view_loss", loss_dict["loss_range_view"].item(), step)
                        if "mean_chamfer_distance" in loss_dict:
                            writer.add_scalar("train/chamfer_distance_loss", loss_dict["mean_chamfer_distance"].item(), step)
                    # ===== END PAPER-METRICS: TRAINING LOSSES =====

                    writer.add_scalar('Loss', nll, step)
                    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)

                    # Anteil gültiger Pixel
                    try:
                        if "loss_dict" in locals() and "valid_ratio" in loss_dict:
                            writer.add_scalar('train/valid_pixel_ratio', loss_dict["valid_ratio"].item(), step)
                        elif "valid_ratio" in locals():
                            writer.add_scalar('train/valid_pixel_ratio', valid_ratio.item(), step)
                    except Exception as e:
                        print(f"[WARN] valid_pixel_ratio not logged: {e}")
                    
                    # Add distance losses
                    if not (batch_idx % cfg["train_params"]["plot_batch_step"] == 0):
                        # Reshape to [B, T, H, W]
                        B, T, H, W = future_ranges.shape
                        
                        # gather gt, mode, mean
                            # get gt
                        gt_all = future_ranges.detach().cpu().numpy()
                            # get modes
                        modes_flat = estimate_mixture_modes(mixture, n_samples=cfg["train_params"]["num_samples"])  # [B*T*H*W]
                        modes_all = modes_flat.view(B, T, H, W).detach().cpu().numpy()
                            # get mean
                        mean_flat = mixture.mean  # shape [B*T*H*W]
                        mean_all = mean_flat.view(B, T, H, W).detach().cpu().numpy()
                        
                    # valid GT range pixels only (invalid are encoded as -1)
                    mask = (gt_all > 0)
                    diff_mean_mean = np.mean(np.abs(mean_all - gt_all)[mask])
                    diff_median_mean = np.median(np.abs(mean_all - gt_all)[mask])
                    
                    diff_mean_mode = np.mean(np.abs(modes_all - gt_all)[mask])
                    diff_median_mode = np.median(np.abs(modes_all - gt_all)[mask])
                    
                    writer.add_scalar('Range/Mean/Mean', diff_mean_mean, step)
                    writer.add_scalar('Range/Mean/Mode', diff_mean_mode, step)
                    writer.add_scalar('Range/Median/Mean', diff_median_mean, step)
                    writer.add_scalar('Range/Median/Mode', diff_median_mode, step)

        # average loss caluclation
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{cfg['train_params']['num_total_epochs']}, Average Training Loss: {avg_loss}")
            
        # save model weights with save interval
        if cfg["train_params"]["with_save"]:
            writer.add_scalar('Loss/Train/Epoch', avg_loss, epoch)
            if epoch % cfg["train_params"]["auto_save_step"] == 0:
                torch.save(model.state_dict(), os.path.join(save_path, "weights", f"model_epoch_{epoch}.pt"))
        
        # --- Validation Loop ---
        # ===== PAPER-METRICS: VALIDATION ACCUMULATORS =====
        val_rv_sum = 0.0          # Summe Range-View-Loss über Val-Batches
        val_cd_sum = 0.0          # Summe Chamfer-Distanzen
        val_mask_sum = 0.0        # Summe Masken-Loss
        val_batch_count = 0       # Anzahl Val-Batches
        val_abs_sum = 0.0     # Sum |pred-gt| über alle gültigen Pixel der ganzen Val-Epoche
        val_valid_sum = 0.0   # Sum gültige Pixel der ganzen Val-Epoche
        # ===== END PAPER-METRICS: VALIDATION ACCUMULATORS =====
        model.eval()
        with torch.no_grad():
            for batch_idx, (hist_xyzd, future_xyz, future_ranges) in enumerate(tqdm(iterable=val_loader, total=len(val_loader))):
                # model's forward gives "output" of shape [B,T,H,W,3K]
                hist_xyzd = hist_xyzd.to(args.device)
                # Range-only input: take range channel (index 3)
                if hist_xyzd.shape[2] == 4:
                    hist_in = hist_xyzd[:, :, 3:4, :, :]   # [B,T,1,H,W]
                elif hist_xyzd.shape[2] == 1:
                    hist_in = hist_xyzd
                else:
                    raise ValueError(f"Unexpected input channels from dataloader: {hist_xyzd.shape}")
                # Paper-KITTI Normalisierung (nur Range-Kanal)
                # range_mean = 10.839
                # range_std  = 9.314

                # hist_in = (hist_in - range_mean) / (range_std + 1e-8)

                # target = future_xyz.to(args.device).clone()

                # x_mean, x_std = 0.005, 11.521
                # y_mean, y_std = 0.494, 8.262
                # z_mean, z_std = -1.13, 0.828

                # target[:, :, 0] = (target[:, :, 0] - x_mean) / (x_std + 1e-8)
                # target[:, :, 1] = (target[:, :, 1] - y_mean) / (y_std + 1e-8)
                # target[:, :, 2] = (target[:, :, 2] - z_mean) / (z_std + 1e-8)

                if batch_idx == 0:
                    print(
                        "[DEBUG VAL hist_in] "
                        f"shape={tuple(hist_in.shape)} | "
                        f"min={hist_in.min().item():.4f} | "
                        f"max={hist_in.max().item():.4f} | "
                        f"mean={hist_in.mean().item():.4f} | "
                        f"std={hist_in.std().item():.4f}"
                    )


                start_time = time.perf_counter()    # fractional time in seconds
                output = model(hist_in)
                curr_time = (time.perf_counter() - start_time) * 1000   # elapsed time in ms
                if batch_idx == 0:
                    fr = future_ranges.to(args.device)
                    print(f"[DEBUG VAL future_ranges_m] min={fr.min().item():.4f} max={fr.max().item():.4f} mean={fr.mean().item():.4f} std={fr.std().item():.4f}")
                    print(f"[DEBUG VAL output_m]       min={output.min().item():.4f} max={output.max().item():.4f} mean={output.mean().item():.4f} std={output.std().item():.4f}")

                    print("[DEBUG] gt_mean:", future_ranges.to(args.device).mean().item())
                    print("[DEBUG] out_mean:", output.mean().item())

                    print("hist invalid ratio:", (hist_xyzd[:, :, 3] <= 0).float().mean().item())
                    print("fut  invalid ratio:", (future_ranges <= 0).float().mean().item())


                # build & compute 1D‐range loss
                # mixture, ok = build_range_mixture_distribution(cfg, output)
                if cfg["model_params"].get("use_mdn", True):
                    mixture, ok = model.build_mixture(cfg, output)
                    if not ok:
                        continue
                    target = future_xyz.to(args.device)
                    loss_dict = criterion(output, target, mode="val", epoch_number=epoch)

                    loss_tensor = loss_dict["loss"]
                    nll = loss_dict["loss_range_view"]
                    valid_ratio = loss_dict["valid_ratio"]

                    # ===== PAPER-METRICS: UPDATE VALIDATION SUMS =====
                    if isinstance(loss_dict, dict):
                        val_batch_count += 1

                        # Range-View-Loss (L1) aufsummieren
                        if "loss_range_view" in loss_dict:
                            val_rv_sum += float(loss_dict["loss_range_view"])

                        # Chamfer Distance (mittlere Chamfer-L1) aufsummieren
                        if "mean_chamfer_distance" in loss_dict:
                            val_cd_sum += float(loss_dict["mean_chamfer_distance"])

                        # Masken-Loss (optional, falls vorhanden)
                        if "loss_mask" in loss_dict:
                            val_mask_sum += float(loss_dict["loss_mask"])
                    # ===== END PAPER-METRICS: UPDATE VALIDATION SUMS =====

                # else:
                #     # Direkte Range-Regression ohne Gaußparameter
                #     future_ranges_n = (future_ranges.to(args.device) - range_mean) / (range_std + 1e-8)
                #     loss_tensor = torch.nn.functional.l1_loss(output, future_ranges_n)
                #     nll = loss_tensor.item()
                else:
                    fr = future_ranges.to(args.device)  # [B,F,H,W]

                    print("[DBG] fr min/max:", fr.min().item(), fr.max().item())
                    print("[DBG] ratio <=0:", (fr <= 0.0).float().mean().item())
                    print("[DBG] ratio  0:", (fr ==  0.0).float().mean().item())
                    print("[DBG] ratio >0:", (fr  >  0.0).float().mean().item())

                    valid = (fr > 0.0)
                    diff = (output - fr).abs()

                    loss_tensor = (diff * valid).sum() / (valid.sum().clamp_min(1e-8))
                    nll = loss_tensor.item()
                    # --- Epoch-Accumulator (pixel-gewichtet, glättet Zickzack) ---
                    # Wichtig: float() damit Summen stabil sind
                    val_abs_sum   += (diff * valid).sum().detach().item()
                    val_valid_sum += valid.sum().detach().item()

                # =============================================================
                # Logge nur alle N Batches (Loss + ACC-Metriken synchron)
                # =============================================================
                if batch_idx % cfg["train_params"]["tensorboard_log_interval"] == 0:
                    step = epoch * len(train_loader) + batch_idx
                    # --- Loss ---
                    if cfg["train_params"]["with_save"]:
                        writer.add_scalar('Loss/Validation', float(nll), step)

                    # === NEU: ACC-Range-Metriken berechnen & (optional) in TensorBoard loggen ===
                    # Ziel-Range braucht Form [B,T,1,H,W]
                    future_for_metrics = future_ranges
                    if future_for_metrics.ndim == 4:
                        future_for_metrics = future_for_metrics.unsqueeze(2)
                    future_for_metrics = future_for_metrics.to(args.device)

                    # output kann dict (mit 'rv') ODER Tensor sein
                    if isinstance(output, dict) and ("rv" in output):
                        out_for_metrics = output
                    elif torch.is_tensor(output):
                        out_for_metrics = {"rv": output}
                    else:
                        out_for_metrics = None  # sollte nicht passieren

                    if out_for_metrics is not None:
                        step = epoch * len(train_loader) + batch_idx
                        if cfg["train_params"]["with_save"]:
                            model.compute_and_log_metrics(
                                output=out_for_metrics,
                                future=future_for_metrics,
                                writer=writer,
                                global_step=step,
                                prefix="val"
                            )
                        else:
                            # ohne Writer: nur berechnen (kein Log)
                            _ = model.compute_and_log_metrics(
                                output=out_for_metrics,
                                future=future_for_metrics,
                                writer=None,
                                prefix="val"
                            )
                    # === ENDE NEU ===

                    # === Chamfer-Metrik (optional, wie im Paper) ===========
                    compute_cd = cfg["train_params"].get("compute_chamfer_metric", False)
                    cd_interval = cfg["train_params"].get("chamfer_metric_interval", 10)

                    # Nur falls aktiviert + nur alle cd_interval-Batches
                    if compute_cd and (batch_idx % cd_interval == 0):
                        # a) Output für Chamfer vorbereiten:
                        #    - ACC-Modelle liefern Dict mit "rv" + "mask_logits"
                        #    - sonst Dummy-mask_logits verwenden
                        if isinstance(output, dict) and ("rv" in output):
                            if "mask_logits" in output:
                                out_for_cd = output
                            else:
                                out_for_cd = {
                                    "rv": output["rv"],
                                    "mask_logits": torch.zeros_like(output["rv"]),
                                }
                        elif torch.is_tensor(output):
                            out_for_cd = {
                                "rv": output,
                                "mask_logits": torch.zeros_like(output),
                            }
                        else:
                            out_for_cd = None

                        if out_for_cd is not None:
                            # b) Target für Chamfer: [B,T,4,H,W] = [range, x, y, z]
                            target_for_cd = torch.cat(
                                [
                                    future_ranges.unsqueeze(2).to(args.device),  # [B,T,1,H,W]
                                    future_xyz.to(args.device),                 # [B,T,3,H,W]
                                ],
                                dim=2,
                            )  # -> [B,T,4,H,W]

                            # c) Downsampling-Parameter (aus TEST-Config, fallback -1 = kein Downsampling)
                            n_ds = cfg.get("TEST", {}).get("N_DOWNSAMPLED_POINTS_CD", -1)

                            cd_dict, cd_tensor = chamfer_metric(
                                out_for_cd,      # erwartet "rv" und "mask_logits"
                                target_for_cd,
                                n_ds,            # n_samples / n_downsampled_points
                            )
                            # cd_tensor: [T, B] – Chamfer pro Zeitschritt & Batch
                            cd_mean = cd_tensor.mean().item()

                            if cfg["train_params"]["with_save"]:
                                # globaler Chamfer-Wert
                                writer.add_scalar(
                                    "val/chamfer_distance/mean", cd_mean, step
                                )

                                # optional: pro Zukunfts-Horizont (t0, t1, ...)
                                for t_idx, cd_val in cd_dict.items():
                                    writer.add_scalar(
                                        f"val/chamfer_distance/t{t_idx}",
                                        cd_val.item(),
                                        step,
                                    )
                
                print(f"inference took {curr_time} ms.\tloss: {float(nll):.6f}\t@Epoch {epoch+1}/{cfg['train_params']['num_total_epochs']}")
                
                # add batch's validation loss to overall loss
                total_loss_val += float(nll)
            
            # average loss caluclation
            avg_loss_val = total_loss_val / len(val_loader)
            print(f"Epoch {epoch + 1}/{cfg['train_params']['num_total_epochs']}, Average Validation Loss: {avg_loss_val}")

            # ===== PAPER-L1 EPOCH METRIC (pixel-weighted) =====
            if cfg["train_params"]["with_save"] and val_valid_sum > 0:
                rv_epoch = val_abs_sum / max(val_valid_sum, 1e-8)

                # "paper/" Name (klarer Vergleich)
                writer.add_scalar("paper/val/range_view_metric_L1_epoch", rv_epoch, epoch)

                # optional: auch unter deinem bisherigen Key
                writer.add_scalar("val/range_view_metric_L1_epoch", rv_epoch, epoch)

                print(f"[VAL][EPOCH] range_view_metric_L1_epoch (pixel-weighted): {rv_epoch:.6f}")
            # ===== END =====

            # ===== PAPER-METRICS: LOG VALIDATION EPOCH MEANS =====
            if cfg["train_params"]["with_save"] and val_batch_count > 0:
                rv_mean = val_rv_sum / val_batch_count
                writer.add_scalar("val/range_view_metric_L1", rv_mean, epoch)

                if val_cd_sum > 0.0:
                    cd_mean = val_cd_sum / val_batch_count
                    writer.add_scalar("val/chamfer_distance_metric_L1", cd_mean, epoch)

                if val_mask_sum > 0.0:
                    mask_mean = val_mask_sum / val_batch_count
                    writer.add_scalar("val/loss_mask", mask_mean, epoch)
            # ===== END PAPER-METRICS: LOG VALIDATION EPOCH MEANS =====
        
        if cfg["train_params"]["with_save"]:
            writer.add_scalar('Loss/Validation/Epoch', avg_loss_val, epoch)
            
        # Update learning rate scheduler
        # scheduler.step()
        # scheduler.step(avg_loss_val)  # using ReduceLROnPlateau

    # ----------------------------------------------------
    # Finaler Test-Loop (nur einmal am Ende des Trainings)
    # ----------------------------------------------------
    if split_type == "predefined" and 'test_loader' in locals() and test_loader is not None:
        print("\n===== RUN FINAL EVALUATION ON TEST SET =====")
        model.eval()
        last_epoch_idx = cfg["train_params"]["num_total_epochs"] - 1

        test_loss_sum = 0.0
        test_nll_sum = 0.0
        test_batches = 0

        # Aggregatoren für Range-View-Metriken (ACC)
        metrics_sums = {
            "mae_mean": 0.0,
            "rmse_mean": 0.0,
            "logrmse_mean": 0.0,
            "acc_0.1_mean": 0.0,
            "acc_0.2_mean": 0.0,
            "acc_0.5_mean": 0.0,
            "tv_time": 0.0,
            "vel_mae": 0.0,
            "acc_mae": 0.0,
            "valid_ratio_mean": 0.0,
        }

        with torch.no_grad():
            for batch_idx, (hist_xyzd, future_xyz, future_ranges) in enumerate(
                    tqdm(test_loader, total=len(test_loader))):

                # hist_xyz: [B, T_in, 3/4, H, W]
                # future_xyz: [B, T_out, 3, H, W]
                # future_ranges: [B, T_out, H, W]
                hist_xyzd = hist_xyzd.to(args.device)
                future_xyz = future_xyz.to(args.device)
                future_ranges = future_ranges.to(args.device)

                # Range-only input: take range channel (index 3)
                if hist_xyzd.shape[2] == 4:
                    hist_in = hist_xyzd[:, :, 3:4, :, :]   # [B,T,1,H,W]
                elif hist_xyzd.shape[2] == 1:
                    hist_in = hist_xyzd
                else:
                    raise ValueError(f"Unexpected input channels from dataloader: {hist_xyzd.shape}")

                # Vorwärtsdurchlauf
                start_time = time.perf_counter()
                output = model(hist_in)
                curr_time = (time.perf_counter() - start_time) * 1000.0  # ms

                # -------- Loss-Berechnung (wie im Validation-Loop) --------
                if cfg["model_params"].get("use_mdn", True):
                    mixture, ok = model.build_mixture(cfg, output)
                    if not ok:
                        continue

                    target = future_xyz  # Loss arbeitet auf xyz
                    loss_dict = criterion(output, target, mode="test", epoch_number=last_epoch_idx)
                    loss_tensor = loss_dict["loss"]
                    nll = loss_dict.get("loss_range_view", loss_tensor).item()
                else:
                    # Direkte Range-Regression ohne Gaußparameter
                    loss_tensor = torch.nn.functional.l1_loss(
                        output, future_ranges.to(args.device)
                    )
                    nll = loss_tensor.item()

                test_loss_sum += loss_tensor.item()
                test_nll_sum += nll
                test_batches += 1

                # ---------------- ACC-Range-Metriken berechnen ----------------
                # Ziel-Range braucht Form [B,T,1,H,W]
                future_for_metrics = future_ranges
                if future_for_metrics.ndim == 4:
                    future_for_metrics = future_for_metrics.unsqueeze(2)
                future_for_metrics = future_for_metrics.to(args.device)

                # output kann dict (mit 'rv') ODER Tensor sein
                if isinstance(output, dict) and ("rv" in output):
                    out_for_metrics = output
                elif torch.is_tensor(output):
                    out_for_metrics = {"rv": output}
                else:
                    out_for_metrics = None  # sollte nicht passieren

                if out_for_metrics is not None:
                    # Kein Writer -> gibt dict m zurück
                    m = model.compute_and_log_metrics(
                        output=out_for_metrics,
                        future=future_for_metrics,
                        writer=None,
                        prefix="test"
                    )
                    # Wichtige Metriken aufsummieren
                    for k in metrics_sums.keys():
                        if k in m:
                            metrics_sums[k] += float(m[k])

        if test_batches > 0:
            test_loss_mean = test_loss_sum / test_batches
            test_nll_mean = test_nll_sum / test_batches
            metrics_mean = {k: v / test_batches for k, v in metrics_sums.items()}
        else:
            test_loss_mean = float("nan")
            test_nll_mean = float("nan")
            metrics_mean = {k: float("nan") for k in metrics_sums.keys()}

        # ---------- Optional: alles einmal in TensorBoard loggen ----------
        if cfg["train_params"].get("with_save", False):
            # x-Achse: "letzter Step" – hier einfach gesamte Train-Steps
            final_step = cfg["train_params"]["num_total_epochs"] * len(train_loader)

            writer.add_scalar("test/loss", test_loss_mean, final_step)
            writer.add_scalar("test/nll",  test_nll_mean,  final_step)

            # gleiche Tags wie bei val/train, aber mit Prefix "test"
            writer.add_scalar("test/mae_mean",         metrics_mean["mae_mean"], final_step)
            writer.add_scalar("test/rmse_mean",        metrics_mean["rmse_mean"], final_step)
            writer.add_scalar("test/logrmse_mean",     metrics_mean["logrmse_mean"], final_step)
            writer.add_scalar("test/acc_0.1_mean",     metrics_mean["acc_0.1_mean"], final_step)
            writer.add_scalar("test/acc_0.2_mean",     metrics_mean["acc_0.2_mean"], final_step)
            writer.add_scalar("test/acc_0.5_mean",     metrics_mean["acc_0.5_mean"], final_step)
            writer.add_scalar("test/tv_time",          metrics_mean["tv_time"], final_step)
            writer.add_scalar("test/vel_mae",          metrics_mean["vel_mae"], final_step)
            writer.add_scalar("test/acc_mae",          metrics_mean["acc_mae"], final_step)
            writer.add_scalar("test/valid_ratio_mean", metrics_mean["valid_ratio_mean"], final_step)    
            
    if cfg["train_params"]["with_save"]:
        torch.save(model.state_dict(), os.path.join(save_path, "weights", "model_final.pt"))
    
    if cfg["train_params"]["plot_examples"]:
        plt.ioff()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visu",
        type=bool,
        default=True
    )
    
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/home/devuser/workspace/src/configs/semanticKitti_default.yaml" # thab_default  semanticKitti_default
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    
    parser.add_argument(
        "--dataloader_device",
        type=str,
        default="cpu"   # cpu or cuda
    )
    
    args = parser.parse_args()
    main(args)
