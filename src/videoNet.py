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
#from model import RangeMixtureVideoModel
from models.rangeMixtureSwinTrans import RangeMixtureSwinTransformerModel

# Helper functions
from helper.pointcloud_visualization import pointcloud_from_expected_range
from helper.dataloader_helper import make_sequences, build_dataloaders

import torch.optim as optim
import cv2
import copy
import open3d as o3d
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
import time

import threading
import multiprocessing as mp

import matplotlib
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


def main(args):
    global show_pc_flag, show_pdf_flag, show_ray_flag
    with open(args.cfg_path) as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    base = args.data_dir  # parent directory
    all_seqs = make_sequences(base)
    # Prepare rotary splits
    rotary_loaders = build_dataloaders(
        all_seqs,
        cfg,
        args.dataloader_device,
        split_type='rotary' # zu predefined umändern und unten den scheduler der auskommentiert ist dann verwenden. test dataset ist 0006
    )  # list of (holdout_id, train_loader, val_loader)
    n_splits = len(rotary_loaders)

    # model definition
    # from torchvision baseline "Video Classification" models, see https://pytorch.org/vision/main/models.html#video-classification
    #model = RangeMixtureVideoModel(cfg)
    model = RangeMixtureSwinTransformerModel(cfg)
    # load weights
    try:
        if os.path.isfile(cfg["train_params"]["pre_train_weights"]):    # throws TypeError if NoneType provided
            weights = torch.load(cfg["train_params"]["pre_train_weights"])
            model.load_state_dict(weights)
    except Exception as ex:
            print("no custom pretrained weights found, use default vanilla")
    model.to(args.device)
    
    # Define optimizer
    #optimizer = optim.Adam(model.parameters(), lr=cfg["train_params"]["start_learning_rate"])
    
    # Optimizer: AdamW with decoupled weight decay, 
    # see https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train_params"].get("start_learning_rate", 1e-3),
        weight_decay=cfg["train_params"].get("weight_decay", 1e-4)  # typical: 1e-2 -> 1e-4
    )
    # num_epochs = cfg["train_params"].get("num_epochs", 50)
    # steps_per_epoch = len(dataloader_train)
    # total_steps = max(1, num_epochs * steps_per_epoch)

    # warmup_epochs = cfg["train_params"].get("num_warmup_epochs", 2)
    # warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    # base_lr = optimizer.param_groups[0]["lr"]
    # eta_min = cfg["train_params"].get("learning_rate_min", 5e-6)
    # warmup_start = 0.3  # start at 30% of base LR

    # def lr_lambda(global_step: int):
    #     if global_step < warmup_steps:  # linear warmup (per-iter)
    #         return warmup_start + (1.0 - warmup_start) * (global_step / warmup_steps)
    #     # cosine decay to eta_min
    #     t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    #     cos = 0.5 * (1 + math.cos(math.pi * t))
    #     return (eta_min / base_lr) + (1 - eta_min / base_lr) * cos

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Linear warm-up over first N epochs
    num_warmup_epochs = cfg["train_params"].get("num_warmup_epochs", 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,   # start at 10% of base LR
        end_factor=1.0,     # ramp to 100% of base LR
        total_iters=num_warmup_epochs
    )
    #Cosine annealing with restarts thereafter
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,           # first restart after T_0 epochs
        T_mult=1,         # no increase in cycle length
        eta_min=1e-6      # floor LR
    )  
    #Chain them: warm-up then cosine restarts
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_epochs]
    ) 

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # TensorBoard
    if cfg["train_params"]["with_save"]:
        t = time.gmtime()
        time_start = time.strftime("%y-%m-%d_%H-%M-%S", t)  # Changed format to avoid colons
        save_path = os.path.join(cfg["train_params"]["logs_save_dir"], time_start)
        os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
        writer = SummaryWriter(save_path)
        #save_path ="/home/devuser/workspace/LidarGaussianVideoView/logs"
        # save current training config file
        with open(os.path.join(save_path, "thab_default.yaml"), "w") as file:
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
                im = ax.imshow(
                    np.zeros((H, W)),
                    aspect='equal',
                    vmin=0, vmax=50,
                    cmap='turbo'
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
    
    for epoch in range(cfg["train_params"]["num_total_epochs"]):
        # Select split based on epoch (round-robin)
        split_idx = epoch % n_splits
        holdout_id, train_loader, val_loader = rotary_loaders[split_idx]
        print(f"Epoch {epoch+1}/{cfg['train_params']['num_total_epochs']}, training on sequence {holdout_id}")

        total_loss = 0.0
        total_loss_val = 0.0
        # --- Training Loop ---
        for batch_idx, (hist_xyz, future_xyz, future_ranges) in enumerate(tqdm(iterable=train_loader, total=len(train_loader))):
            # hist_xyz      = [B, T_in,     3,  H, W], [B, T_in,     4,  H, W]
            # future_xyz    = [B, T_out,    3,  H, W]
            # future_ranges = [B, T_out,        H, W]
        
            model.train()
            #model.eval()
            
            # model's forward gives "output" of shape [B,T,H,W,3K]
            hist_xyz = hist_xyz.to(args.device)
            
            start_time = time.perf_counter()    # fractional time in seconds
            output = model(hist_xyz)
            curr_time = (time.perf_counter() - start_time) * 1000   # elapsed time in ms
            
            # build & compute 1D‐range loss
            # mixture, ok = build_range_mixture_distribution(cfg, output)
            mixture, ok = model.build_mixture(cfg, output)
            if not ok:
                continue
            loss_tensor, nll = compute_nll_range_loss(cfg, mixture, future_ranges.to('cuda'))
            
            # add batch's train loss to overall loss
            total_loss += nll
                
            mixture_cpu = mixture_to_cpu(mixture)
            #r_exp = mixture.mean
            # loss_occ = occlusion_penalty(r_exp, future_ranges)
            # loss_tensor = loss_tensor + loss_occ
            #loss_tensor = loss_tensor
            
            print(f"inference took {curr_time:.3f} ms.\tLR: {optimizer.param_groups[0]['lr']}\tloss: {nll:.3f}\t@Epoch {epoch+1}/{cfg['train_params']['num_total_epochs']}")
            
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            #scheduler.step()
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
                theta_range = cfg['model_params'].get('theta_range', [-np.pi/16, np.pi/16])
                phi_grid, theta_grid = make_angle_grids(H, W, theta_range, device="cpu")
                
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

                # VISUALIZATION
                b = 0   # which batch‐element to show
                flip = cfg["train_params"].get("plot_time_vertically", True)
                
                if flip:
                    # rows = time, cols = [GT,Mean,Mode]
                    for row in range(T):
                        im_handles[row][0].set_data(gt_all[b,   row])
                        cb_handles[row][0].update_normal(im_handles[row][0])

                        im_handles[row][1].set_data(mean_all[b, row])
                        cb_handles[row][1].update_normal(im_handles[row][1])

                        im_handles[row][2].set_data(modes_all[b, row])
                        cb_handles[row][2].update_normal(im_handles[row][2])
                else:
                    # rows = [GT,Mean,Mode], cols=time
                    for col in range(T):
                        im_handles[0][col].set_data(gt_all[b,   col])
                        cb_handles[0][col].update_normal(im_handles[0][col])

                        im_handles[1][col].set_data(mean_all[b, col])
                        cb_handles[1][col].update_normal(im_handles[1][col])

                        im_handles[2][col].set_data(modes_all[b, col])
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
                    writer.add_scalar('Loss', nll, step)
                    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)
                    
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
                        
                    mask = (gt_all != 0)
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
        with torch.no_grad():
            for batch_idx, (hist_xyz, future_xyz, future_ranges) in enumerate(tqdm(iterable=val_loader, total=len(val_loader))):
                # model's forward gives "output" of shape [B,T,H,W,3K]
                hist_xyz = hist_xyz.to(args.device)
                
                start_time = time.perf_counter()    # fractional time in seconds
                output = model(hist_xyz)
                curr_time = (time.perf_counter() - start_time) * 1000   # elapsed time in ms
                
                # build & compute 1D‐range loss
                # mixture, ok = build_range_mixture_distribution(cfg, output)
                mixture, ok = model.build_mixture(cfg, output)
                if not ok:
                    continue
                loss_tensor, nll = compute_nll_range_loss(cfg, mixture, future_ranges.to('cuda'))
                
                if cfg["train_params"]["with_save"]: 
                    if batch_idx % cfg["train_params"]["tensorboard_log_interval"] == 0:
                        step = epoch * len(train_loader) + batch_idx
                        writer.add_scalar('Loss/Validation', nll, step)
                
                print(f"inference took {curr_time} ms.\tloss: {nll}\t@Epoch {epoch+1}/{cfg['train_params']['num_total_epochs']}")
                
                # add batch's validation loss to overall loss
                total_loss_val += nll
            
            # average loss caluclation
            avg_loss_val = total_loss_val / len(val_loader)
            print(f"Epoch {epoch + 1}/{cfg['train_params']['num_total_epochs']}, Average Validation Loss: {avg_loss_val}")
        
        if cfg["train_params"]["with_save"]:
            writer.add_scalar('Loss/Validation/Epoch', avg_loss_val, epoch)
            
        # Update learning rate scheduler
        scheduler.step()
        # scheduler.step(avg_loss_val)  # using ReduceLROnPlateau
        
            
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
        "--data_dir",
        type=str,
        default="/home/devuser/workspace/data/datasets/SemanticTHAB/sequences"
    )
    
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/home/devuser/workspace/src/configs/thab_default.yaml"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    
    parser.add_argument(
        "--dataloader_device",
        type=str,
        default="cuda"   # cpu or cuda
    )
    
    args = parser.parse_args()
    main(args)