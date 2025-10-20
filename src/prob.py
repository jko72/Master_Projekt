import torch
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical
import numpy as np

import torch
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

def visualize_pc_with_ray_confidence_open3d(
    mixture: torch.distributions.MixtureSameFamily,
    future_xyz: torch.Tensor,
    b: int,
    t: int,
    j: int,
    phi_grid: torch.Tensor,
    theta_grid: torch.Tensor,
    H: int,
    W: int,
    N_r: int = 50,
    r_max: float = 50.0,
    device: str = "cpu"
):
    """
    Show a ground-truth point cloud (future_xyz[b,t]) together with a selected
    vertical “ray” at column j, sampled N_r times up to r_max meters.  Each
    sample is colored by the mixture’s PDF value at that range.
    
    Args:
        mixture    : MixtureSameFamily over [B*T*H*W] distributions
        future_xyz : [B, T, 3, H, W] ground-truth 3D points
        b, t, j    : batch-, time-, and azimuth‐column indices
        phi_grid   : [H, W] azimuth angles per pixel
        theta_grid : [H, W] elevation angles per pixel
        H, W       : image dims
        N_r        : samples per ray
        r_max      : max range (m)
        device     : torch device
    """
    # 1) Flatten index base for this (b,t) slice
    base_idx = (b * mixture.batch_shape[0] // (future_xyz.numel()//(b+1)))  # unused
    # Actually easier: idx = (b*T + t)*H*W + i*W + j, we'll compute per-i below

    # 2) Build range samples
    r_axis = torch.linspace(0.0, r_max, N_r, device=device)  # [N_r]
    # Expand to shape [N_r, N_batch]
    N_batch = mixture.batch_shape[0]
    r_mat = r_axis.view(N_r, 1).expand(N_r, N_batch)          # [N_r, N_batch]
    # Compute log‐pdf for all distributions at all sample ranges
    logp = mixture.log_prob(r_mat)                            # [N_r, N_batch]
    pdf = torch.exp(logp)                                      # [N_r, N_batch]

    # 3) For each elevation i, compute the sample points and collect PDF
    sample_pts = []
    sample_vals = []
    # row‐wise: i in [0,H)
    for i in range(H):
        idx = (b * future_xyz.shape[1] + t) * (H*W) + i*W + j
        # direction vector for pixel (i,j)
        phi   = phi_grid[i, j].to(device)
        theta = theta_grid[i, j].to(device)
        dir_vec = torch.tensor([
            torch.cos(theta) * torch.cos(phi),
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta)
        ], device=device)                                    # [3]

        # sample points along ray: [N_r,3]
        pts_ij = (r_axis.unsqueeze(1) * dir_vec.unsqueeze(0))  # [N_r,3]
        sample_pts.append(pts_ij)
        sample_vals.append(pdf[:, idx])                        # [N_r]

    # stack over i: [H*N_r,3] and [H*N_r]
    sample_pts = torch.cat(sample_pts, dim=0)   # [H*N_r,3]
    sample_vals = torch.cat(sample_vals, dim=0) # [H*N_r]

    # 4) Build ground-truth point cloud
    gt_pts = future_xyz[b, t].permute(1, 2, 0).reshape(-1, 3)  # [H*W,3]
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts.cpu().numpy())
    pcd_gt.paint_uniform_color([0.7,0.7,0.7])

    # 5) Build confidence‐colored sample cloud
    cmap = cm.get_cmap("turbo")
    cols = cmap(sample_vals.clamp(min=0, max=pdf.max()).cpu().numpy())
    cols = cols[:, :3]  # drop alpha
    pcd_conf = o3d.geometry.PointCloud()
    pcd_conf.points = o3d.utility.Vector3dVector(sample_pts.cpu().numpy())
    pcd_conf.colors = o3d.utility.Vector3dVector(cols)

    # 6) Visualize both together
    o3d.visualization.draw_geometries(
        [pcd_gt, pcd_conf],
        window_name=f"Batch {b}, Timestep {t}, Ray j={j}",
        width=800, height=600
    )



# Function to plot mixture of Gaussians for selected pixels
def visualize_mixture_pdfs(
    mixture: MixtureSameFamily,
    B: int,
    T: int,
    H: int,
    W: int,
    phi_grid: torch.Tensor,
    theta_grid: torch.Tensor,
    alpha_threshold: float,
    use_density_threshold: bool = False,
    density_threshold: float = None,
    device: str = 'cpu'
) -> list:
    """
    Non‐blocking Matplotlib: shows up to two pixels’ component PDFs,
    using the same masking as the point‐cloud generator.
    Returns a list of Figure handles.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    comp = mixture.component_distribution
    cat  = mixture.mixture_distribution
    K    = comp.loc.size(-1)

    # reshape & move to device
    mus    = comp.loc.view(B, T, H, W, K).to(device).detach()
    alphas = cat.probs.view(B, T, H, W, K).to(device).detach()
    sigmas = comp.scale.view(B, T, H, W, K).to(device).detach()

    # sort so we can pick the first valid component
    sorted_mus    , indices    = torch.sort(mus,    dim=-1)
    sorted_alphas = torch.gather(alphas, -1, indices)
    sorted_sigmas = torch.gather(sigmas, -1, indices)

    # build the exact same mask as in generate_point_clouds_from_mixture
    mask = sorted_alphas >= alpha_threshold
    if use_density_threshold:
        if density_threshold is None:
            raise ValueError("density_threshold must be set when using density threshold")
        log_densities = -0.5 * (
            torch.log(torch.tensor(2 * torch.pi, device=device))
            + 2 * torch.log(sorted_sigmas)
        )
        mask &= (log_densities >= density_threshold)

    # first‐true per pixel
    cumsum_mask = torch.cumsum(mask.int(), dim=-1)
    first_true  = (cumsum_mask == 1) & mask
    chosen_r    = (sorted_mus * first_true.float()).sum(dim=-1)  # [B,T,H,W]

    # batch=0, t=0 only for PDF viz
    rmap      = chosen_r[0,0]            # [H,W]
    valid_idx = torch.nonzero(rmap > 0, as_tuple=False)
    if valid_idx.numel() == 0:
        print("No valid pixels to visualize PDF")
        return []

    sel = valid_idx[:2]  # up to 2 pixels
    r_axis = np.linspace(0, 100, 200)   # clamp 0–100 m

    plt.ion()
    figs = []
    for (i, j) in sel.tolist():
        mu_k    = sorted_mus[0,0,i,j].cpu().numpy()
        sigma_k = sorted_sigmas[0,0,i,j].cpu().numpy()
        alpha_k = sorted_alphas[0,0,i,j].cpu().numpy()

        # component PDFs
        pdfs = []
        for k in range(K):
            pdf_k = alpha_k[k] / (sigma_k[k] * np.sqrt(2*np.pi)) * \
                    np.exp(-0.5 * ((r_axis - mu_k[k]) / sigma_k[k])**2)
            pdfs.append(pdf_k)
        mix_pdf = np.sum(pdfs, axis=0)

        # new figure per pixel
        fig = plt.figure(figsize=(6,4))
        for k, pdf_k in enumerate(pdfs):
            plt.plot(r_axis, pdf_k, label=f"comp {k}")
            plt.axvline(mu_k[k], linestyle="--", color="gray")
        cr = rmap[i,j].item()
        plt.plot(r_axis, mix_pdf, 'k-', lw=2, label="mixture")
        plt.axvline(cr, color="red", lw=2, label=f"chosen r={cr:.2f} m")
        plt.title(f"Mixture PDF at pixel ({i},{j})")
        plt.xlabel("Range (m)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        figs.append(fig)

    return figs

def generate_point_clouds_from_mixture(
    mixture: MixtureSameFamily,
    B: int,
    T: int,
    H: int,
    W: int,
    phi_grid: torch.Tensor,
    theta_grid: torch.Tensor,
    alpha_threshold: float = 0.1,
    device: str = "cpu",
    use_density_threshold: bool = False,
    density_threshold: float = None
):
    """
    Convert MDN range outputs into a single 3D point cloud per time-step, one point per pixel.

    Vectorized: picks the first ascending mean per pixel with weight >= threshold.

    Returns a list of B*T tensors of shape [M,3].
    """
    # Component params: comp.loc [N,K], cat.probs [N,K]
    comp = mixture.component_distribution
    cat = mixture.mixture_distribution
    N = B * T * H * W
    K = comp.loc.size(-1)

    # reshape to [B,T,H,W,K]
    mus = comp.loc.view(B, T, H, W, K).detach()             # [B,T,H,W,K]
    alphas = cat.probs.view(B, T, H, W, K).detach()      # [B,T,H,W,K]

    if mus.device.type != device:
        mus = mus.to(device)
        alphas = alphas.to(device)
    
    # sort means and align weights
    sorted_mus, indices = torch.sort(mus, dim=-1)
    sorted_alphas = torch.gather(alphas, -1, indices)

    # If using density threshold, compute densities
    if use_density_threshold:
        if density_threshold is None:
            raise ValueError("density_threshold must be set when using density threshold")
        sigmas = comp.scale.view(B, T, H, W, K).to(device).detach()
        sorted_sigmas = torch.gather(sigmas, -1, indices)
        
        # value of the log probability density function (pdf) of each Gaussian at its own mean
        # p(x;mu;sigma) = 1/(sigma* /sqrt(2pi)) * exp( -1/2 *  ((x - mu)/sigma )**2 )
        # maximum accurs at x=mu, giving:
        # log p(mu;mu;sigma) = log 1/(sigma* /sqrt(2pi)) 
        log_densities = -0.5 * (torch.log(torch.tensor(2*torch.pi)) + 2 * torch.log(sorted_sigmas))
        mask = (alphas >= alpha_threshold) & (log_densities >= density_threshold)

    else:
        mask = sorted_alphas >= alpha_threshold  # [B,T,H,W,K]

    # find first valid index per pixel
    # make mask float and cumulative logic
    # valid_mask = mask.float()
    # # convert to a large negative for invalid
    # neg_inf = torch.full_like(valid_mask, float('-inf'))
    # use mask to build weights for argmax: want first True => index=0 if first, etc
    # instead, create a mask of first occurrence: for each pixel, first_true = (cumsum(mask, dim=-1)==1) & mask
    cumsum_mask = torch.cumsum(mask.int(), dim=-1)
    first_true = (cumsum_mask == 1) & mask  # [B,T,H,W,K]

    # get range per pixel: multiply by sorted_mus and sum along K
    chosen_r = (sorted_mus * first_true.float()).sum(dim=-1)  # [B,T,H,W]

    # generate 3D coords
    # expand phi/theta to [B,T,H,W]
    phi = phi_grid.to(device=device).unsqueeze(0).unsqueeze(0).expand(B, T, H, W)
    theta = theta_grid.to(device=device).unsqueeze(0).unsqueeze(0).expand(B, T, H, W)

    x = chosen_r * torch.cos(theta) * torch.cos(phi)
    y = chosen_r * torch.cos(theta) * torch.sin(phi)
    z = chosen_r * torch.sin(theta)

    # build point clouds list
    pointclouds = []
    for b in range(B):
        for t in range(T):
            # stack H*W points
            pts_bt = torch.stack((
                    x[b,t].reshape(-1),
                    y[b,t].reshape(-1),
                    z[b,t].reshape(-1)), dim=1)  # [H*W,3]
            # optionally filter zeros (where no mask)
            mask_bt = chosen_r[b,t].reshape(-1) > 0
            pts_bt = pts_bt[mask_bt]
            pointclouds.append(pts_bt)
    return pointclouds

############################################################################################
############################################################################################

def build_range_mixture_distribution(
    cfg,
    output,
    alpha_temp: torch.Tensor
):
    """
    Construct a GMM distribution over future range per-pixel.
    """
    B, T, H, W, _ = output.shape
    K = cfg["model_params"]["mdn_num_gaussians"]
    eps_mu = cfg["train_params"]["build_mixture_variance_mu"]
    eps_sigma = cfg["train_params"]["build_mixture_variance_epsilon"]

    # split components: mu, log-sigma, logits
    raw_mu     = output[..., :K]
    raw_logsig = output[..., K:2*K]
    raw_logits = output[..., 2*K:3*K]

    # positive mean via softplus
    mu_deltas = F.softplus(raw_mu) + eps_mu     # positive increments
    mu = torch.cumsum(mu_deltas, dim=-1)        # ascending means
    # positive sigma via exp
    sigma = torch.exp(raw_logsig) + eps_sigma
    sigma = sigma.clamp(max=10.0)               # keeps variance reasonable while head is immature
    # temperature-scaled mixture weights
    temp = F.softplus(alpha_temp) + 1e-6
    alpha = F.softmax(raw_logits / temp, dim=-1)

    # flatten for distribution
    N = B * T * H * W
    mu_flat     = mu.reshape(N, K)
    sigma_flat  = sigma.reshape(N, K)
    alpha_flat  = alpha.reshape(N, K)

    cat  = Categorical(probs=alpha_flat)
    comp = Normal(loc=mu_flat, scale=sigma_flat)
    mix  = MixtureSameFamily(cat, comp)
    return mix, True


def compute_nll_range_loss(
    cfg,
    mixture_dist,
    target_ranges,
    with_sigma_regularization: bool = True,
    with_alpha_regularization: bool = True
):
    target_flat = target_ranges.reshape(-1)
    mask = target_flat > 0
    log_p = mixture_dist.log_prob(target_flat)
    nll = -log_p[mask].mean()

    reg = 0.0
    if with_sigma_regularization:
        q = Normal(mixture_dist.component_distribution.loc,
                mixture_dist.component_distribution.scale)
        p = Normal(q.loc, torch.ones_like(q.scale))
        kl = torch.distributions.kl_divergence(q, p).mean()
        reg += kl * cfg["train_params"].get("lambda_sigma", 1e-3)
    if with_alpha_regularization:
        alphas = mixture_dist.mixture_distribution.probs
        entropy = -(alphas * torch.log(alphas + 1e-12)).sum(dim=-1).mean()
        reg -= entropy * cfg["train_params"].get("lambda_alpha", 5e-2)
        # Shannon entropy, measures the uncertainty or spread of the categorical distribution over components.
        # If one α_k ​is near 1 and the others near 0, H(α)≈0: the model is very "confident" in a single component.
        # If all α_k ​are equal (=1/K), H(α)=logK: maximal uncertainty or "flat" mixture.
            # regularize alpha: 
            # with (+)-sign: encourage peaked distributions (low entropy)
            # with (-)-sign: encourage mixture diversity


    loss = nll + reg
    return loss, nll.item()


# def build_range_mixture_distribution(
#     cfg,
#     output,
#     posthoc_sort: bool = True
# ):
#     """
#     Args:
#         output : Tensor of shape [B, T, H, W, 3*K]
#                 last‐dim = [ mu_1 … mu_K | raw_s_1 … raw_s_K | logits_alpha_1 … logits_alpha_K ]
#         num_gaussians : K

#     Returns:
#         mixture : a MixtureSameFamily of 1D Normals, 
#                     batch‐flattened to shape [B*T*H*W]
#         valid   : bool (False if nan's cropped us)
        
#     Help:
#         SoftPlus: smooth approximation to the ReLU function, constrains the output to always be positive.
#     """
#     B, T, H, W, _ = output.shape
#     K = cfg["model_params"]["mdn_num_gaussians"]
#     eps_mu = cfg["train_params"]["build_mixture_variance_mu"]
#     eps_sigma = cfg["train_params"]["build_mixture_variance_epsilon"]

#     # split into parameters
#     mu_raw     = output[...,    :  K]        # [B,T,H,W,K]
#     sigma_raw  = output[...,   K:2*K]     # [B,T,H,W,K]
#     logits_a   = output[..., 2*K:3*K]           # [B,T,H,W,K]
    
#     # means mu >0
#     mu_pos = F.softplus(mu_raw) + eps_mu
#     # enforce ascending-order means. This removes permutation ambiguity and stabilizes training.
#     #mu_ordered = torch.cumsum(mu_pos, dim=-1)

#     # variance >0
#     sigma = F.softplus(sigma_raw) + eps_sigma
#     # mixture weights alpha [0,1]
#     alpha = F.softmax(logits_a, dim=-1)    # [B,T,H,W,K]

#     # flatten out the spatial & time dims
#     N = B*T*H*W
#     mu_flat    = mu_pos     .reshape(N, K)   # [(B*T*H*W), K], test with mu_pos and mu_ordered
#     sigma_flat = sigma      .reshape(N, K)   # [(B*T*H*W), K]
#     alpha_flat = alpha      .reshape(N, K)   # [(B*T*H*W), K]

#     if posthoc_sort:
#         # sort means and align sigma, alpha
#         mu_flat, idx = torch.sort(mu_flat, dim=-1)
#         sigma_flat = torch.gather(sigma_flat, -1, idx)
#         alpha_flat = torch.gather(alpha_flat, -1, idx)

#     # build the MixtureSameFamily of 1D Normals
#     try:
#         cat = Categorical(probs=alpha_flat)
#         comp = Normal(loc=mu_flat, scale=sigma_flat)
#         mixture = MixtureSameFamily(cat, comp)
#     except Exception:
#         return None, False

#     return mixture, True


# def compute_nll_range_loss(
#     cfg,
#     mixture_dist,
#     target_ranges,
#     with_sigma_regualization: bool = False,
#     with_alpha_regualization: bool = False,
#     lambda_sigma=1e-3, 
#     lambda_alpha=2e-1
# ):
#     """
#     Args:
#         mixture_dist : MixtureSameFamily over [(B*T*H*W),K]
#         target_ranges: Tensor [B, T, H, W] of true beam‐ranges

#     Returns:
#         loss         : scalar Tensor = NLL
#         nll_value    : float
#     """
#     # flatten the targets to match mixture's batch‐axis
#     target_flat = target_ranges.reshape(-1)                 # N=[(B*T*H*W)]
#     # build mask of valid pixels (pixels with no data==0 gets dropped)
#     mask = (target_flat>0)                  
#     # log‐prob & NLL (only on valid entries)
#     log_p       = mixture_dist.log_prob(target_flat)        # [M]                 
#     nll         = -log_p[mask] .mean()                      # [M<=N]

#     sigma_reg   = 0
#     entropy     = 0
#     if with_sigma_regualization:
#         sig = mixture_dist.component_distribution.scale  # [N,K]
#         sigma_reg = sig.mean()
#     if with_alpha_regualization:
#         # regularize alpha: encourage peaked distributions (low entropy)
#         # Shannon entropy, measures the uncertainty or spread of the categorical distribution over components.
#         # If one α_k ​is near 1 and the others near 0, H(α)≈0: the model is very "confident" in a single component.
#         # If all α_k ​are equal (=1/K), H(α)=logK: maximal uncertainty or "flat" mixture.
#         alphas = mixture_dist.mixture_distribution.probs  # [N,K]
#         entropy = -(alphas * torch.log(alphas + 1e-12)).sum(dim=-1).mean()
    
#     if with_sigma_regualization and with_alpha_regualization:
#         loss = nll + lambda_sigma * sigma_reg + lambda_alpha * entropy
#     elif with_sigma_regualization and (not with_alpha_regualization):
#         loss = nll + lambda_sigma * sigma_reg
#     elif (not with_sigma_regualization) and with_alpha_regualization:
#         loss = nll + lambda_alpha * entropy
#     else:
#         loss = nll
        
#     return loss, nll.item()
