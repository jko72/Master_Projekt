import torch
import numpy as np

def make_angle_grids(height, width, theta_range, device="cpu"):
    """
    Create per-pixel angle grids for spherical projection.

    Args:
        height: number of elevation bins (H)
        width:  number of azimuth bins (W)
        theta_range: (tmin, tmax) elevation bounds
        device: compute device

    Returns:
        phi_grid   : Tensor [H, W] azimuth angles
        theta_grid : Tensor [H, W] elevation angles
    """
    tmin, tmax = theta_range
    pmin, pmax = -np.pi, np.pi
    th_vals = torch.linspace(tmax, tmin, steps=height, device=device)
    ph_vals = torch.linspace(pmin, pmax, steps=width,  device=device)
    theta_grid, phi_grid = torch.meshgrid(th_vals, ph_vals, indexing='ij')
    return phi_grid, theta_grid

def to_deflection_coordinates(x, y, z):
    """
    Convert Cartesian (x,y,z) to 
    - phi ∈ [–π, π]
    - theta ∈ [0, π]
    """
    p   = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    theta = -torch.atan2(p, z) + (torch.pi / 2)
    return phi, theta

def spherical_projection(pc, height=None, width=None,
                         theta_range=None, th=1.0,
                         sort_largest_first=False,
                         device="cpu"):
    """
    Args:
        pc            : Tensor [B, 3, H, W] (or [3, H, W]). Channels are (x,y,z).
        height, width : output image size; if None, uses H, W of pc.
        theta_range   : (min, max) to clamp theta; if None, computed from data.
        th            : range threshold (currently unused; you can insert filtering).
        sort_largest_first : if True, far points override near; else, near override far.

    Returns:
        pj_img   : [B, 3, height, width]  spherical‐projected point‐cloud image
        alpha    : [height, width]        per‐pixel distance in (phi,theta)‐space
        (tmin,tmax) : tuple of floats, theta range used
        (pmin,pmax) : tuple of floats, phi range used (= (–π, π))
    """
    # ensure batch‐dim
    if pc.device.type != device:
        pc = pc.to(device)
    single = (pc.dim() == 3)
    if single:
        pc = pc.unsqueeze(0)  # [1,3,H,W]
    B, C, H, W = pc.shape
    if height is None: height = H
    if width  is None: width  = W

    device = device
    # flatten spatial dims
    pc_flat = pc.view(B, C, H*W).permute(0, 2, 1)  # [B, N, 3], N=H*W

    # compute range r and sort
    x, y, z = pc_flat.unbind(-1)
    r = torch.sqrt(x**2 + y**2 + z**2)            # [B, N]
    order = torch.argsort(r, dim=1)               # ascending r
    if not sort_largest_first:
        order = order.flip(dims=[1])              # descending r if we want near last
    # reorder points
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    pc_sorted = pc_flat[batch_idx, order]         # [B, N, 3]
    x_s, y_s, z_s = pc_sorted.unbind(-1)

    # angles
    phi, theta = to_deflection_coordinates(x_s, y_s, z_s)  # each [B, N]

    # determine theta range
    if theta_range is None:
        tmin = float(theta.min())
        tmax = float(theta.max())
    else:
        tmin, tmax = theta_range

    # phi spans full circle
    pmin, pmax = -torch.pi, torch.pi

    # make ascending bin boundaries
    bins_h_asc = torch.linspace(tmin, tmax, steps=height, device=device)
    bins_w_asc = torch.linspace(pmin, pmax, steps=width,  device=device)

    # bucketize angle -> bin index (0..H-1), then flip for top‐down
    idx_h = torch.bucketize(theta, bins_h_asc) - 1      # [B, N]
    idx_h = idx_h.clamp(0, height-1)
    row   = (height - 1) - idx_h                        # [B, N]

    # same for phi, but flip to have leftmost=phi_max
    idx_w = torch.bucketize(phi, bins_w_asc) - 1
    idx_w = idx_w.clamp(0, width-1)
    col   = (width - 1) - idx_w                         # [B, N]

    # scatter back into image grid
    pj = torch.zeros((B, C, height, width), device=device, dtype=pc.dtype)
    # linear index for spatial dim
    lin = row * width + col                             # [B, N]
    # for each batch, scatter along flattened spatial dimension
    # prepare output
    pj = torch.zeros((B, C, height, width),
                    device=device, dtype=pc.dtype)
    pj_flat = pj.view(B, C, -1)   # [B, C, N]

    # build a batched “src” of shape [B, C, N]
    src_all = pc_sorted.permute(0, 2, 1)    # [B, N, C] → [B, C, N]

    # expand lin to [B, C, N] so it lines up with src_all
    idx_all = lin.unsqueeze(1).expand(-1, C, -1)  # [B, 1, N] → [B, C, N]

    # now a single scatter_ along the spatial dim (dim=2)
    pj_flat.scatter_(2, idx_all, src_all)

    pj_img = pj  # [B, 3, height, width]

    # # build alpha grid (same for all batches)
    # th_vals = torch.linspace(tmax, tmin, steps=height, device=device)
    # ph_vals = torch.linspace(pmax, pmin, steps=width,  device=device)
    # theta_grid, phi_grid = torch.meshgrid(th_vals, ph_vals, indexing='ij')
    # #alpha = torch.sqrt(theta_grid**2 + phi_grid**2)     # [H, W]

    if single:
        pj_img = pj_img.squeeze(0)  # back to [3,H,W]
    return pj_img
    #return pj_img, alpha, (tmin, tmax), (pmin, pmax)