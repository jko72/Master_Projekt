import numpy as np
import torch
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
# ---------------------------------------------------------------------
# Paper-style range projection (SemanticKITTI / KITTI Odometry)
# ---------------------------------------------------------------------

def _range_projection_paper(
    pc: np.ndarray,
    proj_H: int,
    proj_W: int,
    fov_up_deg: float,
    fov_down_deg: float,
    max_range: float = 85.0,
    min_range: float = 0.0,
):
    """
    Paper-style range projection.

    Args:
        pc: np.ndarray (N, C) where first 3 columns are x,y,z. More channels allowed (e.g. intensity).
        proj_H, proj_W: output image size
        fov_up_deg, fov_down_deg: LiDAR vertical FoV in degrees (e.g. 3.0, -25.0)
        max_range: clip max range (e.g. 85.0)
        min_range: clip min range (usually 0.0)

    Returns:
        proj_xyz:   (H,W,3) float32 xyz image, invalid pixels = 0
        proj_range: (H,W)   float32 range image, invalid pixels = -1
        proj_idx:   (H,W)   int32 indices into original point cloud, invalid = -1
    """
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"pc must be (N, >=3). Got {pc.shape}")

    pc = pc.astype(np.float32)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    depth = np.sqrt(x * x + y * y + z * z)

    # filter
    valid = (depth > min_range) & (depth < max_range)
    if valid.sum() == 0:
        proj_xyz = np.zeros((proj_H, proj_W, 3), dtype=np.float32)
        proj_range = np.full((proj_H, proj_W), -1.0, dtype=np.float32)
        proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)
        return proj_xyz, proj_range, proj_idx

    x = x[valid]
    y = y[valid]
    z = z[valid]
    depth = depth[valid]
    idxs = np.nonzero(valid)[0].astype(np.int32)

    # angles
    yaw = np.arctan2(y, x)  # [-pi, pi]
    pitch = np.arcsin(z / np.maximum(depth, 1e-8))  # [-pi/2, pi/2]

    # FoV in radians
    fov_up = fov_up_deg / 180.0 * np.pi
    fov_down = fov_down_deg / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    # project to [0,1]
    proj_x = 0.5 * (yaw / np.pi + 1.0)              # [0,1]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov    # [0,1]

    # scale to image size
    proj_x *= proj_W
    proj_y *= proj_H

    proj_x = np.floor(proj_x).astype(np.int32)
    proj_y = np.floor(proj_y).astype(np.int32)

    proj_x = np.clip(proj_x, 0, proj_W - 1)
    proj_y = np.clip(proj_y, 0, proj_H - 1)

    # init outputs
    proj_xyz = np.zeros((proj_H, proj_W, 3), dtype=np.float32)
    proj_range = np.full((proj_H, proj_W), -1.0, dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)

    # IMPORTANT: overwrite order
    # We want nearest point to win. Sort by depth DESC so nearer gets written LAST.
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    x = x[order]; y = y[order]; z = z[order]
    idxs = idxs[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    proj_xyz[proj_y, proj_x, 0] = x
    proj_xyz[proj_y, proj_x, 1] = y
    proj_xyz[proj_y, proj_x, 2] = z
    proj_range[proj_y, proj_x] = depth
    proj_idx[proj_y, proj_x] = idxs

    return proj_xyz, proj_range, proj_idx


# ---------------------------------------------------------------------
# Drop-in replacement: keep your old function name/signature
# ---------------------------------------------------------------------

def spherical_projection(
    pc,
    height=64,
    width=2048,
    theta_range=None,
    th=0.0,
    sort_largest_first=False,
    bins_h=None,
    max_range=None,
):
    """
    DROP-IN replacement for your old spherical_projection.

    Your old code returned:
        pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max)

    This patch returns the SAME tuple shape, so your calling code doesn't break.
    BUT internally it uses paper-style yaw/pitch projection.
    - pj_img: (H,W,C) where channels are taken from input pc columns.
             If pc is (N,3) -> pj_img is (H,W,3)
             If pc is (N,4) -> pj_img is (H,W,4) (e.g., intensity kept)
    - alpha, theta/phi ranges are dummies here (kept for compatibility).
    """
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"spherical_projection expects pc (N,>=3). Got {pc.shape}")

    height = int(height)
    width = int(width)
    C = pc.shape[1]

    pj_img = np.zeros((height, width, C), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    phi_min, phi_max = -np.pi, np.pi

    if pc.shape[0] == 0:
        if theta_range is None:
            theta_min, theta_max = -np.pi / 16.0, np.pi / 16.0
        else:
            theta_min, theta_max = float(theta_range[0]), float(theta_range[1])
        return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max)

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    valid = np.isfinite(r) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if max_range is not None:
        valid &= (r < float(max_range))
    if not np.any(valid):
        if theta_range is None:
            theta_min, theta_max = -np.pi / 16.0, np.pi / 16.0
        else:
            theta_min, theta_max = float(theta_range[0]), float(theta_range[1])
        return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max)

    pc = pc[valid]
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    r = np.sqrt(x * x + y * y + z * z)

    # Old behavior: sort by range and resolve collisions by overwrite order.
    order = np.argsort(r)  # ascending
    if not sort_largest_first:
        order = order[::-1]  # descending => near points are written last
    pc_sorted = pc[order]
    x_s = pc_sorted[:, 0]
    y_s = pc_sorted[:, 1]
    z_s = pc_sorted[:, 2]

    phi = np.arctan2(y_s, x_s)  # [-pi, pi]
    theta = np.arcsin(z_s / np.maximum(np.sqrt(x_s * x_s + y_s * y_s + z_s * z_s), 1e-8))

    if theta_range is None:
        theta_min = float(theta.min())
        theta_max = float(theta.max())
    else:
        theta_min, theta_max = float(theta_range[0]), float(theta_range[1])
        if theta_max <= theta_min:
            raise ValueError(f"theta_range must satisfy max > min, got {theta_range}")

    # Avoid degenerate bucket boundaries.
    if theta_max - theta_min < 1e-8:
        theta_max = theta_min + 1e-8

    bins_h_asc = np.linspace(theta_min, theta_max, num=height, dtype=np.float32)
    bins_w_asc = np.linspace(phi_min, phi_max, num=width, dtype=np.float32)

    idx_h = np.searchsorted(bins_h_asc, theta, side="right") - 1
    idx_h = np.clip(idx_h, 0, height - 1)
    row = (height - 1) - idx_h

    idx_w = np.searchsorted(bins_w_asc, phi, side="right") - 1
    idx_w = np.clip(idx_w, 0, width - 1)
    col = (width - 1) - idx_w

    pj_img[row, col, :] = pc_sorted

    # Keep compatibility alpha output.
    th_vals = np.linspace(theta_max, theta_min, num=height, dtype=np.float32)
    ph_vals = np.linspace(phi_max, phi_min, num=width, dtype=np.float32)
    theta_grid, phi_grid = np.meshgrid(th_vals, ph_vals, indexing='ij')
    alpha = np.sqrt(theta_grid * theta_grid + phi_grid * phi_grid).astype(np.float32)

    return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max)

# import torch
# import numpy as np

# def make_angle_grids(height, width, theta_range, device="cpu"):
#     """
#     Create per-pixel angle grids for spherical projection.

#     Args:
#         height: number of elevation bins (H)
#         width:  number of azimuth bins (W)
#         theta_range: (tmin, tmax) elevation bounds
#         device: compute device

#     Returns:
#         phi_grid   : Tensor [H, W] azimuth angles
#         theta_grid : Tensor [H, W] elevation angles
#     """
#     tmin, tmax = theta_range
#     pmin, pmax = -np.pi, np.pi
#     th_vals = torch.linspace(tmax, tmin, steps=height, device=device)
#     ph_vals = torch.linspace(pmin, pmax, steps=width,  device=device)
#     theta_grid, phi_grid = torch.meshgrid(th_vals, ph_vals, indexing='ij')
#     return phi_grid, theta_grid

# def to_deflection_coordinates(x, y, z, eps=1e-8):
#     """
#     NumPy version.
#     x,y,z: np.ndarray shape (N,)
#     returns:
#       phi   azimuth  in [-pi, pi]
#       theta elevation in [-pi/2, pi/2]
#     """
#     phi = np.arctan2(y, x)
#     p = np.sqrt(x**2 + y**2 + eps)
#     theta = np.arctan2(z, p)
#     return phi, theta

# # def to_deflection_coordinates(x, y, z):
# #     """
# #     Convert Cartesian (x,y,z) to 
# #     - phi ∈ [–π, π]
# #     - theta ∈ [0, π]
# #     """
# #     p   = torch.sqrt(x**2 + y**2)
# #     phi = torch.atan2(y, x)
# #     theta = -torch.atan2(p, z) + (torch.pi / 2)
# #     return phi, theta

# def spherical_projection(pc, height=64, width=2048, theta_range=None, th=1.0, sort_largest_first=False, bins_h=None, max_range=None):
#     '''spherical projection 
#     Args:
#         pc: point cloud, dim: N*C
#     Returns:
#         pj_img: projected spherical iamges, shape: h*w*C
#     '''

#     # filter all small range values to avoid overflows in theta min max calculation
#     #if isinstance(theta_range, type(None)):
        
#     r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
#     arr1inds = r.argsort()
#     if sort_largest_first:
#         pc = pc[arr1inds]
#     else:
#         pc = pc[arr1inds[::-1]]
#     #pc = pc[arr1inds]
#     # r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
#     # if not isinstance(max_range,type(None)):
#     #     indices = np.where((r > th)*(r<=max_range))
#     # else:
#     #     indices = np.where(r > th)
#     # pc = pc[indices]
        
#     x = pc[:, 0]
#     y = pc[:, 1]
#     z = pc[:, 2]

#     r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
#     phi, theta = to_deflection_coordinates(x,y,z)

#     #indices = np.where(r > th)
#     if isinstance(theta_range, type(None)):
#         theta_min, theta_max = [theta.min(), theta.max()]
#     else: 
#         theta_min, theta_max = theta_range
        
#     phi_min, phi_max = [-np.pi, np.pi]
    
#     # assuming uniform distribution of rays
#     if isinstance(bins_h, type(None)):
#         bins_h = np.linspace(theta_min, theta_max, height)[::-1]
        
#     bins_w = np.linspace(phi_min, phi_max, width)[::-1]
    
#     theta_img = np.stack(width*[bins_h], axis=-1)
#     phi_img = np.stack(height*[bins_w], axis=0)

#     idx_h = np.digitize(theta, bins_h)-1
#     idx_w = np.digitize(phi, bins_w)-1
#     #idx_h = np.clip(idx_h, 0, height - 1)
#     #idx_w = np.clip(idx_w, 0, width  - 1)

    
#     pj_img = np.zeros((height, width, pc.shape[1])).astype(np.float32)

    
#     pj_img[idx_h, idx_w, :] = pc

   
#     alpha = np.sqrt(np.square(theta_img)+np.square(phi_img))
   
#     return pj_img, alpha, (theta_min, theta_max), (phi_min, phi_max) 


# def spherical_projection(pc, height=None, width=None,
#                          theta_range=None, th=1.0,
#                          sort_largest_first=False,
#                          device="cpu"):
#     """
#     Args:
#         pc            : Tensor [B, 3, H, W] (or [3, H, W]). Channels are (x,y,z).
#         height, width : output image size; if None, uses H, W of pc.
#         theta_range   : (min, max) to clamp theta; if None, computed from data.
#         th            : range threshold (currently unused; you can insert filtering).
#         sort_largest_first : if True, far points override near; else, near override far.

#     Returns:
#         pj_img   : [B, 3, height, width]  spherical‐projected point‐cloud image
#         alpha    : [height, width]        per‐pixel distance in (phi,theta)‐space
#         (tmin,tmax) : tuple of floats, theta range used
#         (pmin,pmax) : tuple of floats, phi range used (= (–π, π))
#     """
#     # ensure batch‐dim
#     if pc.device.type != device:
#         pc = pc.to(device)
#     single = (pc.dim() == 3)
#     if single:
#         pc = pc.unsqueeze(0)  # [1,3,H,W]
#     B, C, H, W = pc.shape
#     if height is None: height = H
#     if width  is None: width  = W

#     device = device
#     # flatten spatial dims
#     pc_flat = pc.view(B, C, H*W).permute(0, 2, 1)  # [B, N, 3], N=H*W

#     # compute range r and sort
#     x, y, z = pc_flat.unbind(-1)
#     r = torch.sqrt(x**2 + y**2 + z**2)            # [B, N]
#     order = torch.argsort(r, dim=1)               # ascending r
#     if not sort_largest_first:
#         order = order.flip(dims=[1])              # descending r if we want near last
#     # reorder points
#     batch_idx = torch.arange(B, device=device).unsqueeze(1)
#     pc_sorted = pc_flat[batch_idx, order]         # [B, N, 3]
#     x_s, y_s, z_s = pc_sorted.unbind(-1)

#     # angles
#     phi, theta = to_deflection_coordinates(x_s, y_s, z_s)  # each [B, N]

#     # determine theta range
#     if theta_range is None:
#         tmin = float(theta.min())
#         tmax = float(theta.max())
#     else:
#         tmin, tmax = theta_range

#     # phi spans full circle
#     pmin, pmax = -torch.pi, torch.pi

#     # make ascending bin boundaries
#     bins_h_asc = torch.linspace(tmin, tmax, steps=height, device=device)
#     bins_w_asc = torch.linspace(pmin, pmax, steps=width,  device=device)

#     # bucketize angle -> bin index (0..H-1), then flip for top‐down
#     idx_h = torch.bucketize(theta, bins_h_asc) - 1      # [B, N]
#     idx_h = idx_h.clamp(0, height-1)
#     row   = (height - 1) - idx_h                        # [B, N]

#     # same for phi, but flip to have leftmost=phi_max
#     idx_w = torch.bucketize(phi, bins_w_asc) - 1
#     idx_w = idx_w.clamp(0, width-1)
#     col   = (width - 1) - idx_w                         # [B, N]

#     # scatter back into image grid
#     pj = torch.zeros((B, C, height, width), device=device, dtype=pc.dtype)
#     # linear index for spatial dim
#     lin = row * width + col                             # [B, N]
#     # for each batch, scatter along flattened spatial dimension
#     # prepare output
#     pj = torch.zeros((B, C, height, width),
#                     device=device, dtype=pc.dtype)
#     pj_flat = pj.view(B, C, -1)   # [B, C, N]

#     # build a batched “src” of shape [B, C, N]
#     src_all = pc_sorted.permute(0, 2, 1)    # [B, N, C] → [B, C, N]

#     # expand lin to [B, C, N] so it lines up with src_all
#     idx_all = lin.unsqueeze(1).expand(-1, C, -1)  # [B, 1, N] → [B, C, N]

#     # now a single scatter_ along the spatial dim (dim=2)
#     pj_flat.scatter_(2, idx_all, src_all)

#     pj_img = pj  # [B, 3, height, width]

#     # # build alpha grid (same for all batches)
#     # th_vals = torch.linspace(tmax, tmin, steps=height, device=device)
#     # ph_vals = torch.linspace(pmax, pmin, steps=width,  device=device)
#     # theta_grid, phi_grid = torch.meshgrid(th_vals, ph_vals, indexing='ij')
#     # #alpha = torch.sqrt(theta_grid**2 + phi_grid**2)     # [H, W]

#     if single:
#         pj_img = pj_img.squeeze(0)  # back to [3,H,W]
#     return pj_img
#     #return pj_img, alpha, (tmin, tmax), (pmin, pmax)
