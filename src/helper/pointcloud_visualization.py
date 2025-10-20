import torch

def pointcloud_from_expected_range(
    r_exp: torch.Tensor,           # [B, T, H, W]
    phi_grid: torch.Tensor,        # [H, W]
    theta_grid: torch.Tensor,      # [H, W]
    alpha: torch.Tensor = None,    # [B, T, H, W, K], optional confidences
    alpha_thresh: float = 0.1
):
    """
    Returns a list of length B*T of [Ni,3] point tensors.
    """
    B, T, H, W = r_exp.shape
    pcs = []
    # optionally compute max‐alpha per pixel
    if alpha is not None:
        conf = alpha.max(dim=-1)[0]  # [B,T,H,W]
    else:
        conf = None

    for b in range(B):
        for t in range(T):
            r_bt = r_exp[b,t]               # [H,W]
            mask  = r_bt > 0
            if conf is not None:
                mask &= (conf[b,t] >= alpha_thresh)
            idx = mask.nonzero(as_tuple=False)  # [N,2]
            if idx.numel()==0:
                pcs.append(torch.empty((0,3), dtype=torch.float32))
                continue
            i,j = idx[:,0], idx[:,1]
            r_vals = r_bt[i,j]
            phi = phi_grid[i,j]
            theta = theta_grid[i,j]
            x = r_vals * torch.cos(theta) * torch.cos(phi)
            y = r_vals * torch.cos(theta) * torch.sin(phi)
            z = r_vals * torch.sin(theta)
            pcs.append(torch.stack([x,y,z], dim=-1))
    return pcs