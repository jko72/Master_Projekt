from torch.utils.data import Dataset
import torch
import numpy as np
from utils_torch import spherical_projection


class RandomWindowSeqDataset(Dataset):
    def __init__(self, sequences, cfg, device='cpu', with_projection=True, theta_range=[-np.pi/16, np.pi/16]):
        """
        sequences: output of make_sequences()
        cfg:       config dict with
                   cfg['model_params']['input_horizon']
                   cfg['train_params']['output_horizons']
                   cfg['model_params']['grid_height'], grid_width
        """
        self.seqs          = sequences
        self.history       = cfg['model_params']['input_horizon']
        self.future_offs   = cfg['train_params']['output_horizons']
        self.max_off       = max(self.future_offs)
        self.device        = device
        self.out_H         = cfg['model_params']['grid_height']
        self.out_W         = cfg['model_params']['grid_width']
        self.with_projection = cfg['model_params']['preserve_ray_position']
        self.theta_range = theta_range

        # precompute valid windows per sequence
        self.windows = []
        for s_id, seq in enumerate(self.seqs):
            L = len(seq['paths'])
            n_windows = L - (self.history - 1) - self.max_off
            for start in range(n_windows):
                self.windows.append((s_id, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        s_id, start = self.windows[idx]
        seq_paths = self.seqs[s_id]['paths']
        seq_poses = self.seqs[s_id]['poses']  # list of numpy 4×4
        T, H_org, W_org = self.history, 128, 2048   # TODO: no hard encoding

        # --- 1) load & stack history xyz ---
        hist_xyz = []
        for j in range(start, start+T):
            pc_path, _ = seq_paths[j]
            # the (x, y, z, intensity) are stored in binary
            xyzi = np.fromfile(pc_path, dtype=np.float32).reshape(-1,4)
            pts = xyzi[:, :-1].reshape(H_org, W_org, 3)  # [H_org=128, W_org=2048, 3]
            xyz = torch.from_numpy(pts).permute(2, 0, 1).to(self.device)
            
            # project image to target sensor configuration (self.out_H: #layers, self.out_W: #horizontal laser positions)
            xyz = spherical_projection(
                xyz, self.out_H, self.out_W,
                theta_range=self.theta_range,
                device=self.device
            )
            
            hist_xyz.append(xyz)
        hist_xyz = torch.stack(hist_xyz, dim=0)  # [T,3,H,W]
        # mask, where projected points/pixels are out of scope
        mask = (hist_xyz == 0).all(dim=1, keepdim=True)
        # --- 2) get and invert last‐pose ---
        last_pose = torch.tensor(seq_poses[start+T-1],
                                device=self.device,
                                dtype=torch.float32)  # [4,4]
        inv_last  = torch.inverse(last_pose)     # [4,4]
        inv_all  = inv_last.unsqueeze(0).expand(T,4,4)
        
        # --- 3) align history & project ---
        H, W = hist_xyz.shape[-2:]
        xyz_h = torch.cat([
            hist_xyz.unsqueeze(0),                                # [1,T,3,H,W]
            torch.ones((1,T,1,H,W), device=self.device)
        ], dim=2).view(T,4,H * W)                                     # [T,4,N]
        pose_seq = torch.stack([
            torch.tensor(p, device=self.device, dtype=torch.float32)
                for p in seq_poses[start:start+T]] \
            , dim=0)                                              # [T,4,4]
        
        # transform ego to world coordinates
        world    = torch.bmm(pose_seq, xyz_h)                     # [T,4,N]
        
        # transform from world to current frame with ego motion compensated
        aligned  = torch.bmm(inv_all, world)                      # [T,4,N]
        
        hist_pc = aligned[:, :3].view(T,3,H,W)                   # [T,3,H,W]
        hist_pc = hist_pc.masked_fill(mask, 0)
        if self.with_projection:
            # re-project transformed points to reference frame -> pixel/ ray direction vector is consistent over time -> image will get warped/ distorted
                # (+) Keeps theta/phi constant across time, so the model can treat each pixel as a fixed ray. 
                # (+) Reduces spatial drift, simplifying the subsequent per‑pixel attention.
                # (-) Re‑projection introduces sparse holes where no historical point lands
            hist_pc = spherical_projection(
                hist_pc, self.out_H, self.out_W,
                theta_range=self.theta_range,
                device=self.device
            )                                                         # [T,3,out_H,out_W]

        # --- 4) align & project futures, plus range images ---
        future_xyzs, future_ranges = [], []
        for off in self.future_offs:
            j = start + T - 1 + off
            pc_path, _ = seq_paths[j]
            xyzi = np.fromfile(pc_path, dtype=np.float32).reshape(-1,4)
            pts = xyzi[:, :-1].reshape(H_org, W_org, 3)  # [H_org=128, W_org=2048, 3]
            xyz = torch.from_numpy(pts).permute(2, 0, 1).to(self.device)
            
            # project image to target sensor configuration (self.out_H: #layers, self.out_W: #horizontal laser positions)
            xyz = spherical_projection(
                xyz, self.out_H, self.out_W,
                theta_range=self.theta_range,
                device=self.device
            )
            # mask, where projected points/pixels are out of scope
            mask = (xyz == 0).all(dim=0, keepdim=True)
            H, W = xyz.shape[-2:]
            
            fh = torch.cat([
                xyz.unsqueeze(0),                                # [1,T,3,H,W]
                torch.ones((1,1,H,W), device=self.device)
            ], dim=1).view(1,4,H * W)    
            
            pose_j = torch.tensor(seq_poses[j],
                                device=self.device, dtype=torch.float32).unsqueeze(0)  # [1,4,4]
            
            # transform ego to world coordinates
            worldf = torch.bmm(pose_j, fh)                 # [1,4,N]
            
            # transform from world to current frame with ego motion compensated
            alignedf = torch.bmm(inv_last.unsqueeze(0), worldf)  # [1,4,N]
            
            future_pc = alignedf[:, :3].view(3,H,W)             # [3,H,W]
            future_pc = future_pc.masked_fill(mask, 0)
            
            if self.with_projection:
                # re-project transformed points to reference frame -> pixel/ ray direction vector is consistent over time -> image will get warped/ distorted
                    # (+) Keeps theta/phi constant across time, so the model can treat each pixel as a fixed ray. 
                    # (+) Reduces spatial drift, simplifying the subsequent per‑pixel attention.
                    # (-) Re‑projection introduces sparse holes where no historical point lands
                future_pc = spherical_projection(
                    future_pc, self.out_H, self.out_W,
                    theta_range=self.theta_range,
                    device=self.device
                )
            future_xyzs.append(future_pc)                       # [3,out_H,out_W]
            future_ranges.append(torch.norm(future_pc, dim=0)) # [out_H,out_W]

        future_xyz   = torch.stack(future_xyzs,   dim=0)  # [F,3,out_H,out_W]
        future_ranges= torch.stack(future_ranges, dim=0)  # [F,out_H,out_W]
        
        # stack xyz with range
        hist_xyzd = torch.cat([hist_pc, torch.norm(hist_pc, dim=1, keepdim=True)],   dim=1)        # [T, 3, H, W] + [T, 1, H, W]

        return hist_xyzd, future_xyz, future_ranges