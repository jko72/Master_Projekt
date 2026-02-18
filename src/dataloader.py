from torch.utils.data import Dataset
import torch
import numpy as np
from utils_torch import spherical_projection


class RandomWindowSeqDataset(Dataset):
    def __init__(self, sequences, cfg, device='cpu', with_projection=True, theta_range=None):
        """
        sequences: output of make_sequences()
        cfg:       config dict with
                   cfg['model_params']['input_horizon']
                   cfg['train_params']['output_horizons']
                   cfg['model_params']['grid_height'], grid_width
        """
        self.seqs = sequences
        self.history = cfg['model_params']['input_horizon']
        self.future_offs = cfg['train_params']['output_horizons']
        self.max_off = max(self.future_offs)
        self.device = device
        self.out_H = cfg['model_params']['grid_height']
        self.out_W = cfg['model_params']['grid_width']
        self.org_H = cfg['model_params'].get('org_grid_height', 128)
        self.org_W = cfg['model_params'].get('org_grid_width', 2048)
        self.with_projection = cfg['model_params']['preserve_ray_position']
        if theta_range is not None:
            self.theta_range = [float(theta_range[0]), float(theta_range[1])]
        else:
            fov_up = float(cfg['model_params'].get('FOV_UP', 3.0))
            fov_down = float(cfg['model_params'].get('FOV_DOWN', -25.0))
            self.theta_range = [fov_down * np.pi / 180.0, fov_up * np.pi / 180.0]

        # precompute valid windows per sequence
        self.windows = []
        for s_id, seq in enumerate(self.seqs):
            L = len(seq['paths'])
            n_windows = L - (self.history - 1) - self.max_off
            for start in range(n_windows):
                self.windows.append((s_id, start))

    def __len__(self):
        return len(self.windows)

    def _load_organized_xyz(self, pc_path: str) -> torch.Tensor:
        """
        Load scan and return organized xyz image [3, org_H, org_W].
        Works for:
        - organized scans (direct reshape)
        - unorganized scans (project to fixed org grid first)
        """
        xyzi = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        expected = self.org_H * self.org_W

        if xyzi.shape[0] == expected:
            pts = xyzi[:, :-1].reshape(self.org_H, self.org_W, 3)
        else:
            pj_img, _, _, _ = spherical_projection(
                xyzi,
                height=self.org_H,
                width=self.org_W,
                theta_range=self.theta_range
            )  # [org_H, org_W, 4]
            pts = pj_img[:, :, :3]

        return torch.from_numpy(pts.astype(np.float32)).permute(2, 0, 1).to(self.device)

    def _project_xyz_tensor(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper:
        old code projected torch tensors [3,H,W] or [T,3,H,W];
        current spherical_projection expects NumPy (N,>=3).
        """
        if xyz.dim() == 3:
            pts = (
                xyz.permute(1, 2, 0)
                .reshape(-1, xyz.shape[0])
                .contiguous()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            pj_img, _, _, _ = spherical_projection(
                pts,
                height=self.out_H,
                width=self.out_W,
                theta_range=self.theta_range
            )
            return torch.from_numpy(pj_img).permute(2, 0, 1).to(self.device)

        if xyz.dim() == 4:
            out = []
            for t in range(xyz.shape[0]):
                out.append(self._project_xyz_tensor(xyz[t]))
            return torch.stack(out, dim=0)

        raise ValueError(f"_project_xyz_tensor expects [3,H,W] or [T,3,H,W], got {tuple(xyz.shape)}")

    def __getitem__(self, idx):
        s_id, start = self.windows[idx]
        seq_paths = self.seqs[s_id]['paths']
        seq_poses = self.seqs[s_id]['poses']  # list of numpy 4x4
        T, H_org, W_org = self.history, self.org_H, self.org_W

        # --- 1) load & stack history xyz ---
        hist_xyz = []
        for j in range(start, start+T):
            pc_path, _ = seq_paths[j]
            xyz = self._load_organized_xyz(pc_path)
            xyz = self._project_xyz_tensor(xyz)
            hist_xyz.append(xyz)
        hist_xyz = torch.stack(hist_xyz, dim=0)

        # mask, where projected points/pixels are out of scope
        mask = (hist_xyz == 0).all(dim=1, keepdim=True)

        # --- 2) get and invert last-pose ---
        last_pose = torch.tensor(
            seq_poses[start + T - 1],
            device=self.device,
            dtype=torch.float32
        )  # [4,4]
        inv_last = torch.inverse(last_pose)
        inv_all = inv_last.unsqueeze(0).expand(T, 4, 4)

        # --- 3) align history & project ---
        H, W = hist_xyz.shape[-2:]
        xyz_h = torch.cat([
            hist_xyz.unsqueeze(0),  # [1,T,3,H,W]
            torch.ones((1, T, 1, H, W), device=self.device)
        ], dim=2).view(T, 4, H * W)  # [T,4,N]
        pose_seq = torch.stack([
            torch.tensor(p, device=self.device, dtype=torch.float32)
            for p in seq_poses[start:start + T]
        ], dim=0)  # [T,4,4]

        world = torch.bmm(pose_seq, xyz_h)
        aligned = torch.bmm(inv_all, world)

        hist_pc = aligned[:, :3].view(T, 3, H, W)  # [T,3,H,W]
        hist_pc = hist_pc.masked_fill(mask, 0)
        if self.with_projection:
            hist_pc = self._project_xyz_tensor(hist_pc)

        # --- 4) align & project futures, plus range images ---
        future_xyzs, future_ranges = [], []
        for off in self.future_offs:
            j = start + T - 1 + off
            pc_path, _ = seq_paths[j]
            xyz = self._load_organized_xyz(pc_path)
            xyz = self._project_xyz_tensor(xyz)

            # mask, where projected points/pixels are out of scope
            mask = (xyz == 0).all(dim=0, keepdim=True)
            H, W = xyz.shape[-2:]

            fh = torch.cat([
                xyz.unsqueeze(0),  # [1,T,3,H,W]
                torch.ones((1, 1, H, W), device=self.device)
            ], dim=1).view(1, 4, H * W)

            pose_j = torch.tensor(
                seq_poses[j], device=self.device, dtype=torch.float32
            ).unsqueeze(0)  # [1,4,4]

            worldf = torch.bmm(pose_j, fh)
            alignedf = torch.bmm(inv_last.unsqueeze(0), worldf)

            future_pc = alignedf[:, :3].view(3, H, W)  # [3,H,W]
            future_pc = future_pc.masked_fill(mask, 0)

            if self.with_projection:
                future_pc = self._project_xyz_tensor(future_pc)

            future_xyzs.append(future_pc)                        # [3,out_H,out_W]
            future_ranges.append(torch.norm(future_pc, dim=0))  # [out_H,out_W]

        future_xyz = torch.stack(future_xyzs, dim=0)      # [F,3,out_H,out_W]
        future_ranges = torch.stack(future_ranges, dim=0)  # [F,out_H,out_W]

        # stack xyz with range
        hist_xyzd = torch.cat(
            [hist_pc, torch.norm(hist_pc, dim=1, keepdim=True)],
            dim=1
        )  # [T,4,H,W]

        return hist_xyzd, future_xyz, future_ranges
