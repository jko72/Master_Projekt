import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import random

#from pyTorchChamferDistance.chamfer_distance import ChamferDistance
try:
    from pyTorchChamferDistance.chamfer_distance import ChamferDistance
except ModuleNotFoundError:
    class ChamferDistance:
        def __init__(self):
            print("[WARN] ChamferDistance module not found – using dummy placeholder.")
        def __call__(self, x, y):
            # Gibt Dummywerte zurück (0 statt echten Distanzwert)
            return torch.zeros_like(x[..., 0]), torch.zeros_like(y[..., 0])

#from utils.projection import projection
try:
    from models.projection import projection
except ModuleNotFoundError:
    class projection:
        def __init__(self, cfg=None):
            print("[WARN] utils.projection not found – using dummy projection.")
        def get_target_mask_from_range_view(self, range_view):
            # Gibt Maske mit 1 für alle gültigen Pixel (alles aktiv)
            return torch.ones_like(range_view, dtype=torch.float32)
        def get_masked_range_view(self, output):
            # Gibt Range-View direkt zurück
            return output["rv"]
        def get_valid_points_from_range_view(self, range_view):
            # Dummy-Valid-Punkte (keine echte Projektion)
            pts = torch.nonzero(range_view > 0, as_tuple=False).float()
            return pts
from models.chamfer import cham_dist

def diff_div_reg(pred_y, batch_y, tau=0.1, eps=1e-12):
        pred_y = pred_y["rv"].clone()
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()

class Loss(nn.Module):
    """Combined loss for point cloud prediction"""

    def __init__(self, cfg):
        """Init"""
        super().__init__()
        self.cfg = cfg
        self.n_future_steps = self.cfg.get("MODEL", self.cfg.get("model_params", {})).get("forecast_horizon", 3)

        #  Unterstützt sowohl TRAIN als auch train_params (robust)
        train_cfg = self.cfg.get("TRAIN", self.cfg.get("train_params", {}))

        #  Sichere Zugriffsmethode mit Standardwerten
        self.loss_weight_cd   = train_cfg.get("LOSS_WEIGHT_CHAMFER_DISTANCE", 0.0)
        self.loss_weight_rv   = train_cfg.get("LOSS_WEIGHT_RANGE_VIEW", 1.0)
        self.loss_weight_mask = train_cfg.get("LOSS_WEIGHT_MASK", 1.0)

        self.alpha  = 0.1
        self.loss_range = loss_range(self.cfg)
        self.chamfer_distance = cham_dist(self.cfg)
        self.loss_mask = loss_mask(self.cfg)

    def forward(self, output, target, mode, epoch_number=40):
        """Forward pass with multiple loss components

        Args:
        output (dict): Predicted mask logits and ranges
        target (torch.tensor): Target range image
        mode (str): Mode (train,val,test)

        Returns:
        dict: Dict with loss components
        """
        # print(f"The output shape {output['rv'].shape}")
        # print(f"The target shape {target.shape}")
        target_range_image = target[:,:, 0, :, :]

        # Range view
        loss_range_view, loss_range_timestep, valid_ratio = self.loss_range(output, target_range_image)

        # Mask
        loss_mask = self.loss_mask(output, target_range_image)

        # Chamfer Distance
        if (epoch_number>=100 and self.loss_weight_cd > 0.0) or mode == "val" or mode == "test":
            chamfer_distance, chamfer_distances_tensor = self.chamfer_distance(
                output, target, self.cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"]
            )
            loss_chamfer_distance = sum([cd for cd in chamfer_distance.values()]) / len(
                chamfer_distance
            )
            detached_chamfer_distance = {
                step: cd.detach() for step, cd in chamfer_distance.items()
            }
        else:
            chamfer_distance = dict(
                (step, torch.zeros(1).type_as(target_range_image))
                for step in range(self.n_future_steps)
            )
            chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)
            loss_chamfer_distance = torch.zeros_like(loss_range_view)
            detached_chamfer_distance = chamfer_distance

        loss = (
            self.loss_weight_cd * loss_chamfer_distance
            + self.loss_weight_rv * loss_range_view
            + self.loss_weight_mask * loss_mask
        )
        
        loss_dict = {
            "loss": loss,
            "chamfer_distance": detached_chamfer_distance,
            "chamfer_distances_tensor": chamfer_distances_tensor.detach(),
            "mean_chamfer_distance": loss_chamfer_distance.detach(),
            "final_chamfer_distance": chamfer_distance[
                self.n_future_steps - 1
            ].detach(),
            "loss_range_view": loss_range_view.detach(),
            "loss_range_timestep": loss_range_timestep.detach(),
            "loss_mask": loss_mask.detach(),
            "valid_ratio": valid_ratio.detach(),
        }
        return loss_dict


class loss_mask(nn.Module):
    """Binary cross entropy loss for prediction of valid mask"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.projection = projection(self.cfg)

    def forward(self, output, target_range_view):
        target_mask = self.projection.get_target_mask_from_range_view(target_range_view)
        loss = self.loss(output["mask_logits"], target_mask)
        return loss


class loss_range(nn.Module):
    """
    L1-Loss für die Range-Vorhersage mit Maskierung:
    Pixel mit Range == 0 oder == -1 werden ignoriert,
    da sie keine gültigen LiDAR-Entfernungen repräsentieren.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # L1Loss mit 'none', damit wir pro Pixel den Fehler berechnen können
        self.loss = nn.L1Loss(reduction="none")

    def forward(self, output, target_range_image):
        """
        Args:
            output: dict mit 'rv' (predicted range view), [B, T, H, W]
            target_range_image: Ground Truth ranges, [B, T, H, W]

        Returns:
            loss: gemittelter Loss über alle gültigen Pixel
            timestep_loss: Loss pro Zeitschritt
        """

        # 1 Vorhersage kopieren, damit das Original nicht verändert wird
        pred = output["rv"].clone()

        # 2 Gültigkeitsmaske: != -1 verwenden
        valid_mask = (target_range_image != -1.0)

        # 3 Pixelweise L1-Differenz
        pixelwise_loss = torch.abs(pred - target_range_image)

        # 4 Maskieren: Nur gültige Pixel beibehalten
        masked_loss = pixelwise_loss * valid_mask

        # 5 Mittelwert nur über gültige Pixel
        loss = masked_loss.sum() / (valid_mask.sum() + 1e-8)

        # 6 Optional: Loss pro Zeitschritt (für Logging)
        timestep_loss = torch.zeros(target_range_image.shape[1], device=pred.device)
        for i in range(target_range_image.shape[1]):
            valid_t = (target_range_image[:, i, :, :] != -1.0)
            step_loss = torch.abs(pred[:, i, :, :] - target_range_image[:, i, :, :]) * valid_t
            timestep_loss[i] = step_loss.sum() / (valid_t.sum() + 1e-8)
        
        # 7 Anteil gültiger Pixel (Monitoring)
        total_pixels = target_range_image.numel()
        valid_ratio = valid_mask.sum().float() / total_pixels

        return loss, timestep_loss, valid_ratio


class chamfer_distance(nn.Module):
    """Chamfer distance loss. Additionally, the implementation allows the evaluation
    on downsampled point cloud (this is only for comparison to other methods but not recommended,
    because it is a bad approximation of the real Chamfer distance.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = ChamferDistance()
        self.projection = projection(self.cfg)

    def forward(self, output, target, n_samples):
        batch_size, n_future_steps, H, W = output["rv"].shape
        masked_output = self.projection.get_masked_range_view(output)
        chamfer_distances = {}
        chamfer_distances_tensor = torch.zeros(n_future_steps, batch_size)
        for s in range(n_future_steps):
            chamfer_distances[s] = 0
            for b in range(batch_size):
                output_points = self.projection.get_valid_points_from_range_view(
                    masked_output[b, s, :, :]
                ).view(1, -1, 3)
                target_points = target[b, s, 1:4, :, :].permute(1, 2, 0)
                target_points = target_points[target[b, s, 0, :, :] > 0.0].view(
                    1, -1, 3
                )

                if n_samples != -1:
                    n_output_points = output_points.shape[1]
                    n_target_points = target_points.shape[1]
                    n_samples = min(n_samples, n_output_points, n_target_points)

                    sampled_output_indices = random.sample(
                        range(n_output_points), n_samples
                    )
                    sampled_target_indices = random.sample(
                        range(n_target_points), n_samples
                    )

                    output_points = output_points[:, sampled_output_indices, :]
                    target_points = target_points[:, sampled_target_indices, :]

                dist1, dist2 = self.loss(output_points, target_points)
                dist_combined = torch.mean(dist1) + torch.mean(dist2)
                chamfer_distances[s] += dist_combined
                chamfer_distances_tensor[s, b] = dist_combined
            chamfer_distances[s] = chamfer_distances[s] / batch_size
        return chamfer_distances, chamfer_distances_tensor

