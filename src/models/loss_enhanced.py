import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math

# -- optionale Fallbacks für externe Abhängigkeiten
try:
    from pyTorchChamferDistance.chamfer_distance import ChamferDistance  # nicht genutzt, nur Kompatibilität
except ModuleNotFoundError:
    class ChamferDistance:
        def __init__(self):
            print("[WARN] ChamferDistance module not found – (not used).")
        def __call__(self, x, y):
            return torch.zeros_like(x[..., 0]), torch.zeros_like(y[..., 0])

try:
    from utils.projection import projection
except ModuleNotFoundError:
    class projection:
        def __init__(self, cfg=None):
            print("[WARN] utils.projection not found – using dummy projection.")
        def get_target_mask_from_range_view(self, range_view):
            return (range_view > 0.0).float() * (range_view != -1.0).float()
        def get_masked_range_view(self, output):
            return output["rv"]
        def get_valid_points_from_range_view(self, range_view):
            pts = torch.nonzero(range_view > 0, as_tuple=False).float()
            return pts

# ---------------------------
# Hilfsfunktionen
# ---------------------------

def spatial_gradients(img: torch.Tensor):
    """
    img: [B, T, H, W]
    returns:
        dx: [B, T, H, W-1]
        dy: [B, T, H-1, W]
    """
    dx = img[..., :, 1:] - img[..., :, :-1]
    dy = img[..., 1:, :] - img[..., :-1, :]
    return dx, dy

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """gemittelter Wert nur über maskierte Elemente"""
    s = (x * mask).sum()
    n = mask.sum().clamp_min(eps)
    return s / n

def downscale_mask_for_dx(mask: torch.Tensor):
    # valid für dx wenn beide benachbarten Pixel gültig sind
    return (mask[..., :, 1:] * mask[..., :, :-1]).float()

def downscale_mask_for_dy(mask: torch.Tensor):
    return (mask[..., 1:, :] * mask[..., :-1, :]).float()

def temporal_tv(pred: torch.Tensor, valid_mask: torch.Tensor):
    """
    pred: [B, T, H, W]
    valid_mask: [B, T, H, W] (bool/0-1)
    """
    if pred.size(1) < 2:
        return pred.new_tensor(0.0)
    dt = torch.abs(pred[:, 1:] - pred[:, :-1])  # [B, T-1, H, W]
    valid_t = (valid_mask[:, 1:] * valid_mask[:, :-1]).float()
    return masked_mean(dt, valid_t)

def diff_div_reg(pred_y, batch_y, tau=0.1, eps=1e-12):
    """Vorhanden, standardmäßig NICHT in Loss integriert (kann per Gewicht aktiviert werden)."""
    pred_y = pred_y["rv"].clone()
    B, T, C = pred_y.shape[:3] if pred_y.ndim == 5 else (pred_y.shape[0], pred_y.shape[1], 1)
    if T <= 2:
        return pred_y.new_tensor(0.0)
    gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
    gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
    softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
    softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
    loss_gap = softmax_gap_p * torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
    return loss_gap.mean()

# ---------------------------
# Mask-Loss
# ---------------------------

class loss_mask(nn.Module):
    """Masken-Loss: BCEWithLogits oder optional Focal-BCE (per Config)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        train_cfg = self.cfg.get("TRAIN", self.cfg.get("train_params", {}))
        self.use_focal = bool(train_cfg.get("USE_FOCAL_MASK", False))
        self.focal_alpha = float(train_cfg.get("FOCAL_ALPHA", 0.25))
        self.focal_gamma = float(train_cfg.get("FOCAL_GAMMA", 2.0))
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.projection = projection(self.cfg)

    def forward(self, output, target_range_view):
        target_mask = self.projection.get_target_mask_from_range_view(target_range_view)  # [B,T,H,W]
        logits = output["mask_logits"]

        if not self.use_focal:
            return self.bce(logits, target_mask)

        # Focal BCE
        with torch.no_grad():
            y = target_mask
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        focal = -alpha * y * (1 - p).pow(gamma) * torch.log(p) \
                - (1 - alpha) * (1 - y) * p.pow(gamma) * torch.log(1 - p)
        return focal.mean()

# ---------------------------
# Range-Loss (Smooth-L1 + Gradient-Konsistenz)
# ---------------------------

class loss_range(nn.Module):
    """
    Smooth-L1 (Huber) auf Range-View mit Gültigkeitsmaske
    + zusätzlicher Gradient-Loss, um Kanten zu erhalten.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        train_cfg = self.cfg.get("TRAIN", self.cfg.get("train_params", {}))
        # Huber/ Smooth-L1 Parameter
        self.huber_beta = float(train_cfg.get("SMOOTHL1_BETA", 0.5))
        # Gewicht des Gradienten-Loss wird im Haupt-Loss gewichtet; hier nur vorbereitet

    def forward(self, output, target_range_image):
        """
        output["rv"]: [B, T, H, W]
        target_range_image: [B, T, H, W]
        returns:
            base_range_loss (Smooth-L1, maskiert)
            timestep_loss (Smooth-L1 pro T)
            grad_loss (L1 der Gradienten, maskiert)
            valid_ratio
        """
        pred = output["rv"].clone()                       # [B,T,H,W]
        gt   = target_range_image                         # [B,T,H,W]

        valid_mask = ((gt > 0.0) & (gt != -1.0)).float()

        # Smooth-L1 pro Pixel
        # (PyTorch SmoothL1Loss mit reduction='none' arbeitet elementweise als Huber)
        base_pixel = F.smooth_l1_loss(pred, gt, beta=self.huber_beta, reduction="none")
        base_loss  = masked_mean(base_pixel, valid_mask)

        # Loss pro Zeitschritt (nur base)
        T = gt.shape[1]
        timestep_loss = pred.new_zeros(T)
        for i in range(T):
            valid_t = ((gt[:, i] > 0.0) & (gt[:, i] != -1.0)).float()
            pix_t = F.smooth_l1_loss(pred[:, i], gt[:, i], beta=self.huber_beta, reduction="none")
            timestep_loss[i] = masked_mean(pix_t, valid_t)

        # Gradient-Konsistenz (|∇pred - ∇gt|, maskiert)
        dx_p, dy_p = spatial_gradients(pred)
        dx_g, dy_g = spatial_gradients(gt)

        valid_dx = downscale_mask_for_dx(valid_mask)
        valid_dy = downscale_mask_for_dy(valid_mask)

        grad_loss = masked_mean(torch.abs(dx_p - dx_g), valid_dx) \
                  + masked_mean(torch.abs(dy_p - dy_g), valid_dy)

        valid_ratio = valid_mask.mean()

        return base_loss, timestep_loss, grad_loss, valid_ratio

# ---------------------------
# Haupt-Loss Container
# ---------------------------

class Loss(nn.Module):
    """Kombinierter Loss ohne Chamfer und ohne Gauß/NLL."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_future_steps = self.cfg.get("MODEL", self.cfg.get("model_params", {})).get("N_FUTURE_STEPS", 3)

        train_cfg = self.cfg.get("TRAIN", self.cfg.get("train_params", {}))

        # Gewichte
        self.loss_weight_rv        = float(train_cfg.get("LOSS_WEIGHT_RANGE_VIEW", 1.0))
        self.loss_weight_mask      = float(train_cfg.get("LOSS_WEIGHT_MASK", 1.0))
        self.loss_weight_rv_grad   = float(train_cfg.get("LOSS_WEIGHT_RANGE_GRADIENT", 0.05))   # neu
        self.loss_weight_tv_time   = float(train_cfg.get("LOSS_WEIGHT_TEMPORAL_TV", 0.02))      # neu
        self.loss_weight_divreg    = float(train_cfg.get("LOSS_WEIGHT_DIV_REG", 0.0))           # optional, default aus

        self.loss_range = loss_range(self.cfg)
        self.loss_mask  = loss_mask(self.cfg)

    def forward(self, output, target, mode, epoch_number=0):
        """
        output: dict mit 'rv': [B,T,H,W], 'mask_logits': [B,T,H,W]
        target: Tensor mit mindestens range-kanal in [:,:,0,:,:]
        mode: 'train'|'val'|'test' (nur fürs Logging relevant)
        """
        # Ziel-Range extrahieren
        target_range_image = target[:, :, 0, :, :]  # [B,T,H,W]

        # Range (Smooth-L1 + Grad)
        base_range_loss, loss_range_timestep, range_grad_loss, valid_ratio = self.loss_range(
            output, target_range_image
        )

        # Mask
        loss_mask_val = self.loss_mask(output, target_range_image)

        # Zeitliche TV (nur auf gültigen Pixels)
        valid_mask = ((target_range_image > 0.0) & (target_range_image != -1.0))
        tv_time = temporal_tv(output["rv"], valid_mask)

        # Optionaler vorhandener Regularizer (default 0.0)
        div_reg = diff_div_reg(output, target_range_image) if self.loss_weight_divreg > 0.0 else output["rv"].new_tensor(0.0)

        # Endloss (ohne Chamfer, ohne Gauß)
        loss = (
            self.loss_weight_rv      * base_range_loss
          + self.loss_weight_mask    * loss_mask_val
          + self.loss_weight_rv_grad * range_grad_loss
          + self.loss_weight_tv_time * tv_time
          + self.loss_weight_divreg  * div_reg
        )

        loss_dict = {
            "loss": loss,
            "loss_range_view": base_range_loss.detach(),
            "loss_range_timestep": loss_range_timestep.detach(),
            "loss_range_grad": range_grad_loss.detach(),
            "loss_tv_time": tv_time.detach(),
            "loss_mask": loss_mask_val.detach(),
            "valid_ratio": valid_ratio.detach(),
            # keine Chamfer-Felder mehr
        }
        return loss_dict
