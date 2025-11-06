#!/usr/bin/env python3
# @brief: Simplified Base Model (no Lightning, no utils dependency)

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

# === Metrics helpers (Range-View) ===========================================

def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    return (x * mask).sum() / mask.sum().clamp_min(eps)

def _valid_mask_from_range(gt_rv: torch.Tensor):
    # gültig: >0 und != -1  (wie in deiner loss.py)
    return ((gt_rv > 0.0) & (gt_rv != -1.0))

@torch.no_grad()
def compute_range_metrics(pred_rv: torch.Tensor,
                          gt_rv: torch.Tensor,
                          *,
                          thresholds=(0.1, 0.2, 0.5),
                          distance_bins=((0,10),(10,30),(30,60),(60,1e9))):
    """
    pred_rv, gt_rv: [B,T,H,W]
    returns dict mit:
      mae_mean, rmse_mean, logrmse_mean,
      mae_t, rmse_t (je T),
      acc_<thr>_mean für thr in thresholds,
      mae_bin_<lo>_<hi> für Distanz-Bins,
      tv_time, vel_mae, acc_mae,
      valid_ratio_mean
    """
    device = pred_rv.device
    valid = _valid_mask_from_range(gt_rv)

    abs_err = (pred_rv - gt_rv).abs()
    sq_err  = (pred_rv - gt_rv).pow(2)

    mae_mean  = _masked_mean(abs_err, valid.float())
    rmse_mean = _masked_mean(sq_err,  valid.float()).sqrt()

    # log-RMSE (relativer, distanzsensitiver)
    log_pred = torch.log1p(pred_rv.clamp_min(0))
    log_gt   = torch.log1p(gt_rv.clamp_min(0))
    log_err  = (log_pred - log_gt).pow(2)
    logrmse_mean = _masked_mean(log_err, valid.float()).sqrt()

    B, T, H, W = pred_rv.shape
    mae_t  = pred_rv.new_zeros(T)
    rmse_t = pred_rv.new_zeros(T)
    for t in range(T):
        v_t = valid[:, t]
        if v_t.any():
            mae_t[t]  = _masked_mean(abs_err[:, t], v_t.float())
            rmse_t[t] = _masked_mean(sq_err[:, t],  v_t.float()).sqrt()

    # Threshold-Accuracy
    accs = {}
    for thr in thresholds:
        inside = ((pred_rv - gt_rv).abs() < thr) & valid
        accs[f"acc_{thr}_mean"] = inside.float().sum() / valid.float().sum().clamp_min(1)

    # Distanz-Bins
    bin_maes = {}
    for lo, hi in distance_bins:
        bin_mask = valid & (gt_rv >= lo) & (gt_rv < hi)
        key = f"mae_bin_{int(lo)}_{int(hi) if hi<1e9 else 'inf'}"
        if bin_mask.any():
            bin_maes[key] = _masked_mean(abs_err, bin_mask.float())
        else:
            bin_maes[key] = torch.tensor(0.0, device=device)

    # Zeitliche Stabilität
    if T >= 2:
        dt = (pred_rv[:,1:] - pred_rv[:,:-1]).abs()
        valid_t = (valid[:,1:] & valid[:,:-1]).float()
        tv_time = _masked_mean(dt, valid_t)

        v_pred = pred_rv[:,1:] - pred_rv[:,:-1]
        v_gt   = gt_rv[:,1:]   - gt_rv[:,:-1]
        vel_mae = _masked_mean((v_pred - v_gt).abs(), valid_t)

        acc_mae = torch.tensor(0.0, device=device)
        if T >= 3:
            a_pred = v_pred[:,1:] - v_pred[:,:-1]
            a_gt   = v_gt[:,1:]   - v_gt[:,:-1]
            valid_a = (valid[:,2:] & valid[:,1:] & valid[:,:-1]).float()
            acc_mae = _masked_mean((a_pred - a_gt).abs(), valid_a)
    else:
        tv_time = torch.tensor(0.0, device=device)
        vel_mae = torch.tensor(0.0, device=device)
        acc_mae = torch.tensor(0.0, device=device)

    valid_ratio_mean = valid.float().mean()

    return {
        "mae_mean": mae_mean, "rmse_mean": rmse_mean, "logrmse_mean": logrmse_mean,
        "mae_t": mae_t, "rmse_t": rmse_t,
        "tv_time": tv_time, "vel_mae": vel_mae, "acc_mae": acc_mae,
        "valid_ratio_mean": valid_ratio_mean,
        **accs, **bin_maes
    }

# --- Optional Fallbacks für fehlende Pakete ---
try:
    import lightning.pytorch as pl
except ImportError:
    # Fallback: einfache Ersatzklasse, falls Lightning nicht installiert ist
    class _DummyModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs): pass
    pl = type("pl", (), {"LightningModule": _DummyModule})

# Dummy-Klassen für fehlende Utility-Module (werden hier NICHT genutzt)
class DummyProjection:
    def __init__(self, cfg): pass
    def __call__(self, *args, **kwargs): pass

class DummyLogger:
    def __init__(self): pass
    def log_point_clouds(self, *args, **kwargs): pass
    def save_range_and_mask(self, *args, **kwargs): pass
    def save_point_clouds(self, *args, **kwargs): pass


# =============================================================================
# Vereinfachte BasePredictionModel (Lightning-frei)
# =============================================================================
class BasePredictionModel(pl.LightningModule):
    """Base class for ACC models (simplified, Lightning-free)"""

    def __init__(self, cfg):
        super(BasePredictionModel, self).__init__()
        self.cfg = cfg
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(self.cfg)

        # --- Data config fallback ---
        data_cfg = cfg.get("DATA_CONFIG", {})
        self.height = data_cfg.get("HEIGHT", 64)
        self.width = data_cfg.get("WIDTH", 512)
        self.min_range = data_cfg.get("MIN_RANGE", 0.0)
        self.max_range = data_cfg.get("MAX_RANGE", 80.0)
        mean = data_cfg.get("MEAN", [0.0])
        std = data_cfg.get("STD", [1.0])
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        # --- Model info ---
        model_cfg = cfg.get("MODEL", {})
        self.n_past_steps = model_cfg.get("N_PAST_STEPS", 5)
        self.n_future_steps = model_cfg.get("N_FUTURE_STEPS", 5)
        use_cfg = model_cfg.get("USE", {"XYZ": False, "INTENSITY": False})
        self.use_xyz = use_cfg.get("XYZ", False)
        self.use_intensity = use_cfg.get("INTENSITY", False)

        # --- I/O Info ---
        self.inputs = [0]
        if self.use_xyz:
            self.inputs.extend([1, 2, 3])
        if self.use_intensity:
            self.inputs.append(4)
        self.n_inputs = len(self.inputs)

        # Placeholder attributes (nicht zwingend benötigt)
        self.projection = DummyProjection(cfg)
        self.logger = DummyLogger()
        self.chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)

    # -------------------------------------------------------------
    # Diese Methoden werden im aktuellen Projekt NICHT aufgerufen,
    # bleiben aber kompatibel, falls du sie später brauchst.
    # -------------------------------------------------------------
    def forward(self, x):
        raise NotImplementedError("BasePredictionModel.forward() should be implemented in derived classes.")

    def configure_optimizers(self):
        """Standard-Optimizer; Lightning-kompatibel, aber auch ohne nutzbar."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
        return [optimizer], [scheduler]
    
    # ---------------------------------------------------------------------
    # Convenience: Metriken berechnen & optional in TensorBoard loggen
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def compute_and_log_metrics(self,
                                output: dict,
                                future: torch.Tensor,
                                *,
                                writer=None,
                                global_step: int = None,
                                prefix: str = "val"):
        """
        output: dict mit 'rv' -> [B,T,H,W]
        future: Tensor mit Range im Kanal 0 -> [B,T, C, H, W], C>=1
        writer: TensorBoard SummaryWriter (optional)
        global_step: globaler Step für Logs (optional)
        prefix: 'val'|'test'|...
        """
        pred_rv = output["rv"]
        gt_rv   = future[:, :, 0, :, :]

        m = compute_range_metrics(pred_rv, gt_rv)

        # Falls kein Writer: nur zurückgeben
        if writer is None:
            return m

        # Scalars
        writer.add_scalar(f"{prefix}/mae_mean",  m["mae_mean"].item(),  global_step or 0)
        writer.add_scalar(f"{prefix}/rmse_mean", m["rmse_mean"].item(), global_step or 0)
        writer.add_scalar(f"{prefix}/logrmse_mean", m["logrmse_mean"].item(), global_step or 0)

        for t in range(m["mae_t"].numel()):
            writer.add_scalar(f"{prefix}/mae_t/step_{t}",  m["mae_t"][t].item(),  global_step or 0)
            writer.add_scalar(f"{prefix}/rmse_t/step_{t}", m["rmse_t"][t].item(), global_step or 0)

        for thr in (0.1, 0.2, 0.5):
            writer.add_scalar(f"{prefix}/acc_{thr}_mean", m[f"acc_{thr}_mean"].item(), global_step or 0)

        for k, v in m.items():
            if k.startswith("mae_bin_"):
                writer.add_scalar(f"{prefix}/{k}", v.item(), global_step or 0)

        writer.add_scalar(f"{prefix}/tv_time", m["tv_time"].item(), global_step or 0)
        writer.add_scalar(f"{prefix}/vel_mae", m["vel_mae"].item(), global_step or 0)
        writer.add_scalar(f"{prefix}/acc_mae", m["acc_mae"].item(), global_step or 0)
        writer.add_scalar(f"{prefix}/valid_ratio/mean", m["valid_ratio_mean"].item(), global_step or 0)

        # Optionale Beispiel-Images (Batch 0, t=0)
        try:
            b = 0
            pred_img = pred_rv[b, 0].detach().clamp_min(0)
            gt_img   = gt_rv[b, 0].detach().clamp_min(0)
            err_img  = (pred_img - gt_img).abs()

            def _norm(x):
                mx = x.max().clamp_min(1e-6)
                return (x / mx).unsqueeze(0)  # [1,H,W]

            writer.add_image(f"{prefix}_vis/example/pred_t0", _norm(pred_img), global_step or 0)
            writer.add_image(f"{prefix}_vis/example/gt_t0",   _norm(gt_img),   global_step or 0)
            writer.add_image(f"{prefix}_vis/example/err_t0",  _norm(err_img),  global_step or 0)
        except Exception as e:
            print(f"[{prefix.upper()} VIS] skipping image log:", repr(e))

        return m


