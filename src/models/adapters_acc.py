import torch
import torch.nn as nn
import torch.nn.functional as F

from models.acc_models import Model1 as ACC_Model1
from models.acc_models import Model2 as ACC_Model2
from prob import build_range_mixture_distribution  # bereits im Swin-Modell

def _xyz_to_range(xyz):  # [B, T, 3, H, W] -> [B, T, 1, H, W]
    r = torch.sqrt(torch.clamp(xyz[:, :, 0]**2 + xyz[:, :, 1]**2 + xyz[:, :, 2]**2, min=1e-9))
    return r.unsqueeze(2)

def _time_resample(x, target_frames):  # [B, T_in, C, H, W] -> [B, target_frames, C, H, W]
    B, T, C, H, W = x.shape
    if isinstance(target_frames, (list, tuple)):
        target_frames = target_frames[0]
    elif not isinstance(target_frames, int):
        target_frames = int(target_frames)

    if T == target_frames:
        return x

    x = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, T]
    x = F.interpolate(x.reshape(B * H * W * C, 1, T), size=target_frames, mode="linear", align_corners=True)
    x = x.reshape(B, H, W, C, target_frames).permute(0, 4, 3, 1, 2).contiguous()
    return x

def _ensure_acc_cfg_shape(cfg, F, H, W):
    # ACC-Modelle erwarten bestimmte Felder im cfg; wir füllen Dummy-Werte nach
    if "DATA_CONFIG" not in cfg:
        cfg["DATA_CONFIG"] = {
            "HEIGHT": H, "WIDTH": W, "MIN_RANGE": 0.0, "MAX_RANGE": 80.0,
            "MEAN": [0.0], "STD": [1.0],
        }
    if "MODEL" not in cfg:
        cfg["MODEL"] = {}
    cfg["MODEL"].setdefault("N_PAST_STEPS", F)          # wird von ACC nicht hart genutzt hier
    cfg["MODEL"].setdefault("N_FUTURE_STEPS", F)
    cfg["MODEL"].setdefault("USE", {"XYZ": False, "INTENSITY": False})
    cfg["MODEL"].setdefault("NORM", "batch")
    cfg["MODEL"].setdefault("N_CHANNELS_PER_GROUP", 2)
    cfg.setdefault("TRAIN", {"LR": 1e-3, "LR_EPOCH": 100000, "LR_DECAY": 1.0})

class _AccToMDN_Base(nn.Module):
    """
    Kapselt ACC-Modelle (Model1/Model2) und liefert MDN-kompatiblen Output [B,F,H,W,3K].
    """
    def __init__(self, cfg, acc_core: nn.Module):
        super().__init__()
        mp = cfg["model_params"]
        self.F = mp["forecast_horizon"]
        self.K = mp["mdn_num_gaussians"]
        self.H = mp["grid_height"]
        self.W = mp["grid_width"]
        _ensure_acc_cfg_shape(cfg, self.F, self.H, self.W)
        self.acc = acc_core

        # Globale, lernbare σ und α-Temperatur wie im Swin-Modell
        init_log_sigma = cfg.get("train_params", {}).get("acc_sigma_init", -0.3)  # σ≈0.74 m
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=torch.float32))
        init_alpha_temp = cfg.get("train_params", {}).get("alpha_temperature_init", 0.25)
        self.alpha_temp = nn.Parameter(torch.tensor(init_alpha_temp, dtype=torch.float32))

    def forward(self, hist_xyz):  # [B, T_in, 3, H, W]
        B, T_in, C, H, W = hist_xyz.shape

        # 1) XYZ -> Range-Sequenz
        rv_seq = _xyz_to_range(hist_xyz)         # [B, T_in, 1, H, W]
        rv_seq = _time_resample(rv_seq, self.F)  # [B, F, 1, H, W]

        # 2) ACC-Modell aufrufen → Dict mit "rv" [B,F,H,W], "mask_logits" [B,F,H,W]
        out = self.acc(rv_seq)
        mu = out["rv"]            # [B, F, H, W]
        mask_logits = out.get("mask_logits", None)

        # 3) MDN-Parameter bauen: [B,F,H,W,3K] (μ, logσ, α-logits)
        mu_k     = mu.unsqueeze(-1).repeat(1, 1, 1, 1, self.K)   # [B,F,H,W,K]
        logsig_k = self.log_sigma.expand_as(mu_k)                # [B,F,H,W,K]
        alpha_k  = torch.zeros_like(mu_k)                        # logits ~ uniform

        # Optionale α-Modulation durch Mask-Logits
        if mask_logits is not None:
            a = torch.tanh(mask_logits).unsqueeze(-1).expand_as(alpha_k)
            alpha_k = alpha_k + 0.2 * a

        packed = torch.cat([mu_k, logsig_k, alpha_k], dim=-1)    # [B,F,H,W,3K]
        return packed

    def build_mixture(self, cfg, output):
        return build_range_mixture_distribution(cfg, output, self.alpha_temp)

class AccurateM1Adapter(_AccToMDN_Base):
    def __init__(self, cfg):
        mp = cfg["model_params"]
        shape_in = (mp["forecast_horizon"], 1, mp["grid_height"], mp["grid_width"])
        super().__init__(cfg, ACC_Model1(cfg, shape_in))

class AccurateM2Adapter(_AccToMDN_Base):
    def __init__(self, cfg):
        mp = cfg["model_params"]
        shape_in = (mp["forecast_horizon"], 1, mp["grid_height"], mp["grid_width"])
        super().__init__(cfg, ACC_Model2(cfg, shape_in))
