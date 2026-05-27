import torch
import torch.nn as nn
import torch.nn.functional as F

from models.acc_models import Model1 as ACC_Model1
from models.acc_models import Model2 as ACC_Model2
from prob import build_range_mixture_distribution

from models.acc_base import BasePredictionModel


def _prepare_acc_input(x: torch.Tensor) -> torch.Tensor:
    """
    Pass through supported channel layouts unchanged.
    Expected input shape: [B, T, C, H, W], where C in {1, 3, 4}.
    """
    if x.dim() != 5:
        raise ValueError(f"_prepare_acc_input expects 5D tensor [B,T,C,H,W], got {x.shape}")

    C = x.shape[2]
    if C in (1, 3, 4):
        return x

    raise ValueError(
        f"_prepare_acc_input got unsupported channel size C={C}, shape={x.shape}. "
        "Expected C in {1,3,4}."
    )


def _time_resample(x: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    [B, T_in, C, H, W] -> [B, target_frames, C, H, W]
    """
    B, T, C, H, W = x.shape

    if isinstance(target_frames, (list, tuple)):
        target_frames = int(target_frames[0])
    else:
        target_frames = int(target_frames)

    if T == target_frames:
        return x

    x = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, T]
    x = F.interpolate(
        x.reshape(B * H * W * C, 1, T),
        size=target_frames,
        mode="linear",
        align_corners=True
    )
    x = x.reshape(B, H, W, C, target_frames).permute(0, 4, 3, 1, 2).contiguous()
    return x


def _as_len5(name: str, v, default):
    """
    Ensure MEAN/STD have length 5 (Range, X, Y, Z, Intensity),
    because the paper checkpoint stores mean/std as vectors of length 5.
    """
    if v is None:
        return list(default)

    if isinstance(v, (int, float)):
        v = [float(v)]

    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{name} must be list/tuple/float, got {type(v)}")

    v = list(v)
    if len(v) == 5:
        return v

    # If only range is given, pad to 5 (XYZ/intensity dummy)
    if len(v) == 1:
        if name.lower() == "mean":
            return [float(v[0]), 0.0, 0.0, 0.0, 0.0]
        if name.lower() == "std":
            return [float(v[0]), 1.0, 1.0, 1.0, 1.0]

    raise ValueError(
        f"{name} must have length 5 (or 1 for range-only). Got len={len(v)}: {v}"
    )


def _map_my_schema_to_acc_cfg(cfg: dict) -> dict:
    """
    Map YOUR project schema (model_params/data_params/train_params)
    into the keys expected by the ACC paper code (DATA_CONFIG/MODEL/TRAIN).

    This must run BEFORE:
      - ACC_Model1/2(cfg, ...)
      - BasePredictionModel.__init__ (registers mean/std buffers)
    """
    mp = cfg.get("model_params", {})
    dp = cfg.get("data_params", {})
    tp = cfg.get("train_params", {})

    # --- Spatial sizes: use original KITTI width if you have it; else fall back ---
    H = int(mp.get("org_grid_height", mp.get("grid_height", 64)))
    W = int(mp.get("org_grid_width",  mp.get("grid_width",  512)))

    # --- Dataset stats (preferred: dp.stats.mean/std) ---
    stats = dp.get("stats", {})
    mean = stats.get("mean", None)
    std  = stats.get("std",  None)

    # If you haven't added stats yet, we keep placeholders but WARN loudly
    if mean is None or std is None:
        # Fallback: try old places (optional)
        mean = mean if mean is not None else mp.get("MEAN", None)
        std  = std  if std  is not None else mp.get("STD",  None)

    if mean is None or std is None:
        print(
            "[WARN][ACC_CFG] No data_params.stats.mean/std found. "
            "Using dummy mean/std -> Paper checkpoint will NOT reproduce correctly. "
            "Please add KITTI mean/std vectors (len=5) to your config."
        )
        mean = [0.0, 0.0, 0.0, 0.0, 0.0]
        std  = [1.0, 1.0, 1.0, 1.0, 1.0]

    mean = _as_len5("mean", mean, default=[0, 0, 0, 0, 0])
    std  = _as_len5("std",  std,  default=[1, 1, 1, 1, 1])

    # --- Range limits (for projection/logging; keep your values if you want) ---
    min_range = float(stats.get("min_range", mp.get("MIN_RANGE", 0.0)))
    max_range = float(stats.get("max_range", mp.get("MAX_RANGE", 80.0)))

    # --- Build/Update ACC expected blocks ---
    if "DATA_CONFIG" not in cfg:
        cfg["DATA_CONFIG"] = {}
    cfg["DATA_CONFIG"].setdefault("HEIGHT", H)
    cfg["DATA_CONFIG"].setdefault("WIDTH", W)

    # These must be correct for checkpoint compatibility:
    cfg["DATA_CONFIG"]["MEAN"] = mean
    cfg["DATA_CONFIG"]["STD"]  = std

    cfg["DATA_CONFIG"].setdefault("MIN_RANGE", min_range)
    cfg["DATA_CONFIG"].setdefault("MAX_RANGE", max_range)

    # FOV values (optional but nice)
    cfg["DATA_CONFIG"].setdefault("FOV_UP",   float(mp.get("FOV_UP", 3.0)))
    cfg["DATA_CONFIG"].setdefault("FOV_DOWN", float(mp.get("FOV_DOWN", -25.0)))

    if "MODEL" not in cfg:
        cfg["MODEL"] = {}

    cfg["MODEL"]["N_PAST_STEPS"]   = int(mp.get("input_horizon", mp.get("N_PAST_STEPS", 5)))
    cfg["MODEL"]["N_FUTURE_STEPS"] = int(mp.get("forecast_horizon", mp.get("N_FUTURE_STEPS", 5)))

    # Paper model flags
    cfg["MODEL"].setdefault("USE", {"XYZ": False, "INTENSITY": False})
    cfg["MODEL"].setdefault("NORM", "batch")
    cfg["MODEL"].setdefault("N_CHANNELS_PER_GROUP", int(mp.get("N_CHANNELS_PER_GROUP", 16)))
    cfg["MODEL"]["MASK_THRESHOLD"] = float(mp.get("MASK_THRESHOLD", 0.5))
    cfg["MODEL"].setdefault("CIRCULAR_PADDING", True)

    # Minimal TRAIN block so paper components don't crash
    if "TRAIN" not in cfg:
        cfg["TRAIN"] = {}
    cfg["TRAIN"].setdefault("LR", float(tp.get("start_learning_rate", tp.get("learning_rate", 3e-4))))
    cfg["TRAIN"].setdefault("LR_EPOCH", int(tp.get("LR_EPOCH", 1)))
    cfg["TRAIN"].setdefault("LR_DECAY", float(tp.get("LR_DECAY", 0.99)))

    return cfg


class _AccToMDN_Base(BasePredictionModel):
    """
    Adapter for ACC models (Model1/Model2)
    Produces MDN-compatible output or pure range regression depending on cfg['model_params']['use_mdn'].
    """
    def __init__(self, cfg, acc_core: nn.Module):
        # IMPORTANT: make cfg ACC-compatible BEFORE BasePredictionModel registers buffers
        _map_my_schema_to_acc_cfg(cfg)
        super().__init__(cfg)

        mp = cfg["model_params"]
        self.F = int(mp["forecast_horizon"])
        self.K = int(mp.get("mdn_num_gaussians", 3))
        self.H = int(mp.get("grid_height", 64))
        self.W = int(mp.get("grid_width", 512))
        self.use_mdn = bool(mp.get("use_mdn", True))
        self.P = int(mp["input_horizon"])      # past steps

        self.acc = acc_core

        if self.use_mdn:
            init_log_sigma = cfg.get("train_params", {}).get("acc_sigma_init", -0.3)
            self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=torch.float32))
            init_alpha_temp = cfg.get("train_params", {}).get("alpha_temperature_init", 0.25)
            self.alpha_temp = nn.Parameter(torch.tensor(init_alpha_temp, dtype=torch.float32))

    def forward(self, hist_xyz: torch.Tensor):
        """
        hist_xyz: [B, T_in, 1/3/4, H, W]
        Returns:
          - MDN:        [B, F, H, W, 3K]
          - Regression: [B, F, H, W]
        """
        seq = _prepare_acc_input(hist_xyz)       # [B, T_in, C, H, W]
        B, T_in, C, H, W = seq.shape
        if T_in > self.P:
            seq = seq[:, -self.P:]                       # last P frames
        elif T_in < self.P:
            # pad at front with -1 (paper invalid) to reach P
            pad = seq.new_full((B, self.P - T_in, C, H, W), -1.0)
            seq = torch.cat([pad, seq], dim=1)
        # Match ACC temporal dimension (built with forecast_horizon/self.F).
        seq = _time_resample(seq, self.F)         # [B, F, C, H, W]

        # Do not treat negative XYZ as invalid; derive invalidity channel-aware.
        seq_in = seq.clone()
        if C == 1:
            invalid_in = (seq <= 0.0)
            seq_in[invalid_in] = 0.0
        elif C == 4:
            invalid_in = (seq[:, :, 3:4] <= 0.0)
            range_ch = seq_in[:, :, 3:4]
            range_ch[invalid_in] = 0.0
            seq_in[:, :, 3:4] = range_ch
        else:  # C == 3
            invalid_in = (seq[:, :, 0:3].abs().sum(dim=2, keepdim=True) <= 1e-8)

        out = self.acc(seq_in)                    # ACC expects meters
        mu = out["rv"]                           # [B, F, H, W] meters
        mask_logits = out.get("mask_logits", None)


        if not hasattr(self, "_dbg_once"):
            self._dbg_once = True
            print("[DBG ADAPTER] input seq shape:", tuple(hist_xyz.shape))
            print(f"[DBG ADAPTER] input channels C={hist_xyz.shape[2]}")
            print("[DBG ADAPTER] ACC core input shape after time resample:", tuple(seq_in.shape))
            print("[DBG ADAPTER] invalid ratio:", invalid_in.float().mean().item())
            print("[DBG ADAPTER] mu shape:", tuple(mu.shape), "F(expected)=", self.F)
            print("[DBG ADAPTER] mu (meters) mean/std:",
                mu.mean().item(), mu.std().item())
        # if not hasattr(self, "_dbg_once"):
        #     self._dbg_once = True
        #     valid_in = (rv_seq != -1.0)
        #     valid_mu = (mu != -1.0)

        #     print("[DBG ADAPTER] input rv_seq (meters) mean/std:",
        #         rv_seq[valid_in].mean().item(), rv_seq[valid_in].std().item())
        #     print("[DBG ADAPTER] mu (meters) valid mean/std:",
        #         mu[valid_mu].mean().item(), mu[valid_mu].std().item())


        if self.use_mdn:
            mu_k = mu.unsqueeze(-1).repeat(1, 1, 1, 1, self.K)
            logsig_k = self.log_sigma.expand_as(mu_k)
            alpha_k = torch.zeros_like(mu_k)

            if mask_logits is not None:
                a = torch.tanh(mask_logits).unsqueeze(-1).expand_as(alpha_k)
                alpha_k = alpha_k + 0.2 * a

            packed = torch.cat([mu_k, logsig_k, alpha_k], dim=-1)
            return packed

        return mu

    def build_mixture(self, cfg, output):
        if not getattr(self, "use_mdn", True):
            return None, True
        return build_range_mixture_distribution(cfg, output, self.alpha_temp)


class AccurateM1Adapter(_AccToMDN_Base):
    def __init__(self, cfg):
        # Ensure cfg mapped BEFORE creating core model
        _map_my_schema_to_acc_cfg(cfg)

        mp = cfg["model_params"]
        C_in = int(mp.get("grid_channels", 4))
        shape_in = (int(mp["forecast_horizon"]), C_in, int(mp["grid_height"]), int(mp["grid_width"]))
        acc_core = ACC_Model1(cfg, shape_in)
        super().__init__(cfg, acc_core)


class AccurateM2Adapter(_AccToMDN_Base):
    def __init__(self, cfg):
        # Ensure cfg mapped BEFORE creating core model
        _map_my_schema_to_acc_cfg(cfg)

        mp = cfg["model_params"]
        C_in = int(mp.get("grid_channels", 4))
        shape_in = (int(mp["forecast_horizon"]), C_in, int(mp["grid_height"]), int(mp["grid_width"]))
        acc_core = ACC_Model2(cfg, shape_in)
        super().__init__(cfg, acc_core)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.acc_models import Model1 as ACC_Model1
# from models.acc_models import Model2 as ACC_Model2
# from prob import build_range_mixture_distribution  # bereits im Swin-Modell

# # BasePredictionModel importieren (Pfad ggf. anpassen, falls anderes Package-Layout)

# from models.acc_base import BasePredictionModel


# def _xyz_to_range(xyz):
#     """
#     Accepts either:
#       - xyz:   [B, T, 3, H, W]  (x,y,z)
#       - range: [B, T, 1, H, W]  (already range)
#     Returns:
#       - range: [B, T, 1, H, W]
#     """
#     if xyz.dim() != 5:
#         raise ValueError(f"_xyz_to_range expects 5D tensor [B,T,C,H,W], got {xyz.shape}")

#     C = xyz.shape[2]

#     # already range-only
#     if C == 1:
#         return xyz

#     # xyz -> range
#     if C >= 3:
#         r = torch.sqrt(torch.clamp(
#             xyz[:, :, 0]**2 + xyz[:, :, 1]**2 + xyz[:, :, 2]**2,
#             min=1e-9
#         ))
#         return r.unsqueeze(2)

#     raise ValueError(f"_xyz_to_range got unsupported channel size C={C}, shape={xyz.shape}")


# def _time_resample(x, target_frames):  # [B, T_in, C, H, W] -> [B, target_frames, C, H, W]
#     B, T, C, H, W = x.shape
#     if isinstance(target_frames, (list, tuple)):
#         target_frames = target_frames[0]
#     elif not isinstance(target_frames, int):
#         target_frames = int(target_frames)

#     if T == target_frames:
#         return x

#     x = x.permute(0, 3, 4, 2, 1)  # [B, H, W, C, T]
#     x = F.interpolate(x.reshape(B * H * W * C, 1, T), size=target_frames, mode="linear", align_corners=True)
#     x = x.reshape(B, H, W, C, target_frames).permute(0, 4, 3, 1, 2).contiguous()
#     return x

# def _ensure_acc_cfg_shape(cfg, F, H, W):
#     # ACC-Modelle erwarten bestimmte Felder im cfg; wir füllen Dummy-Werte nach
#     if "DATA_CONFIG" not in cfg:
#         cfg["DATA_CONFIG"] = {
#             "HEIGHT": H, "WIDTH": W, "MIN_RANGE": 0.0, "MAX_RANGE": 80.0,
#             "MEAN": [0.0], "STD": [1.0],
#         }
#     if "MODEL" not in cfg:
#         cfg["MODEL"] = {}
#     cfg["MODEL"].setdefault("N_PAST_STEPS", F)          # wird von ACC nicht hart genutzt hier
#     cfg["MODEL"].setdefault("N_FUTURE_STEPS", F)
#     cfg["MODEL"].setdefault("USE", {"XYZ": False, "INTENSITY": False})
#     cfg["MODEL"].setdefault("NORM", "batch")
#     cfg["MODEL"].setdefault("N_CHANNELS_PER_GROUP", 2)
#     cfg.setdefault("TRAIN", {"LR": 1e-3, "LR_EPOCH": 100000, "LR_DECAY": 1.0})

# # CHANGE: von nn.Module -> BasePredictionModel erben
# class _AccToMDN_Base(BasePredictionModel):
#     """
#     Adapter für ACC-Modelle (Model1/Model2)
#     Erzeugt MDN-kompatiblen Output oder reine Range-Regression
#     je nach cfg['model_params']['use_mdn']
#     """
#     def __init__(self, cfg, acc_core: nn.Module):
#         # NEW: BasePredictionModel initialisieren (legt u.a. Buffers & Meta an)
#         super().__init__(cfg)
#         mp = cfg["model_params"]
#         self.F = mp["forecast_horizon"]
#         self.K = mp["mdn_num_gaussians"]
#         self.H = mp["grid_height"]
#         self.W = mp["grid_width"]
#         self.use_mdn = mp.get("use_mdn", True)  # 🔹 Schalter zwischen MDN und direkter Regression

#         _ensure_acc_cfg_shape(cfg, self.F, self.H, self.W)
#         self.acc = acc_core

#         # Nur MDN-Parameter, wenn aktiv
#         if self.use_mdn:
#             init_log_sigma = cfg.get("train_params", {}).get("acc_sigma_init", -0.3)
#             self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=torch.float32))
#             init_alpha_temp = cfg.get("train_params", {}).get("alpha_temperature_init", 0.25)
#             self.alpha_temp = nn.Parameter(torch.tensor(init_alpha_temp, dtype=torch.float32))

#     def forward(self, hist_xyz):
#         """
#         hist_xyz: [B, T_in, 3, H, W]
#         Rückgabe: [B, F, H, W, 3K] (MDN) oder [B, F, H, W] (Regression)
#         """
#         B, T_in, C, H, W = hist_xyz.shape

#         # 1) XYZ -> Range-Sequenz
#         rv_seq = _xyz_to_range(hist_xyz)         # [B, T_in, 1, H, W]
#         rv_seq = _time_resample(rv_seq, self.F)  # [B, F, 1, H, W]

#         # 2) ACC-Core aufrufen
#         out = self.acc(rv_seq)
#         mu = out["rv"]            # [B, F, H, W]
#         mask_logits = out.get("mask_logits", None)

#         # 3) Fall A: MDN aktiv
#         if self.use_mdn:
#             mu_k     = mu.unsqueeze(-1).repeat(1, 1, 1, 1, self.K)
#             logsig_k = self.log_sigma.expand_as(mu_k)
#             alpha_k  = torch.zeros_like(mu_k)

#             if mask_logits is not None:
#                 a = torch.tanh(mask_logits).unsqueeze(-1).expand_as(alpha_k)
#                 alpha_k = alpha_k + 0.2 * a

#             packed = torch.cat([mu_k, logsig_k, alpha_k], dim=-1)  # [B,F,H,W,3K]
#             return packed

#         # 4) Fall B: Nur Range-Regression (keine Gauß-Parameter)
#         else:
#             return mu  # [B,F,H,W]

#     def build_mixture(self, cfg, output):
#         """Nur aktiv, wenn use_mdn=True"""
#         if not getattr(self, "use_mdn", True):
#             return None, True
#         return build_range_mixture_distribution(cfg, output, self.alpha_temp)

# class AccurateM1Adapter(_AccToMDN_Base):
#     def __init__(self, cfg):
#         mp = cfg["model_params"]
#         shape_in = (mp["forecast_horizon"], 1, mp["grid_height"], mp["grid_width"])
#         super().__init__(cfg, ACC_Model1(cfg, shape_in))

# class AccurateM2Adapter(_AccToMDN_Base):
#     def __init__(self, cfg):
#         mp = cfg["model_params"]
#         shape_in = (mp["forecast_horizon"], 1, mp["grid_height"], mp["grid_width"])
#         super().__init__(cfg, ACC_Model2(cfg, shape_in))
