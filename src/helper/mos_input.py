"""Channel conventions and optional normalization for MOS inputs."""

import numpy as np

BASE_CHANNELS = ("x", "y", "z", "range", "intensity")
PAPER_MEAN = {"x": 10.88, "y": 0.23, "z": -1.04, "range": 12.12, "intensity": 0.21}
PAPER_STD = {"x": 11.47, "y": 6.91, "z": 0.86, "range": 12.32, "intensity": 0.16}
MODES = {"range": 1, "residual": 0, "range_residual": 1,
         "range_xyz": 4, "range_xyz_residual": 4,
         "range_xyz_intensity": 5, "range_xyz_intensity_residual": 5,
         "range_xyz_normal": 7, "range_xyz_normal_residual": 7}


def channel_count(mode, residual_offsets=()):
    if mode not in MODES:
        raise ValueError(f"Unsupported MOS input_mode={mode!r}; supported: {sorted(MODES)}")
    return MODES[mode] + (len(residual_offsets) if "residual" in mode else 0)


def normalization(cfg, mode):
    value = cfg.get("mos_data_params", {}).get("normalization", "none")
    if isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, dict):
        raise ValueError("mos_data_params.normalization must be a mode or mapping")
    method = value.get("mode", "none")
    if method not in {"none", "intensity_only", "paper_normalization"}:
        raise ValueError(f"Unsupported MOS normalization mode: {method}")
    if method != "none" and "intensity" not in mode:
        raise ValueError(f"Normalization {method} requires an intensity input mode")
    if method == "none":
        return {"mode": method, "mean": {}, "std": {}}
    names = ("intensity",) if method == "intensity_only" else BASE_CHANNELS
    mean = {name: float(value.get("mean", {}).get(name, PAPER_MEAN[name])) for name in names}
    std = {name: float(value.get("std", {}).get(name, PAPER_STD[name])) for name in names}
    if any(not np.isfinite(mean[n]) or not np.isfinite(std[n]) or std[n] <= 0 for n in names):
        raise ValueError("MOS normalization needs finite means and positive finite standard deviations")
    return {"mode": method, "mean": mean, "std": std}


def normalize_channels(channels, names, valid, spec):
    """Normalize named channels and leave empty or invalid pixels at zero."""
    for index, name in enumerate(names):
        if name in spec["mean"]:
            channels[index] = np.where(valid, (channels[index] - spec["mean"][name]) /
                                       spec["std"][name], 0).astype(np.float32)
    return channels


def normalize_feature_tensor(feature, valid, spec):
    """Normalize [C,H,W] XYZR(I) features after projection."""
    if spec["mode"] == "none":
        return feature
    feature = feature.clone()
    for index, name in enumerate(BASE_CHANNELS[:feature.shape[0]]):
        if name in spec["mean"]:
            feature[index] = (feature[index] - spec["mean"][name]) / spec["std"][name]
    return feature.masked_fill(~valid.bool(), 0)
