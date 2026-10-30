"""CPU-only, train-only augmentation for temporal MOS (raw XYZR[I])."""

import math
from numbers import Real

import torch
import numpy as np


def augmentation_settings(value=None, input_mode="range_xyz"):
    """Resolve defaults and reject invalid/unknown parameters, even when disabled."""
    if value is not None and not isinstance(value, dict):
        raise ValueError('mos_augmentation must be a mapping')
    value = {} if value is None else value.copy()
    mode = value.pop('mode', None)
    if mode not in (None, 'none', 'custom', 'lmnet'):
        raise ValueError('mos_augmentation.mode must be none, custom or lmnet')
    if mode in {'none', 'lmnet'}:
        if value.get('enabled') is not None and value['enabled'] != (mode == 'lmnet'):
            raise ValueError('Conflicting augmentation mode and enabled flag')
        if any(not isinstance(value.get(k, {}), dict) for k in
               ('azimuth_rotation', 'historical_point_dropout', 'intensity_jitter')):
            raise ValueError('Augmentation operations must be mappings')
        if any(value.get(k, {}).get('enabled', False) for k in
               ('azimuth_rotation', 'historical_point_dropout', 'intensity_jitter')):
            raise ValueError('LMNet/none cannot be combined with custom augmentation')
    defaults = {
        "enabled": False,
        "azimuth_rotation": {"enabled": False, "probability": 0.5},
        "historical_point_dropout": {"enabled": False, "probability": 0.5,
                                     "max_dropout_ratio": 0.10},
        "intensity_jitter": {"enabled": False, "probability": 0.5, "std": 0.02},
    }

    def resolve(given, template, path):
        if not isinstance(given, dict):
            raise ValueError(f"{path} must be a mapping")
        if set(given) - set(template):
            raise ValueError(f"Unknown {path} parameters: {sorted(set(given) - set(template))}")
        result = {}
        for key, default in template.items():
            item = given.get(key, default)
            name = f"{path}.{key}"
            if isinstance(default, dict):
                item = resolve(item, default, name)
            elif isinstance(default, bool):
                if not isinstance(item, bool):
                    raise ValueError(f"{name} must be boolean")
            else:
                if isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item):
                    raise ValueError(f"{name} must be a finite number")
                item = float(item)
                if item < 0 or (key != "std" and item > 1):
                    raise ValueError(f"{name} must be {'nonnegative' if key == 'std' else 'in [0,1]'}")
            result[key] = item
        return result

    if mode is not None:
        value.setdefault('enabled', mode != 'none')
    settings = resolve(value, defaults, "mos_augmentation")
    if mode is not None:
        settings['mode'] = mode
    if settings["enabled"] and settings["intensity_jitter"]["enabled"] and "intensity" not in input_mode:
        raise ValueError("mos_augmentation.intensity_jitter requires input_mode=range_xyz_intensity")
    return settings


class TemporalMOSAugmentation:
    def __init__(self, settings, split):
        self.settings = settings
        self.training = split == "train" and settings["enabled"]

    def applies(self, name):
        spec = self.settings[name]
        # Short circuit before drawing: disabled augmentation consumes no RNG state.
        return self.settings.get('mode') != 'lmnet' and self.training and spec["enabled"] and spec["probability"] > 0 and bool(
            torch.rand(()) < spec["probability"])

    def drop_history(self, points):
        """Call once per ego-compensated historical scan, before projection."""
        if self.applies("historical_point_dropout"):
            maximum = self.settings["historical_point_dropout"]["max_dropout_ratio"]
            ratio = float(torch.rand(())) * maximum
            keep = torch.rand(len(points)) >= ratio
            return points[keep.numpy()]
        return points

    def jitter(self, feature, mask):
        if self.applies("intensity_jitter"):
            valid = mask[0].bool()
            intensity = feature[4, valid]
            # Raw SemanticKITTI remission is nominally [0,1], before normalization.
            # Bound only changed pixels; never clamp invalid pixels or other channels.
            noise = torch.randn(intensity.shape, dtype=torch.float64)
            values = intensity.double() + noise * self.settings["intensity_jitter"]["std"]
            feature[4, valid] = values.clamp(0.0, 1.0).to(feature.dtype)
        return feature

    @staticmethod
    def rotate(features, masks, y, shift):
        """Rigid XY rotation plus a common cylindrical image roll.

        utils_torch.spherical_projection uses phi=atan2(y,x), reversed columns,
        and W linspace boundaries (spacing 2*pi/(W-1)). We define a periodic
        W-column image with angle=-2*pi*shift/W, including the seam. This is
        internally periodic and norm preserving, but is NOT exactly raw-point
        reprojection under the existing endpoint-inclusive binning convention.
        """
        width = features[0].shape[-1]
        angle = -2 * math.pi * (shift % width) / width
        c, s = math.cos(angle), math.sin(angle)
        rotated, rolled_masks = [], []
        for feature, mask in zip(features, masks):
            feature = torch.roll(feature, shift, dims=-1)
            mask = torch.roll(mask, shift, dims=-1)
            valid = mask[0].bool()
            x, yy = feature[0, valid].clone(), feature[1, valid].clone()
            feature[0, valid] = c * x - s * yy
            feature[1, valid] = s * x + c * yy
            rotated.append(feature)
            rolled_masks.append(mask)
        return rotated, rolled_masks, (torch.roll(y, shift, dims=-1) if y is not None else None)

    def range_views(self, features, masks, y):
        features = [self.jitter(feature, mask) for feature, mask in zip(features, masks)]
        if self.applies("azimuth_rotation"):
            shift = int(torch.randint(features[0].shape[-1], ()))
            return self.rotate(features, masks, y, shift)
        return features, masks, y

    def raw_scans(self, scans, labels=None, parameters=None):
        """One geometry draw for all already aligned scans; deletion only current.

        parameters supports deterministic tests: flip, translation, drop_indices.
        Sampling follows parser.py/laserscan.py distributions, not its RNG stream.
        """
        if not self.training or self.settings.get('mode') != 'lmnet':
            return scans, labels
        if parameters is None:
            if float(torch.rand(())) >= .5:
                return scans, labels
            translate = float(torch.rand(())) < .5
            flip = float(torch.rand(())) < .5
            delta = np.array([float(torch.rand(()))*10-5, float(torch.rand(()))*6-3,
                              float(torch.rand(()))-1], np.float32) if translate else np.zeros(3, np.float32)
            n = len(scans[-1])
            q = float(torch.rand(()))*.5
            drops = torch.randint(n-1, (int(n*q),)).numpy() if n > 1 else np.empty(0, np.int64)
            parameters = dict(flip=flip, translation=delta, drop_indices=drops)
        transformed = []
        for points in scans:
            points = points.copy()
            if parameters['flip']:
                points[:, 1] *= -1
            points[:, :3] += np.asarray(parameters['translation'], np.float32)
            transformed.append(points)
        drops = np.asarray(parameters['drop_indices'], np.int64)
        transformed[-1] = np.delete(transformed[-1], drops, axis=0)
        if labels is not None:
            labels = np.delete(labels, drops)
        return transformed, labels
