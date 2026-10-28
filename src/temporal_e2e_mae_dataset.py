"""Label-free masked reconstruction samples using the MOS input pipeline."""
from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from helper.mos_augmentation import TemporalMOSAugmentation, augmentation_settings
from helper.temporal_e2e_input import TemporalE2EInputMixin
from mos_dataset import MOSFrameDataset


def mask_settings(value):
    defaults = dict(type='patch', patch_h=8, patch_w=16, mask_ratio=.5,
                    mask_only_valid=True, apply_to='current')
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError('mae_params.mask must be a mapping of supported patch parameters')
    spec = {**defaults, **value}
    if spec['type'] != 'patch' or spec['apply_to'] != 'current':
        raise ValueError('Only patch masking of the current frame is supported')
    for name in ('patch_h', 'patch_w'):
        if type(spec[name]) is not int or spec[name] <= 0:
            raise ValueError(f'mae_params.mask.{name} must be a positive integer')
    if type(spec['mask_only_valid']) is not bool:
        raise ValueError('mask_only_valid must be boolean')
    ratio = spec['mask_ratio']
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('mask_ratio must be finite and in [0,1]')
    return spec


def patch_mask(valid, spec, generator):
    """Random nonempty patches until the requested valid-pixel budget is reached.

    Whole-patch selection can overshoot by at most one patch. With
    mask_only_valid, invalid holes are excluded from the returned boolean mask.
    """
    h, w = valid.shape
    mask = torch.zeros_like(valid, dtype=torch.bool)
    ratio = spec['mask_ratio']
    if ratio == 0 or not valid.any():
        return mask
    patches = []
    for top in range(0, h, spec['patch_h']):
        for left in range(0, w, spec['patch_w']):
            region = (slice(top, min(top+spec['patch_h'], h)), slice(left, min(left+spec['patch_w'], w)))
            count = int(valid[region].sum()) if spec['mask_only_valid'] else mask[region].numel()
            if count:
                patches.append((region, count))
    target = max(1, round(ratio * (int(valid.sum()) if spec['mask_only_valid'] else h*w)))
    selected = 0
    for index in torch.randperm(len(patches), generator=generator).tolist():
        region, count = patches[index]
        mask[region] = True
        selected += count
        if selected >= target:
            break
    return mask & valid if spec['mask_only_valid'] else mask


def masked_input(x, mask):
    """Never send the unmasked current target through the encoder or its skips."""
    if x.ndim != 5 or mask.shape != x[:, -1, 0].shape:
        raise ValueError('Expected x[B,T,C,H,W] and mask[B,H,W]')
    masked = x.clone()
    masked[:, -1] = masked[:, -1].masked_fill(mask[:, None].bool(), 0)
    return masked


class TemporalE2EMAEDataset(TemporalE2EInputMixin, Dataset):
    _extract_scan_path = staticmethod(MOSFrameDataset._extract_scan_path)
    _normalize_seq_id = staticmethod(MOSFrameDataset._normalize_seq_id)

    def __init__(self, sequences, cfg, split='train'):
        if split not in {'train', 'val'}:
            raise ValueError('MAE supports train and val only; test sequences must not be pretrained')
        self.cfg, self.split, self.epoch = cfg, split, 0
        data, mp = cfg['mos_data_params'], cfg['model_params']
        self.temporal_input_mode = data['input_mode']
        if self.temporal_input_mode != 'range_xyz_intensity':
            raise ValueError('E2E MAE requires range_xyz_intensity')
        self.offsets = sorted({int(v) for v in data['previous_offsets']}, reverse=True)
        self.scan_rate_hz = float(data.get('scan_rate_hz', 10.))
        if not self.offsets or min(self.offsets) <= 0 or not math.isfinite(self.scan_rate_hz) or self.scan_rate_hz <= 0:
            raise ValueError('MAE needs positive offsets and a positive finite scan rate')
        self.H, self.W = int(mp['grid_height']), int(mp['grid_width'])
        self.theta_range = [np.deg2rad(mp.get('FOV_DOWN', -25.)), np.deg2rad(mp.get('FOV_UP', 3.))]
        self.min_range = float(mp.get('min_range', .1))
        self.max_range = mp.get('max_range', 80.)
        self.augmentation = TemporalMOSAugmentation(augmentation_settings(cfg.get('mos_augmentation'), self.temporal_input_mode), split)
        self.mask_spec = mask_settings(cfg['mae_params'].get('mask', {}))
        self.seed = int(cfg['mae_train_params']['seed'])
        self.sequence_map = {self._normalize_seq_id(s['seq_id']): s for s in sequences}
        forbidden = {self._normalize_seq_id(s) for s in data.get('test_sequences', [])}
        if forbidden & set(self.sequence_map):
            raise ValueError('Test sequence passed to MAE dataset')
        self.samples = []
        for seq_id, seq in self.sequence_map.items():
            for i in range(max(self.offsets), min(len(seq['paths']), len(seq['poses']))):
                indices = [i-off for off in self.offsets] + [i]
                if not all(self._extract_scan_path(seq['paths'][j]) is not None for j in indices):
                    continue
                path = self._extract_scan_path(seq['paths'][i])
                self.samples.append(dict(seq_id=seq_id, frame_index=i, scan_path=path, frame_stem=Path(path).stem))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, valid, dt, _, meta = self._load_temporal_input(self.samples[index])
        # Fixed validation masks permit comparisons of epochs/checkpoints.
        epoch = self.epoch if self.split == 'train' else 0
        generator = torch.Generator().manual_seed(self.seed + 1_000_003*epoch + 9_973*index)
        mask = patch_mask(valid[-1], self.mask_spec, generator)
        return dict(x=x, valid=valid, dt=dt, mask=mask,
                    target_current=x[-1].clone(), meta=meta)
