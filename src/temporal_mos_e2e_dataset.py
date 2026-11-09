"""Past-to-current SemanticKITTI MOS samples with explicit projection occupancy."""

from __future__ import annotations

import numpy as np
import torch

from helper.mos_augmentation import augmentation_settings, TemporalMOSAugmentation
from mos_dataset import MOSFrameDataset
from helper.temporal_e2e_input import TemporalE2EInputMixin
from helper.temporal_offset_sampling import validate_temporal_offset_sampling


class TemporalMOSE2EDataset(TemporalE2EInputMixin, MOSFrameDataset):
    """Returns x[T,C,H,W], valid[T,H,W], dt[T], current y[H,W], meta.

    With mos_auxiliary.enabled, returns (x, valid, dt, y_mos, y_movable,
    meta), with both targets int64 [H,W] in {-1,0,1}. Test sequences return
    None for both targets and require no .label files. The disabled path
    keeps the original five-element tuple.

    For task=residual the sixth-element contract is (x, valid, dt, y_mos,
    aux, meta), where aux={task, target:float32[H,W], valid_mask:bool[H,W],
    stats}. Test inference returns None targets and computes no supervision.

    `make_sequences` provides Velodyne-coordinate poses (camera odometry
    conjugated by calibration), so inv(current_pose) @ past_pose maps past
    Velodyne points into the current Velodyne coordinate system.
    """

    def __init__(self, sequences, cfg, split="train", **kwargs):
        from helper.mos_auxiliary import validate_auxiliary_config
        self.auxiliary_settings = validate_auxiliary_config(cfg)
        self.include_movable = self.auxiliary_settings['enabled'] and self.auxiliary_settings['task'] == 'movable'
        self.include_residual = self.auxiliary_settings['enabled'] and self.auxiliary_settings['task'] == 'residual'
        data = cfg["mos_data_params"]
        self.offset_sampling = validate_temporal_offset_sampling(cfg)
        offsets = data.get("previous_offsets", [3])
        self.offsets = sorted({int(v) for v in offsets}, reverse=True)
        if not self.offsets or any(v <= 0 for v in self.offsets):
            raise ValueError("previous_offsets must contain positive frame offsets")
        self.scan_rate_hz = float(data.get("scan_rate_hz", 10.0))
        if self.scan_rate_hz <= 0:
            raise ValueError("scan_rate_hz must be positive")
        self.sequence_map = {self._normalize_seq_id(s["seq_id"]): s for s in sequences}
        self.temporal_input_mode = data.get("input_mode", "range_xyz")
        if self.temporal_input_mode not in {"range_xyz", "range_xyz_intensity"}:
            raise ValueError("Temporal E2E input_mode must be range_xyz or range_xyz_intensity")
        self.augmentation = TemporalMOSAugmentation(
            augmentation_settings(cfg.get("mos_augmentation"), self.temporal_input_mode), split)
        super().__init__(sequences, cfg, split=split, input_mode=self.temporal_input_mode,
                         residual_offsets=(), mos_label_folder=data.get("mos_label_folder", "mos_labels"),
                         **kwargs)
        kept = []
        for sample in self.samples:
            seq = self.sequence_map[sample["seq_id"]]
            i = sample["frame_index"]
            if self.lmnet:
                if i >= len(seq['poses']):
                    raise ValueError(f'Missing current pose: {sample["scan_path"]}')
                kept.append(sample)
                continue
            if i < max(self.offsets) or i >= len(seq["poses"]):
                continue
            if all(self._extract_scan_path(seq["paths"][i - off]) is not None
                   and i - off < len(seq["poses"]) for off in self.offsets):
                kept.append(sample)
        self.samples = kept
        self.summary["used_frames"] = len(kept)

    def _sample_previous_offset(self, sample):
        """Draw once per sample from existing, same-sequence scans and poses."""
        seq = self.sequence_map[sample['seq_id']]
        choices, weights = [], []
        for off, probability in zip(self.offset_sampling['offsets'], self.offset_sampling['probabilities']):
            if self._lmnet_history_path(seq, sample['frame_index'] - off) is not None:
                choices.append(off)
                weights.append(probability)
        if not choices:
            return None
        weights = torch.tensor(weights, dtype=torch.float64, device='cpu')
        # Zero-weight-only subsets have no conditional distribution. Retain a
        # real history using a uniform draw, rather than discard current frames.
        if not weights.sum():
            weights.fill_(1.)
        return choices[int(torch.multinomial(weights / weights.sum(), 1))]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.include_residual:
            return self._load_lmnet_residual_input(sample)
        if self.lmnet:
            if self.split == 'train' and self.offset_sampling['enabled']:
                offset = self._sample_previous_offset(sample)
                return self._load_lmnet_temporal_input(
                    sample, offsets=(self.offsets[0] if offset is None else offset,),
                    empty_history=offset is None)
            return self._load_lmnet_temporal_input(sample)
        if self.include_movable:
            from helper.mos_online import point_labels, is_test
            from helper.mos_label_helper import _project_point_indices
            if is_test(sample):
                x, valid, dt, _, meta = self._load_temporal_input(sample)
                return x, valid, dt, None, None, meta
            points = self._read_points(sample['scan_path'])
            labels = point_labels(sample, include_movable=True)
            if len(points) != len(labels):
                raise ValueError('Point/label length mismatch')
            indices, _ = _project_point_indices(points, self.cfg)
            targets = np.full((2, self.H, self.W), -1, np.int64)
            occupied = indices >= 0
            targets[:, occupied] = labels[indices[occupied]].T
            x, valid, dt, targets, meta = self._load_temporal_input(sample, torch.from_numpy(targets))
            return x, valid, dt, targets[0], targets[1], meta
        y = torch.from_numpy(self._load_mos_label(sample["mos_label_path"]).astype(np.int64))
        return self._load_temporal_input(sample, y)
