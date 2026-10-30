"""Past-to-current SemanticKITTI MOS samples with explicit projection occupancy."""

from __future__ import annotations

import numpy as np
import torch

from helper.mos_augmentation import augmentation_settings, TemporalMOSAugmentation
from mos_dataset import MOSFrameDataset
from helper.temporal_e2e_input import TemporalE2EInputMixin


class TemporalMOSE2EDataset(TemporalE2EInputMixin, MOSFrameDataset):
    """Returns x[T,C,H,W], valid[T,H,W], dt[T], current y[H,W], meta.

    `make_sequences` provides Velodyne-coordinate poses (camera odometry
    conjugated by calibration), so inv(current_pose) @ past_pose maps past
    Velodyne points into the current Velodyne coordinate system.
    """

    def __init__(self, sequences, cfg, split="train", **kwargs):
        data = cfg["mos_data_params"]
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

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.lmnet:
            return self._load_lmnet_temporal_input(sample)
        y = torch.from_numpy(self._load_mos_label(sample["mos_label_path"]).astype(np.int64))
        return self._load_temporal_input(sample, y)
