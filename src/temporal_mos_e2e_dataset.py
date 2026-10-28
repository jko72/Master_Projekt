"""Past-to-current SemanticKITTI MOS samples with explicit projection occupancy."""

from __future__ import annotations

import numpy as np
import torch

from helper.residual_helper import transform_points
from helper.mos_input import normalization, normalize_feature_tensor
from helper.mos_augmentation import augmentation_settings, TemporalMOSAugmentation
from mos_dataset import MOSFrameDataset
from temporal_mae_dataset import project_range_features


class TemporalMOSE2EDataset(MOSFrameDataset):
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
            if i < max(self.offsets) or i >= len(seq["poses"]):
                continue
            if all(self._extract_scan_path(seq["paths"][i - off]) is not None
                   and i - off < len(seq["poses"]) for off in self.offsets):
                kept.append(sample)
        self.samples = kept
        self.summary["used_frames"] = len(kept)

    @staticmethod
    def _read_points(path):
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"Scan has invalid float count: {path}")
        return raw.reshape(-1, 4)

    @staticmethod
    def _pose(seq, index):
        pose = np.asarray(seq["poses"][index], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid pose at {seq['seq_id']} frame {index}")
        return pose

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = self.sequence_map[sample["seq_id"]]
        i = sample["frame_index"]
        current_pose = self._pose(seq, i)
        scans = []
        paths = []
        for off in self.offsets:
            past_i = i - off
            path = self._extract_scan_path(seq["paths"][past_i])
            points = self._read_points(path)
            transform = np.linalg.solve(current_pose, self._pose(seq, past_i))
            scans.append(self.augmentation.drop_history(transform_points(points, transform)))
            paths.append(path)
        scans.append(self._read_points(sample["scan_path"]))
        paths.append(sample["scan_path"])
        features, masks = zip(*(project_range_features(
            points, self.H, self.W, self.theta_range, self.min_range,
            self.max_range, include_intensity="intensity" in self.temporal_input_mode, explicit_occupancy=True,
        ) for points in scans))
        y = torch.from_numpy(self._load_mos_label(sample["mos_label_path"]).astype(np.int64))
        features, masks, y = self.augmentation.range_views(features, masks, y)
        spec = normalization(self.cfg, self.temporal_input_mode)
        x = torch.stack([normalize_feature_tensor(feature, mask, spec)
                         for feature, mask in zip(features, masks)])
        valid = torch.stack(masks)[:, 0].bool()
        dt = torch.tensor([-off / self.scan_rate_hz for off in self.offsets] + [0.0], dtype=torch.float32)
        y = y.masked_fill(~valid[-1], -1)
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite projection at {sample['scan_path']}")
        meta = {"seq_id": sample["seq_id"], "frame_index": i,
                "frame_stem": sample["frame_stem"], "scan_path": sample["scan_path"],
                "history_paths": paths[:-1], "previous_offsets": self.offsets,
                "channel_names": ["x", "y", "z", "range"] + (["intensity"] if x.shape[1] == 5 else []),
                "normalization": spec["mode"]}
        return x, valid, dt, y, meta
