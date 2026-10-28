"""Shared label-independent temporal E2E input processing for MOS and MAE."""
import numpy as np
import torch
from helper.residual_helper import transform_points
from helper.mos_input import normalization, normalize_feature_tensor
from temporal_mae_dataset import project_range_features


class TemporalE2EInputMixin:
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

    def _load_temporal_input(self, sample, y=None):
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
        features, masks, y = self.augmentation.range_views(features, masks, y)
        spec = normalization(self.cfg, self.temporal_input_mode)
        x = torch.stack([normalize_feature_tensor(feature, mask, spec)
                         for feature, mask in zip(features, masks)])
        valid = torch.stack(masks)[:, 0].bool()
        dt = torch.tensor([-off / self.scan_rate_hz for off in self.offsets] + [0.0], dtype=torch.float32)
        if y is not None:
            y = y.masked_fill(~valid[-1], -1)
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite projection at {sample['scan_path']}")
        meta = {"seq_id": sample["seq_id"], "frame_index": i,
                "frame_stem": sample["frame_stem"], "scan_path": sample["scan_path"],
                "history_paths": paths[:-1], "previous_offsets": self.offsets,
                "channel_names": ["x", "y", "z", "range"] + (["intensity"] if x.shape[1] == 5 else []),
                "normalization": spec["mode"]}
        return x, valid, dt, y, meta
