"""Shared label-independent temporal E2E input processing for MOS and MAE."""
import numpy as np
import torch
from helper.residual_helper import transform_points
from helper.mos_input import normalization, normalize_feature_tensor
from temporal_mae_dataset import project_range_features


class TemporalE2EInputMixin:
    def _project_lmnet_temporal(self, sample, labels=None):
        """Shared label-independent geometry for supervised MOS and MAE."""
        from helper.mos_projection import project_features
        from pathlib import Path
        seq = self.sequence_map[sample['seq_id']]
        i = sample['frame_index']
        current_pose = self._pose(seq, i)
        scans, paths, available = [], [], []
        for off in self.offsets:
            j = i-off
            path = self._extract_scan_path(seq['paths'][j]) if j >= 0 else None
            present = path is not None and Path(path).is_file() and j < len(seq['poses'])
            if present:
                aligned = transform_points(self._read_points(path), np.linalg.solve(current_pose, self._pose(seq, j)))
                scans.append(self.augmentation.drop_history(aligned))
            else:
                scans.append(np.empty((0, 4), np.float32))
            paths.append(path if present else '')
            available.append(present)
        scans.append(self._read_points(sample['scan_path']))
        if labels is not None and len(labels) != len(scans[-1]):
            raise ValueError(f'Point/label length mismatch: {sample["scan_path"]}')
        scans, labels = self.augmentation.raw_scans(scans, labels)
        projected = [project_features(points, self.H, self.W, self.theta_range,
                                      'intensity' in self.temporal_input_mode) for points in scans]
        features, masks, maps = zip(*projected)
        return features, masks, maps, labels, paths, available

    def _finish_lmnet_temporal(self, sample, features, masks, paths, available, y=None):
        spec = normalization(self.cfg, self.temporal_input_mode)
        features, masks, y = self.augmentation.range_views(features, masks, y)
        x = torch.stack([normalize_feature_tensor(f, m, spec) for f, m in zip(features, masks)])
        valid = torch.stack(masks)[:, 0].bool()
        for j, present in enumerate(available):
            if not present:
                x[j].zero_()
                valid[j].zero_()
        if y is not None:
            y = y.masked_fill(~valid[-1], -1)
        dt = torch.tensor([-off/self.scan_rate_hz for off in self.offsets] + [0.], dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError(f'Non-finite projection at {sample["scan_path"]}')
        meta = dict(sample, history_paths=paths, previous_offsets=self.offsets,
                    history_available=available, normalization=spec['mode'], projection_mode='lmnet',
                    channel_names=['x', 'y', 'z', 'range'] + (['intensity'] if x.shape[1] == 5 else []))
        return x, valid, dt, y, meta

    def _load_lmnet_mae_input(self, sample):
        features, masks, _, _, paths, available = self._project_lmnet_temporal(sample)
        return self._finish_lmnet_temporal(sample, features, masks, paths, available)

    def _load_lmnet_temporal_input(self, sample):
        from helper.mos_projection import project_targets
        from helper.mos_online import point_labels, is_test
        from helper.mos_cache import check_cache
        from pathlib import Path
        online = self.split == 'train' or self.cfg['mos_data_params'].get('online_validation_labels', False)
        labels = point_labels(sample) if online and not is_test(sample) else None
        features, masks, maps, labels, paths, available = self._project_lmnet_temporal(sample, labels)
        if is_test(sample):
            y = None
        elif online:
            y = torch.from_numpy(project_targets(labels, maps[-1]))
        else:
            check_cache(Path(sample['mos_label_path']).parent, self.cfg, 'labels', sample['seq_dir'])
            y = torch.from_numpy(self._load_mos_label(sample['mos_label_path']).astype(np.int64))
        return self._finish_lmnet_temporal(sample, features, masks, paths, available, y)

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
