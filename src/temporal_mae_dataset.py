"""Temporal LiDAR range-view dataset for MAE pretraining."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from helper.residual_helper import transform_points
from mae_dataset import select_sequences
from utils_torch import spherical_projection


class TemporalRangeMAEDataset(Dataset):
    """Build short aligned range-view histories with current residual targets."""

    def __init__(
        self,
        sequences: Sequence[dict],
        cfg: dict,
        split: str = "train",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sequences = list(sequences)
        self.cfg = cfg
        self.split = str(split)

        model_cfg = cfg.get("model_params", {}) or {}
        pretrain_cfg = cfg.get("pretrain_params", {}) or {}
        temporal_cfg = pretrain_cfg.get("temporal", {}) or {}
        mask_cfg = pretrain_cfg.get("mask", {}) or {}
        loss_cfg = pretrain_cfg.get("loss", {}) or {}
        residual_cfg = pretrain_cfg.get("residual_targets", {}) or {}
        auxiliary_cfg = pretrain_cfg.get("auxiliary_tasks", {}) or {}
        normals_cfg = auxiliary_cfg.get("surface_normals", {}) or {}
        train_cfg = cfg.get("train_params", {}) or {}

        self.height = int(model_cfg.get("grid_height", 64))
        self.width = int(model_cfg.get("grid_width", 512))
        self.input_horizon = int(model_cfg.get("input_horizon", temporal_cfg.get("input_horizon", 5)))
        self.fov_up = float(model_cfg.get("FOV_UP", 3.0))
        self.fov_down = float(model_cfg.get("FOV_DOWN", -25.0))
        self.theta_range = (
            self.fov_down * np.pi / 180.0,
            self.fov_up * np.pi / 180.0,
        )
        self.min_range = float(loss_cfg.get("min_range", model_cfg.get("min_range", 0.1)))
        max_range = model_cfg.get("max_range", model_cfg.get("MAX_RANGE", None))
        self.max_range = None if max_range is None else float(max_range)

        self.align_history_to_current = bool(temporal_cfg.get("align_history_to_current", True))
        self.surface_normals_enabled = bool(normals_cfg.get("enabled", True))
        self.channel_names = ["x", "y", "z", "range"]
        if self.surface_normals_enabled:
            self.channel_names += ["nx", "ny", "nz"]
        self.num_channels = len(self.channel_names)
        grid_channels = int(model_cfg.get("grid_channels", self.num_channels))
        if grid_channels != self.num_channels:
            raise ValueError(
                "model_params.grid_channels must match temporal MAE feature channels: "
                f"got {grid_channels}, expected {self.num_channels}."
            )

        self.residual_enabled = bool(residual_cfg.get("enabled", True))
        self.residual_offsets = [int(v) for v in residual_cfg.get("offsets", [1])]
        if self.residual_enabled and not self.residual_offsets:
            raise ValueError("pretrain_params.residual_targets.offsets must contain at least one offset.")
        if any(offset <= 0 for offset in self.residual_offsets):
            raise ValueError(f"Residual offsets must be positive, got {self.residual_offsets}.")
        self.residual_folder_template = str(residual_cfg.get("folder_template", "residual_images_{offset}"))
        self.allow_missing_residuals = bool(
            residual_cfg.get("allow_missing", residual_cfg.get("allow_missing_residuals", False))
        )
        self.residual_names = [f"residual_{offset}" for offset in self.residual_offsets]

        self.mask_type = str(mask_cfg.get("type", "patch")).lower()
        self.patch_h = int(mask_cfg.get("patch_h", 4))
        self.patch_w = int(mask_cfg.get("patch_w", 16))
        self.mask_ratio = float(mask_cfg.get("mask_ratio", 0.5))
        self.mask_only_valid = bool(mask_cfg.get("mask_only_valid", True))
        self.mask_apply_to = str(mask_cfg.get("apply_to", "current")).lower()
        self.seed = int(train_cfg.get("seed", 42) if seed is None else seed)
        self.epoch = 0

        if self.input_horizon <= 0:
            raise ValueError(f"input_horizon must be positive, got {self.input_horizon}.")
        if self.mask_type != "patch":
            raise ValueError(f"Unsupported mask type '{self.mask_type}'. Currently supported: 'patch'.")
        if self.patch_h <= 0 or self.patch_w <= 0:
            raise ValueError("patch_h and patch_w must be positive.")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0,1], got {self.mask_ratio}.")
        if self.mask_apply_to not in {"current", "all"}:
            raise ValueError("pretrain_params.mask.apply_to must be 'current' or 'all'.")

        self.samples: list[dict] = []
        self._build_index()
        target_names = self.channel_names + self.residual_names
        print(
            f"[TemporalRangeMAEDataset:{self.split}] frames={len(self.samples)} "
            f"input_horizon={self.input_horizon} "
            f"shape=[T={self.input_horizon},C={self.num_channels},H={self.height},W={self.width}] "
            f"residuals={self.residual_offsets}"
        )
        print(f"channels={self.channel_names}")
        print(f"targets={target_names}")
        print(f"align_history_to_current={self.align_history_to_current}")

    def _build_index(self) -> None:
        max_offset = max(self.residual_offsets) if self.residual_offsets else 0
        for seq in self.sequences:
            if not isinstance(seq, dict):
                continue
            seq_id = str(seq.get("seq_id", "unknown"))
            paths = list(seq.get("paths", []))
            poses = list(seq.get("poses", []))
            if len(poses) < len(paths):
                raise ValueError(f"Sequence {seq_id} has {len(paths)} paths but only {len(poses)} poses.")

            for frame_index, path_entry in enumerate(paths):
                if frame_index < self.input_horizon - 1 or frame_index < max_offset:
                    continue
                scan_path = self._extract_scan_path(path_entry)
                if scan_path is None:
                    continue
                frame_stem = os.path.splitext(os.path.basename(scan_path))[0]
                residual_paths = self._residual_paths_for_scan(scan_path, frame_stem)
                missing = [path for path in residual_paths if not os.path.isfile(path)]
                if missing and not self.allow_missing_residuals:
                    continue
                history_indices = list(range(frame_index - self.input_horizon + 1, frame_index + 1))
                self.samples.append(
                    {
                        "seq": seq,
                        "seq_id": seq_id,
                        "frame_index": int(frame_index),
                        "frame_stem": frame_stem,
                        "scan_path": scan_path,
                        "history_indices": history_indices,
                        "residual_paths": residual_paths,
                    }
                )

    @staticmethod
    def _extract_scan_path(path_entry) -> str | None:
        if isinstance(path_entry, (tuple, list)):
            return str(path_entry[0]) if path_entry else None
        if isinstance(path_entry, dict):
            for key in ("scan_path", "pc_path", "velodyne_path", "bin_path", "path"):
                if key in path_entry:
                    return str(path_entry[key])
            return None
        if isinstance(path_entry, (str, os.PathLike)):
            return str(path_entry)
        return None

    def _residual_paths_for_scan(self, scan_path: str, frame_stem: str) -> list[str]:
        seq_dir = os.path.dirname(os.path.dirname(scan_path))
        paths = []
        for offset in self.residual_offsets:
            folder = self.residual_folder_template.format(offset=offset)
            paths.append(os.path.join(seq_dir, folder, f"{frame_stem}.npy"))
        return paths

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        seq = sample["seq"]
        current_index = int(sample["frame_index"])
        current_pose = np.asarray(seq["poses"][current_index], dtype=np.float64)

        features = []
        current_valid_mask = None
        for hist_index in sample["history_indices"]:
            scan_path = self._extract_scan_path(seq["paths"][hist_index])
            if scan_path is None:
                raise FileNotFoundError(f"Missing scan path for sequence {sample['seq_id']} frame {hist_index}.")
            points = self._load_scan(scan_path)
            if self.align_history_to_current and hist_index != current_index:
                hist_pose = np.asarray(seq["poses"][hist_index], dtype=np.float64)
                transform = np.linalg.inv(current_pose) @ hist_pose
                points = transform_points(points, transform)
            feature, valid = self._project_features(points)
            features.append(feature)
            if hist_index == current_index:
                current_valid_mask = valid

        if current_valid_mask is None:
            raise RuntimeError("Current frame was not included in temporal history.")

        hist_features = torch.stack(features, dim=0)
        target_current = hist_features[-1].clone()
        target_residuals = self._load_residual_targets(sample["residual_paths"])
        residual_valid_mask = current_valid_mask.clone()

        mask = self._make_patch_mask(current_valid_mask, index)
        masked_hist_features = hist_features.clone()
        mask_bool = mask.bool()
        if self.mask_apply_to == "current":
            masked_hist_features[-1] = masked_hist_features[-1].masked_fill(
                mask_bool.expand_as(masked_hist_features[-1]),
                0.0,
            )
        else:
            masked_hist_features = masked_hist_features.masked_fill(
                mask_bool.unsqueeze(0).expand_as(masked_hist_features),
                0.0,
            )

        return {
            "hist_features": hist_features.float(),
            "masked_hist_features": masked_hist_features.float(),
            "target_current": target_current.float(),
            "target_residuals": target_residuals.float(),
            "mask": mask,
            "valid_mask": current_valid_mask,
            "residual_valid_mask": residual_valid_mask,
            "meta": {
                "seq_id": sample["seq_id"],
                "frame_index": current_index,
                "frame_stem": sample["frame_stem"],
                "history_indices": list(sample["history_indices"]),
                "channel_names": list(self.channel_names),
                "residual_names": list(self.residual_names),
            },
        }

    @staticmethod
    def _load_scan(scan_path: str) -> np.ndarray:
        if not os.path.isfile(scan_path):
            raise FileNotFoundError(f"LiDAR scan not found: {scan_path}")
        points = np.fromfile(scan_path, dtype=np.float32)
        if points.size % 4 != 0:
            raise ValueError(f"Scan float count is not divisible by 4: {scan_path}")
        return points.reshape(-1, 4)

    def _project_features(self, points: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        projected, _, _, _ = spherical_projection(
            points.astype(np.float32),
            height=self.height,
            width=self.width,
            theta_range=self.theta_range,
            max_range=self.max_range,
        )
        if projected.shape[:2] != (self.height, self.width) or projected.shape[-1] < 3:
            raise ValueError(
                f"Unexpected projection shape {projected.shape}; expected [H,W,>=3] "
                f"with H,W={(self.height, self.width)}."
            )

        xyz_img = projected[..., :3].astype(np.float32)
        xyz = torch.from_numpy(xyz_img).permute(2, 0, 1).contiguous()
        range_channel = torch.linalg.vector_norm(xyz, dim=0)
        xyz_is_zero = torch.all(xyz == 0.0, dim=0)
        finite = torch.isfinite(xyz).all(dim=0) & torch.isfinite(range_channel)
        valid = finite & (~xyz_is_zero) & (range_channel > self.min_range)

        feature = torch.cat((xyz, range_channel.unsqueeze(0)), dim=0)
        if self.surface_normals_enabled:
            from helper.normal_helper import build_normal_xyz

            normals_np = build_normal_xyz(xyz_img)
            normals = torch.from_numpy(normals_np).permute(2, 0, 1).contiguous()
            normals = torch.where(valid.unsqueeze(0), normals, torch.zeros_like(normals))
            feature = torch.cat((feature, normals), dim=0)
        feature = torch.where(valid.unsqueeze(0), feature, torch.zeros_like(feature))
        return feature.float(), valid.unsqueeze(0).float()

    def _load_residual_targets(self, residual_paths: Sequence[str]) -> torch.Tensor:
        residuals = []
        for path in residual_paths:
            if not os.path.isfile(path):
                if self.allow_missing_residuals:
                    residuals.append(torch.zeros((self.height, self.width), dtype=torch.float32))
                    continue
                raise FileNotFoundError(f"Residual target not found: {path}")
            residual = np.load(path).astype(np.float32)
            if residual.shape != (self.height, self.width):
                raise ValueError(f"Residual target {path} has shape {residual.shape}, expected {(self.height, self.width)}")
            if not np.isfinite(residual).all():
                raise ValueError(f"Residual target contains non-finite values: {path}")
            residuals.append(torch.from_numpy(residual))
        if not residuals:
            return torch.zeros((0, self.height, self.width), dtype=torch.float32)
        return torch.stack(residuals, dim=0).contiguous()

    def _make_patch_mask(self, valid_mask: torch.Tensor, index: int) -> torch.Tensor:
        valid = valid_mask[0].bool()
        mask = torch.zeros((self.height, self.width), dtype=torch.bool)
        if self.mask_ratio <= 0.0 or (self.mask_only_valid and not bool(valid.any())):
            return mask.unsqueeze(0).float()

        patch_slices = []
        patch_valid_counts = []
        for top in range(0, self.height, self.patch_h):
            bottom = min(top + self.patch_h, self.height)
            for left in range(0, self.width, self.patch_w):
                right = min(left + self.patch_w, self.width)
                count = int(valid[top:bottom, left:right].sum().item())
                if (not self.mask_only_valid) or count > 0:
                    patch_slices.append((top, bottom, left, right))
                    patch_valid_counts.append(count)

        if not patch_slices:
            return mask.unsqueeze(0).float()

        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * self.epoch + 9_973 * int(index))
        order = torch.randperm(len(patch_slices), generator=generator).tolist()

        if self.mask_only_valid:
            target_count = int(round(self.mask_ratio * int(valid.sum().item())))
            if self.mask_ratio > 0.0:
                target_count = max(target_count, 1)
            selected_valid = 0
            for patch_idx in order:
                top, bottom, left, right = patch_slices[patch_idx]
                mask[top:bottom, left:right] = True
                selected_valid += patch_valid_counts[patch_idx]
                if selected_valid >= target_count:
                    break
        else:
            count = int(round(self.mask_ratio * len(patch_slices)))
            if self.mask_ratio > 0.0:
                count = max(count, 1)
            for patch_idx in order[:count]:
                top, bottom, left, right = patch_slices[patch_idx]
                mask[top:bottom, left:right] = True
        return mask.unsqueeze(0).float()


__all__ = ["TemporalRangeMAEDataset", "select_sequences"]
