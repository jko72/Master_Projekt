"""Frame-wise LiDAR range-view dataset for MAE-RangeXYZ pretraining.

Channel convention is always ``[x, y, z, range]``. No labels, residual
images, poses, or future frames are consumed by this dataset.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils_torch import spherical_projection


class RangeXYZMAEDataset(Dataset):
    """Project individual scans and create reproducible patch masks.

    Returned tensors have shapes ``target_xyzd=[4,H,W]``,
    ``masked_xyzd=[4,H,W]``, ``mask=[1,H,W]`` and
    ``valid_mask=[1,H,W]``. A mask value of one means "masked".
    """

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
        mask_cfg = pretrain_cfg.get("mask", {}) or {}
        loss_cfg = pretrain_cfg.get("loss", {}) or {}
        train_cfg = cfg.get("train_params", {}) or {}

        self.height = int(model_cfg.get("grid_height", 64))
        self.width = int(model_cfg.get("grid_width", 512))
        self.fov_up = float(model_cfg.get("FOV_UP", 3.0))
        self.fov_down = float(model_cfg.get("FOV_DOWN", -25.0))
        self.theta_range = (
            self.fov_down * np.pi / 180.0,
            self.fov_up * np.pi / 180.0,
        )
        self.min_range = float(loss_cfg.get("min_range", model_cfg.get("min_range", 0.1)))
        max_range = model_cfg.get("max_range", model_cfg.get("MAX_RANGE", None))
        self.max_range = None if max_range is None else float(max_range)

        self.mask_type = str(mask_cfg.get("type", "patch")).lower()
        self.patch_h = int(mask_cfg.get("patch_h", 4))
        self.patch_w = int(mask_cfg.get("patch_w", 16))
        self.mask_ratio = float(mask_cfg.get("mask_ratio", 0.5))
        self.mask_only_valid = bool(mask_cfg.get("mask_only_valid", True))
        self.seed = int(train_cfg.get("seed", 42) if seed is None else seed)
        self.epoch = 0

        if self.mask_type != "patch":
            raise ValueError(f"Unsupported mask type '{self.mask_type}'. Currently supported: 'patch'.")
        if self.patch_h <= 0 or self.patch_w <= 0:
            raise ValueError("patch_h and patch_w must be positive.")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0,1], got {self.mask_ratio}.")

        self.samples: list[dict] = []
        self._build_index()
        print(
            f"[RangeXYZMAEDataset:{self.split}] frames={len(self.samples)} "
            f"shape=[4,{self.height},{self.width}] channels=[x,y,z,range]"
        )

    def _build_index(self) -> None:
        for seq in self.sequences:
            if not isinstance(seq, dict):
                continue
            seq_id = str(seq.get("seq_id", "unknown"))
            for frame_index, path_entry in enumerate(seq.get("paths", [])):
                scan_path = self._extract_scan_path(path_entry)
                if scan_path is None:
                    continue
                self.samples.append(
                    {
                        "seq_id": seq_id,
                        "frame_index": int(frame_index),
                        "frame_stem": os.path.splitext(os.path.basename(scan_path))[0],
                        "scan_path": scan_path,
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

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic mask realization for an epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        target_xyzd, valid_mask = self._load_xyzd(sample["scan_path"])
        mask = self._make_patch_mask(valid_mask, index)
        masked_xyzd = target_xyzd.masked_fill(mask.expand_as(target_xyzd).bool(), 0.0)
        return {
            "target_xyzd": target_xyzd,
            "masked_xyzd": masked_xyzd,
            "mask": mask,
            "valid_mask": valid_mask,
            "meta": {
                **sample,
                "split": self.split,
                "channel_names": ["x", "y", "z", "range"],
            },
        }

    def _load_xyzd(self, scan_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        if not os.path.isfile(scan_path):
            raise FileNotFoundError(f"LiDAR scan not found: {scan_path}")
        points = np.fromfile(scan_path, dtype=np.float32)
        if points.size % 4 != 0:
            raise ValueError(f"Scan float count is not divisible by 4: {scan_path}")
        points = points.reshape(-1, 4)
        projected, _, _, _ = spherical_projection(
            points,
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

        xyz = torch.from_numpy(projected[..., :3].astype(np.float32)).permute(2, 0, 1).contiguous()
        range_channel = torch.linalg.vector_norm(xyz, dim=0)
        xyz_is_zero = torch.all(xyz == 0.0, dim=0)
        finite = torch.isfinite(xyz).all(dim=0) & torch.isfinite(range_channel)
        valid = finite & (~xyz_is_zero) & (range_channel > self.min_range)

        target = torch.cat((xyz, range_channel.unsqueeze(0)), dim=0)
        target = torch.where(valid.unsqueeze(0), target, torch.zeros_like(target))
        return target.float(), valid.unsqueeze(0).float()

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


def select_sequences(sequences: Iterable[dict], sequence_ids: Sequence[str]) -> list[dict]:
    """Select configured IDs while accepting zero-padded numeric variants."""
    sequences = list(sequences)
    requested = {str(s) for s in sequence_ids}
    requested_numeric = {int(s) for s in requested if str(s).isdigit()}
    selected = []
    for seq in sequences:
        seq_id = str(seq.get("seq_id", ""))
        if seq_id in requested or (seq_id.isdigit() and int(seq_id) in requested_numeric):
            selected.append(seq)
    found = {str(s.get("seq_id", "")) for s in selected}
    missing = [
        s for s in requested
        if s not in found and not (s.isdigit() and any(f.isdigit() and int(f) == int(s) for f in found))
    ]
    if missing:
        available = sorted(str(s.get("seq_id", "")) for s in sequences)
        raise ValueError(f"Sequences not found: {sorted(missing)}. Available: {available}")
    return selected
