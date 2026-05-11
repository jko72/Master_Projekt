from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils_torch import spherical_projection


class MOSFrameDataset(Dataset):
    """Frame-wise dataset for Moving Object Segmentation on range-view grids."""

    ALLOWED_INPUT_MODES = {"range", "residual", "range_residual"}
    ALLOWED_LABEL_VALUES = {-1, 0, 1}

    def __init__(
        self,
        sequences,
        cfg,
        split: str = "train",
        input_mode: str = "range_residual",
        residual_offsets: Sequence[int] = (1,),
        mos_label_folder: str = "mos_labels",
        device: str = "cpu",
        require_moving: bool = False,
        min_moving_pixels: int = 1,
        allow_missing_labels: bool = False,
    ):
        self.sequences = list(sequences)
        self.cfg = cfg
        self.split = str(split)
        self.input_mode = str(input_mode)
        self.residual_offsets = self._parse_residual_offsets(residual_offsets)
        self.mos_label_folder = str(mos_label_folder)
        self.device = str(device)
        self.require_moving = bool(require_moving)
        self.min_moving_pixels = int(min_moving_pixels)
        self.allow_missing_labels = bool(allow_missing_labels)

        if self.input_mode not in self.ALLOWED_INPUT_MODES:
            raise ValueError(
                f"Unsupported input_mode='{self.input_mode}'. "
                f"Use one of {sorted(self.ALLOWED_INPUT_MODES)}."
            )
        if self.input_mode in {"residual", "range_residual"} and len(self.residual_offsets) == 0:
            raise ValueError("Residual input mode needs at least one residual offset.")

        mp = self.cfg["model_params"]
        self.H = int(mp["grid_height"])
        self.W = int(mp["grid_width"])
        fov_up = float(mp.get("FOV_UP", 3.0))
        fov_down = float(mp.get("FOV_DOWN", -25.0))
        self.theta_range = [fov_down * np.pi / 180.0, fov_up * np.pi / 180.0]
        self.label_ignore_value = -1

        self._missing_residual_warned: set[str] = set()
        self._invalid_label_values_warned: set[str] = set()
        self._label_stats_cache: Dict[str, Dict[str, int]] = {}

        self.samples = []
        self._build_index()

        self.summary = {
            "total_frames": int(self._total_frames),
            "used_frames": int(len(self.samples)),
            "frames_with_moving": int(self._frames_with_moving),
            "filtered_out_frames": int(self._filtered_out_frames),
        }

        print(
            "[MOSFrameDataset] total_frames={total_frames} used_frames={used_frames} "
            "frames_with_moving={frames_with_moving} filtered_out_frames={filtered_out_frames}".format(
                **self.summary
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        x_channels: List[np.ndarray] = []
        if self.input_mode in {"range", "range_residual"}:
            range_img = self._load_range_image(sample["scan_path"])
            x_channels.append(range_img)

        residual_paths = []
        if self.input_mode in {"residual", "range_residual"}:
            for off in self.residual_offsets:
                res_path = os.path.join(
                    sample["seq_dir"], f"residual_images_{off}", f"{sample['frame_stem']}.npy"
                )
                residual_paths.append(res_path)
                x_channels.append(self._load_residual_image(res_path))

        x = torch.from_numpy(np.stack(x_channels, axis=0).astype(np.float32))

        mos_label_path = os.path.join(sample["seq_dir"], self.mos_label_folder, f"{sample['frame_stem']}.npy")
        y_np = self._load_mos_label(mos_label_path)
        y = torch.from_numpy(y_np.astype(np.int64))

        pix_stats = self._pixel_counts_from_label(y_np)
        meta = {
            "seq_id": sample["seq_id"],
            "frame_stem": sample["frame_stem"],
            "frame_index": int(sample["frame_index"]),
            "scan_path": sample["scan_path"],
            "residual_paths": residual_paths,
            "mos_label_path": mos_label_path,
            "input_mode": self.input_mode,
            "moving_pixels": int(pix_stats["moving_pixels"]),
            "static_pixels": int(pix_stats["static_pixels"]),
            "ignore_pixels": int(pix_stats["ignore_pixels"]),
        }
        return x, y, meta

    def get_class_stats(self, max_samples: int | None = None) -> Dict[str, float]:
        n = len(self.samples)
        if max_samples is not None:
            n = min(n, int(max_samples))

        ignore_pixels = 0
        static_pixels = 0
        moving_pixels = 0
        frames_with_moving = 0
        moving_per_frame = []

        for i in range(n):
            sample = self.samples[i]
            stats = self._get_cached_or_load_label_stats(sample["mos_label_path"])
            ignore_pixels += int(stats["ignore_pixels"])
            static_pixels += int(stats["static_pixels"])
            moving_pixels += int(stats["moving_pixels"])
            moving_per_frame.append(int(stats["moving_pixels"]))
            if int(stats["moving_pixels"]) > 0:
                frames_with_moving += 1

        total = ignore_pixels + static_pixels + moving_pixels
        total = max(total, 1)

        if moving_per_frame:
            moving_min = int(np.min(moving_per_frame))
            moving_max = int(np.max(moving_per_frame))
            moving_mean = float(np.mean(moving_per_frame))
        else:
            moving_min = 0
            moving_max = 0
            moving_mean = 0.0

        return {
            "num_samples": int(n),
            "ignore_pixels": int(ignore_pixels),
            "static_pixels": int(static_pixels),
            "moving_pixels": int(moving_pixels),
            "ignore_ratio": float(ignore_pixels / total),
            "static_ratio": float(static_pixels / total),
            "moving_ratio": float(moving_pixels / total),
            "frames_with_moving": int(frames_with_moving),
            "moving_pixels_min": int(moving_min),
            "moving_pixels_max": int(moving_max),
            "moving_pixels_mean": float(moving_mean),
            "channel_count": int(self.channel_count),
            "input_mode": self.input_mode,
        }

    @property
    def channel_count(self) -> int:
        if self.input_mode == "range":
            return 1
        if self.input_mode == "residual":
            return len(self.residual_offsets)
        return 1 + len(self.residual_offsets)

    @staticmethod
    def _parse_residual_offsets(residual_offsets) -> List[int]:
        if isinstance(residual_offsets, (int, np.integer)):
            vals = [int(residual_offsets)]
        elif isinstance(residual_offsets, str):
            tokens = residual_offsets.replace(";", ",").replace(" ", ",").split(",")
            vals = [int(t) for t in tokens if t.strip() != ""]
        elif isinstance(residual_offsets, Iterable):
            vals = [int(v) for v in residual_offsets]
        else:
            raise TypeError("residual_offsets must be int, str, or iterable of ints.")

        vals = sorted(set(vals))
        vals = [v for v in vals if v >= 1]
        return vals

    @staticmethod
    def _normalize_seq_id(raw_seq_id) -> str:
        s = str(raw_seq_id)
        if s.isdigit():
            return s.zfill(2)
        return s

    def _build_index(self):
        samples = []
        self._total_frames = 0
        self._frames_with_moving = 0
        self._filtered_out_frames = 0

        for seq in self.sequences:
            if not isinstance(seq, dict):
                continue
            seq_id = self._normalize_seq_id(seq.get("seq_id", "unknown"))
            paths = seq.get("paths", [])
            for frame_index, path_entry in enumerate(paths):
                scan_path = self._extract_scan_path(path_entry)
                if scan_path is None:
                    continue
                frame_stem = os.path.splitext(os.path.basename(scan_path))[0]
                seq_dir = os.path.abspath(os.path.join(os.path.dirname(scan_path), ".."))
                mos_label_path = os.path.join(seq_dir, self.mos_label_folder, f"{frame_stem}.npy")

                self._total_frames += 1
                include = True
                moving_pixels = None
                if self.require_moving:
                    st = self._get_cached_or_load_label_stats(mos_label_path)
                    moving_pixels = int(st["moving_pixels"])
                    if moving_pixels > 0:
                        self._frames_with_moving += 1
                    if moving_pixels < self.min_moving_pixels:
                        include = False
                        self._filtered_out_frames += 1

                if include:
                    samples.append(
                        {
                            "seq_id": seq_id,
                            "frame_index": int(frame_index),
                            "scan_path": scan_path,
                            "seq_dir": seq_dir,
                            "frame_stem": frame_stem,
                            "mos_label_path": mos_label_path,
                            "moving_pixels_cached": moving_pixels,
                        }
                    )
        self.samples = samples

    @staticmethod
    def _extract_scan_path(path_entry) -> str | None:
        if isinstance(path_entry, (list, tuple)):
            if len(path_entry) == 0:
                return None
            return str(path_entry[0])
        if isinstance(path_entry, dict):
            for key in ("scan_path", "pc_path", "velodyne_path", "bin_path", "path"):
                if key in path_entry:
                    return str(path_entry[key])
            return None
        if isinstance(path_entry, (str, os.PathLike)):
            return str(path_entry)
        return None

    def _load_range_image(self, scan_path: str) -> np.ndarray:
        if not os.path.isfile(scan_path):
            raise FileNotFoundError(f"Scan file not found: {scan_path}")
        pts = np.fromfile(scan_path, dtype=np.float32)
        if pts.size % 4 != 0:
            raise ValueError(f"Scan file has invalid float count (not divisible by 4): {scan_path}")
        pts = pts.reshape(-1, 4)
        pj_img, _, _, _ = spherical_projection(
            pts.astype(np.float32),
            height=self.H,
            width=self.W,
            theta_range=self.theta_range,
        )

        if pj_img.ndim != 3:
            raise ValueError(f"spherical_projection returned unexpected shape {pj_img.shape} for {scan_path}")

        if pj_img.shape[2] >= 3:
            xyz = pj_img[:, :, :3].astype(np.float32)
            range_img = np.sqrt(np.sum(xyz * xyz, axis=2, dtype=np.float32)).astype(np.float32)
        else:
            range_img = pj_img[:, :, 0].astype(np.float32)

        invalid = ~np.isfinite(range_img)
        if np.any(invalid):
            range_img[invalid] = 0.0
        range_img = np.where(range_img > 0.0, range_img, 0.0).astype(np.float32)
        return range_img

    def _load_residual_image(self, residual_path: str) -> np.ndarray:
        if not os.path.isfile(residual_path):
            if residual_path not in self._missing_residual_warned:
                warnings.warn(f"Missing residual file: {residual_path}. Using zeros.", RuntimeWarning)
                self._missing_residual_warned.add(residual_path)
            return np.zeros((self.H, self.W), dtype=np.float32)
        arr = np.load(residual_path).astype(np.float32)
        if arr.shape != (self.H, self.W):
            warnings.warn(
                f"Residual shape mismatch at {residual_path}: got {arr.shape}, expected {(self.H, self.W)}. "
                "Using zeros.",
                RuntimeWarning,
            )
            return np.zeros((self.H, self.W), dtype=np.float32)
        return arr

    def _load_mos_label(self, mos_label_path: str) -> np.ndarray:
        if not os.path.isfile(mos_label_path):
            if self.allow_missing_labels:
                warnings.warn(
                    f"Missing MOS label file: {mos_label_path}. Using ignore-only label.", RuntimeWarning
                )
                return np.full((self.H, self.W), self.label_ignore_value, dtype=np.int16)
            raise FileNotFoundError(
                f"Missing MOS label file: {mos_label_path}. "
                "Set allow_missing_labels=True only if you explicitly want ignore-only fallback."
            )

        arr = np.load(mos_label_path)
        if arr.shape != (self.H, self.W):
            raise ValueError(
                f"MOS label shape mismatch at {mos_label_path}: got {arr.shape}, expected {(self.H, self.W)}."
            )
        arr = arr.astype(np.int16)

        unique_vals = set(int(v) for v in np.unique(arr))
        invalid_vals = sorted(v for v in unique_vals if v not in self.ALLOWED_LABEL_VALUES)
        if invalid_vals and mos_label_path not in self._invalid_label_values_warned:
            warnings.warn(
                f"MOS label contains unexpected values at {mos_label_path}: {invalid_vals}. "
                f"Expected only {sorted(self.ALLOWED_LABEL_VALUES)}.",
                RuntimeWarning,
            )
            self._invalid_label_values_warned.add(mos_label_path)
        return arr

    @staticmethod
    def _pixel_counts_from_label(label_img: np.ndarray) -> Dict[str, int]:
        ignore_pixels = int(np.sum(label_img == -1))
        static_pixels = int(np.sum(label_img == 0))
        moving_pixels = int(np.sum(label_img == 1))
        return {
            "ignore_pixels": ignore_pixels,
            "static_pixels": static_pixels,
            "moving_pixels": moving_pixels,
        }

    def _get_cached_or_load_label_stats(self, mos_label_path: str) -> Dict[str, int]:
        if mos_label_path in self._label_stats_cache:
            return self._label_stats_cache[mos_label_path]
        label_img = self._load_mos_label(mos_label_path)
        stats = self._pixel_counts_from_label(label_img)
        self._label_stats_cache[mos_label_path] = stats
        return stats

