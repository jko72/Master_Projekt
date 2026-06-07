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

    ALLOWED_INPUT_MODES = {"range", "residual", "range_residual", "range_xyz", "range_xyz_residual"}
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
        if self.input_mode in {"residual", "range_residual", "range_xyz_residual"} and len(self.residual_offsets) == 0:
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
        if self.input_mode in {"range_xyz", "range_xyz_residual"}:
            print("[MOSFrameDataset] channel convention for range_xyz* modes: [x,y,z,range,(residuals...)]")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        x_channels: List[np.ndarray] = []
        channel_names: List[str] = []
        if self.input_mode in {"range", "range_residual"}:
            range_img = self._load_range_image(sample["scan_path"])
            x_channels.append(range_img)
            channel_names.append("range")
        elif self.input_mode in {"range_xyz", "range_xyz_residual"}:
            range_img, x_img, y_img, z_img = self._load_projected_xyzd(sample["scan_path"])
            x_channels.extend([x_img, y_img, z_img, range_img])
            channel_names.extend(["x", "y", "z", "range"])

        residual_paths = []
        if self.input_mode in {"residual", "range_residual", "range_xyz_residual"}:
            for off in self.residual_offsets:
                res_path = os.path.join(
                    sample["seq_dir"], f"residual_images_{off}", f"{sample['frame_stem']}.npy"
                )
                residual_paths.append(res_path)
                x_channels.append(self._load_residual_image(res_path))
                channel_names.append(f"residual_{int(off)}")

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
            "channel_names": channel_names,
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
        if self.input_mode == "range_residual":
            return 1 + len(self.residual_offsets)
        if self.input_mode == "range_xyz":
            return 4
        if self.input_mode == "range_xyz_residual":
            return 4 + len(self.residual_offsets)
        raise ValueError(
            f"Unsupported input_mode='{self.input_mode}'. "
            f"Use one of {sorted(self.ALLOWED_INPUT_MODES)}."
        )

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
                # Always read label stats so summary counters are valid even when require_moving=False.
                st = self._get_cached_or_load_label_stats(mos_label_path)
                moving_pixels = int(st["moving_pixels"])
                if moving_pixels > 0:
                    self._frames_with_moving += 1

                include = True
                if self.require_moving:
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

    def _load_projected_xyzd(self, scan_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        if pj_img.shape[0] != self.H or pj_img.shape[1] != self.W:
            raise ValueError(
                f"spherical_projection returned unexpected spatial shape {pj_img.shape[:2]} "
                f"for {scan_path}; expected {(self.H, self.W)}"
            )
        if pj_img.shape[2] < 3:
            raise ValueError(
                f"spherical_projection returned {pj_img.shape[2]} channels for {scan_path}, "
                "but at least 3 channels (x,y,z) are required."
            )

        x_img = pj_img[:, :, 0].astype(np.float32)
        y_img = pj_img[:, :, 1].astype(np.float32)
        z_img = pj_img[:, :, 2].astype(np.float32)

        range_img = np.sqrt((x_img * x_img) + (y_img * y_img) + (z_img * z_img)).astype(np.float32)

        invalid = (
            ~np.isfinite(x_img)
            | ~np.isfinite(y_img)
            | ~np.isfinite(z_img)
            | ~np.isfinite(range_img)
            | (range_img <= 0.0)
        )
        if np.any(invalid):
            x_img[invalid] = 0.0
            y_img[invalid] = 0.0
            z_img[invalid] = 0.0
            range_img[invalid] = 0.0

        return range_img.astype(np.float32), x_img.astype(np.float32), y_img.astype(np.float32), z_img.astype(
            np.float32
        )

    def _load_range_image(self, scan_path: str) -> np.ndarray:
        range_img, _, _, _ = self._load_projected_xyzd(scan_path)
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
