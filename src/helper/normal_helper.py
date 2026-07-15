"""Surface normal helpers for projected LiDAR range-view XYZ images."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only without OpenCV installed.
    raise ImportError(
        "OpenCV is required to build surface normals for MAE-RangeXYZ pretraining. "
        "Please install OpenCV, for example with `pip install opencv-python`."
    ) from exc


def build_normal_xyz(xyz, norm_factor=0.25, ksize=3):
    """Compute Scharr surface normals from a projected ``[H,W,3]`` XYZ image."""
    del ksize
    xyz = np.asarray(xyz)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape [H,W,3], got {xyz.shape}")

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    Sxx = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0 / norm_factor)
    Sxy = cv2.Scharr(x.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0 / norm_factor)

    Syx = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0 / norm_factor)
    Syy = cv2.Scharr(y.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0 / norm_factor)

    Szx = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 1, 0, scale=1.0 / norm_factor)
    Szy = cv2.Scharr(z.astype(np.float32), cv2.CV_32FC1, 0, 1, scale=1.0 / norm_factor)

    normal = -np.dstack(
        (
            Syx * Szy - Szx * Syy,
            Szx * Sxy - Szy * Sxx,
            Sxx * Syy - Syx * Sxy,
        )
    ).astype(np.float32)

    n = np.linalg.norm(normal, axis=2) + 1e-10
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal.astype(np.float32, copy=False)
