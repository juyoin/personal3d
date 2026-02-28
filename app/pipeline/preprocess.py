from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_and_resize_image(image_path: Path, max_side: int) -> np.ndarray:
    """Load image from disk and resize while preserving aspect ratio."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth map into [0, 1] with NaN-safe handling."""
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth)

    d_min = np.min(depth[finite])
    d_max = np.max(depth[finite])
    if np.isclose(d_min, d_max):
        return np.zeros_like(depth)

    out = np.zeros_like(depth, dtype=np.float32)
    out[finite] = (depth[finite] - d_min) / (d_max - d_min)
    return out
