from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointcloud import PointCloudData


def save_point_cloud_json(point_cloud: PointCloudData, destination: Path) -> Path:
    """Save point cloud to a compact JSON schema consumed by browser renderer."""
    payload = {
        "points": point_cloud.points.tolist(),
        "colors": point_cloud.colors.tolist(),
        "count": int(point_cloud.points.shape[0]),
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def save_debug_mask(mask: np.ndarray, destination: Path) -> Path:
    import cv2

    img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(destination), img)
    return destination


def save_debug_depth(depth: np.ndarray, destination: Path) -> Path:
    import cv2

    depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    colorized = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(destination), colorized)
    return destination
