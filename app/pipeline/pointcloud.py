from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PointCloudData:
    points: np.ndarray
    colors: np.ndarray


def make_point_cloud(
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
    depth_scale: float,
    x_scale: float,
    y_scale: float,
) -> PointCloudData:
    """Project masked RGB pixels and depth values into a centered 3D point cloud."""
    h, w = depth_map.shape
    yy, xx = np.indices((h, w))

    xx_n = ((xx - w / 2.0) / w) * x_scale
    yy_n = (-(yy - h / 2.0) / h) * y_scale
    zz_n = depth_map * depth_scale

    valid = mask & np.isfinite(zz_n)

    points = np.stack([xx_n[valid], yy_n[valid], zz_n[valid]], axis=1).astype(np.float32)
    colors = (image_rgb[valid].astype(np.float32) / 255.0).astype(np.float32)

    if points.size == 0:
        return PointCloudData(points=np.zeros((0, 3), dtype=np.float32), colors=np.zeros((0, 3), dtype=np.float32))

    points[:, 2] -= points[:, 2].min()
    return PointCloudData(points=points, colors=colors)
