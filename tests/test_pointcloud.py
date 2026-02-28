import numpy as np

from app.pipeline.pointcloud import make_point_cloud


def test_make_point_cloud_respects_mask_and_colors():
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[0.2, 0.7], [0.4, 0.9]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]])

    cloud = make_point_cloud(image, depth, mask, depth_scale=1.0, x_scale=1.0, y_scale=1.0)

    assert cloud.points.shape == (2, 3)
    assert cloud.colors.shape == (2, 3)
    assert np.allclose(cloud.colors[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(cloud.colors[1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
