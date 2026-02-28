from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .depth import DepthEstimator
from .exporters import save_debug_depth, save_debug_mask, save_point_cloud_json
from .pointcloud import make_point_cloud
from .preprocess import load_and_resize_image
from .segmentation import Segmenter


@dataclass(slots=True)
class PipelineOutput:
    point_cloud_json: Path
    mask_image: Path
    depth_image: Path
    label: str
    confidence: float
    point_count: int


class ImageTo3DPipeline:
    """End-to-end pipeline from image file to browser-ready point cloud."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.segmenter = Segmenter(config.segmentation_model, config.min_mask_area_ratio)
        self.depth_estimator = DepthEstimator(config.depth_model)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_path: Path, stem: str) -> PipelineOutput:
        image = load_and_resize_image(image_path, self.config.max_image_size)
        segmentation = self.segmenter.segment_primary_subject(image)
        depth = self.depth_estimator.estimate(image)

        cloud = make_point_cloud(
            image_rgb=image,
            depth_map=depth,
            mask=segmentation.mask,
            depth_scale=self.config.depth_scale,
            x_scale=self.config.x_scale,
            y_scale=self.config.y_scale,
        )

        point_cloud_json = save_point_cloud_json(cloud, self.config.output_dir / f"{stem}_point_cloud.json")
        mask_image = save_debug_mask(segmentation.mask, self.config.output_dir / f"{stem}_mask.png")
        depth_image = save_debug_depth(depth, self.config.output_dir / f"{stem}_depth.png")

        return PipelineOutput(
            point_cloud_json=point_cloud_json,
            mask_image=mask_image,
            depth_image=depth_image,
            label=segmentation.label,
            confidence=segmentation.score,
            point_count=int(cloud.points.shape[0]),
        )
