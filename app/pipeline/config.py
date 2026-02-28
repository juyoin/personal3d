from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    """Central configuration for the 2D -> 3D reconstruction pipeline."""

    output_dir: Path = Path("outputs")
    max_image_size: int = 1024
    segmentation_model: str = "facebook/detr-resnet-50-panoptic"
    depth_model: str = "Intel/dpt-large"
    min_mask_area_ratio: float = 0.03
    depth_scale: float = 1.5
    x_scale: float = 1.0
    y_scale: float = 1.0


DEFAULT_CONFIG = PipelineConfig()
