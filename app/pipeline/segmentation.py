from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
except Exception:  # pragma: no cover - import fallback
    torch = None
    AutoImageProcessor = None
    AutoModelForUniversalSegmentation = None


@dataclass(slots=True)
class SegmentationResult:
    mask: np.ndarray
    label: str
    score: float


class Segmenter:
    """Semantic foreground segmenter with robust fallback when models are unavailable."""

    def __init__(self, model_name: str, min_area_ratio: float = 0.03):
        self.model_name = model_name
        self.min_area_ratio = min_area_ratio
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self._load_attempted = False

    def _ensure_model_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if AutoImageProcessor is None or AutoModelForUniversalSegmentation is None:
            return

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForUniversalSegmentation.from_pretrained(self.model_name).to(self.device)
        except Exception:
            # Fall back silently to deterministic heuristic segmentation.
            self.processor = None
            self.model = None

    def _largest_connected_component(self, image_rgb: np.ndarray) -> np.ndarray:
        """Heuristic fallback for foreground extraction by saturation + edges."""
        hsv = np.asarray(Image.fromarray(image_rgb).convert("HSV"), dtype=np.uint8)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        mask = ((sat > np.percentile(sat, 55)) | (val < np.percentile(val, 35))).astype(np.uint8)

        # Morphological cleanup.
        import cv2

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return np.ones(mask.shape, dtype=bool)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        area_ratio = float(areas.max()) / float(mask.shape[0] * mask.shape[1])
        if area_ratio < self.min_area_ratio:
            return np.ones(mask.shape, dtype=bool)

        return labels == largest_idx

    def segment_primary_subject(self, image_rgb: np.ndarray) -> SegmentationResult:
        """Extract the dominant foreground object mask and metadata."""
        self._ensure_model_loaded()
        if self.model is None or self.processor is None or torch is None:
            mask = self._largest_connected_component(image_rgb)
            return SegmentationResult(mask=mask, label="foreground", score=0.5)

        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[pil.size[::-1]],
            threshold=0.7,
        )[0]

        seg = processed["segmentation"].cpu().numpy()
        segments_info: list[dict[str, Any]] = processed["segments_info"]

        if not segments_info:
            mask = self._largest_connected_component(image_rgb)
            return SegmentationResult(mask=mask, label="foreground", score=0.5)

        # Pick the largest high-confidence segment as primary foreground.
        best_info = max(segments_info, key=lambda s: s.get("score", 0.0) * s.get("area", 1.0))
        mask = seg == best_info["id"]

        if mask.mean() < self.min_area_ratio:
            mask = self._largest_connected_component(image_rgb)
            return SegmentationResult(mask=mask, label="foreground", score=0.5)

        return SegmentationResult(
            mask=mask,
            label=str(best_info.get("label_id", "object")),
            score=float(best_info.get("score", 0.0)),
        )
