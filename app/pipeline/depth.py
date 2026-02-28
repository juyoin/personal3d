from __future__ import annotations

import numpy as np
from PIL import Image

from .preprocess import normalize_depth

try:
    import torch
    from transformers import AutoImageProcessor, DPTForDepthEstimation
except Exception:  # pragma: no cover - import fallback
    torch = None
    AutoImageProcessor = None
    DPTForDepthEstimation = None


class DepthEstimator:
    """Monocular depth estimation with a Sobel-gradient fallback."""

    def __init__(self, model_name: str):
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

        if AutoImageProcessor is not None and DPTForDepthEstimation is not None:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)

    def _fallback_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        depth = 1.0 - normalize_depth(mag)
        return depth

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.model is None or self.processor is None or torch is None:
            return self._fallback_depth(image_rgb)

        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = outputs.predicted_depth
        upsampled = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = upsampled.cpu().numpy()
        return normalize_depth(depth)
