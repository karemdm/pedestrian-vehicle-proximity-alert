"""Monocular relative depth with Depth Anything V2 (Hugging Face transformers).

The raw model output is min-max normalized per frame to ``[0, 1]`` and inverted,
as in the original prototype (intended convention: higher = closer). Depth
Anything V2 already predicts relative *inverse* depth, so the sign convention
should be verified on real footage; the proximity gates only use ``|dz|`` and are
not affected by it.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_inverted(raw_depth: np.ndarray) -> np.ndarray:
    """Min-max normalizes a depth map to ``[0, 1]`` and inverts it.

    Args:
        raw_depth: Float array ``(H, W)`` as produced by the depth pipeline.

    Returns:
        ``1 - (d - min) / (max - min)``; a constant map yields all ones.
    """
    min_val, max_val = float(raw_depth.min()), float(raw_depth.max())
    if max_val - min_val > 1e-5:
        norm = (raw_depth - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(raw_depth)
    return (1.0 - norm).astype(np.float32)


def robust_depth(depth_map: np.ndarray, x: int, y: int, window_size: int = 7) -> float:
    """Median depth of a square patch centered on ``(x, y)``.

    Args:
        depth_map: Normalized depth map ``(H, W)``.
        x: Column of the patch center.
        y: Row of the patch center.
        window_size: Side of the square patch in pixels.

    Returns:
        Median of the patch clipped to the image, which filters single-pixel noise.
    """
    h, w = depth_map.shape[:2]
    half = window_size // 2
    region = depth_map[max(0, y - half) : min(h, y + half + 1), max(0, x - half) : min(w, x + half + 1)]
    if region.size == 0:
        return float(depth_map[min(max(0, y), h - 1), min(max(0, x), w - 1)])
    return float(np.median(region))


def colorize(depth_map: np.ndarray) -> np.ndarray:
    """Converts a ``[0, 1]`` depth map to a BGR INFERNO color image."""
    return cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


class DepthEstimator:
    """Wrapper around the transformers ``depth-estimation`` pipeline.

    Args:
        model_id: Hugging Face model id (downloaded on first use).
        device: Pipeline device (``0`` for the first GPU, ``-1`` for CPU).
            ``None`` selects the GPU when CUDA is available.
    """

    def __init__(self, model_id: str, device: Optional[int] = None) -> None:
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading depth model %s (device=%s)", model_id, device)
        self._pipe = pipeline(task="depth-estimation", model=model_id, device=device)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimates normalized, inverted depth for a BGR frame.

        Args:
            frame_bgr: Input frame ``(H, W, 3)`` in BGR order.

        Returns:
            Float32 map ``(H, W)`` in ``[0, 1]`` (normalized and inverted).
        """
        from PIL import Image

        height, width = frame_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        raw = np.array(self._pipe(pil_img)["depth"], dtype=np.float32)
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        return normalize_inverted(raw)
