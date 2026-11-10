"""Monocular relative depth with Depth Anything V2 (Hugging Face transformers).

Depth convention used throughout the project (``z``):

- Depth Anything V2 predicts *relative inverse depth* (disparity-like): larger
  values mean nearer surfaces.
- The transformers ``depth-estimation`` pipeline returns it as ``"depth"``, an
  8-bit image already resized to the input size and min-max scaled per frame
  (0 = farthest, 255 = nearest surface in that frame).
- :func:`normalize_inverted` min-max scales that image to ``[0, 1]`` and inverts
  it, as in the original prototype. Therefore **``z = 0`` is the nearest and
  ``z = 1`` the farthest surface of the current frame** (higher = farther).

``z`` is a per-frame relative value, quantized to 256 levels and not linear in
metric distance. The proximity gates only use ``|z_person - z_vehicle|``, which
does not depend on the inversion.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_inverted(raw_depth: np.ndarray) -> np.ndarray:
    """Min-max normalizes a relative inverse-depth map to ``[0, 1]`` and inverts it.

    Args:
        raw_depth: Float array ``(H, W)`` of relative inverse depth (higher = nearer),
            as produced by the depth pipeline.

    Returns:
        ``1 - (d - min) / (max - min)`` as float32, so ``0`` is the nearest and ``1``
        the farthest surface of the frame; a constant map yields all ones.
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
        window_size: Side of the square patch in pixels (even values are rounded
            up to the next odd size).

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
    """Converts a ``[0, 1]`` depth map to a BGR INFERNO color image.

    With the ``z`` convention of this module, near surfaces are dark and far
    surfaces bright.
    """
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
        """Estimates the normalized, inverted depth map ``z`` for a BGR frame.

        Args:
            frame_bgr: Input frame ``(H, W, 3)`` in BGR order.

        Returns:
            Float32 map ``(H, W)`` in ``[0, 1]``; ``0`` = nearest, ``1`` = farthest
            surface of this frame (see the module docstring).
        """
        from PIL import Image

        height, width = frame_bgr.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        raw = np.asarray(self._pipe(pil_img)["depth"], dtype=np.float32)
        # The pipeline already resizes to the input size; this is only a safeguard.
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        return normalize_inverted(raw)
