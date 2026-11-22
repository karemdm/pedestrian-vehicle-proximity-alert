"""Configuration dataclass and COCO class constants.

The defaults here are the ones used by the CLI, the setup window and the README.
"""

from __future__ import annotations

from dataclasses import dataclass

PERSON_CLASS_ID = 0
"""COCO class id for ``person``."""

RIDER_VEHICLE_CLASS_IDS = (1, 3)
"""COCO class ids that can carry a rider: bicycle, motorcycle.

They are detected and used to filter out riders, but are not proximity targets.
"""

TARGET_VEHICLE_CLASS_IDS = (2, 5, 7)
"""COCO class ids checked for pedestrian proximity: car, bus, truck (not bicycles or motorcycles)."""

ALLOWED_CLASS_IDS = (PERSON_CLASS_ID,) + RIDER_VEHICLE_CLASS_IDS + TARGET_VEHICLE_CLASS_IDS
"""All classes requested from the detector."""

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass
class Config:
    """Runtime parameters of the proximity alert pipeline.

    Attributes:
        input_path: Path of the input video.
        output_path: Path of the annotated output video (MP4).
        yolo_model: Ultralytics weights path or official name (auto-downloaded).
        depth_model: Hugging Face model id for the depth-estimation pipeline.
        imgsz: YOLO inference resolution.
        conf: YOLO confidence threshold.
        time_window: Length in seconds of the per-person sliding window.
        min_ratio: Fraction (0-1) of close frames in the window that triggers an alert.
        depth_threshold: Maximum |z_person - z_vehicle| on the per-frame normalized
            depth map z in [0, 1] (0 = nearest, 1 = farthest surface of the frame).
        height_factor: 2D gate radius as a multiple of the person bbox height.
        rider_overlap: Minimum fraction of the person area covered by a bicycle or
            motorcycle box for the person to be considered a rider.
        rider_memory: Seconds a person keeps the rider status after the last overlap.
        depth_patch: Side (pixels) of the square patch used for the median depth.
        pip_scale: Size of the depth picture-in-picture relative to the frame.
        display: Show a live preview window (press Q to stop).
        frame_step: Process every Nth frame (1 = every frame); skipped frames are not
            drawn. The output is written at ``fps / frame_step`` so it plays at the
            original speed, and time windows and rider memory use that rate.
    """

    input_path: str = ""
    output_path: str = ""
    yolo_model: str = "yolo26x.pt"
    depth_model: str = DEFAULT_DEPTH_MODEL
    imgsz: int = 1280
    conf: float = 0.10
    time_window: float = 1.0
    min_ratio: float = 0.50
    depth_threshold: float = 0.12
    height_factor: float = 1.0
    rider_overlap: float = 0.50
    rider_memory: float = 2.0
    depth_patch: int = 7
    pip_scale: float = 0.25
    display: bool = True
    frame_step: int = 1


def effective_fps(fps: float, frame_step: int) -> float:
    """Frame rate of the processed frames when only every ``frame_step``-th one is used.

    Time windows, rider memory and the output video must use this rate so that
    seconds stay seconds.

    Args:
        fps: Frame rate of the input video.
        frame_step: Process every Nth frame (>= 1).

    Returns:
        ``fps / frame_step``.

    Raises:
        ValueError: If ``frame_step`` is smaller than 1.
    """
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    return fps / frame_step


def processed_frame_count(total_frames: int, frame_step: int) -> int:
    """Number of frames processed out of ``total_frames`` (frames 0, N, 2N, ...)."""
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    return max(0, (total_frames + frame_step - 1) // frame_step)
