"""Video processing loop: depth, tracking, proximity evaluation and rendering."""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Hashable, List, Tuple

import cv2
import numpy as np

from proximity_alert.config import (
    ALLOWED_CLASS_IDS,
    PERSON_CLASS_ID,
    RIDER_VEHICLE_CLASS_IDS,
    TARGET_VEHICLE_CLASS_IDS,
    Config,
    effective_fps,
    processed_frame_count,
)
from proximity_alert.depth import DepthEstimator, robust_depth
from proximity_alert.geometry import Box, clip_box
from proximity_alert.proximity import ProximityEvaluator
from proximity_alert.rendering import render

logger = logging.getLogger(__name__)

WINDOW_NAME = "Proximity Alert"


class _Detections:
    """Detections of one frame split by role."""

    def __init__(self) -> None:
        self.persons: List[Tuple[Hashable, Box]] = []
        self.rider_vehicles: List[Box] = []
        self.target_vehicles: List[Box] = []


def split_detections(result, width: int, height: int, warn_state: dict) -> _Detections:
    """Converts an Ultralytics tracking result into clipped boxes per role.

    When the tracker returns no ids, the detection index is used as a fallback id.
    Such ids are not stable across frames, so the temporal window and the rider
    memory become unreliable; a warning is logged the first time it happens. Boxes that are empty after
    clipping to the frame are dropped.

    Args:
        result: First element of ``YOLO.track(...)``.
        width: Frame width.
        height: Frame height.
        warn_state: Mutable dict used to log the fallback warning only once per run.

    Returns:
        Persons with ids, rider-vehicle boxes and target-vehicle boxes.
    """
    dets = _Detections()
    boxes_obj = result.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return dets
    boxes = boxes_obj.xyxy.cpu().numpy()
    classes = boxes_obj.cls.int().cpu().numpy()
    if boxes_obj.id is not None:
        track_ids = boxes_obj.id.int().cpu().numpy().tolist()
    else:
        if not warn_state.get("fallback_ids"):
            logger.warning(
                "Tracker returned no ids; using per-frame detection indices. "
                "IDs are unstable, so time windows and rider memory may be wrong."
            )
            warn_state["fallback_ids"] = True
        track_ids = list(range(len(boxes)))

    for box, track_id, cls_id in zip(boxes, track_ids, classes):
        clipped = clip_box(box, width, height)
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue  # entirely outside the frame (or empty) after clipping
        if cls_id == PERSON_CLASS_ID:
            dets.persons.append((int(track_id), clipped))
        elif cls_id in RIDER_VEHICLE_CLASS_IDS:
            dets.rider_vehicles.append(clipped)
        elif cls_id in TARGET_VEHICLE_CLASS_IDS:
            dets.target_vehicles.append(clipped)
    return dets


def run(config: Config) -> int:
    """Processes the input video and writes the annotated output video.

    Args:
        config: Pipeline configuration.

    Returns:
        Process exit code (``0`` on success).
    """
    from ultralytics import YOLO

    cap = cv2.VideoCapture(config.input_path)
    if not cap.isOpened():
        logger.error("Cannot open input video: %s", config.input_path)
        return 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not math.isfinite(fps) or fps <= 0:
        logger.warning("Invalid FPS reported by the input (%s); assuming 30.", fps)
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(config.frame_step))
    out_fps = effective_fps(fps, step)
    total_processed = processed_frame_count(total, step) if total > 0 else 0
    logger.info(
        "Input: %dx%d @ %.2f FPS, %d frames; processing every %d frame(s), output at %.2f FPS",
        width, height, fps, total, step, out_fps,
    )

    try:
        depth_model = DepthEstimator(config.depth_model)
        logger.info("Loading YOLO model %s", config.yolo_model)
        yolo = YOLO(config.yolo_model)
    except Exception:
        cap.release()
        raise

    out_dir = os.path.dirname(os.path.abspath(config.output_path))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(config.output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        logger.error("Cannot open output video for writing: %s", config.output_path)
        cap.release()
        return 1
    evaluator = ProximityEvaluator(config)
    warn_state: dict = {}

    if config.display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_idx, raw_idx, start = 0, 0, time.time()
    try:
        while True:
            if step > 1:
                # Skipped frames are only grabbed (no decode to an array, no inference).
                if raw_idx % step != 0:
                    if not cap.grab():
                        break
                    raw_idx += 1
                    continue
            ok, frame = cap.read()
            if not ok:
                break
            raw_idx += 1
            if frame.shape[0] != height or frame.shape[1] != width:
                # The writer silently drops frames whose size differs from the one it was opened with.
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frame_idx += 1
            timestamp = frame_idx / out_fps  # effective fps: seconds stay seconds

            depth = depth_model(frame)
            result = yolo.track(
                frame,
                persist=True,
                conf=config.conf,
                imgsz=config.imgsz,
                classes=list(ALLOWED_CLASS_IDS),
                verbose=False,
            )[0]
            dets = split_detections(result, width, height, warn_state)

            def depth_at(x: int, y: int, _d: np.ndarray = depth) -> float:
                return robust_depth(_d, x, y, config.depth_patch)

            persons = evaluator.evaluate(
                dets.persons, dets.rider_vehicles, dets.target_vehicles, depth_at, timestamp, (width, height)
            )
            render(frame, persons, dets.rider_vehicles, dets.target_vehicles, depth, config.pip_scale)
            writer.write(frame)

            elapsed = time.time() - start
            percent = frame_idx / total_processed * 100 if total_processed > 0 else 0.0
            sys.stdout.write(
                f"\rProcessing frame: {frame_idx}/{total_processed} ({percent:.1f}%) | "
                f"Speed: {frame_idx / elapsed if elapsed > 0 else 0.0:.1f} FPS"
            )
            sys.stdout.flush()

            if config.display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    logger.info("Stopped by user")
                    break
    finally:
        sys.stdout.write("\n")
        cap.release()
        writer.release()
        if config.display:
            cv2.destroyAllWindows()
    logger.info("Output saved to %s", config.output_path)
    return 0
