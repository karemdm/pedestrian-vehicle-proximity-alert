"""Video processing loop: depth, tracking, proximity evaluation and rendering."""

from __future__ import annotations

import logging
import math
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
)
from proximity_alert.depth import DepthEstimator, colorize, robust_depth
from proximity_alert.geometry import Box, clip_box
from proximity_alert.proximity import ProximityEvaluator
from proximity_alert.rendering import render

logger = logging.getLogger(__name__)

WINDOW_NAME = "Proximity Monitoring System"


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
    memory become unreliable; a warning is logged the first time it happens.

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
    if not fps or math.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Input: %dx%d @ %.2f FPS, %d frames", width, height, fps, total)

    writer = cv2.VideoWriter(config.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    depth_model = DepthEstimator(config.depth_model)
    logger.info("Loading YOLO model %s", config.yolo_model)
    yolo = YOLO(config.yolo_model)
    evaluator = ProximityEvaluator(config)
    warn_state: dict = {}

    if config.display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_idx, start = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = frame_idx / fps

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
            render(frame, persons, dets.rider_vehicles, dets.target_vehicles, colorize(depth), config.pip_scale)
            writer.write(frame)

            elapsed = time.time() - start
            percent = frame_idx / total * 100 if total > 0 else 0.0
            sys.stdout.write(
                f"\rProcessing frame: {frame_idx}/{total} ({percent:.1f}%) | "
                f"Speed: {frame_idx / elapsed if elapsed > 0 else 0.0:.1f} FPS"
            )
            sys.stdout.flush()

            if config.display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
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
