"""Frame annotation: boxes, contact lines, labels and depth picture-in-picture."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from proximity_alert.depth import colorize
from proximity_alert.geometry import Box
from proximity_alert.proximity import PersonResult

# Colors are BGR.
VEHICLE_COLOR = (255, 0, 0)  # blue
RIDER_VEHICLE_COLOR = (255, 255, 0)  # cyan
RIDER_COLOR = (0, 165, 255)  # orange
OK_COLOR = (0, 255, 0)  # green
ALERT_COLOR = (0, 0, 255)  # red
LINE_COLOR = (0, 255, 255)  # yellow
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(frame: np.ndarray, boxes: Sequence[Box], color: tuple) -> None:
    """Draws plain rectangles in place."""
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_person(frame: np.ndarray, person: PersonResult) -> None:
    """Draws one person box, its contact lines and labels in place.

    Riders are drawn in orange with a ``Rider #id`` label. Pedestrians are green
    (OK) or red (ALERT) with the window ratio and the smallest Delta Z.
    """
    x1, y1, x2, y2 = person.box
    if person.is_rider:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RIDER_COLOR, 2)
        cv2.putText(frame, f"Rider #{person.track_id}", (x1, max(20, y1 - 8)), FONT, 0.45, RIDER_COLOR, 2)
        return

    color = ALERT_COLOR if person.alert else OK_COLOR
    line_color = ALERT_COLOR if person.alert else LINE_COLOR
    for p_pt, v_pt in person.connections:
        cv2.line(frame, p_pt, v_pt, line_color, 2, cv2.LINE_AA)
        cv2.circle(frame, v_pt, 4, line_color, -1)

    status = "ALERT" if person.alert else "OK"
    dz_str = f"{person.min_dz:.3f}" if person.min_dz is not None else "N/A"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"Person #{person.track_id} [{status} ({person.ratio * 100:.0f}%)]",
        (x1, max(25, y1 - 22)),
        FONT,
        0.45,
        color,
        2,
    )
    cv2.putText(frame, f"Delta Z: {dz_str} (Z: {person.z:.3f})", (x1, max(10, y1 - 8)), FONT, 0.4, color, 1)


def overlay_depth_pip(frame: np.ndarray, depth_map: np.ndarray, scale: float = 0.25, margin: int = 10) -> None:
    """Pastes a colorized, scaled depth map in the top-right corner, in place.

    The float map is downscaled first and colorized at PiP size, so the colormap
    runs on ``scale**2`` of the pixels instead of the full frame.

    Args:
        frame: Annotated BGR frame.
        depth_map: Normalized depth map ``z`` in ``[0, 1]`` (see :mod:`proximity_alert.depth`).
        scale: PiP size relative to the frame.
        margin: Distance in pixels from the top and right borders.
    """
    height, width = frame.shape[:2]
    pip_w, pip_h = int(width * scale), int(height * scale)
    if pip_w < 2 or pip_h < 2 or pip_w + margin > width or pip_h + margin > height:
        return
    pip = colorize(cv2.resize(depth_map, (pip_w, pip_h), interpolation=cv2.INTER_AREA))
    cv2.rectangle(pip, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)
    frame[margin : margin + pip_h, width - pip_w - margin : width - margin] = pip


def render(
    frame: np.ndarray,
    persons: Sequence[PersonResult],
    rider_vehicle_boxes: Sequence[Box],
    target_vehicle_boxes: Sequence[Box],
    depth_map: np.ndarray,
    pip_scale: float,
) -> np.ndarray:
    """Draws all annotations on ``frame`` (in place) and returns it."""
    draw_boxes(frame, target_vehicle_boxes, VEHICLE_COLOR)
    draw_boxes(frame, rider_vehicle_boxes, RIDER_VEHICLE_COLOR)
    for person in persons:
        draw_person(frame, person)
    overlay_depth_pip(frame, depth_map, pip_scale)
    return frame
