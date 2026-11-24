"""Bounding-box geometry helpers (standard library only).

Boxes are ``(x1, y1, x2, y2)`` tuples in pixel coordinates.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


def clip_box(box: Iterable[float], width: int, height: int) -> Box:
    """Converts a box to integers and clips it to the frame.

    Args:
        box: ``(x1, y1, x2, y2)`` in float or int pixels.
        width: Frame width.
        height: Frame height.

    Returns:
        Integer box with ``0 <= x <= width`` and ``0 <= y <= height``.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def foot_point(box: Box) -> Point:
    """Returns the bottom-center point of a box (the person's feet)."""
    x1, _, x2, y2 = box
    return int((x1 + x2) / 2.0), int(y2)


def closest_point_on_box(box: Box, x: float, y: float) -> Point:
    """Returns the point of ``box`` (inside or on its border) closest to ``(x, y)``.

    Args:
        box: Target box.
        x: Query x coordinate.
        y: Query y coordinate.

    Returns:
        Integer coordinates of the closest point; equals the query point when it
        lies inside the box.
    """
    x1, y1, x2, y2 = box
    return int(max(x1, min(x, x2))), int(max(y1, min(y, y2)))


def contains_point(box: Box, x: float, y: float) -> bool:
    """Whether ``(x, y)`` lies inside ``box`` or on its border."""
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def inner_sample_points(box: Box, margin: float = 0.25) -> list:
    """Returns a 3x3 grid of points spread over the interior of ``box``.

    The box is shrunk by ``margin`` (fraction of its width/height) on every side and
    the grid covers the shrunk box, so no point lies on the border of the original box.

    Args:
        box: Box to sample.
        margin: Fraction (0 to 0.5) of the width/height removed from each side.

    Returns:
        Nine ``(x, y)`` integer points, row by row.
    """
    x1, y1, x2, y2 = box
    ix1, ix2 = x1 + (x2 - x1) * margin, x2 - (x2 - x1) * margin
    iy1, iy2 = y1 + (y2 - y1) * margin, y2 - (y2 - y1) * margin
    xs = [int(ix1 + (ix2 - ix1) * k / 2.0) for k in range(3)]
    ys = [int(iy1 + (iy2 - iy1) * k / 2.0) for k in range(3)]
    return [(x, y) for y in ys for x in xs]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def overlap_ratio(person_box: Box, other_box: Box) -> float:
    """Fraction of the person box area covered by another box.

    This is intersection over the *person* area, not IoU.

    Args:
        person_box: Person box.
        other_box: Box of the potential ridden vehicle.

    Returns:
        Value in ``[0, 1]``; ``0.0`` for degenerate person boxes or no overlap.
    """
    px1, py1, px2, py2 = person_box
    area = (px2 - px1) * (py2 - py1)
    if area <= 0:
        return 0.0
    ix1, iy1 = max(px1, other_box[0]), max(py1, other_box[1])
    ix2, iy2 = min(px2, other_box[2]), min(py2, other_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / float(area)


def is_rider(person_box: Box, rider_vehicle_boxes: Iterable[Box], threshold: float = 0.50) -> bool:
    """Checks whether a person box overlaps any bicycle/motorcycle box enough.

    Args:
        person_box: Person box.
        rider_vehicle_boxes: Boxes of bicycles and motorcycles in the frame.
        threshold: Minimum :func:`overlap_ratio` to count as a rider.

    Returns:
        ``True`` if at least one vehicle covers ``threshold`` of the person area.
    """
    return any(overlap_ratio(person_box, v) >= threshold for v in rider_vehicle_boxes)
