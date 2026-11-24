"""Per-frame pedestrian-vehicle proximity evaluation (standard library only).

Depth is accessed through a ``depth_at(x, y) -> float`` callable, so this module
does not depend on numpy and can be tested with a fake depth function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Hashable, List, Optional, Sequence, Tuple

from proximity_alert.config import Config
from proximity_alert.geometry import (
    Box,
    Point,
    closest_point_on_box,
    contains_point,
    distance,
    foot_point,
    inner_sample_points,
    is_rider,
    overlap_ratio,
)
from proximity_alert.temporal import ProximityHistory, RiderMemory, is_alert

DepthFn = Callable[[int, int], float]

MIN_TRACK_GRACE_SEC = 1.0
"""Minimum time a missing track keeps its sliding window (video seconds)."""

MIN_PERSON_HEIGHT_PX = 10
"""Lower bound of the person height used for the 2D gate."""

VEHICLE_INNER_MARGIN = 0.25
"""Fraction of the vehicle box width/height cut from each side before sampling its depth.

Boxes of cars, buses and trucks include road, shadows and background near the border, and
the middle half of the box is where the body (doors, hood, windows) is. 0.25 keeps the
samples well inside the surface while a 3x3 grid still spans half of the box in each
direction. It is a fixed design constant, not a tuning knob.
"""


def vehicle_depth(vbox: Box, person_box: Box, depth_at: DepthFn, frame_size: Tuple[int, int]) -> float:
    """Depth of the vehicle surface: median of a 3x3 grid inside the shrunk vehicle box.

    Used when the person's foot lies inside the vehicle box, where the closest point of
    the box is the foot itself and would only return the person's own depth. Grid points
    that fall inside the person's box are ignored (they show the person, not the vehicle);
    if every point is covered, all of them are used.

    Args:
        vbox: Vehicle box.
        person_box: Box of the person being evaluated.
        depth_at: Robust normalized depth at an integer pixel.
        frame_size: ``(width, height)`` used to clamp the sampling points.

    Returns:
        Median normalized depth of the sampled vehicle points.
    """
    width, height = frame_size
    points = inner_sample_points(vbox, VEHICLE_INNER_MARGIN)
    free = [p for p in points if not contains_point(person_box, *p)] or points
    return median(depth_at(min(max(0, x), width - 1), min(max(0, y), height - 1)) for x, y in free)


@dataclass
class PersonResult:
    """Proximity state of one person in the current frame.

    Attributes:
        track_id: Tracker id.
        box: Clipped person box.
        foot: Bottom-center point of the box.
        z: Median normalized depth at the foot point, in [0, 1] (0 = nearest,
            1 = farthest surface of the frame; see :mod:`proximity_alert.depth`).
        is_rider: Person is (or recently was) on a bicycle/motorcycle; skipped.
        is_occupant: Person is (or recently was) inside a car/bus/truck box at the
            vehicle's depth (driver or passenger); skipped like a rider.
        currently_close: Passed both the 2D and the depth gate in this frame.
        min_dz: Smallest depth difference among vehicles passing the 2D gate (``None``
            when none did or the frame had no depth evidence).
        connections: ``(foot, vehicle_contact_point)`` pairs that passed both gates.
        ratio: Fraction of close frames in the sliding window.
        alert: Final alert decision.
    """

    track_id: Hashable
    box: Box
    foot: Point
    z: float
    is_rider: bool = False
    is_occupant: bool = False
    currently_close: bool = False
    min_dz: Optional[float] = None
    connections: List[Tuple[Point, Point]] = field(default_factory=list)
    ratio: float = 0.0
    alert: bool = False


class ProximityEvaluator:
    """Stateful evaluator holding the per-track temporal memories.

    Args:
        config: Pipeline configuration (gates, windows, rider parameters).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.history = ProximityHistory(config.time_window)
        self.riders = RiderMemory(config.rider_memory)
        self.occupants = RiderMemory(config.rider_memory)

    def evaluate(
        self,
        persons: Sequence[Tuple[Hashable, Box]],
        rider_vehicle_boxes: Sequence[Box],
        target_vehicle_boxes: Sequence[Box],
        depth_at: DepthFn,
        timestamp: float,
        frame_size: Tuple[int, int],
        depth_valid: bool = True,
    ) -> List[PersonResult]:
        """Evaluates every person of a frame against the target vehicles.

        Args:
            persons: ``(track_id, box)`` of each detected person.
            rider_vehicle_boxes: Bicycle and motorcycle boxes.
            target_vehicle_boxes: Car, bus and truck boxes.
            depth_at: Returns the robust normalized depth at integer pixel ``(x, y)``.
            timestamp: Video time of the frame in seconds.
            frame_size: ``(width, height)`` used to clamp depth sampling points.
            depth_valid: ``False`` when the frame's depth map is constant (no depth
                evidence). Pairs cannot be confirmed then: nobody is close in this
                frame and the sliding windows get no sample for it.

        Returns:
            One :class:`PersonResult` per person, in input order.
        """
        cfg = self.config
        width, height = frame_size
        results: List[PersonResult] = []

        for track_id, box in persons:
            fx, fy = foot_point(box)
            z = depth_at(min(max(0, fx), width - 1), min(max(0, fy), height - 1))
            person = PersonResult(track_id=track_id, box=box, foot=(fx, fy), z=z)
            results.append(person)

            if is_rider(box, rider_vehicle_boxes, cfg.rider_overlap):
                self.riders.mark(track_id, timestamp)
            if self.riders.is_rider(track_id, timestamp):
                person.is_rider = True
                continue

            if self._is_occupant_now(person, target_vehicle_boxes, depth_at, frame_size):
                self.occupants.mark(track_id, timestamp)
            if self.occupants.is_rider(track_id, timestamp):
                person.is_occupant = True
                continue

            if not depth_valid:
                continue  # no depth evidence: the depth gate cannot confirm a pair
            self._check_vehicles(person, target_vehicle_boxes, depth_at, frame_size)
            self.history.update(track_id, timestamp, person.currently_close)

        # Keep tracks that vanished briefly (missed detections); drop them after the grace period.
        self.history.prune_stale(timestamp, max(MIN_TRACK_GRACE_SEC, cfg.time_window))
        self.riders.cleanup(timestamp)
        self.occupants.cleanup(timestamp)

        for person in results:
            if not (person.is_rider or person.is_occupant):
                person.ratio = self.history.ratio(person.track_id)
                person.alert = is_alert(person.ratio, person.currently_close, cfg.min_ratio)
        return results

    def _is_occupant_now(
        self, person: PersonResult, vehicles: Sequence[Box], depth_at: DepthFn, frame_size: Tuple[int, int]
    ) -> bool:
        """Whether the person box is mostly inside a vehicle box *and* at the vehicle's depth.

        The depth condition keeps a pedestrian standing in front of a bus (whose box is
        entirely inside the bus box in the image) from being taken for a passenger.
        """
        for vbox in vehicles:
            if overlap_ratio(person.box, vbox) < self.config.occupant_overlap:
                continue
            if abs(person.z - vehicle_depth(vbox, person.box, depth_at, frame_size)) <= self.config.depth_threshold:
                return True
        return False

    def _check_vehicles(
        self, person: PersonResult, vehicles: Sequence[Box], depth_at: DepthFn, frame_size: Tuple[int, int]
    ) -> None:
        """Applies the 2D gate and the depth gate against every target vehicle.

        The vehicle depth is sampled at the closest point of its box when the foot is
        outside it. When the foot is inside the box that point is the foot itself, so the
        vehicle depth is taken from the box interior instead (:func:`vehicle_depth`).
        """
        _, y1, _, y2 = person.box
        max_dist = max(MIN_PERSON_HEIGHT_PX, y2 - y1) * self.config.height_factor
        for vbox in vehicles:
            contact = closest_point_on_box(vbox, *person.foot)
            if distance(person.foot, contact) > max_dist:
                continue
            if contact == person.foot:
                z_vehicle = vehicle_depth(vbox, person.box, depth_at, frame_size)
            else:
                z_vehicle = depth_at(*contact)
            dz = abs(person.z - z_vehicle)
            if person.min_dz is None or dz < person.min_dz:
                person.min_dz = dz
            if dz <= self.config.depth_threshold:
                person.currently_close = True
                person.connections.append((person.foot, contact))
