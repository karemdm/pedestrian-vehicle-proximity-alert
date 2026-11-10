"""Per-frame pedestrian-vehicle proximity evaluation (standard library only).

Depth is accessed through a ``depth_at(x, y) -> float`` callable, so this module
does not depend on numpy and can be tested with a fake depth function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, List, Optional, Sequence, Tuple

from proximity_alert.config import Config
from proximity_alert.geometry import Box, Point, closest_point_on_box, distance, foot_point, is_rider
from proximity_alert.temporal import ProximityHistory, RiderMemory, is_alert

DepthFn = Callable[[int, int], float]

MIN_PERSON_HEIGHT_PX = 10
"""Lower bound of the person height used for the 2D gate."""


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
        currently_close: Passed both the 2D and the depth gate in this frame.
        min_dz: Smallest depth difference among vehicles passing the 2D gate.
        connections: ``(foot, vehicle_contact_point)`` pairs that passed both gates.
        ratio: Fraction of close frames in the sliding window.
        alert: Final alert decision.
    """

    track_id: Hashable
    box: Box
    foot: Point
    z: float
    is_rider: bool = False
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

    def evaluate(
        self,
        persons: Sequence[Tuple[Hashable, Box]],
        rider_vehicle_boxes: Sequence[Box],
        target_vehicle_boxes: Sequence[Box],
        depth_at: DepthFn,
        timestamp: float,
        frame_size: Tuple[int, int],
    ) -> List[PersonResult]:
        """Evaluates every person of a frame against the target vehicles.

        Args:
            persons: ``(track_id, box)`` of each detected person.
            rider_vehicle_boxes: Bicycle and motorcycle boxes.
            target_vehicle_boxes: Car, bus and truck boxes.
            depth_at: Returns the robust normalized depth at integer pixel ``(x, y)``.
            timestamp: Video time of the frame in seconds.
            frame_size: ``(width, height)`` used to clamp depth sampling points.

        Returns:
            One :class:`PersonResult` per person, in input order.
        """
        cfg = self.config
        width, height = frame_size
        results: List[PersonResult] = []
        active_ids = set()

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

            active_ids.add(track_id)
            self._check_vehicles(person, target_vehicle_boxes, depth_at)
            self.history.update(track_id, timestamp, person.currently_close)

        self.history.prune(active_ids)
        self.riders.cleanup(timestamp)

        for person in results:
            if not person.is_rider:
                person.ratio = self.history.ratio(person.track_id)
                person.alert = is_alert(person.ratio, person.currently_close, cfg.min_ratio)
        return results

    def _check_vehicles(self, person: PersonResult, vehicles: Sequence[Box], depth_at: DepthFn) -> None:
        """Applies the 2D gate and the depth gate against every target vehicle."""
        x1, y1, x2, y2 = person.box
        max_dist = max(MIN_PERSON_HEIGHT_PX, y2 - y1) * self.config.height_factor
        for vbox in vehicles:
            contact = closest_point_on_box(vbox, *person.foot)
            if distance(person.foot, contact) > max_dist:
                continue
            dz = abs(person.z - depth_at(*contact))
            if person.min_dz is None or dz < person.min_dz:
                person.min_dz = dz
            if dz <= self.config.depth_threshold:
                person.currently_close = True
                person.connections.append((person.foot, contact))
