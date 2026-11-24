"""Tests for ProximityEvaluator with a fake depth function."""
import pytest

from proximity_alert.config import Config
from proximity_alert.proximity import ProximityEvaluator

FRAME = (1000, 1000)
CAR = (100, 300, 300, 400)
PERSON_NEAR = (320, 300, 340, 400)  # foot (330, 400), 30 px from the car, height 100


def flat_depth(x, y):
    return 0.5


def split_depth(x, y):
    return 0.9 if x >= 320 else 0.5  # person much closer than the car


def test_close_in_2d_and_depth_triggers_alert():
    ev = ProximityEvaluator(Config())
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, 0.1, FRAME)
    assert p.currently_close and p.alert
    assert p.connections == [((330, 400), (300, 400))]
    assert p.min_dz == 0.0
    assert p.ratio == 1.0


def test_depth_gate_rejects_far_in_depth():
    ev = ProximityEvaluator(Config())
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], split_depth, 0.1, FRAME)
    assert not p.currently_close and not p.alert
    assert abs(p.min_dz - 0.4) < 1e-9


def test_2d_gate_rejects_far_in_image():
    ev = ProximityEvaluator(Config(height_factor=0.2))  # radius 20 px < 30 px
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, 0.1, FRAME)
    assert not p.currently_close
    assert p.min_dz is None


def test_rider_is_excluded_and_remembered():
    ev = ProximityEvaluator(Config())
    bike = (320, 330, 340, 420)  # covers 70 % of the person box
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [bike], [CAR], flat_depth, 0.1, FRAME)
    assert p.is_rider and not p.alert and not p.connections
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, 1.5, FRAME)
    assert p.is_rider
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, 2.2, FRAME)
    assert not p.is_rider and p.alert


def test_window_ratio_keeps_alert_after_contact():
    ev = ProximityEvaluator(Config())
    far = (800, 300, 820, 400)
    for i in range(3):
        ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, i * 0.1, FRAME)
    (p,) = ev.evaluate([(1, far)], [], [CAR], flat_depth, 0.3, FRAME)
    assert not p.currently_close
    assert p.ratio == 0.75 and p.alert


def test_no_detections_returns_empty_list():
    ev = ProximityEvaluator(Config())
    assert ev.evaluate([], [], [], flat_depth, 0.0, FRAME) == []


def test_only_riders_and_no_pedestrians():
    ev = ProximityEvaluator(Config())
    assert ev.evaluate([], [(0, 0, 50, 50)], [CAR], flat_depth, 0.0, FRAME) == []


def test_person_without_vehicles_is_ok_with_no_dz():
    ev = ProximityEvaluator(Config())
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [], flat_depth, 0.0, FRAME)
    assert not p.alert and p.min_dz is None and p.ratio == 0.0


def test_depth_is_sampled_inside_the_frame_when_the_foot_is_on_the_border():
    calls = []

    def depth(x, y):
        calls.append((x, y))
        return 0.5

    ev = ProximityEvaluator(Config())
    ev.evaluate([(1, (10, 900, 30, 1000))], [], [], depth, 0.0, FRAME)  # foot y == frame height
    assert calls == [(20, 999)]


def test_tiny_person_uses_minimum_height_for_the_2d_gate():
    ev = ProximityEvaluator(Config())
    tiny = (89, 395, 91, 399)  # height 4 -> radius max(10, 4) = 10; foot (90, 399), 10 px from the car's left edge
    car = (100, 300, 300, 400)
    (p,) = ev.evaluate([(1, tiny)], [], [car], flat_depth, 0.0, FRAME)
    assert p.currently_close
    far = (310, 395, 312, 399)  # foot (311, 399), 11 px from the car's right edge
    (p,) = ev.evaluate([(2, far)], [], [car], flat_depth, 0.0, FRAME)
    assert not p.currently_close


def test_2d_gate_is_inclusive_at_the_radius():
    ev = ProximityEvaluator(Config(height_factor=0.3))  # radius exactly 30 px
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, 0.0, FRAME)
    assert p.currently_close


def test_depth_gate_is_inclusive_at_the_threshold():
    ev = ProximityEvaluator(Config(depth_threshold=0.25))

    def depth(x, y):
        return 0.75 if x >= 320 else 0.5

    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR], depth, 0.0, FRAME)
    assert p.currently_close


def test_min_dz_is_the_smallest_among_vehicles_passing_the_2d_gate():
    ev = ProximityEvaluator(Config(depth_threshold=0.0))
    other = (340, 300, 500, 400)  # right of the person, also within the gate

    def depth(x, y):
        return 0.9 if x >= 320 and x < 340 else (0.6 if x >= 340 else 0.1)

    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR, other], depth, 0.0, FRAME)
    assert p.min_dz == pytest.approx(0.3) and not p.currently_close


def test_all_vehicles_passing_both_gates_add_a_connection():
    ev = ProximityEvaluator(Config())
    other = (340, 300, 500, 400)
    (p,) = ev.evaluate([(1, PERSON_NEAR)], [], [CAR, other], flat_depth, 0.0, FRAME)
    assert len(p.connections) == 2


def test_alert_ends_when_the_window_forgets_the_contact():
    ev = ProximityEvaluator(Config())
    far = (800, 300, 820, 400)
    ev.evaluate([(1, PERSON_NEAR)], [], [CAR], flat_depth, 0.0, FRAME)
    (p,) = ev.evaluate([(1, far)], [], [CAR], flat_depth, 0.5, FRAME)
    assert p.alert and p.ratio == 0.5  # 1 close of 2 samples
    (p,) = ev.evaluate([(1, far)], [], [CAR], flat_depth, 1.2, FRAME)
    assert not p.alert and p.ratio == 0.0


def region_depth(person_box, person_z, vehicle_z, vehicle_box=CAR, road_z=0.8):
    """Fake depth map: person region, vehicle region (person on top), road elsewhere."""

    def inside(box, x, y):
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    def depth(x, y):
        if inside(person_box, x, y):
            return person_z
        if inside(vehicle_box, x, y):
            return vehicle_z
        return road_z

    return depth


IN_FRONT = (150, 300, 170, 380)  # box inside the car box, foot (160, 380) inside it
OCCUPANT_BOX = (150, 310, 190, 390)  # driver: fully inside the car box


def test_person_in_front_of_a_vehicle_with_a_depth_gap_is_not_alerted():
    ev = ProximityEvaluator(Config())
    depth = region_depth(IN_FRONT, person_z=0.2, vehicle_z=0.6)
    (p,) = ev.evaluate([(1, IN_FRONT)], [], [CAR], depth, 0.0, FRAME)
    assert not p.is_occupant and not p.currently_close and not p.alert
    assert p.min_dz == pytest.approx(0.4)  # vehicle sampled inside the box, not at the foot


def test_person_in_front_of_a_vehicle_is_rejected_by_a_zero_threshold():
    ev = ProximityEvaluator(Config(depth_threshold=0.0))
    depth = region_depth(IN_FRONT, person_z=0.5, vehicle_z=0.6)
    (p,) = ev.evaluate([(1, IN_FRONT)], [], [CAR], depth, 0.0, FRAME)
    assert p.min_dz == pytest.approx(0.1) and not p.currently_close


def test_person_beside_a_vehicle_at_the_same_depth_is_still_alerted():
    ev = ProximityEvaluator(Config())
    beside = (280, 300, 320, 400)  # 50 % of the box over the car: not an occupant; foot (300, 400) on the border
    depth = region_depth(beside, person_z=0.6, vehicle_z=0.6)
    (p,) = ev.evaluate([(1, beside)], [], [CAR], depth, 0.0, FRAME)
    assert not p.is_occupant and p.currently_close and p.alert
    assert p.min_dz == pytest.approx(0.0)


def test_person_overlapping_the_vehicle_edge_at_a_different_depth_is_not_alerted():
    ev = ProximityEvaluator(Config())
    beside = (280, 300, 320, 400)
    depth = region_depth(beside, person_z=0.2, vehicle_z=0.6)
    (p,) = ev.evaluate([(1, beside)], [], [CAR], depth, 0.0, FRAME)
    assert not p.currently_close and not p.alert


def test_occupant_inside_a_vehicle_at_its_depth_is_excluded_and_remembered():
    ev = ProximityEvaluator(Config())
    depth = region_depth(OCCUPANT_BOX, person_z=0.6, vehicle_z=0.6)
    (p,) = ev.evaluate([(1, OCCUPANT_BOX)], [], [CAR], depth, 0.0, FRAME)
    assert p.is_occupant and not p.is_rider and not p.alert and not p.connections
    # detected half outside the vehicle a moment later: still an occupant (memory)
    half_out = (260, 310, 340, 390)
    (p,) = ev.evaluate([(1, half_out)], [], [CAR], depth, 1.0, FRAME)
    assert p.is_occupant and not p.alert
    (p,) = ev.evaluate([(1, half_out)], [], [CAR], depth, 3.5, FRAME)  # memory (2 s) expired
    assert not p.is_occupant


def test_occupant_overlap_threshold_is_configurable():
    depth = region_depth(OCCUPANT_BOX, person_z=0.6, vehicle_z=0.6)
    partly = (260, 310, 340, 390)  # 50 % inside the car box
    (p,) = ProximityEvaluator(Config(occupant_overlap=0.4)).evaluate([(1, partly)], [], [CAR], depth, 0.0, FRAME)
    assert p.is_occupant
    (p,) = ProximityEvaluator(Config()).evaluate([(1, partly)], [], [CAR], depth, 0.0, FRAME)
    assert not p.is_occupant


def test_person_inside_a_vehicle_box_at_a_different_depth_is_not_an_occupant():
    ev = ProximityEvaluator(Config())
    depth = region_depth(OCCUPANT_BOX, person_z=0.1, vehicle_z=0.6)
    (p,) = ev.evaluate([(1, OCCUPANT_BOX)], [], [CAR], depth, 0.0, FRAME)
    assert not p.is_occupant and not p.alert


def test_foot_outside_the_vehicle_box_still_samples_the_border_point():
    calls = []

    def depth(x, y):
        calls.append((x, y))
        return 0.5

    ev = ProximityEvaluator(Config())
    ev.evaluate([(1, PERSON_NEAR)], [], [CAR], depth, 0.0, FRAME)
    assert calls == [(330, 400), (300, 400)]  # foot, then the closest point on the box


def test_vehicle_depth_ignores_grid_points_covered_by_the_person():
    from proximity_alert.proximity import vehicle_depth

    covering = (100, 300, 300, 400)  # covers every grid point: falls back to all of them
    assert vehicle_depth(CAR, covering, lambda x, y: 0.3, FRAME) == pytest.approx(0.3)
    person = (140, 300, 200, 400)
    depth = region_depth(person, person_z=0.0, vehicle_z=0.7)
    assert vehicle_depth(CAR, person, depth, FRAME) == pytest.approx(0.7)
