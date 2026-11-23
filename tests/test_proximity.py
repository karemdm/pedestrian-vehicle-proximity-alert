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
    tiny = (110, 395, 112, 399)  # height 4 -> radius max(10, 4) = 10; foot (111, 399) inside the car box
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


def test_person_with_foot_inside_a_vehicle_box_always_has_zero_dz():
    # Documents a known limitation: the contact point equals the foot, so both depth
    # samples coincide and the depth gate cannot reject the pair.
    ev = ProximityEvaluator(Config(depth_threshold=0.0))
    inside = (150, 300, 170, 380)
    (p,) = ev.evaluate([(1, inside)], [], [CAR], split_depth, 0.0, FRAME)
    assert p.min_dz == 0.0 and p.currently_close
