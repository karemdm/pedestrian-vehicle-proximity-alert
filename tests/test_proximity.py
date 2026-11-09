"""Tests for ProximityEvaluator with a fake depth function."""

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
