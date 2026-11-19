from dataclasses import replace

from proximity_alert.config import Config
from proximity_alert.proximity import ProximityEvaluator
from proximity_alert.temporal import ProximityHistory

FRAME = (1000, 1000)
CAR = (100, 300, 300, 400)
NEAR = (320, 300, 340, 400)
FAR = (700, 300, 720, 400)
DT = 1 / 30.0


def depth(x, y):
    return 0.5


def test_prune_stale_keeps_recent_and_drops_old():
    h = ProximityHistory(1.0)
    h.update("a", 0.0, True)
    h.update("b", 0.9, True)
    h.prune_stale(1.5, 1.0)
    assert "a" not in h and "b" in h


def test_track_missing_two_frames_keeps_ratio():
    ev = ProximityEvaluator(Config())
    for i in range(3):
        ev.evaluate([(1, NEAR)], [], [CAR], depth, i * DT, FRAME)
    ev.evaluate([], [], [CAR], depth, 3 * DT, FRAME)
    ev.evaluate([], [], [CAR], depth, 4 * DT, FRAME)
    (p,) = ev.evaluate([(1, FAR)], [], [CAR], depth, 5 * DT, FRAME)
    # 3 close samples + 1 far sample: history survived the gap.
    assert ev.history.count(1) == 4
    assert p.ratio == 0.75


def test_track_gone_longer_than_grace_is_pruned():
    ev = ProximityEvaluator(Config())
    ev.evaluate([(1, NEAR)], [], [CAR], depth, 0.0, FRAME)
    ev.evaluate([], [], [CAR], depth, 0.5, FRAME)
    assert 1 in ev.history
    ev.evaluate([], [], [CAR], depth, 1.5, FRAME)
    assert 1 not in ev.history


def test_grace_follows_longer_time_window():
    ev = ProximityEvaluator(replace(Config(), time_window=3.0))
    ev.evaluate([(1, NEAR)], [], [CAR], depth, 0.0, FRAME)
    ev.evaluate([], [], [CAR], depth, 2.5, FRAME)
    assert 1 in ev.history
    ev.evaluate([], [], [CAR], depth, 3.5, FRAME)
    assert 1 not in ev.history


def test_rider_memory_survives_gap():
    ev = ProximityEvaluator(Config())
    bike = (320, 300, 340, 400)
    (p,) = ev.evaluate([(1, NEAR)], [bike], [CAR], depth, 0.0, FRAME)
    assert p.is_rider
    ev.evaluate([], [], [CAR], depth, DT, FRAME)
    (p,) = ev.evaluate([(1, NEAR)], [], [CAR], depth, 2 * DT, FRAME)
    assert p.is_rider
