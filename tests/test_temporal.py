"""Tests for ProximityHistory and the alert rule."""

import math

from proximity_alert.temporal import ProximityHistory, is_alert


def test_ratio_of_close_frames():
    hist = ProximityHistory(window_sec=1.0)
    for i, close in enumerate([True, False, True, True]):
        hist.update(5, i * 0.1, close)
    assert math.isclose(hist.ratio(5), 0.75)
    assert hist.count(5) == 4


def test_unknown_track_ratio_is_zero():
    assert ProximityHistory(1.0).ratio("missing") == 0.0


def test_old_samples_expire():
    hist = ProximityHistory(window_sec=1.0)
    hist.update(1, 0.0, True)
    hist.update(1, 0.5, True)
    hist.update(1, 1.0, False)  # 1.0 - 0.0 == window, still kept
    assert hist.count(1) == 3
    hist.update(1, 1.6, False)  # drops 0.0 and 0.5
    assert hist.count(1) == 2
    assert hist.ratio(1) == 0.0


def test_prune_removes_inactive_tracks():
    hist = ProximityHistory(1.0)
    hist.update(1, 0.0, True)
    hist.update(2, 0.0, False)
    hist.prune({2})
    assert 1 not in hist and 2 in hist
    assert len(hist) == 1


def test_alert_rule():
    assert is_alert(0.5, False, 0.5)
    assert not is_alert(0.49, False, 0.5)
    assert is_alert(0.0, True, 0.5)


def test_window_size_is_stable_with_float_timestamps():
    hist = ProximityHistory(window_sec=1.0)
    counts = set()
    for frame_idx in range(1, 200):
        hist.update(1, frame_idx / 30.0, True)
        if frame_idx > 31:
            counts.add(hist.count(1))
    assert counts == {31}  # 1 s at 30 FPS, inclusive window
