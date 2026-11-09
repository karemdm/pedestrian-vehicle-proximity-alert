"""Tests for rider overlap and RiderMemory."""

import math

from proximity_alert.geometry import is_rider, overlap_ratio
from proximity_alert.temporal import RiderMemory

PERSON = (0, 0, 10, 20)  # area 200


def test_overlap_ratio_is_relative_to_person_area():
    assert math.isclose(overlap_ratio(PERSON, (0, 10, 10, 40)), 0.5)
    assert overlap_ratio(PERSON, (0, 0, 100, 100)) == 1.0


def test_overlap_ratio_no_overlap_and_degenerate():
    assert overlap_ratio(PERSON, (20, 20, 30, 30)) == 0.0
    assert overlap_ratio(PERSON, (10, 0, 20, 20)) == 0.0  # touching edge only
    assert overlap_ratio((5, 5, 5, 10), (0, 0, 10, 10)) == 0.0


def test_is_rider_threshold():
    half = (0, 10, 10, 40)
    assert is_rider(PERSON, [half], threshold=0.50)
    assert not is_rider(PERSON, [half], threshold=0.51)
    assert not is_rider(PERSON, [], threshold=0.50)
    assert is_rider(PERSON, [(50, 50, 60, 60), half])


def test_rider_memory_persists_then_expires():
    mem = RiderMemory(memory_sec=2.0)
    mem.mark(1, 10.0)
    assert mem.is_rider(1, 10.0)
    assert mem.is_rider(1, 12.0)
    assert not mem.is_rider(1, 12.01)


def test_unknown_track_is_not_a_rider_at_video_start():
    mem = RiderMemory(memory_sec=2.0)
    assert not mem.is_rider(7, 0.04)
    assert mem.last_seen(7) is None


def test_rider_memory_cleanup_after_twice_memory():
    mem = RiderMemory(memory_sec=2.0)
    mem.mark(1, 0.0)
    mem.mark(2, 3.0)
    mem.cleanup(4.0)
    assert len(mem) == 2
    mem.cleanup(4.5)
    assert mem.last_seen(1) is None
    assert mem.last_seen(2) == 3.0
