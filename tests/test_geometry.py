"""Tests for proximity_alert.geometry."""

import math

from proximity_alert.geometry import clip_box, closest_point_on_box, distance, foot_point


def test_closest_point_inside_box_is_the_point_itself():
    assert closest_point_on_box((10, 10, 50, 50), 20, 30) == (20, 30)


def test_closest_point_outside_box_is_clamped_to_border():
    box = (10, 10, 50, 50)
    assert closest_point_on_box(box, 0, 30) == (10, 30)
    assert closest_point_on_box(box, 100, 100) == (50, 50)
    assert closest_point_on_box(box, 30, 5) == (30, 10)


def test_distance():
    assert distance((0, 0), (3, 4)) == 5.0
    assert math.isclose(distance((1, 1), (1, 1)), 0.0)


def test_foot_point_is_bottom_center():
    assert foot_point((10, 20, 30, 80)) == (20, 80)
    assert foot_point((10, 20, 31, 80)) == (20, 80)


def test_clip_box_converts_to_int_and_clips():
    assert clip_box((-5.7, 3.9, 120.2, 90.0), 100, 80) == (0, 3, 100, 80)
