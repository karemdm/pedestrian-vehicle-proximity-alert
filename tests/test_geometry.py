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


def test_clip_box_box_fully_outside_gives_empty_box():
    x1, y1, x2, y2 = clip_box((120, 10, 150, 40), 100, 80)
    assert x2 <= x1  # callers must drop empty boxes


def test_foot_point_truncates_toward_zero_for_odd_widths():
    assert foot_point((0, 0, 5, 9)) == (2, 9)


def test_closest_point_is_a_corner_for_diagonal_queries():
    box = (10, 10, 50, 50)
    assert closest_point_on_box(box, -5, -5) == (10, 10)
    assert closest_point_on_box(box, 60, 5) == (50, 10)


def test_overlap_is_not_symmetric_and_uses_person_area():
    from proximity_alert.geometry import overlap_ratio

    person, big = (0, 0, 10, 10), (0, 0, 100, 100)
    assert overlap_ratio(person, big) == 1.0
    assert overlap_ratio(big, person) == 0.01


def test_inner_sample_points_form_a_grid_inside_the_shrunk_box():
    from proximity_alert.geometry import contains_point, inner_sample_points

    pts = inner_sample_points((100, 300, 300, 400), 0.25)
    assert pts[0] == (150, 325) and pts[4] == (200, 350) and pts[8] == (250, 375)
    assert len(set(pts)) == 9 and not any(x in (100, 300) or y in (300, 400) for x, y in pts)
    assert contains_point((0, 0, 10, 10), 10, 0) and not contains_point((0, 0, 10, 10), 11, 5)
