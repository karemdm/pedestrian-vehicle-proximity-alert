"""Tests for drawing helpers (synthetic frames)."""

import numpy as np

from proximity_alert.proximity import PersonResult
from proximity_alert.rendering import (
    ALERT_COLOR,
    OK_COLOR,
    RIDER_COLOR,
    draw_person,
    overlay_depth_pip,
    render,
)


def _frame(w=200, h=120):
    return np.zeros((h, w, 3), np.uint8)


def test_pip_is_top_right_and_does_not_touch_the_rest():
    frame = _frame()
    overlay_depth_pip(frame, np.zeros((120, 200), np.float32), scale=0.25, margin=10)
    ys, xs = np.nonzero(frame.any(axis=2))
    assert xs.min() == 200 - 50 - 10 and xs.max() == 200 - 10 - 1
    assert ys.min() == 10 and ys.max() == 10 + 30 - 1


def test_pip_skipped_when_it_does_not_fit_or_is_tiny():
    for scale, w, h in [(0.99, 100, 100), (0.01, 100, 100), (0.5, 12, 12)]:
        frame = np.zeros((h, w, 3), np.uint8)
        overlay_depth_pip(frame, np.zeros((h, w), np.float32), scale=scale, margin=10)
        assert not frame.any()


def test_pip_with_depth_map_of_other_size_still_fits():
    frame = _frame()
    overlay_depth_pip(frame, np.zeros((518, 700), np.float32), scale=0.25)
    assert frame.any()


def _person(**kw):
    base = dict(track_id=1, box=(20, 40, 60, 100), foot=(40, 100), z=0.4)
    base.update(kw)
    return PersonResult(**base)


def test_person_colors_ok_alert_rider():
    for person, color in [
        (_person(), OK_COLOR),
        (_person(alert=True, ratio=1.0, min_dz=0.0), ALERT_COLOR),
        (_person(is_rider=True), RIDER_COLOR),
    ]:
        frame = _frame()
        draw_person(frame, person)
        assert tuple(int(v) for v in frame[70, 20]) == color  # left border of the box


def test_person_at_frame_top_with_no_dz_draws_without_error():
    frame = _frame()
    draw_person(frame, _person(box=(10, 0, 40, 30)))  # min_dz None -> "N/A"; label clamped inside the frame
    assert frame.any()


def test_connection_line_is_drawn():
    frame = _frame()
    draw_person(frame, _person(connections=[((40, 100), (100, 100))], currently_close=True))
    assert frame[100, 80].any()


def test_render_returns_same_frame_and_handles_no_detections():
    frame = _frame()
    out = render(frame, [], [], [], np.ones((120, 200), np.float32), 0.25)
    assert out is frame
