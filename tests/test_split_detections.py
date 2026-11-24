"""Tests for split_detections with a fake Ultralytics result (no models)."""

import numpy as np

from proximity_alert.pipeline import split_detections


class _T:
    """Minimal tensor stand-in supporting ``.int().cpu().numpy()``."""

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def int(self):
        return _T(self.arr.astype(int))

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Boxes:
    def __init__(self, xyxy, cls, ids=None):
        self.xyxy = _T(np.asarray(xyxy, dtype=float))
        self.cls = _T(cls)
        self.id = None if ids is None else _T(ids)

    def __len__(self):
        return len(self.cls.arr)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def test_roles_ids_and_clipping():
    boxes = _Boxes(
        [(-10, 5, 50, 90), (0, 0, 10, 10), (20, 20, 80, 60), (5, 5, 15, 15), (1, 1, 2, 2)],
        [0, 1, 2, 3, 99],
        [7, 8, 9, 10, 11],
    )
    d = split_detections(_Result(boxes), 100, 80, {})
    assert d.persons == [(7, (0, 5, 50, 80))]
    assert d.rider_vehicles == [(0, 0, 10, 10), (5, 5, 15, 15)]
    assert d.target_vehicles == [(20, 20, 80, 60)]  # class 99 is ignored


def test_bus_and_truck_are_targets():
    d = split_detections(_Result(_Boxes([(0, 0, 9, 9), (1, 1, 9, 9)], [5, 7], [1, 2])), 50, 50, {})
    assert len(d.target_vehicles) == 2 and not d.persons


def test_empty_and_none_boxes():
    assert split_detections(_Result(None), 10, 10, {}).persons == []
    assert split_detections(_Result(_Boxes(np.zeros((0, 4)), [])), 10, 10, {}).persons == []


def test_boxes_outside_the_frame_are_dropped():
    boxes = _Boxes([(200, 10, 260, 50), (10, 100, 30, 120), (10, 10, 30, 50)], [0, 2, 0], [1, 2, 3])
    d = split_detections(_Result(boxes), 100, 80, {})
    assert d.persons == [(3, (10, 10, 30, 50))]
    assert d.target_vehicles == []


def test_fallback_ids_warn_once(caplog):
    state = {}
    boxes = _Boxes([(0, 0, 9, 9), (10, 10, 19, 19)], [0, 0], None)
    with caplog.at_level("WARNING"):
        d1 = split_detections(_Result(boxes), 50, 50, state)
        split_detections(_Result(boxes), 50, 50, state)
    assert [pid for pid, _ in d1.persons] == [0, 1]
    assert sum("no ids" in r.message for r in caplog.records) == 1
