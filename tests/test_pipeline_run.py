"""Tests for pipeline.run with fake models and a fake capture (no inference)."""

import sys
import types

import cv2
import numpy as np
import pytest

from proximity_alert import pipeline
from proximity_alert.config import Config
from proximity_alert.proximity import ProximityEvaluator

W, H = 64, 48
RealCapture = cv2.VideoCapture  # pipeline.cv2 is the same module object, so keep the original


class FakeCap:
    def __init__(self, n_frames, fps=10.0, opened=True):
        self.frames = [np.full((H, W, 3), i, np.uint8) for i in range(n_frames)]
        self.fps = fps
        self.opened = opened
        self.pos = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: W,
            cv2.CAP_PROP_FRAME_HEIGHT: H,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
        }[prop]

    def grab(self):
        if self.pos >= len(self.frames):
            return False
        self.pos += 1
        return True

    def read(self):
        if self.pos >= len(self.frames):
            return False, None
        self.pos += 1
        return True, self.frames[self.pos - 1].copy()

    def release(self):
        self.released = True


class FakeDepth:
    def __init__(self, model_id):
        pass

    def __call__(self, frame):
        return np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))


class FakeYolo:
    seen_frames = []

    def __init__(self, name):
        pass

    def track(self, frame, **kwargs):
        FakeYolo.seen_frames.append(int(frame[0, 0, 0]))
        return [types.SimpleNamespace(boxes=None)]


@pytest.fixture
def fakes(monkeypatch):
    FakeYolo.seen_frames = []
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setattr(pipeline, "DepthEstimator", FakeDepth)
    stamps = []
    real = ProximityEvaluator.evaluate

    def spy(self, persons, riders, targets, depth_at, timestamp, frame_size):
        stamps.append(timestamp)
        return real(self, persons, riders, targets, depth_at, timestamp, frame_size)

    monkeypatch.setattr(ProximityEvaluator, "evaluate", spy)
    return stamps


def _cfg(tmp_path, **kw):
    return Config(input_path="in.mp4", output_path=str(tmp_path / "out" / "o.mp4"), display=False, **kw)


def _use_cap(monkeypatch, cap):
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda path: cap)
    return cap


def test_frame_step_processes_every_nth_frame_and_timestamps(monkeypatch, tmp_path, fakes):
    cap = _use_cap(monkeypatch, FakeCap(10, fps=10.0))
    assert pipeline.run(_cfg(tmp_path, frame_step=3)) == 0
    assert FakeYolo.seen_frames == [0, 3, 6, 9]
    assert fakes == pytest.approx([1 / (10 / 3), 2 / (10 / 3), 3 / (10 / 3), 4 / (10 / 3)])
    assert cap.released
    out = RealCapture(str(tmp_path / "out" / "o.mp4"))
    assert out.isOpened() and int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert out.get(cv2.CAP_PROP_FPS) == pytest.approx(10 / 3, rel=0.05)
    out.release()


def test_zero_fps_from_container_falls_back_to_30(monkeypatch, tmp_path, fakes):
    _use_cap(monkeypatch, FakeCap(3, fps=0.0))
    assert pipeline.run(_cfg(tmp_path)) == 0
    assert fakes == pytest.approx([1 / 30, 2 / 30, 3 / 30])


def test_nan_fps_falls_back_to_30(monkeypatch, tmp_path, fakes):
    _use_cap(monkeypatch, FakeCap(2, fps=float("nan")))
    assert pipeline.run(_cfg(tmp_path)) == 0
    assert fakes[0] == pytest.approx(1 / 30)


def test_cannot_open_input_returns_1(monkeypatch, tmp_path, fakes):
    _use_cap(monkeypatch, FakeCap(0, opened=False))
    assert pipeline.run(_cfg(tmp_path)) == 1


def test_no_readable_frames_returns_1(monkeypatch, tmp_path, fakes):
    cap = _use_cap(monkeypatch, FakeCap(0))
    assert pipeline.run(_cfg(tmp_path)) == 1
    assert cap.released


def test_output_equal_to_input_is_refused(monkeypatch, tmp_path, fakes):
    cap = _use_cap(monkeypatch, FakeCap(3))
    cfg = Config(input_path=str(tmp_path / "v.mp4"), output_path=str(tmp_path / "." / "v.mp4"), display=False)
    assert pipeline.run(cfg) == 1
    assert cap.pos == 0


def test_resources_released_when_window_creation_fails(monkeypatch, tmp_path, fakes):
    cap = _use_cap(monkeypatch, FakeCap(3))

    def boom(*a, **k):
        raise cv2.error("no GUI support")

    monkeypatch.setattr(pipeline.cv2, "namedWindow", boom)
    cfg = _cfg(tmp_path)
    cfg.display = True
    with pytest.raises(cv2.error):
        pipeline.run(cfg)
    assert cap.released


def test_frames_of_unexpected_size_are_resized(monkeypatch, tmp_path, fakes):
    cap = FakeCap(2)
    cap.frames[1] = np.zeros((H + 10, W + 6, 3), np.uint8)
    _use_cap(monkeypatch, cap)
    assert pipeline.run(_cfg(tmp_path)) == 0
    out = RealCapture(str(tmp_path / "out" / "o.mp4"))
    assert int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    out.release()
