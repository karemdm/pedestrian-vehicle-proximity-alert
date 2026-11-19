import pytest

from proximity_alert.config import Config, effective_fps, processed_frame_count
from proximity_alert.temporal import ProximityHistory, RiderMemory


def test_effective_fps_divides_by_step():
    assert effective_fps(30.0, 1) == 30.0
    assert effective_fps(30.0, 3) == pytest.approx(10.0)
    assert effective_fps(29.97, 2) == pytest.approx(29.97 / 2)


def test_effective_fps_rejects_invalid_step():
    with pytest.raises(ValueError):
        effective_fps(30.0, 0)
    with pytest.raises(ValueError):
        processed_frame_count(100, -1)


def test_processed_frame_count():
    assert processed_frame_count(100, 1) == 100
    assert processed_frame_count(100, 3) == 34  # frames 0, 3, ..., 99
    assert processed_frame_count(99, 3) == 33
    assert processed_frame_count(0, 5) == 0


def test_default_config_processes_every_frame():
    assert Config().frame_step == 1


def test_output_duration_matches_input_duration():
    fps, step, total = 30.0, 4, 300
    n = processed_frame_count(total, step)
    assert n / effective_fps(fps, step) == pytest.approx(total / fps, abs=step / fps)


def _timestamps(fps, step, n):
    out_fps = effective_fps(fps, step)
    return [i / out_fps for i in range(1, n + 1)]


def test_window_with_effective_fps_keeps_seconds():
    fps, step = 30.0, 3
    hist = ProximityHistory(1.0)
    for t in _timestamps(fps, step, 50):
        hist.update("a", t, True)
    # 10 effective fps and a 1 s inclusive window -> 11 samples, not 31.
    assert hist.count("a") == 11


def test_window_with_raw_fps_would_be_wrong():
    # Guards the regression: using the raw fps for timestamps stretches the window in real time.
    hist = ProximityHistory(1.0)
    for i in range(1, 51):
        hist.update("a", i / 30.0, True)
    assert hist.count("a") == 31


def test_rider_memory_with_effective_fps():
    fps, step = 30.0, 5
    mem = RiderMemory(2.0)
    ts = _timestamps(fps, step, 40)  # 6 effective fps
    mem.mark("p", ts[0])
    assert mem.is_rider("p", ts[0] + 2.0)
    assert not mem.is_rider("p", ts[0] + 2.0 + 1.0 / (fps / step) * 0.5)
