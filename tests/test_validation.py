from dataclasses import replace

from proximity_alert.config import Config
from proximity_alert.validation import config_errors


def test_defaults_are_valid():
    assert config_errors(Config()) == []


def test_invalid_values_are_reported():
    bad = replace(
        Config(),
        imgsz=0,
        conf=1.5,
        time_window=0.0,
        min_ratio=2.0,
        depth_threshold=-0.1,
        height_factor=0.0,
        rider_overlap=-1.0,
        rider_memory=-1.0,
        depth_patch=0,
        pip_scale=1.5,
    )
    assert len(config_errors(bad)) == 10


def test_nan_is_rejected():
    assert config_errors(replace(Config(), conf=float("nan"))) != []
