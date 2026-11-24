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
        frame_step=0,
    )
    assert len(config_errors(bad)) == 11


def test_nan_is_rejected():
    assert config_errors(replace(Config(), conf=float("nan"))) != []


def test_path_errors(tmp_path):
    from proximity_alert.validation import path_errors

    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    assert path_errors(replace(Config(), input_path=str(a), output_path=str(tmp_path / "b.mp4"))) == []
    assert path_errors(replace(Config(), input_path=str(a), output_path=str(tmp_path / "." / "a.mp4"))) != []
    assert path_errors(Config()) == []  # unset paths are reported elsewhere


def test_occupant_overlap_must_be_a_fraction():
    from proximity_alert.config import Config
    from proximity_alert.validation import config_errors

    assert not config_errors(Config(occupant_overlap=0.0)) and not config_errors(Config(occupant_overlap=1.0))
    assert any("occupant" in e for e in config_errors(Config(occupant_overlap=1.5)))
