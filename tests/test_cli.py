"""Tests for the command-line interface (no models)."""

import pytest

from proximity_alert import cli
from proximity_alert.config import Config


def test_defaults_round_trip_through_the_parser():
    cfg = cli.config_from_args(cli.build_parser().parse_args([]))
    assert cfg == Config(display=True)


def test_min_ratio_is_given_in_percent():
    cfg = cli.config_from_args(cli.build_parser().parse_args(["--min-ratio", "30", "--no-display"]))
    assert cfg.min_ratio == pytest.approx(0.30) and not cfg.display


def test_invalid_value_is_rejected_before_the_setup_window_opens(monkeypatch):
    import proximity_alert.gui as gui

    monkeypatch.setattr(gui, "ask_config", lambda c: pytest.fail("window must not open"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--frame-step", "0"])
    assert exc.value.code == 2


def test_output_required_with_input(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        cli.main(["-i", "a.mp4"])
    assert exc.value.code == 2


def test_same_input_and_output_rejected():
    with pytest.raises(SystemExit) as exc:
        cli.main(["-i", "a.mp4", "-o", "./a.mp4"])
    assert exc.value.code == 2


def test_closing_the_setup_window_exits_with_0(monkeypatch):
    import proximity_alert.gui as gui

    monkeypatch.setattr(gui, "ask_config", lambda c: None)
    assert cli.main([]) == 0


def test_main_returns_the_pipeline_exit_code(monkeypatch):
    import proximity_alert.pipeline as pipeline

    seen = {}
    monkeypatch.setattr(pipeline, "run", lambda c: seen.setdefault("cfg", c) and 1)
    assert cli.main(["-i", "a.mp4", "-o", "b.mp4", "--no-display", "--frame-step", "2"]) == 1
    assert seen["cfg"].frame_step == 2 and seen["cfg"].output_path == "b.mp4"
