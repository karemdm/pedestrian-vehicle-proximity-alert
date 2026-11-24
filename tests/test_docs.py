"""Keeps the documentation consistent with the code."""

import re
from pathlib import Path

from proximity_alert.cli import build_parser
from proximity_alert.config import Config

ROOT = Path(__file__).resolve().parents[1]


def _diagram(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(r"```mermaid\n(.*?)```", text, re.S)
    assert match, path
    return match.group(1).replace("\r\n", "\n")


def test_readme_and_how_it_works_share_the_same_diagram():
    assert _diagram("README.md") == _diagram("docs/how-it-works.md")


def test_diagram_is_structurally_sound():
    src = _diagram("README.md")
    assert src.startswith("flowchart TD")
    assert src.count('"') % 2 == 0
    for line in src.splitlines():
        assert line.count("{") == line.count("}") and line.count("[") == line.count("]")
    defined = set(re.findall(r"^\s*([A-Z])[\[{]", src, re.M)) | set(re.findall(r"-->\s*(?:\|[^|]*\|\s*)?([A-Z])[\[{]", src))
    used = set(re.findall(r"-->\s*(?:\|[^|]*\|\s*)?([A-Z])\b", src)) | set(re.findall(r"^\s*([A-Z])\s*(?:-->|-- )", src, re.M))
    assert used <= defined


def test_every_cli_option_is_documented_in_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for action in build_parser()._actions:
        for flag in action.option_strings:
            if flag.startswith("--") and flag != "--help":
                assert f"`{flag}`" in readme or f"`{flag}," in readme or f", {flag}`" in readme, flag


def test_documented_defaults_match_config():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    d = Config()
    for name, value in [
        ("--imgsz", d.imgsz),
        ("--conf", f"{d.conf:.2f}"),
        ("--time-window", d.time_window),
        ("--depth-threshold", d.depth_threshold),
        ("--height-factor", d.height_factor),
        ("--rider-overlap", f"{d.rider_overlap:.2f}"),
        ("--occupant-overlap", f"{d.occupant_overlap:.2f}"),
        ("--rider-memory", d.rider_memory),
        ("--depth-patch", d.depth_patch),
        ("--pip-scale", d.pip_scale),
        ("--frame-step", d.frame_step),
        ("--model", d.yolo_model),
    ]:
        row = next(line for line in readme.splitlines() if line.startswith(f"| `{name}`"))
        assert f"`{value}`" in row, (name, value, row)
