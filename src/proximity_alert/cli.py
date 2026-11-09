"""Command-line interface (``proximity-alert``)."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from proximity_alert import __version__
from proximity_alert.config import Config


def build_parser() -> argparse.ArgumentParser:
    """Creates the argument parser with the prototype defaults."""
    d = Config()
    p = argparse.ArgumentParser(
        prog="proximity-alert",
        description="Flag pedestrians close to cars, buses and trucks using YOLO tracking "
        "and monocular relative depth. Without --input, a setup window opens.",
    )
    p.add_argument("-i", "--input", help="Input video path. If omitted, the setup GUI is opened.")
    p.add_argument("-o", "--output", help="Output MP4 path (required with --input).")
    p.add_argument("--gui", action="store_true", help="Open the setup GUI prefilled with the CLI values.")
    p.add_argument("--model", default=d.yolo_model, help="YOLO weights path or official name (default: %(default)s).")
    p.add_argument("--depth-model", default=d.depth_model, help="Hugging Face depth model id (default: %(default)s).")
    p.add_argument("--imgsz", type=int, default=d.imgsz, help="YOLO inference size (default: %(default)s).")
    p.add_argument("--conf", type=float, default=d.conf, help="YOLO confidence threshold (default: %(default)s).")
    p.add_argument(
        "--time-window", type=float, default=d.time_window, help="Sliding window in seconds (default: %(default)s)."
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=d.min_ratio * 100,
        help="Percentage of close frames in the window that triggers an alert (default: %(default)s).",
    )
    p.add_argument(
        "--depth-threshold",
        type=float,
        default=d.depth_threshold,
        help="Max |z_person - z_vehicle| on the normalized depth map (default: %(default)s).",
    )
    p.add_argument(
        "--height-factor",
        type=float,
        default=d.height_factor,
        help="2D gate radius as a multiple of the person bbox height (default: %(default)s).",
    )
    p.add_argument(
        "--rider-overlap",
        type=float,
        default=d.rider_overlap,
        help="Fraction of the person area covered by a bike/motorcycle to be a rider (default: %(default)s).",
    )
    p.add_argument(
        "--rider-memory", type=float, default=d.rider_memory, help="Seconds a rider status persists (default: %(default)s)."
    )
    p.add_argument(
        "--depth-patch", type=int, default=d.depth_patch, help="Median depth patch size in pixels (default: %(default)s)."
    )
    p.add_argument(
        "--pip-scale", type=float, default=d.pip_scale, help="Depth picture-in-picture scale (default: %(default)s)."
    )
    p.add_argument("--no-display", action="store_true", help="Do not open the live preview window.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """Builds a :class:`Config` from parsed arguments."""
    return Config(
        input_path=args.input or "",
        output_path=args.output or "",
        yolo_model=args.model,
        depth_model=args.depth_model,
        imgsz=args.imgsz,
        conf=args.conf,
        time_window=args.time_window,
        min_ratio=args.min_ratio / 100.0,
        depth_threshold=args.depth_threshold,
        height_factor=args.height_factor,
        rider_overlap=args.rider_overlap,
        rider_memory=args.rider_memory,
        depth_patch=args.depth_patch,
        pip_scale=args.pip_scale,
        display=not args.no_display,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Parses arguments, optionally opens the GUI, and runs the pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = config_from_args(args)

    if args.gui or not args.input:
        from proximity_alert.gui import ask_config

        chosen = ask_config(config)
        if chosen is None:
            logging.getLogger(__name__).info("Setup window closed; nothing to do.")
            return 0
        config = chosen
    elif not args.output:
        parser.error("--output is required when --input is given")

    from proximity_alert.pipeline import run

    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
