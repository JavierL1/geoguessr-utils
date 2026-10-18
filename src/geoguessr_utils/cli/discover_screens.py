"""
CLI entry point for score-screen discovery.

Parses arguments, pairs a concrete layout with its matching detector,
and hands off to `geoguessr_utils.discovery.scanner.discover`.

This is the only module in the codebase that imports `Layout2026B`
and `ScoreScreenDetector2026B` together. Every other module depends
only on their abstract interfaces.
"""

import argparse
import sys
from pathlib import Path

from geoguessr_utils.discovery.scanner import discover
from geoguessr_utils.layouts import Layout2026B
from geoguessr_utils.utils import video
from geoguessr_utils.vision.detectors.score_screen_2026b import (
    ScoreScreenDetector2026B,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast discovery of Geoguessr score screens."
    )
    parser.add_argument(
        "-v",
        "--video",
        required=True,
        help="Path to video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for saved frames",
    )
    parser.add_argument(
        "--start-time",
        type=int,
        nargs=3,
        metavar=("HH", "MM", "SS"),
        default=[0, 0, 0],
        help="Start time in hours, minutes, seconds",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        nargs=3,
        metavar=("HH", "MM", "SS"),
        default=None,
        help="End time in hours, minutes, seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: Video file '{video_path}' not found.", file=sys.stderr)
        sys.exit(1)

    start_seconds = video.parse_hms(args.start_time)
    end_seconds = video.parse_hms(args.end_time) if args.end_time else None

    layout = Layout2026B()
    detector = ScoreScreenDetector2026B(layout.score_screen_discovery_rois)

    discover(
        video_path=video_path,
        output_dir=Path(args.output),
        detector=detector,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


if __name__ == "__main__":
    main()
