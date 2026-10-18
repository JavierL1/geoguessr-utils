"""
CLI entry point for score-screen extraction.

Parses arguments, resolves the layout, and hands off to
`geoguessr_utils.extraction.runner.extract_directory`.

This is the only module in the codebase that imports `Layout2026B`
for the extraction side. Every other module depends only on the
abstract `Layout` interface.
"""

import argparse
import sys
from pathlib import Path

from geoguessr_utils.config import AppConfig
from geoguessr_utils.extraction.runner import extract_directory
from geoguessr_utils.layouts import Layout2026B


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract data from Geoguessr score screens."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="./output_frames",
        help="Directory containing the frames",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tournament_data.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enables debug image storage",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(
            f"Error: Input directory '{input_path}' does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = AppConfig.load(config_path)

    extract_directory(
        input_dir=args.input,
        output_json=args.output,
        layout=Layout2026B(),
        config=config,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
