"""
CLI entry point for classifying and filtering an existing folder of frames.

Runs the specified layout's detector against every image in a directory.
False positives are moved to a separate `_false_positives` subdirectory.
The remaining true positives are renamed to ensure contiguous batch
numbering (e.g., if batch_003 is rejected, batch_004 becomes batch_003).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2

from geoguessr_utils.layouts import Layout2026A, Layout2026B
from geoguessr_utils.vision.detectors import (
    ScoreScreenDetector2026A,
    ScoreScreenDetector2026B,
)

LAYOUT_REGISTRY = {
    "2026a": (Layout2026A, ScoreScreenDetector2026A),
    "2026b": (Layout2026B, ScoreScreenDetector2026B),
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# Matches: batch_001_frame_0012345.jpg
PATTERN = re.compile(r"batch_(\d+)_frame_(\d+)\.(jpg|jpeg|png)", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter false positives and renumber batch frames."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directory containing the frames to filter",
    )
    parser.add_argument(
        "-l",
        "--layout",
        choices=LAYOUT_REGISTRY.keys(),
        default="2026b",
        help="Which UI layout to use for detection (default: 2026b)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without actually moving or renaming files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    layout_cls, detector_cls = LAYOUT_REGISTRY[args.layout]
    layout = layout_cls()
    detector = detector_cls(layout.score_screen_discovery_rois)

    false_positives_dir = input_dir / "_false_positives"
    if not args.dry_run:
        false_positives_dir.mkdir(exist_ok=True)

    # 1. Find and parse all frames
    frame_files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            match = PATTERN.search(p.name)
            if match:
                batch_num = int(match.group(1))
                frame_num = int(match.group(2))
                ext = match.group(3)
                frame_files.append(
                    {
                        "path": p,
                        "batch": batch_num,
                        "frame": frame_num,
                        "ext": ext,
                    }
                )
            else:
                print(f"⚠️ Skipping file with unexpected name format: {p.name}")

    if not frame_files:
        print(f"No valid frames found in '{input_dir}'.")
        return

    # 2. Sort chronologically by original batch, then frame
    frame_files.sort(key=lambda x: (x["batch"], x["frame"]))

    print(f"Classifying {len(frame_files)} frames using {layout_cls.__name__}...")

    true_positives = 0
    false_positives = 0
    new_batch_num = 1

    # 3. Process each frame
    for item in frame_files:
        path = item["path"]
        frame = cv2.imread(str(path))

        if frame is None:
            print(f"⚠️ Could not read {path.name}, skipping.")
            continue

        is_valid = detector.detect(frame)

        if not is_valid:
            false_positives += 1
            print(f"❌ False positive: {path.name}")
            if not args.dry_run:
                shutil.move(str(path), str(false_positives_dir / path.name))
        else:
            true_positives += 1

            # Check if the file needs renaming
            if item["batch"] != new_batch_num:
                new_name = (
                    f"batch_{new_batch_num:03d}_frame_{item['frame']:07d}.{item['ext']}"
                )
                new_path = path.parent / new_name

                print(f"✅ Renaming: {path.name} -> {new_name}")

                if not args.dry_run:
                    path.rename(new_path)

            new_batch_num += 1

    print("\n--------------------------------------------------")
    print(
        f"Results: {true_positives} true positives, {false_positives} false positives."
    )
    if false_positives > 0 and not args.dry_run:
        print(f"False positives moved to: {false_positives_dir}")
    elif args.dry_run:
        print("Dry run complete. No files were moved or renamed.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
