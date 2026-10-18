"""
Score-screen extraction pipeline.

Walks a directory of batch frames, extracts player data from each, and
writes the results to a JSON file. Extraction of individual players is
delegated to `extract_player`; progress formatting is delegated to
`utils.progress`.

Kept separate from `extract_score.py` (the CLI) so the algorithm can be
called from tests or other scripts without touching argparse or sys.exit.
"""

import glob
import json
import os
import re
import time

from geoguessr_utils.config import AppConfig
from geoguessr_utils.extraction.player import extract_player
from geoguessr_utils.frame_item import FrameItem
from geoguessr_utils.layouts import Layout
from geoguessr_utils.utils import video
from geoguessr_utils.utils.progress import print_eta_progress
from geoguessr_utils.vision import image


def _list_frame_items(input_dir: str) -> list[FrameItem]:
    """Return the frames in `input_dir` sorted by (batch, frame index)."""
    pattern = re.compile(r"batch_(\d+)_frame_(\d+)\.jpg")
    frame_items: list[FrameItem] = []

    for filepath in glob.glob(os.path.join(input_dir, "batch_*_frame_*.jpg")):
        match = pattern.search(os.path.basename(filepath))
        if match:
            frame_items.append(
                {
                    "screen_index": match.group(1),
                    "frame_index": int(match.group(2)),
                    "filepath": filepath,
                }
            )

    frame_items.sort(key=lambda item: (int(item["screen_index"]), item["frame_index"]))
    return frame_items


def extract_directory(
    input_dir: str,
    output_json: str,
    layout: Layout,
    config: AppConfig,
    debug: bool = False,
) -> None:
    """Run the extraction pipeline over a directory of batch frames.

    Args:
        input_dir: Directory containing `batch_NNN_frame_MMMMMMM.jpg` files.
        output_json: Path where the results JSON will be written.
        layout: Concrete layout to extract features from.
        config: Runtime settings (FPS is used to convert frames to time codes).
        debug: If True, layout will emit debug crops for each feature.
    """
    layout.debug = debug
    frame_items = _list_frame_items(input_dir)
    total = len(frame_items)

    if total == 0:
        print(f"No frames found in '{input_dir}' matching pattern.")
        return

    results = []
    start_time_real = time.time()

    print(f"Starting extraction across {total} batches...")

    for idx, frame_item in enumerate(frame_items, start=1):
        frame = image.read(frame_item["filepath"])
        if frame is None:
            continue

        print_eta_progress(
            current=idx,
            total=total,
            start_time=start_time_real,
            suffix=f"Speed: {idx / (time.time() - start_time_real):.2f} batch/s",
        )

        results.append(
            {
                "screen_index": frame_item["screen_index"],
                "frame_index": frame_item["frame_index"],
                "source_image": os.path.basename(frame_item["filepath"]),
                "left_player": extract_player(layout, frame, "left"),
                "right_player": extract_player(layout, frame, "right"),
                "time_code": video.frame_index_to_time(
                    frame_item["frame_index"], config.FPS
                ),
            }
        )

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    total_elapsed = time.time() - start_time_real
    print(
        f"\n\nExtraction complete! Saved {len(results)} batches to {output_json} "
        f"in {total_elapsed:.1f} seconds."
    )
