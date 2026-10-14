import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

from geoguessr_utils.config import AppConfig
from geoguessr_utils.parsers import parsing
from geoguessr_utils.utils import video
from geoguessr_utils.vision import ocr, preprocess


def extract_player(
    frame, rois, side: str, known_players: list[str], debug: bool = False
) -> dict:
    def crop(roi_key: str):
        roi = rois[roi_key]
        return frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]

    def dbg(field: str) -> str | None:
        return f"{side}_{field}" if debug else None

    return {
        "overall_score": parsing.extract_numbers(
            ocr.read_overall_score(
                crop(f"overall_score_{side}"), debug_name=dbg("overall")
            ),
            mode="int",
        ),
        "username": parsing.match_username(
            ocr.read_username(crop(f"username_{side}"), debug_name=dbg("username")),
            known_players,
        ),
        "round_score": parsing.extract_numbers(
            ocr.read_standard(crop(f"round_score_{side}"), debug_name=dbg("round")),
            mode="int",
        ),
        "distance": parsing.parse_distance(
            ocr.read_distance(crop(f"distance_{side}"), debug_name=dbg("distance"))
        ),
        "health": parsing.extract_numbers(
            ocr.read_health(crop(f"health_{side}"), debug_name=dbg("health")),
            mode="int",
        ),
    }


def _group_batches(input_dir: str) -> defaultdict[str, list[tuple[int, str]]]:
    """Return {batch_id: [(frame_index, path), ...]} sorted-ready."""
    batches: dict[str, list[tuple[int, str]]] = defaultdict(list)
    pattern = re.compile(r"batch_(\d+)_frame_(\d+)\.jpg")

    for path in glob.glob(os.path.join(input_dir, "batch_*_frame_*.jpg")):
        match = pattern.search(os.path.basename(path))
        if match:
            batches[match.group(1)].append((int(match.group(2)), path))

    return batches


def _print_progress(idx: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    avg_per_item = elapsed / idx if idx else 0
    remaining = total - idx
    eta = remaining * avg_per_item

    elapsed_mins, elapsed_secs = divmod(int(elapsed), 60)
    eta_mins, eta_secs = divmod(int(eta), 60)
    pct = (idx / total) * 100
    speed = idx / elapsed if elapsed > 0 else 0

    sys.stdout.write(
        f"\rProgress: [{idx}/{total}] ({pct:5.1f}%) | "
        f"Speed: {speed:4.2f} batch/s | "
        f"Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d} | "
        f"ETA: {eta_mins:02d}:{eta_secs:02d} "
    )
    sys.stdout.flush()


def process_directory(
    input_dir: str, output_json: str, config: AppConfig, debug: bool = False
) -> None:
    batches = _group_batches(input_dir)

    sorted_batch_keys = sorted(batches.keys())
    total_batches = len(sorted_batch_keys)

    if total_batches == 0:
        print(f"No batches found in '{input_dir}' matching pattern.")
        return

    results = []
    start_time_real = time.time()

    print(f"Starting extraction across {total_batches} batches...")

    for idx, batch_num in enumerate(sorted_batch_keys, start=1):
        batches[batch_num].sort(key=lambda x: x[0])
        last_frame_path = batches[batch_num][-1][1]
        frame_index = batches[batch_num][-1][0]

        frame = cv2.imread(last_frame_path)
        if frame is None:
            continue

        _print_progress(idx, total_batches, start_time_real)

        frame_data = {
            "batch": batch_num,
            "source_image": os.path.basename(last_frame_path),
            "left_player": extract_player(
                frame,
                config.ROIS,
                "left",
                config.KNOWN_PLAYERS,
                debug=debug,
            ),
            "right_player": extract_player(
                frame,
                config.ROIS,
                "right",
                config.KNOWN_PLAYERS,
                debug=debug,
            ),
            "time_code": video.frame_index_to_time(frame_index, config.FPS),
        }

        results.append(frame_data)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    total_elapsed = time.time() - start_time_real
    print(
        f"\n\nExtraction complete! Saved {len(results)} batches to {output_json} "
        f"in {total_elapsed:.1f} seconds."
    )


def main() -> None:
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
        "-o", "--output", default="tournament_data.json", help="Output JSON file path"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="Path to JSON config file containing ROIs",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enables debug image storage",
    )

    args = parser.parse_args()

    if args.debug:
        preprocess.DEBUG = True

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    config = AppConfig.load(Path(args.config))

    missing = config.missing_rois()
    if missing:
        print(f"Error: Missing ROIs in config: {missing}", file=sys.stderr)
        print("Run configure_rois.py first.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    process_directory(args.input, args.output, config, debug=args.debug)


if __name__ == "__main__":
    main()
