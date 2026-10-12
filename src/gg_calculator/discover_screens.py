# /// script
# dependencies = [
#     "opencv-python",
#     "numpy",
# ]
# ///

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def parse_time_in_seconds(hours: int, minutes: int, seconds: int) -> int | None:
    if (
        not isinstance(hours, int)
        or not isinstance(minutes, int)
        or not isinstance(seconds, int)
    ):
        return None
    return hours * 60 * 60 + minutes * 60 + seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast discovery of Geoguessr score screens."
    )
    parser.add_argument("-v", "--video", required=True, help="Path to video file")
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for saved frames"
    )
    parser.add_argument(
        "-c", "--config", default="config.json", help="Path to config.json file"
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


def calc_crop_from_roi(frame, roi):
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def calc_ratios_from_crop(crop, roi):
    _x, _y, w, h = roi
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_ratio = cv2.countNonZero(thresh_white) / (w * h)
    dark_mask = cv2.inRange(gray, 0, 100)
    dark_ratio = cv2.countNonZero(dark_mask) / (w * h)

    return (white_ratio, dark_ratio)


def is_score_screen(
    frame, roi_left, roi_right, roi_center, roi_health_left, roi_health_right
) -> bool:
    left_crop = calc_crop_from_roi(frame, roi_left)
    right_crop = calc_crop_from_roi(frame, roi_right)
    center_crop = calc_crop_from_roi(frame, roi_center)
    health_left_crop = calc_crop_from_roi(frame, roi_health_left)
    health_right_crop = calc_crop_from_roi(frame, roi_health_right)

    xc, yc, wc, hc = roi_center

    # Bounds Safety Check
    empty_crops = [
        c.size == 0
        for c in [
            left_crop,
            right_crop,
            center_crop,
            health_left_crop,
            health_left_crop,
        ]
    ]

    if any(empty_crops):
        frame_h, frame_w = frame.shape[:2]
        raise ValueError(
            f"\nROI is out of bounds! Video resolution is {frame_w}x{frame_h}.\n"
            f"Left ROI: {roi_left} | Right ROI: {roi_right} | Center ROI: {roi_center}\n"
            f"Left health ROI: {roi_health_left} | Right health ROI: {roi_health_right}\n"
            f"Please adjust your config.json."
        )

    (left_white_ratio, left_dark_ratio) = calc_ratios_from_crop(left_crop, roi_left)
    (right_white_ratio, right_dark_ratio) = calc_ratios_from_crop(right_crop, roi_right)
    (health_left_white_ratio, _health_left_dark_ratio) = calc_ratios_from_crop(
        health_left_crop, roi_health_left
    )
    (health_right_white_ratio, _health_right_dark_ratio) = calc_ratios_from_crop(
        health_right_crop, roi_health_right
    )

    # 3. Central Pill Check
    hsv_center = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    lower_purple = np.array([110, 50, 50])
    upper_purple = np.array([160, 255, 255])
    # Added Orange/Yellow range for 5k tie pills
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([30, 255, 255])

    green_mask = cv2.inRange(hsv_center, lower_green, upper_green)
    purple_mask = cv2.inRange(hsv_center, lower_purple, upper_purple)
    orange_mask = cv2.inRange(hsv_center, lower_orange, upper_orange)

    green_ratio = cv2.countNonZero(green_mask) / (wc * hc)
    purple_ratio = cv2.countNonZero(purple_mask) / (wc * hc)
    orange_ratio = cv2.countNonZero(orange_mask) / (wc * hc)

    # --- Final Evaluation ---
    has_white_text = left_white_ratio > 0.10 and right_white_ratio > 0.10
    has_dark_bg = left_dark_ratio > 0.20 and right_dark_ratio > 0.20
    has_visible_health = (
        health_left_white_ratio > 0.10 and health_right_white_ratio > 0.10
    )

    # Accept either the standard green+purple combination OR the orange tie pill
    has_standard_pill = green_ratio > 0.02 and purple_ratio > 0.02
    has_tie_pill = orange_ratio > 0.05

    has_pill_colors = has_standard_pill or has_tie_pill

    return has_visible_health and has_white_text and has_dark_bg and has_pill_colors


def main():
    args = parse_args()

    # Load Configuration
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
        rois = config.get("ROIS", {})
        roi_left = rois["discover_left"]
        roi_right = rois["discover_right"]
        roi_center = rois["discover_center"]
        roi_health_left = rois["health_left"]
        roi_health_right = rois["health_right"]
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cap = cv2.VideoCapture(
        args.video,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
    )
    if not cap.isOpened():
        print(f"Failed to open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Parse boundaries
    start_seconds = parse_time_in_seconds(*args.start_time) or 0
    if args.end_time:
        end_seconds = parse_time_in_seconds(*args.end_time)
    else:
        end_seconds = total_frames / fps if fps > 0 else 0

    start_frame_idx = int(start_seconds * fps)
    end_frame_idx = int(end_seconds * fps) if end_seconds else total_frames

    if start_frame_idx >= total_frames:
        print("Error: Start time exceeds video length.", file=sys.stderr)
        sys.exit(1)

    # State Machine Variables
    search_step_frames = int(max(1, fps * 1))
    collect_step_frames = int(max(1, fps * 0.25))

    is_collecting = False
    batch_count = 0
    last_valid_frame = None
    last_valid_idx = None

    current_frame_idx = start_frame_idx
    total_frames_to_process = end_frame_idx - start_frame_idx

    # Performance tracking
    loop_iterations = 0
    start_time_real = time.time()

    print(f"Starting discovery on '{args.video}'...")
    print(
        f"Processing window: {start_seconds}s to {end_seconds}s (Total frames to span: {total_frames_to_process})"
    )

    while current_frame_idx < total_frames and current_frame_idx <= end_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()

        if not ret:
            # Save the final batch frame if the video ends while collecting
            if is_collecting and last_valid_frame is not None:
                out_path = os.path.join(
                    args.output,
                    f"batch_{batch_count:03d}_frame_{last_valid_idx:07d}.jpg",
                )
                cv2.imwrite(out_path, last_valid_frame)
                print(
                    f"\n[ {(last_valid_idx / fps):.2f}s ] Saved final frame for Batch {batch_count:03d}"
                )
            break

        detected = is_score_screen(
            frame, roi_left, roi_right, roi_center, roi_health_left, roi_health_right
        )

        if detected:
            if not is_collecting:
                batch_count += 1
                print(
                    f"\n[ {(current_frame_idx / fps):.2f}s ] Score screen detected! Started Batch {batch_count:03d}"
                )
                is_collecting = True

            # Continuously overwrite with the latest valid frame of this batch
            last_valid_frame = frame.copy()
            last_valid_idx = current_frame_idx

            current_frame_idx += collect_step_frames
        else:
            if is_collecting:
                # Score screen just ended, flush the last recorded frame to disk
                if last_valid_frame is not None:
                    out_path = os.path.join(
                        args.output,
                        f"batch_{batch_count:03d}_frame_{last_valid_idx:07d}.jpg",
                    )
                    cv2.imwrite(out_path, last_valid_frame)
                    print(
                        f"\n[ {(last_valid_idx / fps):.2f}s ] Score screen ended. Saved final frame for Batch {batch_count:03d}"
                    )

                is_collecting = False
                last_valid_frame = None
                last_valid_idx = None

            current_frame_idx += search_step_frames

        # Progress Update
        loop_iterations += 1
        if loop_iterations % 5 == 0:
            elapsed_real = time.time() - start_time_real
            progress_frames = current_frame_idx - start_frame_idx

            if elapsed_real > 0 and progress_frames > 0:
                pct = min(100.0, (progress_frames / total_frames_to_process) * 100)
                read_fps = loop_iterations / elapsed_real
                video_speed = (progress_frames / fps) / elapsed_real

                remaining_frames = total_frames_to_process - progress_frames
                eta_sec = remaining_frames / (progress_frames / elapsed_real)
                mins, secs = divmod(int(eta_sec), 60)

                sys.stdout.write(
                    f"\rProgress: {pct:5.1f}% | Batches: {batch_count} | Read: {read_fps:4.1f} fps | Speed: {video_speed:4.1f}x | ETA: {mins:02d}:{secs:02d} "
                )
                sys.stdout.flush()

    cap.release()
    print("\n\nDiscovery complete.")


if __name__ == "__main__":
    main()
