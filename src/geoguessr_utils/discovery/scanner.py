# src/geoguessr_utils/discovery/scanner.py
"""
Score-screen discovery: the algorithm, not the CLI.

Walks through a video, detects score-screen segments, and saves the
last frame of each contiguous batch to disk.

Detection is delegated to a `ScreenDetector` supplied by the caller.
The scanner does not know which heuristic is in use — it just asks
"is this a score screen?" once per sampled frame. This keeps the
scanner independent of layout-specific detection logic.

Kept separate from `discover_screens.py` (the CLI) so this can be
called from tests or other scripts without touching argparse or
sys.exit.
"""

import sys
import time
from pathlib import Path

import cv2

from geoguessr_utils.discovery.collector import ScoreScreenCollector
from geoguessr_utils.layouts import Layout
from geoguessr_utils.utils.progress import print_eta_progress
from geoguessr_utils.vision.detectors.base import ScreenDetector


def discover(
    video_path: Path,
    output_dir: Path,
    layout: Layout,
    detector: ScreenDetector,
    start_seconds: int,
    end_seconds: int | None,
) -> None:
    """Walk a video and save the last frame of each score-screen batch.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where batch frames will be written.
        layout: Layout providing feature ROIs (used downstream by the
            extraction pipeline — the scanner does not touch them
            directly, but accepts the layout to keep the pairing
            explicit at the call site).
        detector: Screen detector paired with `layout`. Controls which
            frames count as "score screen" frames.
        start_seconds: First second of the video to scan.
        end_seconds: Last second to scan, or None for end-of-video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(
        str(video_path),
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
    )
    if not cap.isOpened():
        print(f"Failed to open '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_seconds is None:
        end_seconds = int(total_frames / fps) if fps > 0 else 0

    start_frame_idx = int(start_seconds * fps)
    end_frame_idx = int(end_seconds * fps)

    if start_frame_idx >= total_frames:
        print("Error: Start time exceeds video length.", file=sys.stderr)
        sys.exit(1)

    # Walk step sizes: skip 1s while searching, sample every 0.25s inside a batch.
    search_step = max(1, int(fps * 1))
    collect_step = max(1, int(fps * 0.25))

    collector = ScoreScreenCollector(output_dir, fps)

    current_frame_idx = start_frame_idx
    total_frames_to_process = end_frame_idx - start_frame_idx
    loop_iterations = 0
    start_time_real = time.time()

    print(f"Starting discovery on '{video_path}'...")
    print(f"Processing window: {start_seconds}s to {end_seconds}s")

    while current_frame_idx < total_frames and current_frame_idx <= end_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()

        if not ret:
            collector.flush()
            break

        if detector.detect(frame):
            collector.on_detected(frame, current_frame_idx)
            current_frame_idx += collect_step
        else:
            collector.on_not_detected()
            current_frame_idx += search_step

        loop_iterations += 1
        if loop_iterations % 5 == 0:
            elapsed = time.time() - start_time_real
            progress_frames = current_frame_idx - start_frame_idx
            speed = progress_frames / fps / elapsed if elapsed > 0 else 0
            print_eta_progress(
                current=progress_frames,
                total=total_frames_to_process,
                start_time=start_time_real,
                suffix=f"Batches: {collector.batch_count} | Speed: {speed:.1f}x",
            )

    cap.release()
    print(f"\n\nDiscovery complete. Total batches: {collector.batch_count}")
