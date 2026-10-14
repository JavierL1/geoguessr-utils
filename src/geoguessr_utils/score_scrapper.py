import argparse
import os
import sys
import cv2
import difflib
import pytesseract
import numpy as np
import json
import datetime

from dataclasses import dataclass


KNOWN_PLAYERS = ["BLINKY", "KAPTEN"]


@dataclass
class ScrapperArgs:
    video_path: str
    score_roi_a: tuple[int, int, int, int]
    score_roi_b: tuple[int, int, int, int]
    player_roi_a: tuple[int, int, int, int]
    player_roi_b: tuple[int, int, int, int]
    timer_roi: tuple[int, int, int, int]
    start_sec: float
    end_sec: float | None
    interval_sec: float
    debug: True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape round scores from a Geoguessr match recording using OpenCV and Tesseract OCR."
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--score-roi-a",
        "-sra",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Region of interest bounding box coordinates for score a: X Y Width Height (e.g., 1200 850 150 50).",
    )
    parser.add_argument(
        "--score-roi-b",
        "-srb",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Region of interest bounding box coordinates for score b: X Y Width Height (e.g., 1200 850 150 50).",
    )
    parser.add_argument(
        "--player-roi-a",
        "-pra",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Region of interest bounding box coordinates for player name a: X Y Width Height (e.g., 1200 850 150 50).",
    )
    parser.add_argument(
        "--player-roi-b",
        "-prb",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Region of interest bounding box coordinates for player name b: X Y Width Height (e.g., 1200 850 150 50).",
    )
    parser.add_argument(
        "--timer-roi",
        "-tr",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Region of interest bounding box coordinates for the round timer: X Y Width Height (e.g., 1200 850 150 50).",
    )
    parser.add_argument(
        "--start-time",
        "-s",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        metavar=("H", "m", "s"),
        help="Hour, minute, second mark to start extracting scores (e.g., 1 30 20 -> 01:30:20).",
    )
    parser.add_argument(
        "--end-time",
        "-e",
        type=int,
        nargs=3,
        metavar=("H", "m", "s"),
        help="Hour, minute, second mark to stop extracting scores (e.g., 1 30 20 -> 01:30:20).",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Sample interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--debug",
        "-d",
        type=bool,
        default=False,
        help="Enables debug logging.",
    )
    return parser.parse_args()


def parse_time_in_seconds(hours: int, minutes: int, seconds: int) -> int | None:
    if (
        not isinstance(hours, int)
        or not isinstance(minutes, int)
        or not isinstance(seconds, int)
    ):
        return None

    return hours * 60 * 60 + minutes * 60 + seconds


def preprocess_score(roi_frame):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # 2. Upscale (3x) using cubic interpolation to smooth rounded font edges
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 3. Apply Otsu's Binarization + Invert (THRESH_BINARY_INV)
    # This turns white numbers into crisp black digits on a white canvas
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Optional: Light morphological blur to remove small pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned


def detect_score_text(frame, roi):
    x, y, w, h = roi
    # Crop Region of Interest
    roi_frame = frame[y : y + h, x : x + w]

    # Tesseract Configuration parameters:
    # --oem 3 : Use Default / Neural Nets LSTM engine
    # --psm 7 : Treat image as a single text line
    # -c tessedit_char_whitelist=0123456789 : Restrict characters strictly to digits
    custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

    # Execution
    processed_roi = preprocess_score(roi_frame)
    return pytesseract.image_to_string(processed_roi, config=custom_config).strip()


def preprocess_player_name(roi_frame):
    # 1. Upscale BEFORE color conversion (4x makes thin font strokes thick enough for OCR)
    scaled = cv2.resize(roi_frame, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to HSV
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)

    # 3. Isolate Pure White / Bright Pixels
    # Low saturation (0-40) and high brightness (200-255) isolates white text
    # regardless of background color/gradient
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 4. Invert so letters are BLACK and background is WHITE
    inverted = cv2.bitwise_not(mask)

    # Thin out black text strokes to separate merged letters like 'LI' -> 'U'
    kernel = np.ones((2, 1), np.uint8)  # Vertical kernel thins horizontal stroke connections
    padded_mask = cv2.erode(inverted, kernel, iterations=1)

    # 5. Add a 10px solid white border around the crop.
    # Tesseract FAILS if black characters touch the absolute edge of the image array!
    padded = cv2.copyMakeBorder(
        padded_mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    return padded


def clean_player_name(ocr_output: str) -> str:
    cleaned = ocr_output.upper().strip()
    
    # Find the closest match from known players (cutoff = 0.5 allows 1-2 character errors)
    matches = difflib.get_close_matches(cleaned, KNOWN_PLAYERS, n=1, cutoff=0.5)
    
    if matches:
        return matches[0]  # Transforms "BUNKY" -> "BLINKY"
    return cleaned  # Fallback if no match found


def detect_player_text(frame, roi):
    x, y, w, h = roi
    # Crop Region of Interest
    roi_frame = frame[y : y + h, x : x + w]

    # OEM 3 (Default LSTM), PSM 7 (Single text line)
    # Whitelist: Uppercase A-Z, lowercase a-z, numbers 0-9, dash, underscore, space
    custom_config = (
        r'--oem 3 --psm 8 '
        r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    # Execution
    processed_roi = preprocess_player_name(roi_frame)
    ocr_output = pytesseract.image_to_string(processed_roi, config=custom_config).strip()
    return clean_player_name(ocr_output)


def preprocess_timer(roi_frame):
    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # 2. Define color range for the bright yellow timer text
    # Yellow Hue is roughly 15 to 35 in OpenCV's HSV scale (0-180)
    lower_yellow = np.array([15, 120, 150])
    upper_yellow = np.array([35, 255, 255])

    # 3. Threshold to get only yellow pixels (makes yellow white, everything else black)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 4. Upscale (2x-3x) to smooth out the blocky font edges
    resized = cv2.resize(yellow_mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 5. Invert to get black text on a white background (Tesseract preference)
    inverted = cv2.bitwise_not(resized)

    # 6. Apply morphological close to fill small inner gaps in stylized numbers
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    return cleaned


def detect_timer_text(frame, roi):
    x, y, w, h = roi
    roi_frame = frame[y : y + h, x : x + w]

    custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

    processed_roi = preprocess_timer(roi_frame)
    return pytesseract.image_to_string(processed_roi, config=custom_config).strip()


def extract_scores(args: ScrapperArgs):
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video_path}'.", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0

    # Bounds check for start / end parameters
    if args.start_sec >= video_duration:
        print(
            f"Error: Start time ({args.start_sec}s) exceeds total video length ({video_duration:.2f}s).",
            file=sys.stderr,
        )
        cap.release()
        sys.exit(1)

    # Convert seek step into frames based on requested sample interval
    step_frames = int(max(1, fps * args.interval_sec))

    # Fast Seek directly to start timestamp in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    scores = []
    print(
        f"Processing '{args.video_path}' from {args.start_sec:.1f}s to "
        f"{f'{args.end_sec:.1f}s' if args.end_sec else 'end'} (interval: {args.interval_sec}s)..."
    )

    while cap.isOpened():
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = current_msec / 1000.0

        # Stop if end time threshold is reached
        if args.end_sec is not None and current_sec > args.end_sec:
            break

        ret, frame = cap.read()
        if not ret:
            break

        score_text_a = detect_score_text(frame, args.score_roi_a)
        score_text_b = detect_score_text(frame, args.score_roi_b)
        if args.debug:
            print(f"Raw Score A: {score_text_a}")
            print(f"Raw Score B: {score_text_b}")

        if score_text_a.isdigit() and score_text_b.isdigit():
            score_a = int(score_text_a)
            score_b = int(score_text_b)
            player_a = detect_player_text(frame, args.player_roi_a)
            player_b = detect_player_text(frame, args.player_roi_b)
            players = (player_a, player_b)
            timer_text = detect_timer_text(frame, args.timer_roi)

            scores.append({
                "score_a": score_a,
                "score_b": score_b,
                "player_a": player_a,
                "player_b": player_b,
                "timer": timer_text,
            })
            print(f"[{current_sec:06.1f}s] Detected Score: {(score_a, score_b)}, Players: {players}, Second: {current_sec}, Timer: {timer_text}")

        # 4. Efficient Skip: Jump forward by skipping intermediate frames
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        next_frame = (
            current_frame + step_frames - 1
        )  # -1 because cap.read() already advanced 1 frame

        if next_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    cap.release()
    return scores


def main():
    args = parse_args()
    print(args)
    start_seconds = parse_time_in_seconds(*args.start_time)
    end_seconds = parse_time_in_seconds(*args.end_time)

    scraper_args = ScrapperArgs(
        video_path=args.video,
        score_roi_a=tuple(args.score_roi_a),
        score_roi_b=tuple(args.score_roi_b),
        start_sec=start_seconds,
        end_sec=end_seconds,
        interval_sec=args.interval,
        timer_roi=tuple(args.timer_roi),
        player_roi_a=tuple(args.player_roi_a),
        player_roi_b=tuple(args.player_roi_b),
        debug=args.debug,
    )

    scores = extract_scores(scraper_args)
    current_dt = datetime.datetime.now()

    with open(f"scores_{scraper_args.video_path}_{current_dt}.json", mode="w", encoding="utf-8") as write_file:
        json.dump(scores, write_file)

    print("\n--- Final Extracted Scores ---")
    print(scores)


if __name__ == "__main__":
    main()
