# /// script
# dependencies = [
#     "opencv-python",
#     "pytesseract",
#     "numpy",
# ]
# ///

import argparse
import difflib
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

# --- SPECIALIZED PREPROCESSING PIPELINES ---
import cv2
import numpy as np
import pytesseract

DEBUG = False
# 126 nulls
# 106 nulls


def process_distance_green(crop, debug_name=None):
    """
    Uses the Red channel and morphological erosion to forcefully thin
    bloated text and prevent characters from merging.
    """
    # 1. Load the image
    img = crop

    # 2. Upscale the image (6x is better for tiny text)
    img_scaled = cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)

    # 3. Convert to grayscale
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

    # 4. Binarization (Better approach for white text on green)
    # Instead of Otsu, we use a fixed threshold.
    # White text is very bright (> 180), green background is darker.
    # THRESH_BINARY_INV makes the text black (0) and background white (255)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # 5. Morphological Dilation (Thickens the text slightly to help OCR)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 6. Add a white border (padding)
    # Tesseract struggles when text touches the edge of the image.
    padded = cv2.copyMakeBorder(dilated, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_green_padded.jpg", padded)

    # 7. Configure Tesseract
    # --psm 6: Assume a single uniform block of text (handles spaces better than psm 7)
    # We removed the whitelist. Sometimes a whitelist prevents Tesseract from
    # correctly recognizing the space or the separate "m".
    custom_config = r"--psm 7 -c tessedit_char_whitelist=0123456789kmKM,. "

    # 8. Run Tesseract
    return pytesseract.image_to_string(padded, config=custom_config)


def process_distance_purple(crop, debug_name=None):
    """Minimum channel logic tailored for the dim purple pill."""
    resized = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Min channel trick works beautifully for the darker purple
    min_channel = np.min(resized, axis=2)
    _, thresh = cv2.threshold(min_channel, 130, 255, cv2.THRESH_BINARY)
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_purple_thresh.jpg", thresh)

    # 4. EROSION: Define a 3x3 pixel kernel and shave off the outer edges of the white text.
    # This physically separates letters that are bleeding together.
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=1)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_purple_thinned.jpg", thinned)

    inverted = cv2.bitwise_not(thinned)

    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    config = "--psm 7 -c tessedit_char_whitelist=0123456789kmKM,. "

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_purple.jpg", padded)

    return pytesseract.image_to_string(padded, config=config).strip()


def process_distance_orange(crop, debug_name=None):
    """Pipeline tailored for the orange/gold 5k tie pill background."""
    resized = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Blue channel provides a strong contrast for orange backgrounds
    blue_channel = resized[:, :, 0]
    _, thresh = cv2.threshold(blue_channel, 180, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)

    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    config = "--psm 7 -c tessedit_char_whitelist=0123456789kmKM,. "

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_orange.jpg", padded)

    return pytesseract.image_to_string(padded, config=config).strip()


def detect_background_color(crop):
    """
    Analyzes the Hue channel to determine if the widget is green, orange, or purple.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    median_hue = np.median(hsv[:, :, 0])

    # Orange/yellow fall roughly between 10 and 30 in OpenCV's 0-179 Hue space
    if 10 <= median_hue <= 30:
        return "orange"
    elif median_hue < 100:
        return "green"
    else:
        return "purple"


def process_distance(crop, debug_name=None):
    """
    Routes the crop to the correct OCR pipeline based on its background color.
    """
    bg_color = detect_background_color(crop)

    if bg_color == "green":
        raw_text = process_distance_green(crop, debug_name=debug_name)
    elif bg_color == "orange":
        raw_text = process_distance_orange(crop, debug_name=debug_name)
    else:
        raw_text = process_distance_purple(crop, debug_name=debug_name)

    if raw_text:
        return parse_distance(raw_text)

    return None


def detect_digit_geometrically(crop, debug_name=None):
    """
    For massive, thick standalone digits.
    Uses basic grayscale thresholding and PSM 10 (single character).
    """
    # 1. Upscale (2x is fine, 3x is better for thick fonts)
    resized = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Thresholding
    # Assuming the original is WHITE text on a DARK background:
    # THRESH_BINARY makes white pixels 255, dark pixels 0.
    # This results in WHITE text on a BLACK background.
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}.jpg", thresh)

    # 3. Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return "Empty"

    # 4. Find the largest contour (the digit itself)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if h == 0 or w == 0:
        return "Empty"

    # 5. Check for holes -> It's a '0'
    has_hole = False
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:  # Has a parent contour, meaning it's a hole
            if cv2.contourArea(contour) > 20:
                has_hole = True
                break

    if has_hole:
        return "0"

    # 6. Differentiate between '1' and '2' using Bottom Width Ratio
    # Look at the bottom 30% of the bounding box
    bottom_start_y = y + int(h * 0.7)
    bottom_slice = thresh[bottom_start_y : y + h, x : x + w]

    # Find the columns (x-coordinates) that contain white pixels in this bottom slice
    cols_with_white = np.where(bottom_slice == 255)[1]

    if len(cols_with_white) == 0:
        return "1"

    # Calculate how wide the bottom is compared to the total width
    bottom_width = cols_with_white[-1] - cols_with_white[0]
    bottom_ratio = bottom_width / float(w)

    # A '1' has a narrower bottom (< 0.60 in your specific font)
    # A '2' has a wide flat base (> 0.80 in your specific font)
    if bottom_ratio < 0.60:
        return "1"
    else:
        return "2"


def process_overall_score(crop, debug_name=None):
    """
    For massive, thick standalone digits.
    Uses basic grayscale thresholding and PSM 10 (single character).
    """
    # 1. Upscale (2x is fine, 3x is better for thick fonts)
    resized = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Thresholding
    # Assuming the original is WHITE text on a DARK background:
    # THRESH_BINARY makes white pixels 255, dark pixels 0.
    # This results in WHITE text on a BLACK background.
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 4. INVERT (CRITICAL STEP)
    # Tesseract needs BLACK text on a WHITE background.
    # We removed the erosion step entirely. It was damaging the '0'.
    inverted = cv2.bitwise_not(thresh)

    # 5. Add padding
    padded = cv2.copyMakeBorder(
        inverted, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255
    )

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}.jpg", padded)

    # 6. Run Tesseract
    # --psm 10: Treat the image as a single character.
    # Whitelist: Only allow 0, 1, 2, and O (in case it misreads 0 as O).
    config = "--psm 10 -c tessedit_char_whitelist=012O"

    text = pytesseract.image_to_string(padded, config=config).strip()

    # Clean up common OCR mistakes
    if text.upper() == "O":
        return "0"
    return text


def process_standard(crop, debug_name=None):
    """
    For Usernames, Round Scores, and Health Points.
    Standard upscaling and high-contrast thresholding.
    """
    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=2)

    inverted = cv2.bitwise_not(thinned)

    # Add clean padding around the tight crop
    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}.jpg", padded)

    text = (
        pytesseract.image_to_string(
            padded, config="--psm 7 -c tessedit_char_whitelist=oO0123456789"
        )
        .strip()
        .lower()
        .replace("o", "0")
    )

    return text if text else "0"


def process_username(crop, debug_name=None):
    """
    For Usernames, Round Scores, and Health Points.
    Standard upscaling and high-contrast thresholding.
    """
    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=1)

    inverted = cv2.bitwise_not(thinned)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}.jpg", inverted)

    raw = pytesseract.image_to_string(inverted, config="--psm 7").strip()

    return raw


def process_health(crop, debug_name=None):
    """
    Extracts health points by converting the green health bar fill to black,
    leaving only the bright white text to be thresholded and read.
    """
    if crop is None or crop.size == 0:
        return None

    # 1. Resize for high-resolution OCR
    resized = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to HSV to isolate the green health bar fill
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 1. Define the Yellow Mask (H: 20 to 34)
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([34, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2. Define the Green Mask (H: 35 to 85)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 3. Combine them
    color_mask = cv2.bitwise_or(mask_yellow, mask_green)

    # 3. Transform all green pixels into black
    processed_img = resized.copy()
    processed_img[color_mask > 0] = [0, 0, 0]  # BGR black

    # 4. Convert to grayscale and threshold for the bright white text
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 4. EROSION: Define a 3x3 pixel kernel and shave off the outer edges of the white text.
    # This physically separates letters that are bleeding together.
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=3)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_thinned.jpg", thinned)

    # 5. Add mandatory padding for Tesseract
    padded = cv2.copyMakeBorder(thinned, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_clean.jpg", padded)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"

    raw = pytesseract.image_to_string(padded, config=config).strip()

    return raw


# --- DATA EXTRACTION HELPERS ---


def extract_numbers(text, mode="float"):
    """Strips letters/symbols to return just a float."""
    if text is None:
        return

    numbers = re.sub(r"[^\d\.]", "", text)

    if mode == "float":
        return float(numbers) if numbers else None

    if mode == "int":
        return int(numbers) if numbers else None


def match_username(ocr_text, known_players):
    """Fuzzy matches OCR output to the known player list."""
    if not ocr_text:
        return None
    matches = difflib.get_close_matches(
        ocr_text.upper(), known_players, n=1, cutoff=0.4
    )
    return matches[0] if matches else None


def parse_distance(text):
    cleaned = text.strip().lower()
    unit = None

    if "km" in cleaned:
        unit = "km"

    elif "m" in cleaned:
        unit = "m"

    amount = extract_numbers(cleaned)

    if unit is not None and amount is None:
        return {"amount": "inf", "unit": unit}

    return {"amount": amount, "unit": unit}


# --- MAIN LOGIC ---


def frame_index_to_time(index, fps):
    total_seconds = index / fps
    hours = int(total_seconds // 3600)
    minute_seconds = total_seconds - (hours * 3600)
    minutes = int(minute_seconds // 60)
    seconds = int(minute_seconds - (minutes * 60))
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def process_directory(input_dir, output_json, config):
    rois = config.get("ROIS", {})
    known_players = config.get("KNOWN_PLAYERS", [])
    fps = config.get("FPS", 60)

    batches = defaultdict(list)
    filepaths = glob.glob(os.path.join(input_dir, "batch_*_frame_*.jpg"))
    # filepaths = filter(lambda f: "batch_025_frame_0304885.jpg" in f, filepaths)

    for path in filepaths:
        filename = os.path.basename(path)
        match = re.search(r"batch_(\d+)_frame_(\d+)\.jpg", filename)
        if match:
            batches[match.group(1)].append((int(match.group(2)), path))

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

        # --- Metrics & ETA Calculation ---
        elapsed_real = time.time() - start_time_real
        avg_time_per_batch = elapsed_real / idx if idx > 0 else 0
        remaining_batches = total_batches - idx
        eta_sec = remaining_batches * avg_time_per_batch

        elapsed_mins, elapsed_secs = divmod(int(elapsed_real), 60)
        eta_mins, eta_secs = divmod(int(eta_sec), 60)
        pct = (idx / total_batches) * 100
        speed = idx / elapsed_real if elapsed_real > 0 else 0

        sys.stdout.write(
            f"\rProgress: [{idx}/{total_batches}] ({pct:5.1f}%) | "
            f"Speed: {speed:4.2f} batch/s | "
            f"Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d} | "
            f"ETA: {eta_mins:02d}:{eta_secs:02d} "
        )
        sys.stdout.flush()

        frame_data = {
            "batch": batch_num,
            "source_image": os.path.basename(last_frame_path),
            "left_player": {},
            "right_player": {},
            "time_code": frame_index_to_time(frame_index, fps),
        }

        def crop_roi(roi_key):
            x, y, w, h = rois[roi_key]
            return frame[y : y + h, x : x + w]

        # 1. Overall Score
        frame_data["left_player"]["overall_score"] = extract_numbers(
            detect_digit_geometrically(
                crop_roi("overall_score_left"), debug_name="overall_left"
            ),
            mode="int",
        )
        frame_data["right_player"]["overall_score"] = extract_numbers(
            detect_digit_geometrically(
                crop_roi("overall_score_right"), debug_name="overall_right"
            ),
            mode="int",
        )

        # frame_data["left_player"]["overall_score"] = extract_numbers(
        #     process_overall_score(
        #         crop_roi("overall_score_left"), debug_name="overall_left"
        #     ),
        #     mode="int",
        # )
        # frame_data["right_player"]["overall_score"] = extract_numbers(
        #     process_overall_score(
        #         crop_roi("overall_score_right"), debug_name="overall_right"
        #     ),
        #     mode="int",
        # )

        # 2. Usernames
        frame_data["left_player"]["username"] = match_username(
            process_username(crop_roi("username_left"), debug_name="user_left"),
            known_players,
        )
        frame_data["right_player"]["username"] = match_username(
            process_username(crop_roi("username_right"), debug_name="user_right"),
            known_players,
        )

        # 3. Round Scores
        frame_data["left_player"]["round_score"] = extract_numbers(
            process_standard(crop_roi("round_score_left"), debug_name="round_left"),
            mode="int",
        )
        frame_data["right_player"]["round_score"] = extract_numbers(
            process_standard(crop_roi("round_score_right"), debug_name="round_right"),
            mode="int",
        )

        # 4. Distances
        frame_data["left_player"]["distance"] = process_distance(
            crop_roi("distance_left"), debug_name="dist_left"
        )
        frame_data["right_player"]["distance"] = process_distance(
            crop_roi("distance_right"), debug_name="dist_right"
        )

        # 5. Health Points
        frame_data["left_player"]["health"] = extract_numbers(
            process_health(crop_roi("health_left"), debug_name="health_left"),
            mode="int",
        )
        frame_data["right_player"]["health"] = extract_numbers(
            process_health(crop_roi("health_right"), debug_name="health_right"),
            mode="int",
        )

        results.append(frame_data)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    total_elapsed = time.time() - start_time_real
    print(
        f"\n\nExtraction complete! Saved {len(results)} batches to {output_json} in {total_elapsed:.1f} seconds."
    )


if __name__ == "__main__":
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
        type=bool,
        default=False,
        help="Enables debug logging and image storage",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        exit(1)

    with open(args.config, "r") as f:
        config_data = json.load(f)

    process_directory(args.input, args.output, config_data)
