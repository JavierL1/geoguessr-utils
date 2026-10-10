# /// script
# dependencies = [
#     "opencv-python",
#     "pytesseract",
#     "numpy",
# ]
# ///

import argparse
import os
import cv2
import json
import glob
import re
import difflib
import pytesseract
import numpy as np
from collections import defaultdict

# --- SPECIALIZED PREPROCESSING PIPELINES ---

import cv2
import numpy as np
import pytesseract
import re


def process_distance_green(crop, debug_name=None):
    """
    Uses the Red channel and morphological erosion to forcefully thin 
    bloated text and prevent characters from merging.
    """
    # 1. Extract Red channel (darkens the green background)
    red_channel = crop[:, :, 2]
    
    # 2. Upscale linearly
    resized = cv2.resize(red_channel, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

    if debug_name: 
        cv2.imwrite(f"debug_{debug_name}_green_resized.jpg", resized)
    
    # 3. Apply a strict threshold to create white text on a black background
    _, thresh = cv2.threshold(resized, 200, 255, cv2.THRESH_BINARY)

    if debug_name: 
        cv2.imwrite(f"debug_{debug_name}_green_thresh.jpg", thresh)

    # 4. EROSION: Define a 3x3 pixel kernel and shave off the outer edges of the white text.
    # This physically separates letters that are bleeding together.
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=1)

    if debug_name: 
        cv2.imwrite(f"debug_{debug_name}_green_thinned.jpg", thinned)

    # 5. Invert to black text on a white background
    inverted = cv2.bitwise_not(thresh)

    if debug_name: 
        cv2.imwrite(f"debug_{debug_name}_green_inverted.jpg", inverted)

    # 6. Add mandatory padding
    padded = cv2.copyMakeBorder(inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    if debug_name: 
        cv2.imwrite(f"debug_{debug_name}_green_padded.jpg", padded)
    
    config = "--psm 7 -c tessedit_char_whitelist=0123456789kmKM "
    raw_text = pytesseract.image_to_string(padded, config=config).strip()
    
    return raw_text

def process_distance_purple(crop, debug_name=None):
    """Minimum channel logic tailored for the dim purple pill."""
    resized = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # Min channel trick works beautifully for the darker purple
    min_channel = np.min(resized, axis=2)
    _, thresh = cv2.threshold(min_channel, 120, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    
    padded = cv2.copyMakeBorder(inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789kmKM "

    if debug_name: cv2.imwrite(f"debug_{debug_name}_purple.jpg", padded)

    return pytesseract.image_to_string(padded, config=config).strip()

def detect_background_color(crop):
    """
    Analyzes the Hue channel to determine if the widget is green or purple.
    Uses the median to prevent the bright white text from skewing the result.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Extract the Hue channel (index 0) and find the median
    median_hue = np.median(hsv[:, :, 0])
    
    # Threshold at 100 to separate the green range (<100) from the purple range (>100)
    if median_hue < 100:
        return "green"
    else:
        return "purple"

def process_distance(crop, debug_name=None):
    """
    Routes the crop to the correct OCR pipeline based on its background color.
    """
    bg_color = detect_background_color(crop)
    
    if debug_name:
        print(f"[{debug_name}] Detected background: {bg_color}")
        
    if bg_color == "green":
        raw_text = process_distance_green(crop, debug_name=debug_name)
    else:
        raw_text = process_distance_purple(crop, debug_name=debug_name)
        
    # Return the extracted numbers (make sure your extract_numbers function handles None)
    if raw_text:
        # Assuming you implemented a sanitize/extract function or are using your previous one
        return extract_numbers(raw_text) 
    
    return None

def process_overall_score(crop, debug_name=None):
    """
    For massive, thick standalone digits on dark backgrounds.
    Uses basic grayscale thresholding and PSM 10 (single character).
    """
    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    
    if debug_name: cv2.imwrite(f"debug_{debug_name}.jpg", inverted)
    
    text = pytesseract.image_to_string(inverted, config="--psm 10 -c tessedit_char_whitelist=012O").strip()
    return '0' if text.upper() == 'O' else text

def process_standard(crop, debug_name=None):
    """
    For Usernames, Round Scores, and Health Points.
    Standard upscaling and high-contrast thresholding.
    """
    resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    
    if debug_name: cv2.imwrite(f"debug_{debug_name}.jpg", inverted)
    
    return pytesseract.image_to_string(inverted, config="--psm 7").strip()

# --- DATA EXTRACTION HELPERS ---

def extract_numbers(text):
    """Strips letters/symbols to return just the integer."""
    if text is None:
        return

    numbers = re.sub(r'[^\d]', '', text)
    return int(numbers) if numbers else None

def match_username(ocr_text, known_players):
    """Fuzzy matches OCR output to the known player list."""
    if not ocr_text:
        return None
    matches = difflib.get_close_matches(ocr_text.upper(), known_players, n=1, cutoff=0.4)
    return matches[0] if matches else ocr_text

# --- MAIN LOGIC ---

def process_directory(input_dir, output_json, config):
    rois = config.get("ROIS", {})
    known_players = config.get("KNOWN_PLAYERS", [])
    
    batches = defaultdict(list)
    filepaths = glob.glob(os.path.join(input_dir, "batch_*_frame_*.jpg"))
    
    for path in filepaths:
        filename = os.path.basename(path)
        match = re.search(r'batch_(\d+)_frame_(\d+)\.jpg', filename)
        if match:
            batches[match.group(1)].append((int(match.group(2)), path))
            
    results = []
    
    for batch_num in sorted(batches.keys()):
        batches[batch_num].sort(key=lambda x: x[0])
        last_frame_path = batches[batch_num][-1][1]
        
        frame = cv2.imread(last_frame_path)
        if frame is None:
            continue
            
        print(f"Processing Batch {batch_num} -> {os.path.basename(last_frame_path)}")
        
        frame_data = {
            "batch": batch_num,
            "source_image": os.path.basename(last_frame_path),
            "left_player": {},
            "right_player": {}
        }
        
        def crop_roi(roi_key):
            x, y, w, h = rois[roi_key]
            return frame[y:y+h, x:x+w]

        # 1. Overall Score
        frame_data["left_player"]["overall_score"] = extract_numbers(
            process_overall_score(crop_roi("overall_score_left"), debug_name="overall_left"))
        frame_data["right_player"]["overall_score"] = extract_numbers(
            process_overall_score(crop_roi("overall_score_right"), debug_name="overall_right"))

        # 2. Usernames 
        frame_data["left_player"]["username"] = match_username(
            process_standard(crop_roi("username_left"), debug_name="user_left"), known_players)
        frame_data["right_player"]["username"] = match_username(
            process_standard(crop_roi("username_right"), debug_name="user_right"), known_players)

        # 3. Round Scores
        frame_data["left_player"]["round_score"] = extract_numbers(
            process_standard(crop_roi("round_score_left"), debug_name="round_left"))
        frame_data["right_player"]["round_score"] = extract_numbers(
            process_standard(crop_roi("round_score_right"), debug_name="round_right"))

        # 4. Distances
        frame_data["left_player"]["distance_km"] = process_distance(crop_roi("distance_left"), debug_name="dist_left")
        frame_data["right_player"]["distance_km"] = process_distance(crop_roi("distance_right"), debug_name="dist_right")

        # 5. Health Points
        frame_data["left_player"]["health"] = extract_numbers(
            process_standard(crop_roi("health_left"), debug_name="health_left"))
        frame_data["right_player"]["health"] = extract_numbers(
            process_standard(crop_roi("health_right"), debug_name="health_right"))

        results.append(frame_data)
        
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExtraction complete! Saved {len(results)} batches to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from Geoguessr score screens.")
    parser.add_argument("-i", "--input", default="./output_frames", help="Directory containing the frames")
    parser.add_argument("-o", "--output", default="tournament_data.json", help="Output JSON file path")
    parser.add_argument("-c", "--config", default="config.json", help="Path to JSON config file containing ROIs")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        exit(1)
        
    with open(args.config, 'r') as f:
        config_data = json.load(f)
        
    process_directory(args.input, args.output, config_data)
