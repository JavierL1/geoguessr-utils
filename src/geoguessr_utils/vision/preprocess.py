"""
OpenCV transformation pipelines for Geoguessr OCR.

Each `prepare_*` function takes a raw crop and returns a clean, high-contrast
image optimized for downstream OCR. It does *not* run OCR itself.

Design notes:
- Functions are pure (input image -> output image), easy to unit test.
- Set `DEBUG = True` to dump intermediate images to disk for inspection.
- Screen detection helpers (is_score_screen, etc.) live in `detection.py`.
"""

import cv2
import numpy as np

DEBUG = False


# ---------------------------------------------------------------------------
# Distance pill pipelines (green / purple / orange variants)
# ---------------------------------------------------------------------------


def detect_background_color(crop: np.ndarray) -> str:
    """Classify the distance pill background by dominant hue.

    Returns one of: "green", "orange", "purple".
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    median_hue = np.median(hsv[:, :, 0])

    if 10 <= median_hue <= 30:
        return "orange"
    if median_hue < 100:
        return "green"
    return "purple"


def prepare_distance_green(crop, debug_name=None):
    """White text on bright green. Threshold + dilate to fight merged letters."""
    scaled = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    padded = cv2.copyMakeBorder(dilated, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_green_padded.jpg", padded)
    return padded


def prepare_distance_purple(crop, debug_name=None):
    """Dim purple pill. Min-channel + erosion separates merged glyphs."""
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    min_channel = np.min(scaled, axis=2)
    _, thresh = cv2.threshold(min_channel, 130, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=1)
    inverted = cv2.bitwise_not(thinned)
    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_purple.jpg", padded)
    return padded


def prepare_distance_orange(crop, debug_name=None):
    """Orange/gold 5k tie pill. Blue channel provides best contrast."""
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blue_channel = scaled[:, :, 0]
    _, thresh = cv2.threshold(blue_channel, 180, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(thresh)
    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_orange.jpg", padded)
    return padded


def prepare_distance(crop, debug_name=None):
    """Route a distance crop to the correct variant pipeline automatically."""
    bg = detect_background_color(crop)
    if bg == "green":
        return prepare_distance_green(crop, debug_name)
    if bg == "orange":
        return prepare_distance_orange(crop, debug_name)
    return prepare_distance_purple(crop, debug_name)


# ---------------------------------------------------------------------------
# Overall score (massive standalone digits 0/1/2)
# ---------------------------------------------------------------------------


def detect_digit_by_geometry(crop: np.ndarray, debug_name=None) -> str:
    """Identify a single digit (0/1/2) using pure contour geometry.

    - 0 is the only digit with an enclosed hole.
    - 1 vs 2 is decided by the width of the digit's base.
    """
    scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_thresh.jpg", thresh)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        return "Empty"

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w == 0 or h == 0:
        return "Empty"

    # Hole inside the contour -> it's a '0'
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1 and cv2.contourArea(contours[i]) > 20:
            return "0"

    # Otherwise compare top vs bottom width. A '1' is narrow at the bottom.
    bottom_start = y + int(h * 0.7)
    bottom_slice = thresh[bottom_start : y + h, x : x + w]
    cols = np.where(bottom_slice == 255)[1]
    if len(cols) == 0:
        return "1"

    bottom_ratio = (cols[-1] - cols[0]) / float(w)
    return "1" if bottom_ratio < 0.60 else "2"


# ---------------------------------------------------------------------------
# Standard numbers, usernames, health bars
# ---------------------------------------------------------------------------


def prepare_standard(crop, debug_name=None):
    """Round scores and similar numeric fields."""
    scaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=2)
    inverted = cv2.bitwise_not(thinned)
    padded = cv2.copyMakeBorder(
        inverted, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}.jpg", padded)
    return padded


def prepare_username(crop, debug_name=None):
    """Player name text."""
    scaled = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=1)
    inverted = cv2.bitwise_not(thinned)
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}.jpg", inverted)
    return inverted


def prepare_health(crop, debug_name=None):
    """Health bar. Blanks out the green/yellow fill so only white text remains."""
    if crop is None or crop.size == 0:
        return None

    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([34, 255, 255])
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    color_mask = cv2.bitwise_or(mask_yellow, mask_green)

    processed = scaled.copy()
    processed[color_mask > 0] = [0, 0, 0]

    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(thresh, kernel, iterations=3)

    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_thinned.jpg", thinned)

    padded = cv2.copyMakeBorder(thinned, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    if debug_name and DEBUG:
        cv2.imwrite(f"debug_{debug_name}_clean.jpg", padded)

    return padded
