"""
OpenCV-based screen detection.

Pure functions: given a frame and ROIs, return a bool.
Kept separate from `layout.py` so that "what features exist" and
"how do we recognize them" are independent concerns.
"""

import cv2
import numpy as np

from geoguessr_utils.vision.roi import ROI

# HSV bounds for the pill colors that appear on the center widget.
PILL_GREEN_LOWER = np.array([35, 100, 100])
PILL_GREEN_UPPER = np.array([85, 255, 255])
PILL_PURPLE_LOWER = np.array([110, 50, 50])
PILL_PURPLE_UPPER = np.array([160, 255, 255])
PILL_ORANGE_LOWER = np.array([10, 100, 100])  # 5k tie pill
PILL_ORANGE_UPPER = np.array([30, 255, 255])

# Tuning thresholds — keep them here so tweaking is one place.
WHITE_TEXT_MIN = 0.10
DARK_BG_MIN = 0.20
PILL_STANDARD_MIN = 0.02
PILL_TIE_MIN = 0.05


def _crop(frame: np.ndarray, roi: ROI) -> np.ndarray:
    return frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]


def _white_and_dark_ratios(crop: np.ndarray, roi: ROI) -> tuple[float, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_ratio = cv2.countNonZero(thresh_white) / (roi.w * roi.h)

    dark_mask = cv2.inRange(gray, 0, 100)
    dark_ratio = cv2.countNonZero(dark_mask) / (roi.w * roi.h)

    return white_ratio, dark_ratio


def _pill_color_ratios(crop: np.ndarray, roi: ROI) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    area = roi.w * roi.h

    green = (
        cv2.countNonZero(cv2.inRange(hsv, PILL_GREEN_LOWER, PILL_GREEN_UPPER)) / area
    )
    purple = (
        cv2.countNonZero(cv2.inRange(hsv, PILL_PURPLE_LOWER, PILL_PURPLE_UPPER)) / area
    )
    orange = (
        cv2.countNonZero(cv2.inRange(hsv, PILL_ORANGE_LOWER, PILL_ORANGE_UPPER)) / area
    )

    return green, purple, orange


def is_score_screen(frame: np.ndarray, rois: dict[str, ROI]) -> bool:
    """Return True if `frame` looks like a Geoguessr score screen.

    Expects keys: discover_left, discover_right, discover_center,
    health_left, health_right.

    Raises ValueError if any ROI is out of bounds for the frame.
    """
    left = _crop(frame, rois["discover_left"])
    right = _crop(frame, rois["discover_right"])
    center = _crop(frame, rois["discover_center"])
    health_left = _crop(frame, rois["health_left"])
    health_right = _crop(frame, rois["health_right"])

    if min(c.size for c in [left, right, center, health_left, health_right]) == 0:
        frame_h, frame_w = frame.shape[:2]
        raise ValueError(
            f"\nROI(s) out of bounds for frame size {frame_w}x{frame_h}.\n"
            f"Please adjust your discovery ROIs."
        )

    left_white, left_dark = _white_and_dark_ratios(left, rois["discover_left"])
    right_white, right_dark = _white_and_dark_ratios(right, rois["discover_right"])
    health_left_white, _ = _white_and_dark_ratios(health_left, rois["health_left"])
    health_right_white, _ = _white_and_dark_ratios(health_right, rois["health_right"])

    green, purple, orange = _pill_color_ratios(center, rois["discover_center"])

    return (
        left_white > WHITE_TEXT_MIN
        and right_white > WHITE_TEXT_MIN
        and left_dark > DARK_BG_MIN
        and right_dark > DARK_BG_MIN
        and health_left_white > WHITE_TEXT_MIN
        and health_right_white > WHITE_TEXT_MIN
        and (
            (green > PILL_STANDARD_MIN and purple > PILL_STANDARD_MIN)
            or orange > PILL_TIE_MIN
        )
    )
