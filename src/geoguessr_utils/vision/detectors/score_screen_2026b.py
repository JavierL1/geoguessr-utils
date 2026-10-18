# src/geoguessr_utils/vision/detectors/score_screen_2026b.py
"""
Score-screen detector for the 2026B Geoguessr layout.

Recognizes the standard post-round score screen by checking five
features simultaneously:

  1. White text in the left player area (round score, username)
  2. White text in the right player area (round score, username)
  3. Dark backgrounds behind both players (the score screen has a
     dark panel, unlike the live gameplay screen)
  4. Visible health bars for both players
  5. A center pill matching one of the known color palettes:
       - green + purple (standard match)
       - orange (5k tie)

All five conditions must hold. This is intentionally strict — a false
positive (detecting a score screen when there isn't one) is worse than
a false negative (skipping one score screen), because the pipeline
would emit garbage data downstream.
"""

import cv2
import numpy as np

from geoguessr_utils.vision.detectors.base import ScreenDetector
from geoguessr_utils.vision.roi import ROI

# HSV bounds for the three pill colors that appear on the center widget.
# Values are in OpenCV's 0-179 / 0-255 / 0-255 HSV space.
PILL_GREEN_LOWER = np.array([35, 100, 100])
PILL_GREEN_UPPER = np.array([85, 255, 255])
PILL_PURPLE_LOWER = np.array([110, 50, 50])
PILL_PURPLE_UPPER = np.array([160, 255, 255])
PILL_ORANGE_LOWER = np.array([10, 100, 100])  # 5k tie pill
PILL_ORANGE_UPPER = np.array([30, 255, 255])

# Tuning thresholds. If detection starts missing or over-triggering,
# these are the knobs to adjust.
WHITE_TEXT_MIN = 0.10
DARK_BG_MIN = 0.20
PILL_STANDARD_MIN = 0.02
PILL_TIE_MIN = 0.05

# Every key the constructor requires. Exposed so the CLI can validate
# before constructing, and so tests can build synthetic ROI dicts.
REQUIRED_ROIS = (
    "discover_left",
    "discover_right",
    "discover_center",
    "health_left",
    "health_right",
)


class ScoreScreenDetector2026B(ScreenDetector):
    """Detects the 2026B score screen via HSV ratios and text density."""

    def __init__(self, rois: dict[str, ROI]):
        missing = [key for key in REQUIRED_ROIS if key not in rois]
        if missing:
            raise ValueError(
                f"ScoreScreenDetector2026B is missing required ROIs: {missing}. "
                f"Expected keys: {list(REQUIRED_ROIS)}"
            )
        self._rois = rois

    def detect(self, frame: np.ndarray) -> bool:
        crops = self._crop_all(frame)

        left_white, left_dark = self._white_and_dark_ratios(
            crops["discover_left"], self._rois["discover_left"]
        )
        right_white, right_dark = self._white_and_dark_ratios(
            crops["discover_right"], self._rois["discover_right"]
        )
        health_left_white, _ = self._white_and_dark_ratios(
            crops["health_left"], self._rois["health_left"]
        )
        health_right_white, _ = self._white_and_dark_ratios(
            crops["health_right"], self._rois["health_right"]
        )

        green, purple, orange = self._pill_color_ratios(
            crops["discover_center"], self._rois["discover_center"]
        )

        has_white_text = left_white > WHITE_TEXT_MIN and right_white > WHITE_TEXT_MIN
        has_dark_bg = left_dark > DARK_BG_MIN and right_dark > DARK_BG_MIN
        has_visible_health = (
            health_left_white > WHITE_TEXT_MIN and health_right_white > WHITE_TEXT_MIN
        )
        has_standard_pill = green > PILL_STANDARD_MIN and purple > PILL_STANDARD_MIN
        has_tie_pill = orange > PILL_TIE_MIN

        return (
            has_white_text
            and has_dark_bg
            and has_visible_health
            and (has_standard_pill or has_tie_pill)
        )

    # -- internals --------------------------------------------------------

    def _crop_all(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Crop every required ROI, validating the frame bounds first."""
        frame_h, frame_w = frame.shape[:2]
        crops: dict[str, np.ndarray] = {}

        for key, roi in self._rois.items():
            if roi.x + roi.w > frame_w or roi.y + roi.h > frame_h:
                raise ValueError(
                    f"ROI '{key}' ({roi}) is out of bounds for frame "
                    f"of size {frame_w}x{frame_h}. Check that the layout "
                    f"matches the video resolution."
                )
            crops[key] = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]

        return crops

    @staticmethod
    def _white_and_dark_ratios(crop: np.ndarray, roi: ROI) -> tuple[float, float]:
        """Fraction of bright-white pixels and dark pixels in a crop."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = cv2.countNonZero(thresh_white) / (roi.w * roi.h)

        dark_mask = cv2.inRange(gray, 0, 100)
        dark_ratio = cv2.countNonZero(dark_mask) / (roi.w * roi.h)

        return white_ratio, dark_ratio

    @staticmethod
    def _pill_color_ratios(crop: np.ndarray, roi: ROI) -> tuple[float, float, float]:
        """Fraction of the center pill occupied by green, purple, and orange."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        area = roi.w * roi.h

        green = (
            cv2.countNonZero(cv2.inRange(hsv, PILL_GREEN_LOWER, PILL_GREEN_UPPER))
            / area
        )
        purple = (
            cv2.countNonZero(cv2.inRange(hsv, PILL_PURPLE_LOWER, PILL_PURPLE_UPPER))
            / area
        )
        orange = (
            cv2.countNonZero(cv2.inRange(hsv, PILL_ORANGE_LOWER, PILL_ORANGE_UPPER))
            / area
        )

        return green, purple, orange
