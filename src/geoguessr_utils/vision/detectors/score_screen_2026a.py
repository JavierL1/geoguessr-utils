"""
Score-screen detector for the 2026A Geoguessr layout.

Recognizes the 2026A score screen by checking for:
  1. Huge, bright white round scores at the top left and right.
  2. The "WORLD CHAMPIONSHIPS" logo in the top center.
  3. Dark backgrounds behind the UI elements.

Gameplay screens are rejected because their top left/right areas only
contain tiny match scores (0 or 1), not the massive round scores.
"""

import cv2
import numpy as np

from geoguessr_utils.vision.detectors.base import ScreenDetector
from geoguessr_utils.vision.roi import ROI

# Tuning thresholds.
WHITE_TEXT_MIN = 0.1  # The top scores are huge, so they take up a lot of area.
DARK_BG_MIN = 0.80
LOGO_WHITE_MIN = 0.29

# Required ROIs for this specific layout.
REQUIRED_ROIS = (
    "top_score_left",  # Bounding box around the top-left huge number (e.g., 4997)
    "top_score_right",  # Bounding box around the top-right huge number (e.g., 5000)
    "center_logo",  # Bounding box around the "WORLD CHAMPIONSHIPS BERLIN" text
    "bg_left",  # A blank dark area on the left side
    "bg_right",  # A blank dark area on the right side
)


class ScoreScreenDetector2026A(ScreenDetector):
    """Detects the 2026A World Championships score screen."""

    def __init__(self, rois: dict[str, ROI]):
        missing = [key for key in REQUIRED_ROIS if key not in rois]
        if missing:
            raise ValueError(
                f"ScoreScreenDetector2026A is missing required ROIs: {missing}. "
                f"Expected keys: {list(REQUIRED_ROIS)}"
            )
        self._rois = rois

    def detect(self, frame: np.ndarray) -> bool:
        crops = self._crop_all(frame)

        # Check the massive top scores
        top_left_white, _ = self._white_and_dark_ratios(
            crops["top_score_left"], self._rois["top_score_left"]
        )
        top_right_white, _ = self._white_and_dark_ratios(
            crops["top_score_right"], self._rois["top_score_right"]
        )

        # Check the center logo
        logo_white, _ = self._white_and_dark_ratios(
            crops["center_logo"], self._rois["center_logo"]
        )

        # Check background darkness
        _, left_dark = self._white_and_dark_ratios(
            crops["bg_left"], self._rois["bg_left"]
        )
        _, right_dark = self._white_and_dark_ratios(
            crops["bg_right"], self._rois["bg_right"]
        )

        print(f"top_left_white: {top_left_white}")
        print(f"top_right_white: {top_right_white}")

        # Evaluate conditions
        # In a score screen, these top ROIs will be >20% filled with white text.
        # In gameplay, they will be <5% (just the tiny 0 or 1).
        has_top_scores = (
            top_left_white > WHITE_TEXT_MIN and top_right_white > WHITE_TEXT_MIN
        )

        print(f"logo_white: {logo_white}")
        print(f"left_dark: {left_dark}")
        print(f"right_dark: {right_dark}")

        has_logo = logo_white > LOGO_WHITE_MIN
        has_dark_bg = left_dark > DARK_BG_MIN and right_dark > DARK_BG_MIN

        return has_top_scores and has_logo and has_dark_bg

    # -- internals --------------------------------------------------------

    def _crop_all(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        frame_h, frame_w = frame.shape[:2]
        crops: dict[str, np.ndarray] = {}

        for key, roi in self._rois.items():
            if roi.x + roi.w > frame_w or roi.y + roi.h > frame_h:
                raise ValueError(
                    f"ROI '{key}' ({roi}) is out of bounds for frame "
                    f"of size {frame_w}x{frame_h}."
                )
            crops[key] = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]

        return crops

    @staticmethod
    def _white_and_dark_ratios(crop: np.ndarray, roi: ROI) -> tuple[float, float]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Threshold for bright white text
        _, thresh_white = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        white_ratio = cv2.countNonZero(thresh_white) / (roi.w * roi.h)

        # Threshold for dark background
        dark_mask = cv2.inRange(gray, 0, 100)
        dark_ratio = cv2.countNonZero(dark_mask) / (roi.w * roi.h)

        return white_ratio, dark_ratio
