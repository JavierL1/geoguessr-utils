"""
Score-screen detector for the 2026A Geoguessr layout.

Recognizes the 2026A score screen by checking for:
  1. Huge, bright white round scores at the top left and right.
  2. The "WORLD CHAMPIONSHIPS" logo in the top center.
  3. Dark backgrounds behind the UI elements.
  4. Rejects screens where the top scores have an orange or blue hue
     (e.g., round win or multiplier glows).
"""

import cv2
import numpy as np

from geoguessr_utils.vision.detectors.base import ScreenDetector
from geoguessr_utils.vision.roi import ROI

# Tuning thresholds.
WHITE_TEXT_MIN = 0.1
DARK_BG_MIN = 0.80
LOGO_WHITE_MIN = 0.29
COLOR_HUE_MAX = 0.02

# Orange/Yellow HSV bounds
ORANGE_LOWER = np.array([10, 100, 150])
ORANGE_UPPER = np.array([35, 255, 255])

# Cyan/Blue HSV bounds (Targeting the bright glow, not the dark background)
BLUE_LOWER = np.array([80, 100, 150])
BLUE_UPPER = np.array([130, 255, 255])

REQUIRED_ROIS = (
    "top_score_left",
    "top_score_right",
    "center_logo",
    "bg_top_left",
    "bg_top_right",
    "bg_left",
    "bg_right",
    "bg_bottom_left",
    "bg_bottom_right",
    "bg_middle_left",
    "bg_middle_right",
    "distance_left",
    "distance_right",
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

        # 1. Check for orange hue in the scores
        left_orange = self._color_ratio(
            crops["top_score_left"],
            self._rois["top_score_left"],
            ORANGE_LOWER,
            ORANGE_UPPER,
        )
        right_orange = self._color_ratio(
            crops["top_score_right"],
            self._rois["top_score_right"],
            ORANGE_LOWER,
            ORANGE_UPPER,
        )
        # print(f"left_orange: {left_orange}")
        # print(f"right_orange: {right_orange}")

        # 2. Check for blue/cyan hue in the scores
        left_blue = self._color_ratio(
            crops["top_score_left"],
            self._rois["top_score_left"],
            BLUE_LOWER,
            BLUE_UPPER,
        )
        right_blue = self._color_ratio(
            crops["top_score_right"],
            self._rois["top_score_right"],
            BLUE_LOWER,
            BLUE_UPPER,
        )

        # 3. Check the massive top scores (White/Gray)
        top_left_white = self._white_ratio(
            crops["top_score_left"], self._rois["top_score_left"], 130
        )
        top_right_white = self._white_ratio(
            crops["top_score_right"], self._rois["top_score_right"], 130
        )
        distance_left_white = self._white_ratio(
            crops["distance_left"], self._rois["distance_left"], 75
        )
        distance_right_white = self._white_ratio(
            crops["distance_right"], self._rois["distance_right"], 75
        )
        # print(f"distance_left_white: {distance_left_white}")
        # print(f"distance_right_white: {distance_right_white}")

        # 4. Check the center logo
        logo_white = self._white_ratio(
            crops["center_logo"], self._rois["center_logo"], 130
        )

        # 5. Check background darkness
        bg_keys = [
            "bg_top_left",
            "bg_top_right",
            "bg_left",
            "bg_right",
            "bg_bottom_left",
            "bg_bottom_right",
            "bg_middle_left",
            "bg_middle_right",
        ]
        dark_ratios = {
            bg_key: self._dark_ratio(crops[bg_key], self._rois[bg_key])
            for bg_key in bg_keys
        }

        has_white_text = (
            top_left_white > WHITE_TEXT_MIN
            and top_right_white > WHITE_TEXT_MIN
            and distance_left_white > WHITE_TEXT_MIN
            and distance_right_white > WHITE_TEXT_MIN
        )
        has_logo = logo_white > LOGO_WHITE_MIN
        has_dark_bg = all([ratio > DARK_BG_MIN for ratio in dark_ratios.values()])

        # Reject if any of the scores have a glowing color
        has_colored_glow = (
            left_orange > COLOR_HUE_MAX
            or right_orange > COLOR_HUE_MAX
            or left_blue > COLOR_HUE_MAX
            or right_blue > COLOR_HUE_MAX
        )

        return has_white_text and has_logo and has_dark_bg and not has_colored_glow

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
    def _color_ratio(
        crop: np.ndarray, roi: ROI, lower: np.ndarray, upper: np.ndarray
    ) -> float:
        """Fraction of the crop occupied by pixels within the given HSV bounds."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return cv2.countNonZero(mask) / (roi.w * roi.h)

    @staticmethod
    def _white_ratio(crop: np.ndarray, roi: ROI, thresh: float) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh_white = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        return cv2.countNonZero(thresh_white) / (roi.w * roi.h)

    @staticmethod
    def _dark_ratio(crop: np.ndarray, roi: ROI) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray, 0, 100)

        return cv2.countNonZero(dark_mask) / (roi.w * roi.h)
