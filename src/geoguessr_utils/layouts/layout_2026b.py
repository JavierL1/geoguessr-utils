# src/geoguessr_utils/layouts/layout_2026b.py
"""
Concrete layout for the 2026B Geoguessr UI.

This layout describes *where* to find each player's data on the score
screen and *how* to extract it. It does not know how the score screen
is detected — that responsibility lives in
`geoguessr_utils.vision.detectors.score_screen_2026b` and is paired
with this layout by the CLI.

Coordinate data is hardcoded for 1080p video. If support for other
resolutions is added, this class is the natural place to branch (e.g.,
via a `resolution` constructor argument that scales ROIs).
"""

import numpy as np

from geoguessr_utils.layouts.layout import (
    FeatureROIS,
    Layout,
    ScreenFeature,
    ScreenFeatureROIS,
    ScreenSide,
)
from geoguessr_utils.parsers import parsing
from geoguessr_utils.vision import ocr
from geoguessr_utils.vision.roi import ROI


class Layout2026B(Layout):
    """Concrete layout for the 2026B Geoguessr UI at 1080p."""

    @property
    def known_usernames(self) -> list[str]:
        """Known player usernames to narrow detections."""
        return [
            "BLINKY",
            "KAPTEN",
            "CLEMENT",
            "KFL",
            "RADU C",
            "CONSUS",
            "MK",
            "MANUELSON",
            "STREFAN",
            "KRATSOOO",
            "MOO",
            "JAMABI",
            "BULGARIAENJ.",
            "LENNLI",
            "DEBRE",
            "NOKSI",
        ]

    @property
    def score_screen_discovery_rois(self) -> dict[str, ROI]:
        """ROIs consumed by the paired ScoreScreenDetector2026B."""
        return {
            "discover_left": ROI(x=632, y=127, w=89, h=57),
            "discover_right": ROI(x=1283, y=130, w=74, h=54),
            "discover_center": ROI(x=866, y=159, w=258, h=50),
            "health_left": ROI(x=480, y=931, w=144, h=63),
            "health_right": ROI(x=1365, y=932, w=144, h=63),
        }

    @property
    def score_screen_feature_rois(self) -> ScreenFeatureROIS:
        """ROIs for each player-facing feature on the score screen."""
        return ScreenFeatureROIS(
            left=FeatureROIS(
                match_score=ROI(x=878, y=23, w=60, h=70),
                round_score=ROI(x=560, y=123, w=270, h=60),
                distance=ROI(x=873, y=166, w=101, h=36),
                health_points=ROI(x=480, y=931, w=144, h=63),
                username=ROI(x=493, y=852, w=296, h=62),
            ),
            right=FeatureROIS(
                match_score=ROI(x=1058, y=26, w=61, h=65),
                round_score=ROI(x=1151, y=122, w=270, h=61),
                distance=ROI(x=1019, y=163, w=101, h=36),
                health_points=ROI(x=1365, y=932, w=144, h=63),
                username=ROI(x=1198, y=845, w=311, h=70),
            ),
        )

    def detect_match_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Return the parsed match score, or None if not readable."""
        feature: ScreenFeature = "match_score"
        return parsing.extract_numbers(
            ocr.read_match_score(
                self.crop_feature(frame, side, feature),
                debug_name=self.feature_name(side, feature),
            ),
            mode="int",
        )

    def detect_round_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Return the parsed round score, or None if not readable."""
        feature: ScreenFeature = "round_score"
        return parsing.extract_numbers(
            ocr.read_standard(
                self.crop_feature(frame, side, feature),
                debug_name=self.feature_name(side, feature),
            ),
            mode="int",
        )

    def detect_distance(
        self, frame: np.ndarray, side: ScreenSide
    ) -> parsing.Distance | None:
        """Return the parsed distance, or None if not readable."""
        feature: ScreenFeature = "distance"
        return parsing.parse_distance(
            ocr.read_distance(
                self.crop_feature(frame, side, feature),
                debug_name=self.feature_name(side, feature),
            )
        )

    def detect_health_points(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Return the parsed health points, or None if not readable."""
        feature: ScreenFeature = "health_points"
        return parsing.extract_numbers(
            ocr.read_health_points(
                self.crop_feature(frame, side, feature),
                debug_name=self.feature_name(side, feature),
            ),
            mode="int",
        )

    def detect_username(self, frame: np.ndarray, side: ScreenSide) -> str | None:
        """Return the fuzzy-matched username, or None if not readable."""
        feature: ScreenFeature = "username"
        return parsing.match_username(
            ocr.read_username(
                self.crop_feature(frame, side, feature),
                debug_name=self.feature_name(side, feature),
            ),
            self.known_usernames,
        )
