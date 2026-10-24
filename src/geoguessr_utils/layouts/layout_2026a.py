"""
Concrete layout for the 2026A Geoguessr UI.

Currently implements only the detection ROIs. Extraction methods are
stubbed out and will be implemented in the next phase.
"""

import numpy as np

from geoguessr_utils.layouts.layout import (
    FeatureROIS,
    Layout,
    ScreenFeatureROIS,
    ScreenSide,
)
from geoguessr_utils.parsers import parsing
from geoguessr_utils.vision.roi import ROI


class Layout2026A(Layout):
    """Concrete layout for the 2026A Geoguessr UI at 1080p."""

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
            "SHIINA",
            "FINBARR",
            "THE GYMTIMIDATOR",
        ]

    @property
    def score_screen_discovery_rois(self) -> dict[str, ROI]:
        """ROIs consumed by the paired ScoreScreenDetector2026A."""
        return {
            "top_score_left": ROI(x=530, y=98, w=270, h=94),
            "top_score_right": ROI(x=1224, y=98, w=270, h=94),
            "center_logo": ROI(x=875, y=68, w=263, h=134),
            "bg_left": ROI(x=524, y=194, w=325, h=65),
            "bg_right": ROI(x=1203, y=194, w=325, h=65),
        }

    @property
    def score_screen_feature_rois(self) -> ScreenFeatureROIS:
        """ROIs for extraction. (Not implemented yet)."""
        dummy_roi = ROI(x=0, y=0, w=1, h=1)
        dummy_features = FeatureROIS(
            match_score=dummy_roi,
            round_score=dummy_roi,
            distance=dummy_roi,
            username=dummy_roi,
            health_points=dummy_roi,
        )
        return ScreenFeatureROIS(left=dummy_features, right=dummy_features)

    # -- Feature extraction (Stubs) --------------------------------------

    def detect_match_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        return None

    def detect_round_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        return None

    def detect_distance(
        self, frame: np.ndarray, side: ScreenSide
    ) -> parsing.Distance | None:
        return None

    def detect_health_points(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        return None

    def detect_username(self, frame: np.ndarray, side: ScreenSide) -> str | None:
        return None
