"""
Concrete layout for the 2026A Geoguessr UI (World Championships).

Differences from 2026B:
  - No match score is displayed (returns None).
  - Distances are plain text at the top (e.g., 7,027 km).
  - Health points are inside the bottom bars (0, 3948).
  - Usernames are below the webcams.
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
from geoguessr_utils.vision.recipes import layout_2026a as recipes
from geoguessr_utils.vision.recipes.base import run_recipe
from geoguessr_utils.vision.roi import ROI


class Layout2026A(Layout):
    """Concrete layout for the 2026A Geoguessr UI at 1080p."""

    @property
    def known_usernames(self) -> list[str]:
        """Known player usernames to narrow detections."""
        return [
            # World League - Europe
            "BLINKY",
            "CONSUS",
            "DEBRE",
            "BULGARIAENJOYER",
            "KAPTEN",
            "FINBARR",
            "CLEMENT",
            "MANUELSON",
            # World League - Americas
            "MK",
            "RADU C",
            "MOO",
            "ECAL",
            # World League - APAC
            "SHIINA",
            "LINGDING",
            "KIRANIA",
            "TRABOTA",
            # Play-in
            "LENNLI",
            "NOKSI",
            "KRATSOOO",
            "THE GYMTIMIDATOR",
            "STREFAN",
            "JAMABI",
            # Wildcards
            "MALFUNCTIONN",
            "KFL",
        ]

    @property
    def score_screen_discovery_rois(self) -> dict[str, ROI]:
        """ROIs consumed by the paired ScoreScreenDetector2026A."""
        return {
            "top_score_left": ROI(x=530, y=98, w=270, h=94),
            "top_score_right": ROI(x=1224, y=98, w=270, h=94),
            "center_logo": ROI(x=875, y=68, w=263, h=134),
            "bg_top_left": ROI(x=542, y=74, w=68, h=22),
            "bg_top_right": ROI(x=1413, y=74, w=68, h=22),
            "bg_middle_left": ROI(x=799, y=5, w=78, h=249),
            "bg_middle_right": ROI(x=1137, y=5, w=78, h=249),
            "bg_left": ROI(x=489, y=101, w=41, h=98),
            "bg_right": ROI(x=1486, y=101, w=41, h=98),
            "bg_bottom_left": ROI(x=524, y=194, w=325, h=65),
            "bg_bottom_right": ROI(x=1203, y=194, w=325, h=65),
            "distance_left": ROI(x=576, y=55, w=53, h=29),
            "distance_right": ROI(x=1392, y=55, w=53, h=29),
        }

    @property
    def score_screen_feature_rois(self) -> ScreenFeatureROIS:
        """ROIs for each player-facing feature on the score screen."""
        return ScreenFeatureROIS(
            left=FeatureROIS(
                match_score=ROI(x=0, y=0, w=1, h=1),  # Not used on 2026A
                round_score=ROI(x=526, y=97, w=313, h=101),
                distance=ROI(x=569, y=46, w=190, h=40),
                username=ROI(x=146, y=345, w=327, h=51),
                health_points=ROI(x=670, y=343, w=132, h=56),
            ),
            right=FeatureROIS(
                match_score=ROI(x=0, y=0, w=1, h=1),  # Not used on 2026A
                round_score=ROI(x=1186, y=97, w=313, h=101),
                distance=ROI(x=1234, y=46, w=211, h=40),
                username=ROI(x=1540, y=345, w=327, h=51),
                health_points=ROI(x=1211, y=343, w=119, h=55),
            ),
        )

    @property
    def supported_features(self) -> list[ScreenFeature]:
        return ["round_score", "distance", "username", "health_points"]

    # -- Feature extraction ----------------------------------------------

    def detect_match_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """2026A does not display the match score on the score screen."""
        return None

    def detect_round_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Return the parsed round score, or 0 if not readable."""
        feature: ScreenFeature = "round_score"
        round_score = parsing.extract_numbers(
            run_recipe(
                self.crop_feature(frame, side, feature),
                recipes.ROUND_SCORE,
                debug_name=self.feature_name(side, feature),
            ),
            mode="int",
        )
        return round_score if isinstance(round_score, int) else 0

    def detect_distance(
        self, frame: np.ndarray, side: ScreenSide
    ) -> parsing.Distance | None:
        """Return the parsed distance, or None if not readable."""
        feature: ScreenFeature = "distance"
        return parsing.parse_distance(
            run_recipe(
                self.crop_feature(frame, side, feature),
                recipes.DISTANCE,
                debug_name=self.feature_name(side, feature),
            )
        )

    def detect_health_points(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Return the parsed health points, or 0 if not readable."""
        feature: ScreenFeature = "health_points"
        health_points = parsing.extract_numbers(
            run_recipe(
                self.crop_feature(frame, side, feature),
                recipes.HEALTH,
                debug_name=self.feature_name(side, feature),
            ),
            mode="int",
        )
        return health_points if isinstance(health_points, int) else 0

    def detect_username(self, frame: np.ndarray, side: ScreenSide) -> str | None:
        """Return the fuzzy-matched username, or None if not readable."""
        feature: ScreenFeature = "username"
        return parsing.match_username(
            run_recipe(
                self.crop_feature(frame, side, feature),
                recipes.USERNAME,
                debug_name=self.feature_name(side, feature),
            ),
            self.known_usernames,
        )
