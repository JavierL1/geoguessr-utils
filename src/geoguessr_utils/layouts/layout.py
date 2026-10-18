# src/geoguessr_utils/layouts/layout.py
"""
Abstract base class for screen layouts.

A `Layout` describes *what* to extract and *where* to find it on a
given screen. It exposes the ROIs the OCR pipeline needs and routes
each feature through the appropriate OCR + parsing function.

Layouts are deliberately unaware of:

  - How a screen is *detected*. Detection lives in
    `geoguessr_utils.vision.detectors` and is paired with a layout
    by the CLI. This keeps the two axes of variation independent.

  - Video IO, the discovery state machine, progress reporting, and
    the CLI. Those live downstream.

A concrete layout is expected to be constructable with no arguments
and to be safe to instantiate multiple times. It carries no mutable
state beyond the debug flag.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from geoguessr_utils.parsers import parsing
from geoguessr_utils.vision.roi import ROI

ScreenSide = Literal["left", "right"]
ScreenFeature = Literal[
    "match_score", "round_score", "distance", "username", "health_points"
]


@dataclass
class FeatureROIS:
    """ROIs for the five player-facing features, for one player."""

    match_score: ROI
    round_score: ROI
    distance: ROI
    username: ROI
    health_points: ROI


@dataclass
class ScreenFeatureROIS:
    """Feature ROIs for both players on a score screen."""

    left: FeatureROIS
    right: FeatureROIS


class Layout(ABC):
    """Abstract base class for concrete screen layouts."""

    def __init__(self):
        self._debug: bool = False

    @property
    def debug(self) -> bool:
        """Whether debug image dumps are enabled."""
        return self._debug

    @debug.setter
    def debug(self, value: bool = False) -> None:
        self._debug = value

    # -- Data contracts ---------------------------------------------------

    @property
    @abstractmethod
    def known_usernames(self) -> list[str]:
        """Known player usernames to narrow OCR detections."""

    @property
    @abstractmethod
    def score_screen_discovery_rois(self) -> dict[str, ROI]:
        """ROIs used by the score-screen detector.

        Every concrete layout must provide keys that match the detector
        it is paired with. See the detector's `REQUIRED_ROIS` for the
        exact set this layout's sibling detector expects.
        """

    @property
    @abstractmethod
    def score_screen_feature_rois(self) -> ScreenFeatureROIS:
        """ROIs used by the extraction pipeline for each feature."""

    # -- Feature extraction ----------------------------------------------

    @abstractmethod
    def detect_match_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Extract and parse a player's match score."""

    @abstractmethod
    def detect_round_score(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Extract and parse a player's round score."""

    @abstractmethod
    def detect_distance(
        self, frame: np.ndarray, side: ScreenSide
    ) -> parsing.Distance | None:
        """Extract and parse a player's distance reading."""

    @abstractmethod
    def detect_health_points(self, frame: np.ndarray, side: ScreenSide) -> int | None:
        """Extract and parse a player's remaining health points."""

    @abstractmethod
    def detect_username(self, frame: np.ndarray, side: ScreenSide) -> str | None:
        """Extract and fuzzy-match a player's username."""

    # -- Shared helpers --------------------------------------------------

    def crop_feature(
        self, frame: np.ndarray, side: ScreenSide, feature: ScreenFeature
    ) -> np.ndarray:
        """Crop a feature ROI from the frame."""
        roi = self.score_screen_feature_rois[side][feature]
        return frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]

    def feature_name(self, side: ScreenSide, feature: ScreenFeature) -> str | None:
        """Debug filename for a feature crop, or None if debug is off."""
        return f"{side}_{feature}" if self._debug else None
