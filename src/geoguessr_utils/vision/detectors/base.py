# src/geoguessr_utils/vision/detectors/base.py
"""
Abstract base class for screen detectors.

A `ScreenDetector` answers exactly one question: "does this frame
correspond to the screen I'm looking for?" It has no knowledge of
video IO, layouts, extraction, or the CLI. Everything it needs is
passed in via the constructor.

Detectors are paired with a concrete `Layout` at the CLI level. The
layout supplies the ROIs; the detector supplies the algorithm. This
keeps the two axes of variation (coordinates vs. heuristic) independent.
"""

from abc import ABC, abstractmethod

import numpy as np


class ScreenDetector(ABC):
    """Base class for all screen detectors.

    Subclasses must implement `detect`. They are expected to be
    stateless after `__init__` — all state (ROIs, thresholds, masks)
    is baked in at construction time, so `detect` can be called in any
    order on any number of frames.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> bool:
        """Return True if `frame` matches the screen this detector targets.

        Args:
            frame: BGR image as returned by cv2.imread / cap.read.
                   Shape is (H, W, 3), dtype uint8.

        Returns:
            True if the frame is judged to be a match, False otherwise.

        Raises:
            ValueError: If the frame's dimensions are incompatible with
                the ROIs the detector was built from. Implementations
                should raise here rather than silently returning False,
                so that misconfiguration is caught early.
        """
