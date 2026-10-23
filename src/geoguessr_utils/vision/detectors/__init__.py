# src/geoguessr_utils/vision/detectors/__init__.py
"""
Screen detectors for Geoguessr layouts.

Each concrete detector answers "is this frame the screen I'm looking
for?" for one specific layout. Detectors are paired with a `Layout`
at the CLI level (see `discover_screens.py` for the wiring).
"""

from .base import ScreenDetector
from .score_screen_2026b import ScoreScreenDetector2026B

__all__ = ["ScreenDetector", "ScoreScreenDetector2026B"]
