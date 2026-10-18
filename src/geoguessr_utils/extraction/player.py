"""
Per-player feature extraction from a single score-screen frame.

This module knows how to route a layout's `detect_*` methods into a
`PlayerResult` dict. It has no knowledge of file IO, progress, or the
CLI — a frame goes in, a structured result comes out.
"""

import numpy as np

from geoguessr_utils.layouts import Layout, ScreenSide
from geoguessr_utils.player_result import PlayerResult


def extract_player(layout: Layout, frame: np.ndarray, side: ScreenSide) -> PlayerResult:
    """Extract every player-facing feature from a score-screen frame.

    Args:
        layout: Concrete layout to extract from.
        frame: BGR score-screen frame.
        side: Which player's data to extract.

    Returns:
        A `PlayerResult` with one entry per feature.
    """
    return {
        "match_score": layout.detect_match_score(frame, side),
        "username": layout.detect_username(frame, side),
        "round_score": layout.detect_round_score(frame, side),
        "distance": layout.detect_distance(frame, side),
        "health_points": layout.detect_health_points(frame, side),
    }
