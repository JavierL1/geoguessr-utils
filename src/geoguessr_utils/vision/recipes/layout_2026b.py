"""Recipes for the 2026B layout.

2026B uses shared recipes without modification. If a future tweak
becomes necessary (e.g., a different threshold for one feature),
override here with `dataclasses.replace(SHARED_RECIPE, ...)`.
"""

from geoguessr_utils.vision.recipes.shared import (
    DISTANCE,
    HEALTH,
    ROUND_SCORE,
    USERNAME,
)

__all__ = ["ROUND_SCORE", "DISTANCE", "USERNAME", "HEALTH"]
