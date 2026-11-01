"""Recipes for the 2026A layout."""

from geoguessr_utils.vision import postprocess, preprocess
from geoguessr_utils.vision.ocr import CONFIG_DISTANCE, CONFIG_STANDARD
from geoguessr_utils.vision.recipes import shared
from geoguessr_utils.vision.recipes.base import Recipe

# 2026A distances are plain text above the score (e.g., "7,027 km")
DISTANCE = Recipe(
    prepare=preprocess.prepare_distance_dark,
    config=CONFIG_DISTANCE,
)

# 2026A health points are inside bars, but the text is pure white.
# prepare_round_score works well here. We use the standard config with a fallback.
HEALTH = Recipe(
    prepare=preprocess.prepare_round_score,
    config=CONFIG_STANDARD,
    postprocess=postprocess.zero_and_lower,
    fallback_prepare=preprocess.add_synthetic_context,
)

# Reuse shared recipes for the rest
ROUND_SCORE = Recipe(
    prepare=lambda crop, debug_name=None: preprocess.prepare_round_score(
        crop, threshold=100, debug_name=debug_name
    ),
    config=CONFIG_STANDARD,
    postprocess=postprocess.postprocess_numbers,
    fallback_prepare=preprocess.add_synthetic_context,
)
USERNAME = shared.USERNAME

__all__ = ["ROUND_SCORE", "DISTANCE", "USERNAME", "HEALTH"]
