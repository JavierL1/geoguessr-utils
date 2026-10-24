"""Recipes shared across layouts.

If a layout needs different behavior for a feature, import this and
override with `dataclasses.replace(...)` in that layout's module.
"""

from geoguessr_utils.vision import preprocess
from geoguessr_utils.vision.ocr import (
    CONFIG_DISTANCE,
    CONFIG_HEALTH,
    CONFIG_STANDARD,
    CONFIG_USERNAME,
)
from geoguessr_utils.vision.recipes.base import Recipe


def _zero_and_lower(s: str) -> str:
    return s.lower().replace("o", "0")


ROUND_SCORE = Recipe(
    prepare=preprocess.prepare_standard,
    config=CONFIG_STANDARD,
    postprocess=_zero_and_lower,
    fallback_prepare=preprocess.add_synthetic_context,
)

DISTANCE = Recipe(
    prepare=preprocess.prepare_distance,
    config=CONFIG_DISTANCE,
)

USERNAME = Recipe(
    prepare=preprocess.prepare_username,
    config=CONFIG_USERNAME,
)

HEALTH = Recipe(
    prepare=preprocess.prepare_health,
    config=CONFIG_HEALTH,
)
