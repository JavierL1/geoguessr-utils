"""
Recipes: named pipelines from raw crop to extracted text.

A Recipe bundles the four things that vary between layouts:

  - prepare:           crop -> image ready for Tesseract
  - config:            Tesseract flags
  - postprocess:       text cleanup after Tesseract
  - fallback_prepare:  optional retry when Tesseract returns empty
                       (isolated single digits need synthetic context)

Recipes are immutable data. Layouts pick them from the recipes
package; they are never constructed inline.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytesseract

PrepareFn = Callable[..., np.ndarray | None]
PostprocessFn = Callable[[str], str]


def _identity(s: str) -> str:
    return s


@dataclass(frozen=True)
class Recipe:
    prepare: PrepareFn
    config: str
    postprocess: PostprocessFn = field(default=_identity)
    fallback_prepare: PrepareFn | None = None


def run_recipe(crop: np.ndarray, recipe: Recipe, debug_name: str | None = None) -> str:
    """Execute a Recipe on a crop, returning cleaned text."""
    image = recipe.prepare(crop, debug_name=debug_name)

    if image is None:
        return ""

    text = pytesseract.image_to_string(image, config=recipe.config).strip()
    text = recipe.postprocess(text)

    # Fallback for isolated single digits: retry with synthetic context.
    if not text and recipe.fallback_prepare is not None:
        fallback_image = recipe.fallback_prepare(image)

        if fallback_image is not None:
            text = pytesseract.image_to_string(
                fallback_image, config=recipe.config
            ).strip()
            text = recipe.postprocess(text)

    return text
