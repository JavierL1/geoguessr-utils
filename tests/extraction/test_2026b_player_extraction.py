"""
Golden-file extraction tests for extract_player.

For each .jpg in tests/fixtures/extraction/, a sibling .json file
holds the expected PlayerResult for both the left and right side.

The test asserts exact equality. When you intentionally change
extraction behavior, regenerate the golden file deliberately:

    uv run python scripts/regenerate_extraction_goldens.py

then review the diff before committing.
"""

import json
from pathlib import Path

import cv2
import pytest

from geoguessr_utils.extraction import extract_player
from geoguessr_utils.layouts import Layout2026B

FIXTURES = Path(__file__).parent.parent / "fixtures" / "2026b_frames" / "extraction"


def _discover_fixtures() -> list[Path]:
    """Find all fixture images (excludes .json sidecars)."""
    if not FIXTURES.is_dir():
        pytest.fail(
            f"Fixture directory not found: {FIXTURES}\n"
            f"Create it and add labeled extraction fixtures."
        )
    return sorted(
        p for p in FIXTURES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


@pytest.fixture(scope="module")
def layout():
    return Layout2026B()


@pytest.mark.parametrize("image_path", _discover_fixtures(), ids=lambda p: p.name)
def test_extract_player_matches_golden(layout, image_path):
    """Extraction output must match the checked-in golden JSON."""
    golden_path = image_path.with_suffix(".json")

    frame = cv2.imread(str(image_path))
    assert frame is not None, f"Failed to load fixture image: {image_path}"

    actual = {
        "left": extract_player(layout, frame, "left"),
        "right": extract_player(layout, frame, "right"),
    }

    assert golden_path.exists(), (
        f"Golden file missing for {image_path.name}.\n"
        f"Run scripts/regenerate_extraction_goldens.py to generate it, "
        f"then review and commit."
    )

    expected = json.loads(golden_path.read_text())

    for side in ("left", "right"):
        assert actual[side] == expected[side], (
            f"\nMismatch on {image_path.name} [{side} player]\n"
            f"  Expected: {expected[side]}\n"
            f"  Actual:   {actual[side]}\n"
            f"  If the change is intentional, regenerate the golden file "
            f"and review the diff."
        )
