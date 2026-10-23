# tests/detectors/test_score_screen_2026b_classification.py
"""
Parameterized classification tests for ScoreScreenDetector2026B.

Walks a directory of labeled fixture frames and asserts that the
detector returns the expected boolean for each one.

Fixtures are organized by expectation:

    tests/fixtures/score_screens/*.jpg      -> expected True
    tests/fixtures/non_score_screens/*.jpg  -> expected False

Drop a new frame into either directory and it is automatically picked
up by the test suite on next run. No code changes required.
"""

from pathlib import Path

import cv2
import pytest

from geoguessr_utils.layouts import Layout2026B
from geoguessr_utils.vision.detectors import ScoreScreenDetector2026B

FIXTURES = Path(__file__).parent.parent / "fixtures"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# Each entry: (label, directory, expected_result)
LABELED_DIRS = [
    ("score_screen", FIXTURES / "2026b_frames" / "score_screens", True),
    ("non_score_screen", FIXTURES / "2026b_frames" / "non_score_screens", False),
]


def _collect_fixtures():
    """Build the param list for the test.

    Returns a list of pytest.param entries, each carrying the image
    path, the expected bool, and a readable test id.

    Raises a clear error at collection time if a directory is missing,
    so the test fails loudly instead of silently passing with zero
    cases.
    """
    params = []
    for label, directory, expected in LABELED_DIRS:
        if not directory.is_dir():
            pytest.fail(
                f"Fixture directory not found: {directory}\n"
                f"Create it and add labeled images, or remove this label "
                f"from LABELED_DIRS."
            )

        for image_path in sorted(
            p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            params.append(
                pytest.param(
                    image_path,
                    expected,
                    id=f"{label}/{image_path.name}",
                )
            )

    return params


@pytest.fixture(scope="module")
def detector():
    """One detector instance shared across all classification tests."""
    layout = Layout2026B()
    return ScoreScreenDetector2026B(layout.score_screen_discovery_rois)


@pytest.mark.parametrize("image_path,expected", _collect_fixtures())
def test_classifies_frame(detector, image_path, expected):
    """The detector should return `expected` for this fixture."""
    frame = cv2.imread(str(image_path))
    assert frame is not None, f"Failed to load fixture image: {image_path}"

    result = detector.detect(frame)

    assert result is expected, (
        f"\nMisclassification on {image_path.name}\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {result}\n"
        f"  If this is a new frame, verify the label is correct. "
        f"If the label is correct, the detection heuristic may need tuning."
    )
