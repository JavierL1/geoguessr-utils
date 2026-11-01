from geoguessr_utils.layouts import Layout2026A, Layout2026B
from geoguessr_utils.vision.detectors import (
    ScoreScreenDetector2026A,
    ScoreScreenDetector2026B,
)

LAYOUT_REGISTRY = {
    "2026a": {"layout": Layout2026A, "detector": ScoreScreenDetector2026A},
    "2026b": {"layout": Layout2026B, "detector": ScoreScreenDetector2026B},
}
