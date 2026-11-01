"""
Visualize the ROIs defined in a specific Layout class on a sample image.

This is useful for tuning the coordinates of a layout or debugging why
a certain feature is not being detected/extracted correctly.
"""

import argparse
import sys
from pathlib import Path

import cv2

from geoguessr_utils.layout_registry import LAYOUT_REGISTRY
from geoguessr_utils.vision.roi import ROI


def _draw_roi(
    image,
    roi: ROI,
    label: str,
    color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    """Helper to draw a single ROI box and label."""
    # Draw the rectangle (BGR format)
    cv2.rectangle(image, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)

    # Add a text label just above the rectangle
    label_y = roi.y - 5 if roi.y - 5 > 15 else roi.y + 15

    # Draw a black background for the text so it's readable on any background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(
        image,
        (roi.x, label_y - text_h - 4),
        (roi.x + text_w, label_y + 2),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        label,
        (roi.x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        text_color,
        1,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Layout ROIs on a sample image."
    )
    parser.add_argument(
        "-i", "--image", required=True, help="Path to the sample frame (PNG/JPG)"
    )
    parser.add_argument(
        "-l",
        "--layout",
        choices=LAYOUT_REGISTRY.keys(),
        default="2026b",
        help="Which UI layout to visualize (default: 2026b)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="layout_roi_debug.jpg",
        help="Path to save the annotated image",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image file '{image_path}' not found.", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image '{image_path}'.", file=sys.stderr)
        sys.exit(1)

    # 1. Instantiate the layout
    layout_cls = LAYOUT_REGISTRY[args.layout]["layout"]

    layout = layout_cls()
    print(f"Loaded {layout_cls.__name__}")

    # 2. Draw Discovery ROIs (Red boxes)
    for name, roi in layout.score_screen_discovery_rois.items():
        _draw_roi(
            image, roi, f"disc: {name}", color=(0, 0, 255), text_color=(0, 0, 255)
        )

    # 3. Draw Feature ROIs (Green for Left, Blue for Right)
    for side in ["left", "right"]:
        feature_rois = getattr(layout.score_screen_feature_rois, side)

        # We use vars() to iterate over the dataclass fields dynamically
        for feature_name, roi in vars(feature_rois).items():
            if feature_name.startswith("_"):
                continue  # Skip internal dataclass attributes

            label = f"{side[0]}: {feature_name}"  # e.g., "l: round_score"

            if side == "left":
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue

            _draw_roi(image, roi, label, color=color, text_color=color)

    # Save the output
    cv2.imwrite(args.output, image)
    print(f"Success! Annotated image saved to '{args.output}'.")


if __name__ == "__main__":
    main()
