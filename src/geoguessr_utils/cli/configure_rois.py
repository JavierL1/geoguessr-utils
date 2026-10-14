import argparse
import os
import sys
from pathlib import Path

import cv2

from geoguessr_utils.config import ROI, ROI_NAMES, AppConfig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively setup and update all pipeline ROIs in config.json "
            "using a score screen snapshot."
        )
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        required=True,
        help="Path to the snapshot image or video frame.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json file to update.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not decode image '{args.image}'.", file=sys.stderr)
        sys.exit(1)

    config = AppConfig.load(Path(args.config))

    print("\n==================================================")
    print("          INTERACTIVE ROI CONFIGURATOR          ")
    print("==================================================")
    print("Instructions:")
    print(" 1. A window will pop up sequentially for each ROI.")
    print(" 2. Drag a bounding box around the target element.")
    print(" 3. Press ENTER or SPACE to confirm.")
    print(" 4. Press 'c' to skip an ROI and keep its existing value.\n")

    for roi_name in ROI_NAMES:
        window_name = f"Configuring [{roi_name}] - Press ENTER to save, 'c' to skip"

        x, y, w, h = cv2.selectROI(
            window_name, image, showCrosshair=True, fromCenter=False
        )
        cv2.destroyAllWindows()

        if w == 0 or h == 0:
            print(f"Skipped '{roi_name}'.")
            continue

        config.ROIS[roi_name] = ROI(x=x, y=y, w=w, h=h)

        print(f"Saved '{roi_name}': [ {x}, {y}, {w}, {h} ]")

        cropped = image[y : y + h, x : x + w]
        if cropped.size > 0:
            cv2.imshow(f"Preview: {roi_name} (Any key for next)", cropped)
            cv2.waitKey(400)
            cv2.destroyAllWindows()

    config.save(Path(args.config))

    print("\n--------------------------------------------------")
    print(f"All ROIs successfully updated and saved to '{args.config}'!")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
