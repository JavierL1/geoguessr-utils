import argparse
import json
import os
import sys

import cv2

SIDES = ["left", "right"]
FEATURES = ["match_score", "round_score", "distance", "username", "health_points"]


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
        "-f",
        "--filepath",
        type=str,
        default="configured_rois.json",
        help="Path to json file to write the rois to.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not decode image '{args.image}'.", file=sys.stderr)
        sys.exit(1)

    config = {}

    print("\n==================================================")
    print("          INTERACTIVE ROI CONFIGURATOR          ")
    print("==================================================")
    print("Instructions:")
    print(" 1. A window will pop up sequentially for each ROI.")
    print(" 2. Drag a bounding box around the target element.")
    print(" 3. Press ENTER or SPACE to confirm.")
    print(" 4. Press 'c' to skip an ROI and keep its existing value.\n")

    for side in SIDES:
        config[side] = {}

        for feature in FEATURES:
            roi_name = f"{side} {feature}"
            window_name = f"Configuring [{roi_name}] - Press ENTER to save, 'c' to skip"

            x, y, w, h = cv2.selectROI(
                window_name, image, showCrosshair=True, fromCenter=False
            )
            cv2.destroyAllWindows()

            if w == 0 or h == 0:
                print(f"Skipped '{roi_name}'.")
                continue

            config[side][feature] = [x, y, w, h]

            print(f"Saved '{roi_name}': [ {x}, {y}, {w}, {h} ]")

            cropped = image[y : y + h, x : x + w]
            if cropped.size > 0:
                cv2.imshow(f"Preview: {roi_name} (Any key for next)", cropped)
                cv2.waitKey(400)
                cv2.destroyAllWindows()

    with open(args.filepath, "w") as f:
        json.dump(config, f, indent=4)

    print("\n--------------------------------------------------")
    print(f"All ROIs successfully updated and saved to '{args.filepath}'!")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
