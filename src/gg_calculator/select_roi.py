import argparse
import os
import sys
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Interactively select a Region of Interest (ROI) from a video frame or image snapshot."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the snapshot image or video frame.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not decode image '{args.image}'.", file=sys.stderr)
        sys.exit(1)

    window_name = "Select Score ROI - Press ENTER or SPACE to confirm, 'c' to cancel"

    # Open native OpenCV window to select ROI: returns (x, y, width, height)
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("\nSelection canceled or invalid ROI chosen.")
        sys.exit(0)

    # Verification Step: Show crop preview to verify correctness
    cropped = image[y : y + h, x : x + w]
    cv2.imshow("Selected ROI Crop (Press any key to close)", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n--- ROI Extraction Complete ---")
    print(f"Raw Coordinates: x={x}, y={y}, width={w}, height={h}")
    print("\nRun your extractor with these arguments:")
    print(f"  --roi {x} {y} {w} {h}")


if __name__ == "__main__":
    main()
