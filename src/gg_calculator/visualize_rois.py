# /// script
# dependencies = [
#     "opencv-python",
# ]
# ///

import argparse
import json
import cv2
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Visualize ROIs from config on a sample image.")
    parser.add_argument("-i", "--image", required=True, help="Path to the sample frame (PNG/JPG)")
    parser.add_argument("-c", "--config", default="config.json", help="Path to JSON config file")
    parser.add_argument("-o", "--output", default="roi_debug_output.jpg", help="Path to save the annotated image")
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.config, 'r') as f:
        config_data = json.load(f)
        
    rois = config_data.get("ROIS", {})
    if not rois:
        print("Error: No 'ROIS' key found in the config file.", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image '{args.image}'.", file=sys.stderr)
        sys.exit(1)

    # Draw each ROI
    for name, coords in rois.items():
        if len(coords) != 4:
            print(f"Warning: ROI '{name}' does not have exactly 4 coordinates. Skipping.")
            continue
            
        x, y, w, h = coords
        
        # Draw the red rectangle (BGR format: 0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add a text label just above the rectangle
        label_y = y - 5 if y - 5 > 15 else y + 15
        cv2.putText(
            image, name, (x, label_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

    cv2.imwrite(args.output, image)
    print(f"Success! Annotated image saved to '{args.output}'.")

if __name__ == "__main__":
    main()