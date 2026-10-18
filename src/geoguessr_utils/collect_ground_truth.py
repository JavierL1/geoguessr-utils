import argparse
import glob
import json
import os
import cv2


def prompt_player_data(player_side):
    print(f"\n--- {player_side.upper()} PLAYER ---")

    try:
        match_score = int(input("  Overall Score (int): ").strip())
    except ValueError:
        match_score = 0

    username = input("  Username (str): ").strip()

    try:
        round_score = int(input("  Round Score (int): ").strip())
    except ValueError:
        round_score = 0

    try:
        dist_amount = float(
            input("  Distance Amount (float, e.g. 1.0 or 78.0): ").strip()
        )
    except ValueError:
        dist_amount = 0.0

    dist_unit = input("  Distance Unit ('m' or 'km'): ").strip()

    try:
        health = int(input("  Health Points (int): ").strip())
    except ValueError:
        health = 0

    return {
        "match_score": match_score,
        "username": username,
        "round_score": round_score,
        "distance": {"amount": dist_amount, "unit": dist_unit},
        "health": health,
    }


def collect_ground_truth(input_dir, output_json):
    filepaths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    if not filepaths:
        print(f"No JPG files found in '{input_dir}'.")
        return

    # Load existing annotations if resuming
    results = []
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing annotations from {output_json}")
        except json.JSONDecodeError:
            results = []

    processed_images = {r.get("source_image") for r in results}

    print(f"Found {len(filepaths)} total images in {input_dir}.")
    print(
        "Instructions: Type values when prompted. Press Ctrl+C at any time to exit safely.\n"
    )

    for path in filepaths:
        filename = os.path.basename(path)
        if filename in processed_images:
            continue

        frame = cv2.imread(path)
        if frame is None:
            print(f"Could not read {path}, skipping.")
            continue

        # Display the image window
        cv2.imshow("Ground Truth Annotation Tool", frame)
        cv2.waitKey(1)  # Refresh OpenCV window frame

        print("=" * 50)
        print(f"Annotating Image: {filename}")
        print("=" * 50)

        try:
            left_data = prompt_player_data("left")
            right_data = prompt_player_data("right")

            frame_data = {
                "source_image": filename,
                "left_player": left_data,
                "right_player": right_data,
            }

            results.append(frame_data)

            # Save incrementally after every image
            with open(output_json, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved! Progress updated in {output_json}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress and exiting...")
            break

    cv2.destroyAllWindows()
    print(f"\nAnnotation session ended. Total saved entries: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive ground truth annotation tool for Geoguessr score screens."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="./output_frames",
        help="Directory containing the JPG images",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ground_truth.json",
        help="Output JSON file path for ground truths",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found.")
        exit(1)

    collect_ground_truth(args.input, args.output)
