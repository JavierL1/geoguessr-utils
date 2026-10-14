import argparse
import json
import os
import re


def normalize_string(val):
    """
    Normalizes strings for robust comparison:
    - Converts to string if not already
    - Strips leading/trailing whitespace
    - Converts to lowercase
    - Removes extra inner whitespace and standard punctuation if needed
    """
    if not isinstance(val, str):
        return val
    # Strip whitespace, lowercase, and collapse multiple spaces
    normalized = re.sub(r"\s+", " ", val.strip().lower())
    return normalized


def compare_values(gt_val, pred_val, path=""):
    """
    Recursively compares ground truth values to prediction values.
    Only checks keys present in the ground truth. Normalizes strings during comparison.
    """
    differences = []

    # If ground truth is a dictionary, ensure prediction is also a dictionary and check its keys
    if isinstance(gt_val, dict):
        if not isinstance(pred_val, dict):
            return [
                f"  - Type mismatch at '{path}': Ground Truth is dict, but Prediction is {type(pred_val).__name__}"
            ]

        # ONLY iterate over keys present in the ground truth
        for key in sorted(gt_val.keys()):
            new_path = f"{path}.{key}" if path else key
            if key not in pred_val:
                differences.append(f"  - Key '{new_path}' missing in predictions.")
            else:
                differences.extend(compare_values(gt_val[key], pred_val[key], new_path))

    # If ground truth is a list
    elif isinstance(gt_val, list):
        if not isinstance(pred_val, list):
            return [
                f"  - Type mismatch at '{path}': Ground Truth is list, but Prediction is {type(pred_val).__name__}"
            ]

        if len(gt_val) != len(pred_val):
            differences.append(
                f"  - Length mismatch at '{path}': Ground Truth has {len(gt_val)} items, Prediction has {len(pred_val)} items."
            )
        else:
            for idx, (gt_item, pred_item) in enumerate(zip(gt_val, pred_val)):
                differences.extend(compare_values(gt_item, pred_item, f"{path}[{idx}]"))

    # Scalar values (strings, numbers, booleans, etc.)
    else:
        # Normalize strings if applicable
        norm_gt = normalize_string(gt_val)
        norm_pred = normalize_string(pred_val)

        if norm_gt != norm_pred:
            differences.append(
                f"  - Mismatch at '{path}': Ground Truth = {gt_val!r} | Predicted = {pred_val!r}"
            )

    return differences


def compare_jsons(ground_truth_path, prediction_path):
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file '{ground_truth_path}' not found.")
        return
    if not os.path.exists(prediction_path):
        print(f"Error: Prediction file '{prediction_path}' not found.")
        return

    with open(ground_truth_path, "r") as f:
        gt_data = json.load(f)

    with open(prediction_path, "r") as f:
        pred_data = json.load(f)

    # Index by source_image for fast lookup
    gt_map = {
        item.get("source_image"): item for item in gt_data if "source_image" in item
    }
    pred_map = {
        item.get("source_image"): item for item in pred_data if "source_image" in item
    }

    all_gt_images = sorted(gt_map.keys())

    total_checked = 0
    total_matches = 0
    mismatches = []

    print(
        f"\nComparing {len(all_gt_images)} ground truth images against predictions...\n"
    )

    for img_name in all_gt_images:
        if img_name not in pred_map:
            mismatches.append(
                f"[{img_name}] Present in Ground Truth, but missing from Predictions."
            )
            continue

        total_checked += 1
        gt_obj = gt_map[img_name]
        pred_obj = pred_map[img_name]

        diffs = compare_values(gt_obj, pred_obj)
        if diffs:
            mismatches.append(f"[{img_name}] Mismatches found:")
            mismatches.extend(diffs)
        else:
            total_matches += 1

    # Print Report
    print("=" * 60)
    print(" VALIDATION REPORT SUMMARY (Normalized & GT-Scoped)")
    print("=" * 60)
    print(f"Total GT Images Checked : {total_checked}")
    print(f"Exact Matches           : {total_matches}")
    print(f"Discrepancies Found     : {total_checked - total_matches}")
    print("=" * 60)

    if mismatches:
        print("\nDETAILED DIFFERENCES:")
        for line in mismatches:
            print(line)
    else:
        print("\nAll compared entries match the ground truth perfectly!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare algorithm predictions against ground truth JSON with string normalization."
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        default="ground_truth.json",
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "-p",
        "--predictions",
        default="tournament_data.json",
        help="Path to algorithm output JSON file",
    )

    args = parser.parse_args()
    compare_jsons(args.ground_truth, args.predictions)
