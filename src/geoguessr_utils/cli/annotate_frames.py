import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from geoguessr_utils.layout_registry import LAYOUT_REGISTRY
from geoguessr_utils.layouts import Layout
from geoguessr_utils.parsers import parsing
from geoguessr_utils.player_result import PlayerResult

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively annotate frames to create test fixtures."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Directory containing the frames"
    )
    parser.add_argument(
        "-l", "--layout", choices=LAYOUT_REGISTRY.keys(), default="2026b"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing JSON files"
    )
    return parser.parse_args()


def open_image(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=True)
        else:
            subprocess.run(["xdg-open", path], check=True)
    except Exception:
        print(f"⚠️ Could not open image viewer. Please open manually: {path}")


def _prompt_int(prompt: str) -> int | None:
    while True:
        val = input(prompt).strip()
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            print("Invalid input. Please enter an integer.")


def _prompt_distance(side: str) -> parsing.Distance | None:
    print(f"\n[{side}] Distance:")
    amount_str = input("  Amount (or 'inf', or Enter for None): ").strip()
    if not amount_str:
        return None

    amount: float | str
    if amount_str.lower() == "inf":
        amount = "inf"
    else:
        try:
            amount = float(amount_str)
        except ValueError:
            print("  Invalid amount, defaulting to None.")
            return None

    unit_str = input("  Unit (km/m, or Enter for None): ").strip().lower()

    # Restrict the unit to the exact Literals the TypedDict expects
    unit: Literal["m", "km"] | None = None
    if unit_str == "km":
        unit = "km"
    elif unit_str == "m":
        unit = "m"
    elif unit_str:
        print(
            f"  Warning: '{unit_str}' is not a valid unit (km/m). Defaulting to None."
        )

    # Return the explicitly typed Distance dictionary
    return {"amount": amount, "unit": unit}


def _prompt_username(usernames: list[str], side: str) -> str | None:
    print(f"\n--- Select {side} Username ---")
    print("0: None / Unreadable")
    for i, name in enumerate(usernames, start=1):
        print(f"{i}: {name}")

    while True:
        val = input(f"[{side}] Select number (0-{len(usernames)}): ").strip()
        if not val:
            return None
        try:
            choice = int(val)
            if choice == 0:
                return None
            if 1 <= choice <= len(usernames):
                return usernames[choice - 1]
            print(f"Please enter a number between 0 and {len(usernames)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def prompt_player(side: str, layout: Layout) -> PlayerResult:
    """Prompts the user for all features supported by the given layout."""
    print(f"\n{'=' * 40}")
    print(f" {side.upper()} PLAYER")
    print(f"{'=' * 40}")

    # Initialize all fields to None by default
    result: PlayerResult = {
        "match_score": None,
        "username": None,
        "round_score": None,
        "distance": None,
        "health_points": None,
    }

    features = layout.supported_features

    # Dynamically prompt based on the layout's supported features
    if "match_score" in features:
        result["match_score"] = _prompt_int(
            f"[{side}] Match Score (0, 1, 2, or Enter): "
        )

    if "username" in features:
        result["username"] = _prompt_username(layout.known_usernames, side)

    if "round_score" in features:
        result["round_score"] = _prompt_int(f"[{side}] Round Score (or Enter): ")

    if "distance" in features:
        result["distance"] = _prompt_distance(side)

    if "health_points" in features:
        result["health_points"] = _prompt_int(f"[{side}] Health Points (or Enter): ")

    return result


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input)

    if not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    layout_cls = LAYOUT_REGISTRY[args.layout]["layout"]
    layout = layout_cls()

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"Found {len(image_paths)} images to annotate using {layout_cls.__name__}.")
    print(f"Supported features: {', '.join(layout.supported_features)}")

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            json_path = image_path.with_suffix(".json")

            if json_path.exists() and not args.overwrite:
                overwrite = input(
                    f"\n[{idx}/{len(image_paths)}] {image_path.name} exists. "
                    "Overwrite? (y/N): "
                )
                if overwrite.lower() != "y":
                    continue

            print(f"\n[{idx}/{len(image_paths)}] Opening {image_path.name}...")
            open_image(image_path)

            left_data = prompt_player("left", layout)
            right_data = prompt_player("right", layout)

            golden_data = {"left": left_data, "right": right_data}
            json_path.write_text(json.dumps(golden_data, indent=2) + "\n")
            print(f"✅ Saved {json_path.name}")

    except KeyboardInterrupt:
        print("\n\nAnnotation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
