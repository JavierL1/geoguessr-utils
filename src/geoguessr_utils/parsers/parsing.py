"""
Pure text-parsing utilities for OCR output.
"""

import difflib
import re
from typing import Literal, TypedDict

NumberMode = Literal["int", "float"]


def extract_numbers(text: str | None, mode: NumberMode = "float") -> int | float | None:
    """Strip every non-numeric character and return the number.

    Handles common OCR artifacts like '5O1O' -> '5010' by treating 'O' as '0'
    before stripping. Returns None if the cleaned string has no digits.
    """
    if text is None:
        return None

    # Common OCR confusions before stripping
    cleaned = text.replace("O", "0").replace("o", "0")
    cleaned = re.sub(r"[^\d\.]", "", cleaned)

    if not cleaned:
        return None

    try:
        return int(float(cleaned)) if mode == "int" else float(cleaned)
    except ValueError:
        return None


def match_username(
    ocr_text: str | None, known_players: list[str], cutoff: float = 0.4
) -> str | None:
    """Fuzzy-match OCR output against the known player roster.

    Returns the closest match above `cutoff`, or None if nothing is close enough.
    A low cutoff (0.4) is intentional — OCR of stylized names is often poor.
    """
    if not ocr_text or not known_players:
        return None

    matches = difflib.get_close_matches(
        ocr_text.upper().strip(), known_players, n=1, cutoff=cutoff
    )
    return matches[0] if matches else None


class Distance(TypedDict):
    amount: float | str | None
    unit: Literal["m", "km"] | None


def parse_distance(text: str | None) -> Distance | None:
    """Parse distance strings like '511 km' or '7.4 km' or 'inf m'.

    Returns None if text is empty, otherwise a Distance dict. The 'inf'
    sentinel is used when the distance shown is the "no guess" marker.
    """
    if not text:
        return None

    cleaned = text.strip().lower()

    unit: Literal["m", "km"] | None = None
    if "km" in cleaned:
        unit = "km"
    elif "m" in cleaned:
        unit = "m"

    amount = extract_numbers(cleaned, mode="float")

    if unit is not None and amount is None:
        return {"amount": "inf", "unit": unit}

    return {"amount": amount, "unit": unit}
