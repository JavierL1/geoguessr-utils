"""Thin Tesseract wrapper. Knows which config string to use for which pipeline."""

import pytesseract

from geoguessr_utils.vision import preprocess

CONFIG_DISTANCE = r"--psm 7 -c tessedit_char_whitelist=0123456789kmKM,. "
CONFIG_STANDARD = "--psm 7 -c tessedit_char_whitelist=oO0123456789"
CONFIG_USERNAME = "--psm 7"
CONFIG_HEALTH = "--psm 7 -c tessedit_char_whitelist=0123456789"


def read_distance(crop, debug_name=None) -> str:
    image = preprocess.prepare_distance(crop, debug_name)
    return pytesseract.image_to_string(image, config=CONFIG_DISTANCE).strip()


def read_standard(crop, debug_name=None) -> str:
    image = preprocess.prepare_standard(crop, debug_name)

    raw = pytesseract.image_to_string(image, config=CONFIG_STANDARD).strip().lower()
    raw = raw.replace("o", "0")

    if not raw:
        image_with_context = preprocess.add_synthetic_context(image)
        raw_context = (
            pytesseract.image_to_string(image_with_context, config=CONFIG_STANDARD)
            .strip()
            .lower()
        )
        raw_context = raw_context.replace("o", "0")

        if raw_context.startswith("1"):
            raw = raw_context[1:]
        else:
            raw = raw_context

    return raw


def read_username(crop, debug_name=None) -> str:
    image = preprocess.prepare_username(crop, debug_name)
    return pytesseract.image_to_string(image, config=CONFIG_USERNAME).strip()


def read_match_score(crop, debug_name=None) -> str:
    return preprocess.detect_digit_by_geometry(crop, debug_name)


def read_health_points(crop, debug_name=None) -> str:
    image = preprocess.prepare_health(crop, debug_name)
    if image is None:
        return ""
    return pytesseract.image_to_string(image, config=CONFIG_HEALTH).strip()
