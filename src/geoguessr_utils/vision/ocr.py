"""Thin Tesseract wrapper. Knows which config string to use for which pipeline."""

CONFIG_DISTANCE = r"--psm 7 -c tessedit_char_whitelist=0123456789kmKM,. "
CONFIG_STANDARD = "--psm 7 -c tessedit_char_whitelist=oO0123456789"
CONFIG_USERNAME = "--psm 7"
CONFIG_HEALTH = "--psm 7 -c tessedit_char_whitelist=0123456789"
