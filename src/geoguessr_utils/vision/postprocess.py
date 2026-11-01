def zero_and_lower(raw: str) -> str:
    return raw.lower().replace("o", "0")


def postprocess_numbers(raw: str) -> str:
    raw = zero_and_lower(raw)
    if raw.startswith("1"):
        raw = raw[1:]
    else:
        raw = raw

    return raw
