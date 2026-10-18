"""Typed wrappers for OpenCV's untyped image I/O functions."""

from pathlib import Path

import cv2
import numpy as np


def read(path: str | Path) -> np.ndarray | None:
    """cv2.imread with proper return type.

    Returns the BGR image as a numpy array, or None if the file
    could not be read (missing, corrupt, unsupported format).
    """
    image = cv2.imread(str(path))
    if image is None:
        return None

    return image  # type: ignore[no-any-return]
