from __future__ import annotations

from typing import Tuple

import numpy as np


def crop_center_to_aspect(
    image: np.ndarray, aspect_w: int, aspect_h: int
) -> np.ndarray:
    """Center-crop an image to the requested aspect ratio.

    Args:
        image: Input image as a NumPy array (H, W) or (H, W, C).
        aspect_w: Target aspect width component (e.g., 4 for 4:3).
        aspect_h: Target aspect height component (e.g., 3 for 4:3).

    Returns:
        Cropped image with the requested aspect ratio, centered.
    """
    if aspect_w <= 0 or aspect_h <= 0:
        return image

    h: int = int(image.shape[0])
    w: int = int(image.shape[1])
    if h <= 0 or w <= 0:
        return image

    target_w_from_h: int = int(round(h * (float(aspect_w) / float(aspect_h))))
    target_h_from_w: int = int(round(w * (float(aspect_h) / float(aspect_w))))

    # Choose the crop that fits within the image while maximizing area
    if target_w_from_h <= w:
        crop_w: int = max(1, target_w_from_h)
        crop_h: int = h
    else:
        crop_w = w
        crop_h = max(1, target_h_from_w)

    x0: int = max(0, (w - crop_w) // 2)
    y0: int = max(0, (h - crop_h) // 2)
    x1: int = min(w, x0 + crop_w)
    y1: int = min(h, y0 + crop_h)

    cropped = image[y1 - crop_h : y1, x1 - crop_w : x1]
    if not cropped.flags["C_CONTIGUOUS"]:
        cropped = np.ascontiguousarray(cropped)
    return cropped


CROP_MODES: list[Tuple[int, int]] = [
    (4, 3),
    (1, 1),
]


def crop_mode_name(index: int) -> str:
    """Return a short name for the crop mode index."""
    idx = int(index) % len(CROP_MODES)
    w, h = CROP_MODES[idx]
    return f"{w}:{h}"
