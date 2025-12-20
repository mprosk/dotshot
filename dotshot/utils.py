from __future__ import annotations

from typing import Tuple

import cv2
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


def apply_ordered_dither(
    image: np.ndarray, levels: int, strength: float = 1.0
) -> np.ndarray:
    """Apply ordered dithering (Bayer matrix) to a grayscale image.

    This is much faster than error diffusion dithering as it can be fully vectorized.
    Uses a 4x4 Bayer matrix tiled across the image.

    Args:
        image: Grayscale uint8 image (2D array: H, W) or color image (H, W, C).
        levels: Number of quantization levels (e.g., 2 for black/white, 4, 8, etc.).
        strength: Dithering strength multiplier (1.0 = standard, higher = more dithering).

    Returns:
        Grayscale uint8 image with ordered dithering applied (2D array: H, W).
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            img_gray = image[:, :, 0]
    elif image.ndim == 2:
        img_gray = image
    else:
        raise ValueError(
            f"Unsupported image dimensions: {image.ndim}D (expected 2D grayscale)"
        )

    # Convert to float32 for calculations (avoid unnecessary copy if already uint8)
    if img_gray.dtype == np.uint8:
        img_float = img_gray.astype(np.float32, copy=False)
    else:
        img_float = img_gray.astype(np.uint8).astype(np.float32)

    # Validate and clamp levels
    levels = max(2, min(256, int(levels)))
    palette_step = 255.0 / (levels - 1)

    # 4x4 Bayer matrix (standard ordered dithering matrix), normalized to 0-1
    bayer_4x4 = (
        np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        )
        / 16.0
    )

    height, width = img_float.shape

    # Tile Bayer matrix efficiently using np.tile
    reps_h = (height + 3) // 4
    reps_w = (width + 3) // 4
    threshold_matrix = np.tile(bayer_4x4, (reps_h, reps_w))[:height, :width]

    # Scale threshold matrix by quantization step size and strength
    threshold_matrix = (threshold_matrix - 0.5) * palette_step * strength

    # Add threshold to image values
    dithered = img_float + threshold_matrix

    # Quantize directly: calculate index and map to palette value
    # This is O(pixels) instead of O(levels * pixels)
    quantized_indices = np.round(
        np.clip(dithered / palette_step, 0, levels - 1)
    ).astype(np.int32)
    result = quantized_indices.astype(np.float32) * palette_step

    # Convert to uint8 (already clipped by the index calculation)
    result = result.astype(np.uint8)

    # Ensure contiguous output
    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)

    return result
