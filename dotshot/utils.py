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
        image: Grayscale uint8 image (2D array: H, W).
        levels: Number of quantization levels (e.g., 2 for black/white, 4, 8, etc.).
        strength: Dithering strength multiplier (1.0 = standard, higher = more dithering).

    Returns:
        Grayscale uint8 image with ordered dithering applied (2D array: H, W).
    """
    # Ensure grayscale (2D) - convert if needed
    if image.ndim == 3:
        # If color image, convert to grayscale
        if image.shape[2] == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            # Take first channel if multi-channel
            img_gray = image[:, :, 0]
    elif image.ndim == 2:
        img_gray = image
    else:
        raise ValueError(
            f"Unsupported image dimensions: {image.ndim}D (expected 2D grayscale)"
        )

    # Ensure uint8 and contiguous
    if img_gray.dtype != np.uint8:
        img_u8 = img_gray.astype(np.uint8)
    else:
        img_u8 = img_gray.copy()
    if not img_u8.flags["C_CONTIGUOUS"]:
        img_u8 = np.ascontiguousarray(img_u8)

    # Convert to float32 for calculations
    img_float = img_u8.astype(np.float32)

    # Create quantization palette
    if levels <= 1:
        levels = 2
    if levels > 256:
        levels = 256

    palette_step = 255.0 / (levels - 1) if levels > 1 else 255.0
    palette = np.array([i * palette_step for i in range(levels)], dtype=np.float32)

    # 4x4 Bayer matrix (standard ordered dithering matrix)
    bayer_4x4 = np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    # Normalize to 0-1 range
    bayer_4x4 = bayer_4x4 / 16.0

    height, width = img_float.shape
    bayer_h, bayer_w = bayer_4x4.shape

    # Tile the Bayer matrix across the image
    # Create indices for tiling using modulo
    y_indices = np.arange(height) % bayer_h
    x_indices = np.arange(width) % bayer_w
    # Use advanced indexing to create tiled threshold matrix
    threshold_matrix = bayer_4x4[np.ix_(y_indices, x_indices)]

    # Scale threshold matrix by quantization step size and strength
    # This determines how much the threshold affects quantization
    quantization_step = palette_step
    threshold_matrix = (threshold_matrix - 0.5) * quantization_step * strength

    # Add threshold to image values, then quantize
    dithered = img_float + threshold_matrix

    # Quantize: find nearest palette value for each pixel
    # Vectorized quantization using broadcasting
    dithered_flat = dithered.flatten()
    palette_2d = palette.reshape(-1, 1)  # Shape: (levels, 1)
    dithered_2d = dithered_flat.reshape(1, -1)  # Shape: (1, pixels)

    # Compute distances: (levels, pixels)
    distances = np.abs(palette_2d - dithered_2d)
    # Find index of minimum distance for each pixel
    quantized_indices = np.argmin(distances, axis=0)
    # Map to quantized values
    quantized_flat = palette[quantized_indices]

    # Reshape back to original image dimensions
    result = quantized_flat.reshape(height, width)

    # Clip to valid range and convert to uint8
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    # Ensure contiguous
    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)

    return result
