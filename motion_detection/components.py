"""
Bright-pixel component detection, merging, and scoring.

Pipeline:
  1. find_bright_components  — threshold top-N% pixels to a binary mask.
  2. merge_and_filter        — dilate to join adjacent blobs, remove outliers by area.
  3. best_component          — score every surviving blob and return the single best one.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from frame_analysis import find_bright_components  # project-root module

from .config import (
    INTENSITY_WEIGHT,
    MAX_AREA,
    MAX_DISTANCE,
    MERGE_RADIUS,
    MIN_AREA,
    PROXIMITY_WEIGHT,
    TOP_PERCENT,
)


def detect_mask(gray: np.ndarray) -> np.ndarray:
    """Return a binary mask of the top-N% brightest pixels.

    Args:
        gray: Single-channel uint8 grayscale image.

    Returns:
        uint8 mask (0 or 1) of the same spatial size.
    """
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, mask, *_ = find_bright_components(gray_bgr, top_percent=TOP_PERCENT)
    return mask


def merge_and_filter(
    mask: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Merge adjacent bright clusters and filter by area.

    Adjacent blobs are joined via morphological dilation before connected-
    component labelling.  Components outside [MIN_AREA, MAX_AREA] are dropped.

    Args:
        mask: Binary uint8 mask (0 or 1) from detect_mask().

    Returns:
        A tuple (num_labels, labels, stats, centroids, valid_label_indices)
        as returned by cv2.connectedComponentsWithStats, plus the list of
        label indices that passed the area filter (background label 0 excluded).
    """
    if MERGE_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * MERGE_RADIUS + 1, 2 * MERGE_RADIUS + 1)
        )
        merged = cv2.dilate(mask, kernel)
    else:
        merged = mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        merged, connectivity=8
    )
    valid = [
        lbl for lbl in range(1, num_labels)
        if MIN_AREA <= stats[lbl, cv2.CC_STAT_AREA] <= MAX_AREA
    ]
    return num_labels, labels, stats, centroids, valid


def best_component(
    valid_labels: List[int],
    centroids:    np.ndarray,
    stats:        np.ndarray,
    labels_map:   np.ndarray,
    gray:         np.ndarray,
    predicted:    Optional[Tuple[float, float]],
) -> Optional[int]:
    """Score every candidate component and return the index of the best one.

    Each component is scored by a weighted combination of:
      - Intensity  : mean gray value of the component pixels, normalised to [0, 1].
      - Proximity  : closeness to the Kalman-predicted position, computed as
                     1 / (1 + dist / MAX_DISTANCE).  Falls back to 1.0 for all
                     components when no prior track exists.

    Args:
        valid_labels: Label IDs that passed the area filter.
        centroids:    Centroid array from connectedComponentsWithStats.
        stats:        Stats array from connectedComponentsWithStats.
        labels_map:   Per-pixel label array from connectedComponentsWithStats.
        gray:         Grayscale image used to compute mean intensity.
        predicted:    Camera-motion + velocity prediction (x, y), or None if
                      no track exists yet.

    Returns:
        Index into valid_labels of the highest-scoring component, or None if
        valid_labels is empty.
    """
    if not valid_labels:
        return None

    scores: List[float] = []
    for lbl in valid_labels:
        pixel_vals     = gray[labels_map == lbl]
        norm_intensity = float(pixel_vals.mean()) / 255.0

        if predicted is not None:
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            dist   = ((cx - predicted[0]) ** 2 + (cy - predicted[1]) ** 2) ** 0.5
            norm_proximity = 1.0 / (1.0 + dist / MAX_DISTANCE)
        else:
            norm_proximity = 1.0

        scores.append(INTENSITY_WEIGHT * norm_intensity + PROXIMITY_WEIGHT * norm_proximity)

    return scores.index(max(scores))
