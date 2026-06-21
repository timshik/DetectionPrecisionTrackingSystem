"""
Frame preprocessing and scene-change detection.

Scene cuts are detected using a two-gate approach:
  1. GMC inlier ratio gate : a high fraction of matched optical-flow points
     means the frames share the same scene content (camera motion); skip.
  2. Frame-difference gate : when GMC is unreliable, compare the warp-aligned
     frames.  A large residual after camera compensation indicates a true cut.
"""

from typing import Optional

import cv2
import numpy as np

from .config import (
    GMC_INLIER_RATIO_MIN,
    SCENE_CHANGE_THRESHOLD,
)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR video frame to a single-channel grayscale image.

    Args:
        frame_bgr: Raw BGR frame from cv2.VideoCapture.

    Returns:
        uint8 grayscale image of the same spatial resolution.
    """
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def scene_changed(
    prev_gray:    Optional[np.ndarray],
    curr_gray:    np.ndarray,
    M:            Optional[np.ndarray],
    inlier_ratio: float,
) -> bool:
    """Decide whether a true scene cut occurred between two consecutive frames.

    The function is robust to fast camera pans and zooms by using the GMC
    inlier ratio as its primary signal before falling back to pixel difference.

    Decision logic:
      - inlier_ratio >= GMC_INLIER_RATIO_MIN → camera motion, return False.
      - Otherwise compute the mean absolute pixel difference.  If M is available
        prev_gray is warped to align with curr_gray before diffing (removes the
        camera-motion component).  If the residual exceeds SCENE_CHANGE_THRESHOLD
        return True.

    Args:
        prev_gray:    Grayscale previous frame, or None on the very first call.
        curr_gray:    Grayscale current frame.
        M:            2×3 GMC affine matrix (prev → curr), or None.
        inlier_ratio: Fraction of optical-flow inliers from GMC [0, 1].

    Returns:
        True if a scene cut is detected.
    """
    if prev_gray is None:
        return False

    # High inlier ratio → consistent feature tracking → same scene, camera moving
    if inlier_ratio >= GMC_INLIER_RATIO_MIN:
        return False

    # GMC uncertain: compare frames after aligning for camera motion if possible
    if M is not None:
        h, w = prev_gray.shape[:2]
        warped = cv2.warpAffine(prev_gray, M, (w, h))
        diff = float(
            np.abs(curr_gray.astype(np.int16) - warped.astype(np.int16)).mean()
        )
    else:
        diff = float(
            np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16)).mean()
        )

    return diff > SCENE_CHANGE_THRESHOLD
