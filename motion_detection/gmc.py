"""
Global Motion Compensation (GMC) using sparse optical flow.

The camera motion between consecutive frames is estimated as a 2×3 affine
matrix via Lucas-Kanade optical flow on background (non-bright) pixels,
followed by RANSAC-based affine fitting.  This matrix is used by the tracker
to remove apparent object displacement caused purely by camera movement.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import GMC_MAX_CORNERS, GMC_MIN_DISTANCE, GMC_QUALITY


def apply_affine_to_point(
    pt: Tuple[float, float],
    M: np.ndarray,
) -> Tuple[float, float]:
    """Apply a 2×3 affine matrix to a single (x, y) point.

    Args:
        pt: Source point (x, y).
        M:  2×3 affine transformation matrix.

    Returns:
        Transformed point (nx, ny).
    """
    x, y = pt
    return (
        M[0, 0] * x + M[0, 1] * y + M[0, 2],
        M[1, 0] * x + M[1, 1] * y + M[1, 2],
    )


class CameraMotionEstimator:
    """Estimates per-frame affine camera motion using sparse optical flow.

    Features are detected on background (non-bright) pixels only, so that
    moving foreground objects do not contaminate the camera-motion estimate.

    Usage:
        gmc = CameraMotionEstimator()
        M, inlier_ratio = gmc.update(gray_frame, fg_mask)
    """

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None

    def update(
        self,
        curr_gray: np.ndarray,
        fg_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Estimate the affine transform from the previous frame to the current one.

        Args:
            curr_gray: Current frame as a single-channel grayscale image.
            fg_mask:   uint8 mask where 255 marks bright foreground pixels that
                       should be excluded from the flow computation.

        Returns:
            A tuple (M, inlier_ratio) where:
              M             — 2×3 float32 affine matrix (prev → curr coords), or
                              None on the very first call or when estimation fails.
              inlier_ratio  — Fraction of flow points accepted as RANSAC inliers
                              in [0, 1].  0.0 when M is None.  A high value
                              indicates confident camera motion (not a scene cut).
        """
        if self._prev_gray is None:
            self._prev_gray = curr_gray.copy()
            return None, 0.0

        bg_mask = cv2.bitwise_not(fg_mask)
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=GMC_MAX_CORNERS,
            qualityLevel=GMC_QUALITY,
            minDistance=GMC_MIN_DISTANCE,
            mask=bg_mask,
        )

        M: Optional[np.ndarray] = None
        inlier_ratio = 0.0

        if prev_pts is not None and len(prev_pts) >= 4:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, curr_gray, prev_pts, None
            )
            good_prev = prev_pts[status.ravel() == 1]
            good_curr = curr_pts[status.ravel() == 1]

            if len(good_prev) >= 4:
                M, inlier_mask = cv2.estimateAffinePartial2D(
                    good_prev,
                    good_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )
                if M is not None and inlier_mask is not None:
                    inlier_ratio = float(inlier_mask.sum()) / len(good_prev)

        self._prev_gray = curr_gray.copy()
        return M, inlier_ratio
