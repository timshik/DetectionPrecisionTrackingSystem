"""
Per-track Kalman filter with Global Motion Compensation (GMC) support.

State vector : [x, y, vx, vy]  — position and velocity in image coordinates.
Observation  : [x, y]          — detected centroid position.

Before each predict step the caller must apply the GMC affine matrix so that
camera-induced apparent motion is subtracted from the state.  This keeps vx/vy
as the object's own velocity relative to the scene, not the camera.
"""

from typing import Tuple

import cv2
import numpy as np

from .config import KF_MEASUREMENT_NOISE, KF_PROCESS_NOISE_POS, KF_PROCESS_NOISE_VEL


class KalmanTrack:
    """Kalman filter for a single tracked object.

    The constant-velocity model assumes the object moves with roughly constant
    speed between frames, but the process noise allows it to accelerate/decelerate.
    GMC compensation removes camera motion so only the object's own motion is
    modelled.

    Args:
        cx: Initial x position (pixels).
        cy: Initial y position (pixels).

    Example:
        kf = KalmanTrack(cx=100.0, cy=200.0)
        kf.apply_gmc(M)        # remove camera motion
        px, py = kf.predict()  # where do we expect the object?
        fx, fy = kf.update(detected_x, detected_y)  # fuse with measurement
    """

    def __init__(self, cx: float, cy: float) -> None:
        self._kf = cv2.KalmanFilter(4, 2)

        # State transition: x += vx, y += vy each frame
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # We only observe position, not velocity
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        self._kf.processNoiseCov = np.diag([
            KF_PROCESS_NOISE_POS,
            KF_PROCESS_NOISE_POS,
            KF_PROCESS_NOISE_VEL,
            KF_PROCESS_NOISE_VEL,
        ]).astype(np.float32)

        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASUREMENT_NOISE
        self._kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
        self._kf.statePost           = np.array(
            [[cx], [cy], [0.0], [0.0]], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply_gmc(self, M: np.ndarray) -> None:
        """Shift the internal state by the camera affine transform.

        Must be called before predict() each frame so that vx/vy represents
        the object's own motion, not the sum of object + camera motion.

        Args:
            M: 2×3 affine matrix from CameraMotionEstimator (prev → curr).
        """
        x, y, vx, vy = self._kf.statePost.ravel()
        # Apply full affine (translation + rotation) to position
        nx  = M[0, 0] * x  + M[0, 1] * y  + M[0, 2]
        ny  = M[1, 0] * x  + M[1, 1] * y  + M[1, 2]
        # Apply only rotation/scale to velocity (no translation component)
        nvx = M[0, 0] * vx + M[0, 1] * vy
        nvy = M[1, 0] * vx + M[1, 1] * vy
        self._kf.statePost = np.array([[nx], [ny], [nvx], [nvy]], dtype=np.float32)

    def predict(self) -> Tuple[float, float]:
        """Advance the filter by one time step and return the predicted position.

        Returns:
            Predicted (x, y) position in image coordinates.
        """
        state = self._kf.predict()
        return float(state[0, 0]), float(state[1, 0])

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        """Correct the filter with a new measurement and return the fused position.

        Args:
            cx: Measured x position of the detected centroid.
            cy: Measured y position of the detected centroid.

        Returns:
            Fused (x, y) position — blend of prediction and measurement.
        """
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        state = self._kf.correct(measurement)
        return float(state[0, 0]), float(state[1, 0])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[float, float]:
        """Current best-estimate position (x, y)."""
        s = self._kf.statePost
        return float(s[0, 0]), float(s[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current best-estimate velocity (vx, vy) in pixels/frame."""
        s = self._kf.statePost
        return float(s[2, 0]), float(s[3, 0])
