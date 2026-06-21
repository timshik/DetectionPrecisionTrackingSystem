"""
Motion-compensated multi-object tracker with Kalman filter state estimation.

Each active object is represented by a Track that owns a KalmanTrack instance.
Per-frame update follows six steps:

  1. GMC + predict  : Apply camera motion to every KF state, then predict.
  2. Match          : Greedily match predictions to detections (tight threshold).
  3. KF update      : Fuse matched detections with the Kalman prediction.
  4. Propagate lost : Advance unmatched tracks by their prediction; age them.
  5. Re-attach      : Try a looser threshold before spawning a new track.
  6. Spawn          : Create a new track only if nothing absorbed the detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MAX_DISTANCE, REATTACH_DISTANCE, TRACK_BUFFER
from .gmc import apply_affine_to_point
from .kalman import KalmanTrack


@dataclass
class Track:
    """State of a single tracked object.

    Attributes:
        track_id: Unique integer identifier assigned at track creation.
        kf:       Kalman filter managing position and velocity estimates.
        bbox:     Last matched bounding box (x, y, w, h) in image pixels.
        age:      Frames elapsed since the last successful detection match.
                  0 means the track was matched this frame.
        history:  Ordered list of centroid positions (x, y) over time.
    """

    track_id: int
    kf:       KalmanTrack
    bbox:     Tuple[int, int, int, int] = (0, 0, 0, 0)
    age:      int = 0
    history:  List[Tuple[float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Current best-estimate centroid from the Kalman filter."""
        return self.kf.position

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current best-estimate velocity (px/frame) from the Kalman filter."""
        return self.kf.velocity


class MotionCompensatedTracker:
    """Kalman-filter tracker with Global Motion Compensation.

    Designed for single-object tracking in thermal drone footage:
    at most one detection is passed per frame (the highest-scoring bright
    component), so the tracker focuses on maintaining a single persistent
    track across occlusions and scene motion.

    Attributes:
        _tracks:   Live tracks keyed by track_id.
        _next_id:  Monotonically increasing ID counter.

    Example:
        tracker = MotionCompensatedTracker()
        active = tracker.update(detections, bboxes, M)
        for tid, (cx, cy) in active.items():
            print(f"Track {tid} at ({cx:.1f}, {cy:.1f})")
    """

    def __init__(self) -> None:
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._last_predicted: Dict[int, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, M: Optional[np.ndarray]) -> Dict[int, Tuple[float, float]]:
        """Step 1 — Apply GMC and advance every Kalman filter by one time step.

        Must be called once per frame before update().  The predicted positions
        are returned so the caller can use them (e.g. for component scoring)
        before committing any measurement.

        Args:
            M: 2×3 affine camera-motion matrix from GMC, or None.

        Returns:
            Dict mapping track_id → predicted (x, y) for every live track.
        """
        self._last_predicted = self._predict_all(M)
        return dict(self._last_predicted)

    def update(
        self,
        detections: List[Tuple[float, float]],
        bboxes:     List[Tuple[int, int, int, int]],
    ) -> Dict[int, Tuple[float, float]]:
        """Step 2 — Match detections to predictions and run the Kalman update.

        Must be called after predict().  Uses the predicted positions stored
        by the most recent predict() call.

        Args:
            detections: List of detected centroids [(x, y), ...].
            bboxes:     Matching bounding boxes [(x, y, w, h), ...], same order.

        Returns:
            Dict mapping track_id → (cx, cy) for every track matched this frame.
            Tracks that are alive but unmatched (occluded) are not included.
        """
        predicted = self._last_predicted
        matched, unmatched_dets = self._match(detections, predicted)
        active = self._update_matched(matched, detections, bboxes)
        self._propagate_unmatched(predicted, matched)
        active.update(self._reattach(unmatched_dets, detections, bboxes, matched))
        self._spawn_new(
            [di for di in unmatched_dets if di not in {v for v in matched.values()}],
            detections,
            bboxes,
        )
        return active

    def all_tracks(self) -> Dict[int, Track]:
        """Return all live tracks (matched and occluded)."""
        return self._tracks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_all(
        self, M: Optional[np.ndarray]
    ) -> Dict[int, Tuple[float, float]]:
        """Apply GMC to every track's KF state, then run the predict step."""
        predicted: Dict[int, Tuple[float, float]] = {}
        for tid, track in self._tracks.items():
            if M is not None:
                track.kf.apply_gmc(M)
            predicted[tid] = track.kf.predict()
        return predicted

    def _match(
        self,
        detections: List[Tuple[float, float]],
        predicted:  Dict[int, Tuple[float, float]],
    ) -> Tuple[Dict[int, int], List[int]]:
        """Greedy nearest-neighbour matching within MAX_DISTANCE.

        Returns:
            matched       : {track_id: detection_index}
            unmatched_dets: detection indices that were not claimed
        """
        unmatched_dets = list(range(len(detections)))
        matched: Dict[int, int] = {}

        for tid in list(predicted.keys()):
            if not unmatched_dets:
                break
            px, py = predicted[tid]
            best_dist, best_di = float("inf"), -1
            for di in unmatched_dets:
                dx, dy = detections[di]
                dist = ((px - dx) ** 2 + (py - dy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_di = dist, di
            if best_dist <= MAX_DISTANCE:
                matched[tid] = best_di
                unmatched_dets.remove(best_di)

        return matched, unmatched_dets

    def _update_matched(
        self,
        matched:    Dict[int, int],
        detections: List[Tuple[float, float]],
        bboxes:     List[Tuple[int, int, int, int]],
    ) -> Dict[int, Tuple[float, float]]:
        """Run the Kalman update step for each matched track."""
        active: Dict[int, Tuple[float, float]] = {}
        for tid, di in matched.items():
            cx, cy = detections[di]
            fx, fy = self._tracks[tid].kf.update(cx, cy)
            self._tracks[tid].bbox = bboxes[di]
            self._tracks[tid].history.append((fx, fy))
            self._tracks[tid].age = 0
            active[tid] = (fx, fy)
        return active

    def _propagate_unmatched(
        self,
        predicted: Dict[int, Tuple[float, float]],
        matched:   Dict[int, int],
    ) -> None:
        """Advance unmatched tracks by their prediction and age them out."""
        for tid in list(predicted.keys()):
            if tid in matched:
                continue
            self._tracks[tid].history.append(predicted[tid])
            self._tracks[tid].age += 1
            if self._tracks[tid].age > TRACK_BUFFER:
                del self._tracks[tid]

    def _reattach(
        self,
        unmatched_dets: List[int],
        detections:     List[Tuple[float, float]],
        bboxes:         List[Tuple[int, int, int, int]],
        matched:        Dict[int, int],
    ) -> Dict[int, Tuple[float, float]]:
        """Re-attach detections that missed the tight threshold.

        Uses the track's current centroid (post-predict) with REATTACH_DISTANCE.
        This handles sudden direction or speed changes where the Kalman prediction
        overshot but the object is still nearby.

        Returns:
            Newly matched active tracks {track_id: (cx, cy)}.
        """
        newly_active: Dict[int, Tuple[float, float]] = {}
        still_unmatched: List[int] = []

        for di in unmatched_dets:
            cx, cy = detections[di]
            best_dist, best_tid = float("inf"), -1
            for tid, track in self._tracks.items():
                if tid in matched:
                    continue
                lx, ly = track.centroid
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_tid = dist, tid

            if best_tid != -1 and best_dist <= REATTACH_DISTANCE:
                fx, fy = self._tracks[best_tid].kf.update(cx, cy)
                self._tracks[best_tid].bbox = bboxes[di]
                self._tracks[best_tid].history.append((fx, fy))
                self._tracks[best_tid].age = 0
                newly_active[best_tid] = (fx, fy)
                matched[best_tid] = di
            else:
                still_unmatched.append(di)

        # Replace the original list in-place so _spawn_new sees the remainder
        unmatched_dets[:] = still_unmatched
        return newly_active

    def _spawn_new(
        self,
        unmatched_dets: List[int],
        detections:     List[Tuple[float, float]],
        bboxes:         List[Tuple[int, int, int, int]],
    ) -> None:
        """Create a new track for every detection that could not be re-attached."""
        for di in unmatched_dets:
            cx, cy = detections[di]
            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = Track(
                track_id=new_id,
                kf=KalmanTrack(cx, cy),
                bbox=bboxes[di],
                history=[(cx, cy)],
            )
