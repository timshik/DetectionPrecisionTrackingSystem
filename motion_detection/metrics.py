"""
Tier 1 consistency metrics for the motion detection pipeline.

These metrics validate tracker behaviour without requiring ground truth.
They measure internal consistency: a tracker following the real object
should produce smooth, stable values across all four signals.

Metrics
-------
innovation  : Euclidean distance between the KF prediction and the actual
              detection (pixels).  Low and stable = prediction is accurate.
velocity    : ||vx, vy|| from the Kalman state (pixels/frame).  Should be
              physically plausible and change gradually.
bbox_area   : w × h of the matched bounding box (pixels²).  A single object
              at roughly constant distance should have stable area.
intensity   : Mean pixel brightness of the tracked component [0–255].  Should
              be consistently high (we are tracking the hottest region).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── Per-frame data container ──────────────────────────────────────────────────

@dataclass
class FrameMetrics:
    """All Tier 1 metrics for a single tracked frame.

    Attributes:
        frame:      Video frame index.
        track_id:   ID of the active track.
        cx:         Kalman-filtered centroid x (pixels).
        cy:         Kalman-filtered centroid y (pixels).
        innovation: Distance between KF prediction and detection (pixels).
                    0.0 when the frame had no detection (occluded).
        velocity:   Magnitude of KF velocity state ||vx, vy|| (pixels/frame).
        bbox_area:  Matched bounding box area w × h (pixels²).
        intensity:  Mean pixel brightness of the tracked component [0, 255].
    """
    frame:      int
    track_id:   int
    cx:         float
    cy:         float
    innovation: float
    velocity:   float
    bbox_area:  int
    intensity:  float


# ── Individual metric functions ───────────────────────────────────────────────

def compute_innovation(
    predicted: Optional[Tuple[float, float]],
    detected:  Optional[Tuple[float, float]],
) -> float:
    """Return the Euclidean distance between prediction and detection.

    Args:
        predicted: KF-predicted position (x, y) before the update step.
        detected:  Detected centroid (x, y), or None if no detection this frame.

    Returns:
        Distance in pixels, or 0.0 if either input is None.
    """
    if predicted is None or detected is None:
        return 0.0
    dx = predicted[0] - detected[0]
    dy = predicted[1] - detected[1]
    return float((dx ** 2 + dy ** 2) ** 0.5)


def compute_velocity(vx: float, vy: float) -> float:
    """Return the Euclidean magnitude of the KF velocity vector.

    Args:
        vx: Horizontal velocity component (pixels/frame).
        vy: Vertical velocity component (pixels/frame).

    Returns:
        Speed in pixels/frame.
    """
    return float((vx ** 2 + vy ** 2) ** 0.5)


def compute_intensity(
    gray:       np.ndarray,
    labels_map: np.ndarray,
    label:      int,
) -> float:
    """Return the mean pixel brightness of a connected component.

    Args:
        gray:       Single-channel uint8 grayscale frame.
        labels_map: Per-pixel label array from connectedComponentsWithStats.
        label:      Component label to measure.

    Returns:
        Mean brightness in [0, 255], or 0.0 if the component has no pixels.
    """
    pixels = gray[labels_map == label]
    return float(pixels.mean()) if len(pixels) > 0 else 0.0


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(records: List[FrameMetrics]) -> None:
    """Print a formatted Tier 1 metrics summary to stdout.

    Args:
        records: List of FrameMetrics collected across all frames.
    """
    if not records:
        print("No metric records to summarize.")
        return

    def _stats(values: np.ndarray, label: str, unit: str = "") -> None:
        print(f"  {label}:")
        print(f"    mean = {values.mean():7.2f}{unit}")
        print(f"    std  = {values.std():7.2f}{unit}")
        print(f"    min  = {values.min():7.2f}{unit}  |  max = {values.max():.2f}{unit}")

    total_frames = records[-1].frame + 1 if records else 0
    detected     = [r for r in records if r.innovation > 0]
    innovations  = np.array([r.innovation for r in detected]) if detected else np.array([])
    velocities   = np.array([r.velocity   for r in records])
    areas        = np.array([r.bbox_area  for r in records])
    intensities  = np.array([r.intensity  for r in records])

    print("\n" + "=" * 52)
    print("  Tier 1 Consistency Metrics")
    print("=" * 52)
    print(f"  Tracked frames   : {len(records)} / {total_frames}")
    print(f"  Detection rate   : {len(detected)} / {len(records)}"
          f"  ({100 * len(detected) / len(records):.1f}%)\n")

    if innovations.size:
        _stats(innovations, "Innovation  (prediction error)", " px")
    else:
        print("  Innovation: no frames with a detection.")

    print()
    _stats(velocities, "Velocity magnitude", " px/frame")

    print()
    cv = areas.std() / areas.mean() if areas.mean() > 0 else 0.0
    print(f"  Bounding box area:")
    print(f"    mean = {areas.mean():7.0f} px²")
    print(f"    std  = {areas.std():7.0f} px²")
    print(f"    CV   = {cv:7.3f}  (lower = more stable size)")

    print()
    _stats(intensities, "Component intensity", "")
    print("=" * 52)
