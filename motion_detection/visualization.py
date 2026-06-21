"""
Visualisation utilities for the motion tracking pipeline.

Provides a fixed colour palette for track identities and a draw_frame()
function that renders component bounding boxes, track boxes, and a HUD
overlay onto a copy of the current video frame.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .tracker import MotionCompensatedTracker

# BGR colours cycled over track IDs
PALETTE: List[Tuple[int, int, int]] = [
    (255,  80,  80),
    ( 80, 255,  80),
    ( 80,  80, 255),
    (255, 255,  80),
    (255,  80, 255),
    ( 80, 255, 255),
    (255, 160,  80),
    (160,  80, 255),
    ( 80, 160, 255),
]


def color_for(track_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for the given track ID.

    Args:
        track_id: Integer track identifier.

    Returns:
        (B, G, R) colour tuple.
    """
    return PALETTE[track_id % len(PALETTE)]


def draw_frame(
    frame:        np.ndarray,
    active:       Dict[int, Tuple[float, float]],
    tracker:      MotionCompensatedTracker,
    stats:        np.ndarray,
    labels:       np.ndarray,
    valid_labels: List[int],
    frame_idx:    int,
    total:        int,
    M:            Optional[np.ndarray],
) -> np.ndarray:
    """Render tracking annotations onto a copy of the current frame.

    Draws three layers:
      - Grey rectangle : raw detection bounding box from the component stage.
      - Coloured solid rectangle (thickness 2) : active (matched) track box.
      - Coloured thin rectangle : occluded track box at the Kalman-predicted
        position, labelled with "T{id} lost:{age}".
    A HUD line in the top-left corner shows the frame index and track counts.

    Args:
        frame:        BGR video frame to annotate.
        active:       Dict mapping track_id → current centroid for matched tracks.
        tracker:      Live tracker (used to access all tracks, including occluded).
        stats:        Component stats array from connectedComponentsWithStats.
        labels:       Per-pixel label array from connectedComponentsWithStats.
        valid_labels: Label indices that survived the area filter.
        frame_idx:    Current frame index (for the HUD).
        total:        Total number of frames in the video (for the HUD).
        M:            GMC affine matrix (not used for drawing, kept for signature
                      consistency with the main loop).

    Returns:
        Annotated copy of frame (the original is not modified).
    """
    out = frame.copy()

    # Raw detection box (grey)
    for lbl in valid_labels:
        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(out, (x, y), (x + w, y + h), (120, 120, 120), 1)

    all_tracks = tracker.all_tracks()
    for tid, track in all_tracks.items():
        col = color_for(tid)

        if tid in active:
            # Matched this frame — solid coloured box
            x, y, w, h = track.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
            cv2.putText(
                out, f"T{tid}",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
            )
        else:
            # Occluded — thin box at Kalman-predicted centroid
            px, py = track.centroid
            bw, bh = track.bbox[2], track.bbox[3]
            x = int(px - bw / 2)
            y = int(py - bh / 2)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 1)
            cv2.putText(
                out, f"T{tid} lost:{track.age}",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
            )

    # HUD
    occluded = len(all_tracks) - len(active)
    cv2.putText(
        out,
        f"Frame {frame_idx}/{total}  active:{len(active)}  occluded:{occluded}",
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    return out
