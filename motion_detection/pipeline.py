"""
Main tracking pipeline.

Orchestrates all sub-modules for a single video:

  For each frame:
    1. Preprocess        : convert BGR → grayscale.
    2. Detect components : find top-N% bright pixels → connected blobs.
    3. Estimate GMC      : sparse optical flow on background pixels.
    4. Scene change      : reset tracker on true scene cuts.
    5. Score components  : pick the single best blob based on intensity
                           and proximity to the Kalman prediction.
    6. Track             : Kalman predict + update (or propagate if no match).
    7. Render + write    : annotate and save to output video + CSV.

Entry point:
    from motion_detection.pipeline import run
    run()
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .components import best_component, detect_mask, merge_and_filter
from .config import (
    OUTPUT_CSV,
    OUTPUT_VIDEO,
    SCENE_CHANGE_COOLDOWN,
    VIDEO_PATH,
)
from .gmc import CameraMotionEstimator
from .metrics import FrameMetrics, compute_innovation, compute_intensity, compute_velocity, summarize
from .scene import preprocess, scene_changed
from .tracker import MotionCompensatedTracker
from .visualization import draw_frame


def run() -> None:
    """Run the full motion-detection pipeline on VIDEO_PATH.

    Reads the video frame by frame, detects and tracks the brightest
    moving object using GMC + Kalman filtering, and writes:
      - OUTPUT_VIDEO : annotated MP4 video.
      - OUTPUT_CSV   : CSV log with columns [frame, track_id, cx, cy].
    """
    cap   = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    gmc            = CameraMotionEstimator()
    tracker        = MotionCompensatedTracker()
    records:       List[Dict] = []
    metric_records: List[FrameMetrics] = []
    prev_gray:     Optional[np.ndarray] = None
    cooldown       = 0

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        gray = preprocess(frame)

        # Detect bright components
        mask = detect_mask(gray)
        num_labels, labels, stats, centroids, valid_labels = merge_and_filter(mask)

        # Estimate camera motion
        fg_mask            = (mask * 255).astype(np.uint8)
        M, inlier_ratio    = gmc.update(gray, fg_mask)

        # Scene change detection — reset on true cuts, ignore camera pans
        if cooldown > 0:
            cooldown -= 1
        elif scene_changed(prev_gray, gray, M, inlier_ratio):
            print(
                f"  Scene change at frame {frame_idx} "
                f"(inlier_ratio={inlier_ratio:.2f}) — resetting tracks"
            )
            tracker  = MotionCompensatedTracker()
            gmc      = CameraMotionEstimator()
            M        = None
            cooldown = SCENE_CHANGE_COOLDOWN

        prev_gray = gray.copy()

        # Step 1 — Kalman predict (applies GMC internally, returns predicted positions)
        predictions = tracker.predict(M)

        # Step 2 — Select best component using the KF-predicted position
        # Use the primary (longest-history) track's prediction for scoring
        all_tracks = tracker.all_tracks()
        if predictions and all_tracks:
            primary_tid = max(all_tracks, key=lambda tid: len(all_tracks[tid].history))
            predicted_pos = predictions.get(primary_tid)
        else:
            predicted_pos = None

        best_idx = best_component(
            valid_labels, centroids, stats, labels, gray, predicted_pos
        )

        if best_idx is not None:
            best_lbl   = valid_labels[best_idx]
            detections = [(float(centroids[best_lbl][0]), float(centroids[best_lbl][1]))]
            bboxes     = [(
                int(stats[best_lbl, cv2.CC_STAT_LEFT]),
                int(stats[best_lbl, cv2.CC_STAT_TOP]),
                int(stats[best_lbl, cv2.CC_STAT_WIDTH]),
                int(stats[best_lbl, cv2.CC_STAT_HEIGHT]),
            )]
        else:
            detections, bboxes = [], []

        # Step 3 — Kalman update with the selected detection
        active = tracker.update(detections, bboxes)

        # Log matched positions + Tier 1 metrics
        for tid, (cx, cy) in active.items():
            track = tracker.all_tracks()[tid]
            vx, vy = track.velocity

            detection_pos = detections[0] if detections else None
            inno  = compute_innovation(predicted_pos, detection_pos)
            speed = compute_velocity(vx, vy)
            area  = bboxes[0][2] * bboxes[0][3] if bboxes else 0
            inten = (compute_intensity(gray, labels, valid_labels[best_idx])
                     if best_idx is not None else 0.0)

            records.append({
                "frame":      frame_idx,
                "track_id":   tid,
                "cx":         round(cx, 2),
                "cy":         round(cy, 2),
                "innovation": round(inno, 2),
                "velocity":   round(speed, 2),
                "bbox_area":  area,
                "intensity":  round(inten, 2),
            })
            metric_records.append(FrameMetrics(
                frame=frame_idx, track_id=tid,
                cx=round(cx, 2), cy=round(cy, 2),
                innovation=inno, velocity=speed,
                bbox_area=area, intensity=inten,
            ))

        # Render and write frame
        vis = draw_frame(
            frame, active, tracker,
            stats, labels, valid_labels,
            frame_idx, total, M,
        )
        writer.write(vis)

        if frame_idx % 100 == 0:
            print(
                f"  {frame_idx}/{total}"
                f"  detections:{len(detections)}"
                f"  active tracks:{len(active)}"
            )

    cap.release()
    writer.release()

    with open(OUTPUT_CSV, "w", newline="") as f:
        csv_writer = csv.DictWriter(
            f,
            fieldnames=["frame", "track_id", "cx", "cy",
                        "innovation", "velocity", "bbox_area", "intensity"],
        )
        csv_writer.writeheader()
        csv_writer.writerows(records)

    print(f"\nDone.")
    print(f"  Frames processed : {total}")
    print(f"  Track records    : {len(records)}")
    print(f"  Video saved to   : {OUTPUT_VIDEO}")
    print(f"  Log saved to     : {OUTPUT_CSV}")

    summarize(metric_records)
