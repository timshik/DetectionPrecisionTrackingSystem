"""
Configuration constants for the motion detection pipeline.

All tunable parameters live here so the rest of the code stays
free of magic numbers.  Import individual names or the whole module:

    from motion_detection.config import MAX_DISTANCE, TRACK_BUFFER
"""

# ── I/O ───────────────────────────────────────────────────────────────────────
# Paths are resolved relative to the project root (parent of this package),
# so the pipeline works regardless of the working directory it is launched from.

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

VIDEO_PATH   = str(_PROJECT_ROOT / "videos" / "How to hide from a thermal drone (Ukraine).mp4")
OUTPUT_VIDEO = str(_PROJECT_ROOT / "motion_tracked.mp4")
OUTPUT_CSV   = str(_PROJECT_ROOT / "motion_tracking_log.csv")

# ── Bright-pixel detection ────────────────────────────────────────────────────

TOP_PERCENT  = 0.01   # fraction of brightest pixels treated as foreground
MERGE_RADIUS = 8      # dilation radius (px) used to merge adjacent bright clusters
MIN_AREA     = 10     # minimum component area (px²) — smaller blobs are noise
MAX_AREA     = 1500   # maximum component area (px²) — larger blobs are background

# ── Component scoring ─────────────────────────────────────────────────────────
# Weights must sum to 1.0.
# INTENSITY_WEIGHT  : prefer brighter components
# PROXIMITY_WEIGHT  : prefer components close to the Kalman prediction

INTENSITY_WEIGHT = 0.1
PROXIMITY_WEIGHT = 0.9

# ── Tracking ──────────────────────────────────────────────────────────────────

MAX_DISTANCE      = 40.0   # max px distance (Kalman prediction → detection) for a match
REATTACH_DISTANCE = 120.0  # looser fallback threshold used after direction/speed changes
TRACK_BUFFER      = 50     # frames a track can go unmatched before it is deleted

# ── Scene change detection ────────────────────────────────────────────────────

SCENE_CHANGE_THRESHOLD = 30.0  # mean absolute pixel diff that flags a scene cut
SCENE_CHANGE_COOLDOWN  = 30    # frames to skip checking after a reset (avoids cascades)
GMC_INLIER_RATIO_MIN   = 0.3   # if GMC inlier ratio >= this value it is camera motion, not a cut

# ── Global Motion Compensation (sparse optical flow) ─────────────────────────

GMC_MAX_CORNERS  = 300   # maximum feature points tracked per frame
GMC_QUALITY      = 0.01  # minimum quality for goodFeaturesToTrack
GMC_MIN_DISTANCE = 7     # minimum pixel distance between tracked corners

# ── Kalman filter ─────────────────────────────────────────────────────────────

KF_PROCESS_NOISE_POS = 1.0    # position state noise (px²/frame) — higher = faster adaptation
KF_PROCESS_NOISE_VEL = 0.1    # velocity state noise (px²/frame²)
KF_MEASUREMENT_NOISE = 10.0   # detection noise (px²) — higher = smoother, slower response
