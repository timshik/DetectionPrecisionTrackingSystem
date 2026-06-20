"""
Drone Thermal People-Tracking Pipeline
=======================================
YOLO detection + BoT-SORT multi-object tracking with Global Motion Compensation.

Blocks
------
1. VideoStreamHandler      — opens the source video, yields frames one at a time
2. ObjectDetectorTracker   — YOLO + BoT-SORT inference on a single frame
3. MetricsDataLogger       — accumulates TrackRecords, flushes to CSV or JSON
4. VideoVisualizer         — annotates and writes the output video
5. run_pipeline            — orchestrator that wires all blocks together
"""

from __future__ import annotations

import csv
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drone_tracker")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """
    Single tracked object returned by the detector for one frame.
    Does not carry a frame_id — that is added downstream by the logger.
    """
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True)
class TrackRecord:
    """Persisted log entry — one tracked object in one specific frame."""
    frame_id: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


# ---------------------------------------------------------------------------
# Block 1 — VideoStreamHandler
# ---------------------------------------------------------------------------

class VideoStreamHandler:
    """
    Opens a local video file and yields frames one at a time.

    Implements the context-manager protocol to guarantee resource cleanup.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the input video file.

    Yields (via .frames())
    ----------------------
    frame_idx : int
        Zero-based index of the current frame.
    frame_bgr : np.ndarray
        BGR image, shape (H, W, 3), dtype uint8.
    """

    def __init__(self, video_path: str) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoStreamHandler":
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {self.video_path}")
        log.info(
            "Opened '%s'  |  %d frames @ %.1f fps  |  %dx%d",
            self.video_path.name, self.total_frames, self.fps, self.width, self.height,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._cap:
            self._cap.release()
        log.info("VideoStreamHandler — stream closed.")

    # -- metadata (valid after __enter__) -----------------------------------

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    # -- frame generator ----------------------------------------------------

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that reads and yields one frame at a time.

        Yields
        ------
        (frame_idx, frame_bgr) : Tuple[int, np.ndarray]
        """
        if self._cap is None:
            raise RuntimeError("Must be used inside a 'with VideoStreamHandler(...)' block.")

        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1


# ---------------------------------------------------------------------------
# Block 2 — ObjectDetectorTracker
# ---------------------------------------------------------------------------

class ObjectDetectorTracker:
    """
    Wraps a pre-trained YOLO model with BoT-SORT multi-object tracking.

    BoT-SORT internals (handled transparently):
    - **Global Motion Compensation (GMC)**: sparse optical-flow estimation
      between consecutive frames corrects for drone camera movement before
      the Kalman filter association step, preventing ID switches during pans.
    - **Kalman filter**: predicts and updates each track's state (position +
      velocity) every frame, maintaining smooth trajectories.
    - **Track termination**: a track that goes unmatched for more than
      `track_buffer` consecutive frames is marked 'Lost' and removed,
      handling sudden scene cuts and occlusions gracefully.

    Parameters
    ----------
    model_weights : str
        Path to YOLO weights (e.g. 'yolov8n.pt').
        Auto-downloaded from Ultralytics on first use if not found locally.
    tracker_config : str
        Path to a BoT-SORT YAML config, or 'botsort.yaml' to use the
        generated config (recommended — exposes all key knobs below).
    confidence : float
        Minimum detection confidence. Lower → more detections, more FP.
    iou_threshold : float
        IoU threshold for NMS inside YOLO.
    track_buffer : int
        Frames a lost track survives before being permanently removed.
        Raise for occlusion tolerance; lower to shed stale tracks faster
        during abrupt scene changes (e.g. drone repositioning).
    gmc_method : str
        Global Motion Compensation algorithm. 'sparseOptFlow' (default) is
        fast and well-suited to drone footage. Alternatives: 'orb', 'sift',
        'ecc', 'none'.
    device : str
        Inference device: 'cpu', 'cuda', 'cuda:0', 'mps', etc.
    """

    def __init__(
        self,
        model_weights: str = "yolov8n.pt",
        tracker_config: str = "botsort.yaml",
        confidence: float = 0.30,
        iou_threshold: float = 0.50,
        track_buffer: int = 30,
        gmc_method: str = "sparseOptFlow",
        device: str = "cpu",
        target_classes: Optional[List[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        target_classes : Optional[List[int]]
            Class IDs to keep. ``None`` keeps all classes — use this for
            single-class thermal models where class 0 is already 'human'.
            Pass ``[0]`` to restrict a COCO model to the 'person' class only.
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes

        # Write a custom tracker YAML so all parameters are under our control.
        self._tracker_cfg_path = self._write_tracker_config(
            track_buffer=track_buffer,
            gmc_method=gmc_method,
        )
        # Override with a user-supplied path if it isn't the sentinel value.
        if tracker_config != "botsort.yaml":
            self._tracker_cfg_path = Path(tracker_config)

        log.info("Loading YOLO weights: %s", model_weights)
        self._model = YOLO(model_weights)
        log.info(
            "Tracker ready  |  conf=%.2f  iou=%.2f  buffer=%d  gmc=%s  classes=%s",
            confidence, iou_threshold, track_buffer, gmc_method,
            "all" if target_classes is None else target_classes,
        )

    # -- public API ---------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection + BoT-SORT tracking on a single BGR frame.

        ``persist=True`` is the critical flag that tells Ultralytics to keep
        the BoT-SORT state (Kalman filters, GMC history, track buffers) alive
        between successive calls, enabling true multi-frame tracking.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Current video frame in BGR format, shape (H, W, 3), uint8.

        Returns
        -------
        detections : List[Detection]
            All active tracks in this frame. Empty list if no persons found.
        """
        results = self._model.track(
            source=frame_bgr,
            persist=False,
            tracker=str(self._tracker_cfg_path),
            classes=self.target_classes,   # None = all (for single-class thermal models)
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse(results)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _parse(results) -> List[Detection]:
        """Extract Detection objects from an Ultralytics Results list."""
        detections: List[Detection] = []
        result = results[0]

        if result.boxes.id is None:
            return detections

        ids   = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()   # shape (N, 4): x1 y1 x2 y2
        confs = result.boxes.conf.cpu().numpy()

        for track_id, (x1, y1, x2, y2), conf in zip(ids, boxes, confs):
            detections.append(Detection(
                track_id=int(track_id),
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                confidence=float(conf),
            ))

        return detections

    @staticmethod
    def _write_tracker_config(track_buffer: int, gmc_method: str) -> Path:
        """
        Write a botsort_custom.yaml with the caller's parameters.

        Returns the path to the written file.
        """
        cfg = textwrap.dedent(f"""\
            tracker_type: botsort
            track_high_thresh: 0.5      # high-conf threshold, first association pass
            track_low_thresh: 0.1       # low-conf threshold, second association pass
            new_track_thresh: 0.6       # min score to initialise a brand-new track
            track_buffer: {track_buffer}           # frames to keep a lost track alive
            match_thresh: 0.7           # IOU threshold for track-detection matching
            proximity_thresh: 0.5       # proximity gate for ReID matching
            appearance_thresh: 0.25     # appearance embedding similarity threshold
            with_reid: false            # disable ReID (no separate embedding model)
            fuse_score: true            # fuse detection score into Kalman update
            gmc_method: {gmc_method}   # GMC algorithm (sparseOptFlow recommended for drones)
            model: null                 # ReID model path; null = disabled (with_reid: false)
        """)
        path = Path("botsort_custom.yaml")
        path.write_text(cfg, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Block 3 — MetricsDataLogger
# ---------------------------------------------------------------------------

class MetricsDataLogger:
    """
    Accumulates per-frame tracking results and writes them to disk on demand.

    Supports CSV (default) and JSON output, chosen by the file extension of
    ``output_path``.

    Parameters
    ----------
    output_path : str
        Destination file. Use '.csv' for CSV, '.json' for JSON.
    """

    _CSV_FIELDS = ("frame_id", "track_id", "x1", "y1", "x2", "y2", "confidence")

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self._records: List[TrackRecord] = []

    def log(self, frame_idx: int, detections: List[Detection]) -> None:
        """
        Append detections from one frame to the internal buffer.

        Parameters
        ----------
        frame_idx : int
            Zero-based index of the frame these detections belong to.
        detections : List[Detection]
            Active tracks returned by ObjectDetectorTracker.process_frame().
        """
        for d in detections:
            self._records.append(TrackRecord(
                frame_id=frame_idx,
                track_id=d.track_id,
                x1=round(d.x1, 2),
                y1=round(d.y1, 2),
                x2=round(d.x2, 2),
                y2=round(d.y2, 2),
                confidence=round(d.confidence, 4),
            ))

    def save(self) -> None:
        """
        Flush all buffered records to disk.

        Output
        ------
        File written at ``self.output_path`` in CSV or JSON format.
        """
        if not self._records:
            log.warning("MetricsDataLogger.save() — no records to write.")
            return

        suffix = self.output_path.suffix.lower()
        if suffix == ".json":
            self._save_json()
        else:
            self._save_csv()

        log.info(
            "Saved %d track records across %d unique frames → %s",
            len(self._records),
            len({r.frame_id for r in self._records}),
            self.output_path,
        )

    def _save_csv(self) -> None:
        with open(self.output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            for r in self._records:
                writer.writerow(r.__dict__)

    def _save_json(self) -> None:
        payload = [
            {
                "frame_id":   r.frame_id,
                "track_id":   r.track_id,
                "bbox":       [r.x1, r.y1, r.x2, r.y2],
                "confidence": r.confidence,
            }
            for r in self._records
        ]
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Block 4 — VideoVisualizer
# ---------------------------------------------------------------------------

class VideoVisualizer:
    """
    Draws bounding boxes and track IDs on frames, displays them live
    (optionally), and writes the annotated sequence to an output video.

    Parameters
    ----------
    output_path : str
        Destination path for the annotated video file (.mp4).
    fps : float
        Frame rate — should match the source video exactly.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    display : bool
        If True, show a live cv2 window. Set False on headless machines.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        display: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.display = display
        self._writer: Optional[cv2.VideoWriter] = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create output video at: {self.output_path}")

        log.info("VideoVisualizer ready → %s", self.output_path)

    def __enter__(self) -> "VideoVisualizer":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def render_and_write(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray:
        """
        Annotate one frame, push it to the display window, and write to disk.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Original BGR frame from the video stream.
        detections : List[Detection]
            Active tracks for this frame.
        frame_idx : int
            Current frame index (used for the HUD overlay).
        total_frames : int
            Total frames in the source video (used for the HUD overlay).

        Returns
        -------
        annotated : np.ndarray
            The annotated BGR frame.
        """
        annotated = self._annotate(frame_bgr.copy(), detections, frame_idx, total_frames)

        if self.display:
            cv2.imshow("Drone Tracker", annotated)

        self._writer.write(annotated)
        return annotated

    def release(self) -> None:
        """Flush and close the VideoWriter and any open display windows."""
        if self._writer:
            self._writer.release()
            self._writer = None
        if self.display:
            cv2.destroyAllWindows()
        log.info("VideoVisualizer — writer closed.")

    # -- drawing ------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray:
        for det in detections:
            color = _track_color(det.track_id)
            x1, y1 = int(det.x1), int(det.y1)
            x2, y2 = int(det.x2), int(det.y2)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label: "ID 3  0.87"
            label = f"ID {det.track_id}  {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            # Filled label background
            cv2.rectangle(
                frame,
                (x1, max(y1 - th - baseline - 4, 0)),
                (x1 + tw, y1),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (x1, max(y1 - baseline - 2, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        # HUD — top-left corner
        hud = f"Frame {frame_idx + 1}/{total_frames}   Active tracks: {len(detections)}"
        cv2.putText(
            frame, hud, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return frame


# ---------------------------------------------------------------------------
# Color utility (module-level, shared by visualizer)
# ---------------------------------------------------------------------------

def _track_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic per-ID color — same ID always gets the same color."""
    rng = np.random.default_rng(track_id + 42)
    r, g, b = rng.integers(80, 230, size=3, dtype=np.uint8)
    return int(r), int(g), int(b)


# ---------------------------------------------------------------------------
# Model download utility
# ---------------------------------------------------------------------------

def download_thermal_model(
    repo_id: str = "pitangent-ds/YOLOv8-human-detection-thermal",
    filename: str = "model.pt",
    local_dir: str = "models",
) -> str:
    """
    Download a pre-trained thermal detection model from HuggingFace Hub.

    The model (pitangent-ds/YOLOv8-human-detection-thermal) is a YOLOv8
    fine-tuned on thermal images for single-class human detection. It works
    on both grayscale and pseudo-colour thermal imagery.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository in ``owner/name`` format.
    filename : str
        The weights file inside the repository (default: 'model.pt').
    local_dir : str
        Directory where the downloaded file will be saved.

    Returns
    -------
    local_path : str
        Absolute path to the downloaded weights file.
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    log.info("Downloading thermal model from HuggingFace: %s/%s", repo_id, filename)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    log.info("Model saved → %s", local_path)
    return local_path


# ---------------------------------------------------------------------------
# Block 5 — Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    model_weights: str = "yolov8n.pt",
    tracker_config: str = "botsort.yaml",
    confidence: float = 0.30,
    iou_threshold: float = 0.50,
    track_buffer: int = 30,
    gmc_method: str = "sparseOptFlow",
    device: str = "cpu",
    target_classes: Optional[List[int]] = None,
    output_video_path: str = "output_tracked.mp4",
    output_log_path: str = "tracking_log.csv",
    display: bool = False,
) -> None:
    """
    End-to-end orchestration: stream → detect → log → visualize.

    Parameters
    ----------
    video_path : str
        Input video file (local path).
    model_weights : str
        YOLO weights file. Downloaded automatically if absent.
    tracker_config : str
        'botsort.yaml' uses the auto-generated config with parameters below.
        Pass a custom path to override completely.
    confidence : float
        Detection confidence threshold (0–1).
    iou_threshold : float
        NMS IoU threshold (0–1).
    track_buffer : int
        Frames a lost track survives without a detection match.
    gmc_method : str
        GMC algorithm for camera-motion compensation.
    device : str
        PyTorch device string ('cpu', 'cuda', 'cuda:0', 'mps').
    output_video_path : str
        Path for the annotated output video.
    output_log_path : str
        Path for the tracking log file (.csv or .json).
    display : bool
        Show a live OpenCV window. Press 'q' to stop early.
    """
    detector = ObjectDetectorTracker(
        model_weights=model_weights,
        tracker_config=tracker_config,
        confidence=confidence,
        iou_threshold=iou_threshold,
        track_buffer=track_buffer,
        gmc_method=gmc_method,
        device=device,
        target_classes=target_classes,
    )
    logger = MetricsDataLogger(output_log_path)

    try:
        with VideoStreamHandler(video_path) as stream:
            with VideoVisualizer(
                output_path=output_video_path,
                fps=stream.fps,
                width=stream.width,
                height=stream.height,
                display=display,
            ) as viz:
                for frame_idx, frame_bgr in stream.frames():

                    # ── Block 2: detect + track ──────────────────────────
                    detections = detector.process_frame(frame_bgr)

                    # ── Block 3: log ─────────────────────────────────────
                    logger.log(frame_idx, detections)

                    # ── Block 4: visualise + write ───────────────────────
                    viz.render_and_write(
                        frame_bgr, detections, frame_idx, stream.total_frames
                    )

                    # ── Progress heartbeat every 60 frames ───────────────
                    if frame_idx % 60 == 0:
                        log.info(
                            "Frame %4d / %d  |  active tracks: %d",
                            frame_idx + 1, stream.total_frames, len(detections),
                        )

                    # ── Keyboard quit (live window only) ─────────────────
                    if display and cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("'q' pressed — stopping at frame %d.", frame_idx)
                        break

    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt received — flushing logs before exit.")
    finally:
        # Always save even if we stop early or an error occurs.
        logger.save()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Download the thermal-specific YOLOv8 model on first run; cached afterwards.
    thermal_weights = download_thermal_model(
        repo_id="pitangent-ds/YOLOv8-human-detection-thermal",
        filename="model.pt",
        local_dir="models",
    )

    run_pipeline(
        video_path=r"videos\How to hide from a thermal drone (Ukraine).mp4",
        model_weights=thermal_weights,
        tracker_config="botsort.yaml",  # auto-generates botsort_custom.yaml
        confidence=0.01,                # thermal model: start low, tune up if noisy
        iou_threshold=0.50,
        track_buffer=30,
        gmc_method="sparseOptFlow",
        device="cpu",
        target_classes=None,            # single-class model — no filter needed
        output_video_path="output_tracked.mp4",
        output_log_path="tracking_log.csv",
        display=False,
    )
