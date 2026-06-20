import cv2
import numpy as np
from video_reader import VideoReader
from frame_analysis import find_bright_components


def _match_components(centroids_prev: np.ndarray, centroids_curr: np.ndarray,
                      max_distance: float) -> dict:
    """Greedy nearest-centroid matching. Returns {curr_label: prev_label}."""
    matches = {}
    used_prev = set()

    for ci in range(1, len(centroids_curr)):
        best_dist = max_distance
        best_pi = None
        for pi in range(1, len(centroids_prev)):
            if pi in used_prev:
                continue
            d = np.linalg.norm(centroids_curr[ci] - centroids_prev[pi])
            if d < best_dist:
                best_dist = d
                best_pi = pi
        if best_pi is not None:
            matches[ci] = best_pi
            used_prev.add(best_pi)

    return matches


class ComponentTracker:
    """
    Assigns persistent IDs to connected components across frames.
    A component is considered "tracked" (returned) only when it matches
    a component from the previous frame within max_distance pixels.
    """

    def __init__(self, max_distance: float = 15.0):
        self.max_distance = max_distance
        self._next_id = 1
        self._prev_centroids = None
        self._prev_id_map: dict = {}  # prev_label -> track_id

    def update(self, centroids: np.ndarray) -> dict:
        """
        centroids: output of connectedComponentsWithStats (index 0 = background).
        Returns: {curr_label: track_id} only for components matched from previous frame.
        """
        if self._prev_centroids is None:
            id_map = {label: self._alloc() for label in range(1, len(centroids))}
            self._prev_centroids = centroids
            self._prev_id_map = id_map
            return {}

        matches = _match_components(self._prev_centroids, centroids, self.max_distance)

        id_map = {}
        tracked = {}

        for curr_label, prev_label in matches.items():
            track_id = self._prev_id_map[prev_label]
            id_map[curr_label] = track_id
            tracked[curr_label] = track_id

        for label in range(1, len(centroids)):
            if label not in id_map:
                id_map[label] = self._alloc()

        self._prev_centroids = centroids
        self._prev_id_map = id_map
        return tracked

    def _alloc(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid


def _id_to_color(track_id: int) -> list:
    rng = np.random.default_rng(track_id)
    return rng.integers(80, 255, size=3, dtype=np.uint8).tolist()


def draw_tracked_frame(frame_rgb: np.ndarray, labels: np.ndarray,
                       stats: np.ndarray, centroids: np.ndarray,
                       tracked: dict, frame_idx: int, total_frames: int) -> np.ndarray:
    output = frame_rgb.copy()

    for label, track_id in tracked.items():
        color = _id_to_color(track_id)

        # Filled component overlay at 50% opacity
        component_mask = labels == label
        overlay = output.copy()
        overlay[component_mask] = color
        output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

        # Bounding box + track ID
        x, y, w, h, _ = stats[label]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        cv2.putText(output, f"#{track_id}", (x, max(y - 4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Centroid dot
        cx, cy = int(centroids[label][0]), int(centroids[label][1])
        cv2.circle(output, (cx, cy), 3, color, -1)

    # Frame counter overlay
    cv2.putText(output, f"Frame {frame_idx + 1}/{total_frames}  tracked: {len(tracked)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return output


if __name__ == "__main__":
    VIDEO_PATH = r"videos\How to hide from a thermal drone (Ukraine).mp4"
    OUTPUT_PATH = "tracked_components.mp4"
    NUM_FRAMES = 1200
    TOP_PERCENT = 0.01
    MAX_DISTANCE = 15.0

    reader = VideoReader(VIDEO_PATH).load()

    h, w = reader.frames.shape[1], reader.frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, reader.fps, (w, h))

    tracker = ComponentTracker(max_distance=MAX_DISTANCE)

    for frame_idx in range(NUM_FRAMES):
        frame = reader[frame_idx]
        _, _, num_labels, labels, stats, centroids, threshold = find_bright_components(
            frame, top_percent=TOP_PERCENT
        )

        tracked = tracker.update(centroids)

        vis = draw_tracked_frame(frame, labels, stats, centroids, tracked, frame_idx, NUM_FRAMES)

        # VideoWriter expects BGR
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        print(f"Frame {frame_idx + 1}/{NUM_FRAMES} — components: {num_labels - 1}, "
              f"tracked: {len(tracked)}, threshold: {threshold:.1f}")

    writer.release()
    print(f"\nSaved: {OUTPUT_PATH}")
