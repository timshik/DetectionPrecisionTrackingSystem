import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.frames = None
        self.fps = None
        self.width = None
        self.height = None

    def load(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pre-allocate: (N, H, W, C) uint8 in RGB
        self.frames = np.empty((total, self.height, self.width, 3), dtype=np.uint8)

        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                self.frames = self.frames[:i]
                break
            self.frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()
        print(f"Loaded {len(self.frames)} frames — shape: {self.frames.shape}, fps: {self.fps}")
        return self

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


if __name__ == "__main__":
    VIDEO_PATH = r"videos\How to hide from a thermal drone (Ukraine).mp4"
    reader = VideoReader(VIDEO_PATH).load()
    print(f"Frame 0 shape: {reader[0].shape}, dtype: {reader[0].dtype}")
