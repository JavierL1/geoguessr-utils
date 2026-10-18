from pathlib import Path

import cv2


class ScoreScreenCollector:
    """Tracks score-screen transitions and writes one frame per batch.

    A "batch" begins the moment the score screen appears and ends the
    moment it disappears. We keep overwriting the last valid frame as we
    go, so when the batch ends we save the most recent one — the fully
    rendered final score.
    """

    def __init__(self, output_dir: Path, fps: float):
        self.output_dir = output_dir
        self.fps = fps
        self.batch_count = 0
        self.is_collecting = False
        self._last_frame = None
        self._last_idx: int | None = None

    def on_detected(self, frame, frame_idx: int) -> None:
        if not self.is_collecting:
            self.batch_count += 1
            self.is_collecting = True
            print(
                f"\n[ {frame_idx / self.fps:.2f}s ] "
                f"Started Batch {self.batch_count:03d}"
            )
        self._last_frame = frame.copy()
        self._last_idx = frame_idx

    def on_not_detected(self) -> None:
        if self.is_collecting:
            self._flush()
            self.is_collecting = False

    def flush(self) -> None:
        """Save any pending batch. Call at end of video."""
        if self.is_collecting:
            self._flush()

    def _flush(self) -> None:
        if self._last_frame is None or self._last_idx is None:
            return
        filename = f"batch_{self.batch_count:03d}_frame_{self._last_idx:07d}.jpg"
        cv2.imwrite(str(self.output_dir / filename), self._last_frame)
        print(
            f"\n[ {self._last_idx / self.fps:.2f}s ] "
            f"Saved Batch {self.batch_count:03d} -> {filename}"
        )
        self._last_frame = None
        self._last_idx = None
