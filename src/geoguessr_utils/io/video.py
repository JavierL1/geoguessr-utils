from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class VideoReader:
    """Random-access frame reader backed by PyAV.

    Exposes the small subset of ``cv2.VideoCapture`` that the scanner
    needs — ``fps``, ``total_frames``, and ``read_frame(idx)`` — so the
    calling code stays the same.

    Seeking is done via PTS: we jump to the keyframe at or before the
    target time, then decode forward until we reach the requested frame.
    If the previous read already advanced the decoder to (or past) the
    target, we skip the seek and keep decoding forward, which is much
    cheaper than re-seeking for small step sizes.
    """

    def __init__(self, path: Path) -> None:
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        # ``average_rate`` is usually populated for MP4/WebM; fall back
        # to ``guessed_rate``/``base_rate`` for odd containers.
        rate = (
            self._stream.average_rate
            or self._stream.guessed_rate
            or self._stream.base_rate
        )
        self.fps: float = float(rate) if rate else 0.0

        tb = self._stream.time_base
        self._time_base: Fraction | None = (
            Fraction(tb.numerator, tb.denominator) if tb is not None else None
        )

        # ``stream.frames`` is often 0 in container metadata for
        # streamed sources; fall back to duration * fps.
        if self._stream.frames and self._stream.frames > 0:
            self.total_frames: int = int(self._stream.frames)
        elif self._container.duration:
            self.total_frames = int(self._container.duration / av.time_base * self.fps)
        else:
            self.total_frames = 0

        # State for the "decode forward instead of seeking" optimization.
        self._frame_iter = None
        self._last_seconds: float = -1.0

    def _seek_seconds(self, seconds: float) -> None:
        """Jump to the keyframe at or before ``seconds``."""
        if self._time_base is None:
            self._container.seek(int(seconds * av.time_base), backward=True)
        else:
            self._container.seek(
                int(seconds / self._time_base),
                stream=self._stream,
                backward=True,
                any_frame=False,
            )

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """Return the frame at ``frame_idx`` as a BGR ``np.ndarray``.

        Returns ``None`` if the frame cannot be decoded (EOF, invalid
        index, or a stream with no FPS metadata).
        """
        if self.fps <= 0:
            return None

        target_seconds = frame_idx / self.fps
        tolerance = 0.5 / self.fps  # half a frame of slack for VFR rounding

        # Fast path: continue decoding from where we left off if the
        # target is ahead of us. Avoids paying for a seek on every
        # 0.25s collect step.
        if self._frame_iter is not None and self._last_seconds <= target_seconds:
            for frame in self._frame_iter:
                if frame.time is None:
                    continue
                if frame.time + tolerance >= target_seconds:
                    self._last_seconds = frame.time
                    return frame.to_ndarray(format="bgr24")
            # Generator exhausted; fall through to a seek.
            self._frame_iter = None

        # Slow path: seek and decode forward from the keyframe.
        self._seek_seconds(target_seconds)
        self._frame_iter = self._container.decode(video=0)
        for frame in self._frame_iter:
            if frame.time is None:
                continue
            if frame.time + tolerance >= target_seconds:
                self._last_seconds = frame.time
                return frame.to_ndarray(format="bgr24")

        self._frame_iter = None
        return None

    def release(self) -> None:
        self._frame_iter = None
        self._container.close()
