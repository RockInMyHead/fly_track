"""Sequential video reader — the only safe way to read this clip.

Why this module exists
----------------------
`VID00001.AVI` declares 45000 frames at 30 fps (1500 s) but contains only 36841
frames (1228 s at 30 fps). OpenCV's `CAP_PROP_POS_FRAMES` seek computes a byte
offset from the *declared* frame count, so it lands early, and the error grows with
position:

    seek(200)    actually shows frame   165   (-35 frames,  -1.2 s)
    seek(1595)   actually shows frame  1307   (-288,       -9.6 s)
    seek(5000)   actually shows frame  4095   (-905,      -30.2 s)
    seek(20000)  actually shows frame 16378  (-3622,     -120.7 s)
    seek(36800)  actually shows frame 30133  (-6667,     -222.2 s)

The factor is exactly 36841 / 45000 = 0.8187.

Measured against the transcoded file that the browser plays (`ffmpeg`, which demuxes
sequentially), a sequential read matches to a mean pixel difference of 3.3 while a
seek-based read differs by 64.2. So sequential reading is the correct one, and it is
also the one that agrees with what a human sees in the browser.

Everything that used a seek — `iter_video_at_brain_hz`, `content_motion.read_window`
and every measurement built on them — was reading the wrong frames. This module
replaces them.

How to use
----------
    with VideoReader(path) as vr:
        frames = vr.frames(3180, 3230)      # frame indices, not seconds
        for i, frame in vr.read_all():      # one sequential pass
            ...

Time is `frame_index / declared_fps`, which is exactly the mapping the browser uses.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoReader:
    """Frame-indexed, sequential access to a video. Never seeks."""

    def __init__(self, path: str | Path, declared_fps: float | None = None):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self.declared_fps = float(declared_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        self._count: int | None = None
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------ basics
    def _open(self) -> cv2.VideoCapture:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                raise FileNotFoundError(f"cannot open video: {self.path}")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        """Actual number of decodable frames.

        Counted with `grab()`, which advances the decoder without converting the
        image, so it is far cheaper than counting with `read()`. Cached.
        """
        if self._count is None:
            cap = cv2.VideoCapture(self.path)
            n = 0
            while cap.grab():
                n += 1
            cap.release()
            self._count = n
        return self._count

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.declared_fps

    def t_of(self, frame_index: int) -> float:
        return frame_index / self.declared_fps

    def frame_of(self, t_seconds: float) -> int:
        return int(round(t_seconds * self.declared_fps))

    # ------------------------------------------------------------- sequential
    def read_all(self, start: int = 0, stop: int | None = None):
        """One sequential pass yielding (frame_index, bgr). No seeking."""
        cap = self._open()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= start and (stop is None or i < stop):
                yield i, frame
            i += 1
            if stop is not None and i >= stop:
                break

    def frames(self, first: int, last: int) -> list[np.ndarray]:
        """Frames [first, last). Uses grab() to skip cheaply, then decodes."""
        if last <= first:
            return []
        cap = self._open()
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos > first:
            # cannot rewind without seeking; reopen and restart
            self.close()
            cap = self._open()
            pos = 0
        while pos < first:
            if not cap.grab():
                return []
            pos += 1
        out = []
        for _ in range(last - first):
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame)
        return out

    def frames_at(self, t0: float, t1: float) -> list[np.ndarray]:
        return self.frames(self.frame_of(t0), self.frame_of(t1))


def check_reader(path: str | Path, probe_frames: tuple[int, ...] = (200, 1595, 5000)) -> dict:
    """Confirm that grab-based skipping lands on the same frame as plain reading."""
    vr = VideoReader(path)
    out = {"declared_count": vr.declared_count, "declared_fps": vr.declared_fps,
           "actual_count": vr.frame_count, "checks": []}
    for target in probe_frames:
        if target >= vr.frame_count:
            continue
        # by skip-and-read from the start
        cap = cv2.VideoCapture(str(path))
        for _ in range(target):
            cap.read()
        ok, a = cap.read()
        cap.release()
        # by grab-based skipping
        b = vr.frames(target, target + 1)
        b = b[0] if b else None
        if not ok or b is None:
            continue
        d = float(np.abs(a.astype(float) - b.astype(float)).mean())
        out["checks"].append({"frame": target, "mean_abs_diff": d, "match": d < 8.0})
    vr.close()
    return out
