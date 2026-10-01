"""Fixed-rate brain clock for video → MaleCNS stepping.

Reads sequentially. The previous version sought with `CAP_PROP_POS_MSEC`, which on
this clip lands up to 222 s early because the file declares 45000 frames but contains
36841; see `fly_vo/video_reader.py`. That means every brain run to date was fed the
wrong frames.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .video_reader import VideoReader


def brain_timestamps(start_s: float, end_s: float, dt: float) -> np.ndarray:
    """Uniform brain sample times at 1/dt Hz."""
    if end_s <= start_s:
        return np.array([], dtype=np.float64)
    n = int(np.floor((end_s - start_s) / dt + 1e-9))
    return start_s + np.arange(n + 1, dtype=np.float64) * dt


def iter_video_at_brain_hz(
    video_path: str,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    brain_dt: float = 0.020,
    video_fps: float = 30.0,
) -> Iterator[tuple[float, np.ndarray]]:
    """
    Yield (t_brain, bgr_frame) at fixed brain_dt intervals.

    The video runs at `video_fps`; the brain runs at 1/brain_dt. The nearest source
    frame is chosen for each brain step, so a 30 fps clip driven at 50 Hz repeats
    frames, and that repetition is real and is reported by the caller if it matters.
    Frames are fetched by advancing the decoder, never by seeking.
    """
    if end_seconds is None:
        with VideoReader(video_path, declared_fps=video_fps) as vr:
            end_seconds = vr.duration_s

    ts = brain_timestamps(start_seconds, end_seconds, brain_dt)
    if len(ts) == 0:
        return

    first = int(np.floor(ts[0] * video_fps))
    last = int(np.ceil(ts[-1] * video_fps)) + 1

    # Streamed, never buffered: at 240x320 a full clip is 8.5 GB held in memory,
    # and the brain only ever needs the frame for the current step.
    with VideoReader(video_path, declared_fps=video_fps) as vr:
        it = vr.read_all(start=first, stop=last)
        cur: np.ndarray | None = None
        cur_idx = -1
        for t in ts:
            src = int(round(t * video_fps))
            while cur_idx < src:
                try:
                    cur_idx, cur = next(it)
                except StopIteration:
                    cur = None
                    break
            if cur is None:
                return
            yield float(t), cur
