"""Frozen local-motion measurement — one definition, no threshold.

This module exists because the same event was reported with different numbers
under the same name at different points in the project: different time windows,
different frame areas, different aggregators, different frame rates and three
different estimators. Nothing in this file is tuned; it is the single definition
that all later comparison must use.

Frozen definition
-----------------
    window     exactly [start, end] as declared in the event file
    area       the central 30-70% rows of the frame (the pipeline's own crop)
    frames     native video frame rate, no resampling to the brain clock
    field      the Lucas-Kanade patch field, 16 x 8 cells
    per frame  spatial MEDIAN of the cell displacements      -> s(t)
               spatial MEAN of the cell displacements        -> m(t)
               spatial MEAN of |cell displacement|           -> e(t)
               fraction of tracking points that survived
               the forward-backward check                    -> c(t)
    per window content_dx = median over time of s(t)
               mean_dx    = mean over time of m(t)
               coherence  = |content_dx| / median over time of e(t)
               reliable   = median over time of c(t)

Sign convention
---------------
`content_dx > 0` means the image content moved toward the right edge of the
frame. For a camera yaw the content moves opposite to the camera, so

    content_dx < 0  <->  the camera rotated RIGHT
    content_dx > 0  <->  the camera rotated LEFT

No threshold is applied here, deliberately. A cut-off on coherence or magnitude
would have to be chosen from the results, and that choice must wait until the
blind labels are available. This module reports the numbers and stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .lk_motion_field import LKMotionField

# Frozen geometry.
DEFAULT_FPS = 30.0
ENCODE = (192, 108)
CROP = (0.30, 0.70)
GRID_W = 16
GRID_H = 8


@dataclass
class WindowMeasurement:
    """The frozen measurement of one window. Numbers only, no verdict."""

    event: str = ""
    t0: float = 0.0
    t1: float = 1.0
    n_frames: int = 0

    content_dx: float = 0.0        # spatial median, then median over time
    mean_dx: float = 0.0           # spatial mean, then mean over time
    coherence: float = 0.0         # |content_dx| / median_t(mean_s |dx|)
    reliable_fraction: float = 0.0  # median_t(coverage of the seed grid)
    local_activity: float = 0.0    # median_t(mean_s |dx|), px/frame

    per_frame_median: list[float] = field(default_factory=list)
    per_frame_coverage: list[float] = field(default_factory=list)

    @property
    def content_direction(self) -> str:
        return "RIGHT" if self.content_dx > 0 else ("LEFT" if self.content_dx < 0 else "NONE")

    @property
    def camera_direction(self) -> str:
        return "LEFT" if self.content_dx > 0 else ("RIGHT" if self.content_dx < 0 else "NONE")

    def as_dict(self) -> dict:
        return {
            "event": self.event, "t0": self.t0, "t1": self.t1, "n_frames": self.n_frames,
            "content_dx": self.content_dx, "mean_dx": self.mean_dx,
            "coherence": self.coherence, "reliable_fraction": self.reliable_fraction,
            "local_activity": self.local_activity,
            "content_direction": self.content_direction,
            "camera_direction_if_yaw": self.camera_direction,
        }


def read_window(video_path: str, t0: float, t1: float, fps: float = DEFAULT_FPS):
    """Native frame-rate grayscale frames for [t0, t1), resized to ENCODE.

    Reads sequentially through `VideoReader`. This used to seek with
    `CAP_PROP_POS_FRAMES`, which on this clip lands up to 222 s early because the
    file declares 45000 frames but contains 36841; see `fly_vo/video_reader.py`.
    """
    from .video_reader import VideoReader

    with VideoReader(video_path, declared_fps=fps) as vr:
        bgr = vr.frames_at(t0, t1)
    out = []
    for frame in bgr:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)
    return out


def measure_frames(grays: list[np.ndarray], event: str = "", t0: float = 0.0,
                   t1: float = 1.0) -> WindowMeasurement:
    """Apply the frozen definition to already-loaded frames."""
    m = WindowMeasurement(event=event, t0=t0, t1=t1)
    if not grays:
        return m

    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    s_vals, m_vals, e_vals, c_vals = [], [], [], []
    for g in grays:
        h = g.shape[0]
        band = np.ascontiguousarray(g[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, _dy, diag = lk.process(band)
        flat = dxf.reshape(-1)
        s_vals.append(float(np.median(flat)))
        m_vals.append(float(flat.mean()))
        e_vals.append(float(np.abs(flat).mean()))
        c_vals.append(float(diag.coverage))

    s = np.asarray(s_vals)
    e = np.asarray(e_vals)
    c = np.asarray(c_vals)
    m.n_frames = len(s)
    m.content_dx = float(np.median(s))
    m.mean_dx = float(np.mean(m_vals))
    m.local_activity = float(np.median(e))
    m.coherence = abs(m.content_dx) / max(m.local_activity, 1e-9)
    m.reliable_fraction = float(np.median(c))
    m.per_frame_median = s.tolist()
    m.per_frame_coverage = c.tolist()
    return m


def measure_window(video_path: str, t0: float, t1: float, event: str = "",
                   fps: float = DEFAULT_FPS) -> WindowMeasurement:
    """Read a window from disk and apply the frozen definition."""
    return measure_frames(read_window(video_path, t0, t1, fps), event, t0, t1)
