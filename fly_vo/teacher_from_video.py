"""Estimate a teacher trajectory from camera motion (OpenCV proxy when no user drawing)."""

from __future__ import annotations

import cv2
import numpy as np

from .dopamine import _resample_polyline
from .pfn import body_to_world


def estimate_teacher_from_video(
    video_path: str,
    *,
    heading_deg: float = 90.0,
    n_points: int = 64,
    max_size: int = 480,
    forward_gain: float = 14.0,
    yaw_gain: float = 2.8,
) -> list[tuple[float, float]]:
    """
    Integrate coarse visual odometry: frame diff → forward, phase shift → yaw.
    Returns polyline in the same world frame as the fly UI (heading 90° = up).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    x, y = 0.0, 0.0
    heading = np.radians(heading_deg)
    prev_gray: np.ndarray | None = None
    prev_pan: np.ndarray | None = None
    path = [(x, y)]
    last_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        dt = max(t - last_t, 1.0 / fps) if last_t > 0 else 1.0 / fps
        last_t = t

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        g = gray.astype(np.float32) / 255.0
        motion = 0.0
        pan_shift = 0.0
        if prev_gray is not None:
            motion = float(np.mean(np.abs(g - prev_gray)))
            pan = g.mean(axis=0)
            if prev_pan is not None and len(pan) == len(prev_pan):
                try:
                    (dx, _), _ = cv2.phaseCorrelate(
                        pan.reshape(1, -1).astype(np.float32),
                        prev_pan.reshape(1, -1).astype(np.float32),
                    )
                    pan_shift = float(dx) / max(len(pan), 1)
                except cv2.error:
                    pan_shift = 0.0
            prev_pan = pan

        if prev_gray is not None:
            vx_body = forward_gain * motion
            yaw_rate = -yaw_gain * pan_shift
            heading += yaw_rate * dt
            vx_w, vy_w = body_to_world(
                np.array([vx_body]),
                np.array([0.0]),
                np.array([heading]),
            )
            x += float(vx_w[0] * dt)
            y += float(vy_w[0] * dt)

        path.append((x, y))
        prev_gray = g
        if prev_pan is None:
            prev_pan = g.mean(axis=0)

    cap.release()

    if len(path) < 2:
        return [(0.0, 0.0), (0.0, 1.0)]

    arr = _resample_polyline(np.asarray(path, dtype=np.float64), n_points)
    return [(float(x), float(y)) for x, y in arr]
