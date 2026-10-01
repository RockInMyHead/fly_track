"""Load and preprocess camera video for FlyVis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoData:
    """Grayscale video frames resampled to simulation rate."""

    frames: np.ndarray  # (T, H, W), float32 in [0, 1]
    timestamps: np.ndarray  # (T,), seconds
    source_fps: float
    sim_hz: float


def get_video_info(path: str) -> tuple[float, float, int, int]:
    """Return (duration_sec, fps, width, height)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if n_frames > 0 else 0.0
    cap.release()
    return duration, fps, width, height


def _resize_gray(gray: np.ndarray, max_size: int) -> np.ndarray:
    if max_size and max(gray.shape) > max_size:
        scale = max_size / max(gray.shape)
        return cv2.resize(
            gray,
            (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return gray


def load_video(
    path: str,
    sim_hz: float = 50.0,
    max_seconds: float | None = None,
    max_size: int = 480,
    start_seconds: float = 0.0,
) -> VideoData:
    """Load video segment, convert to grayscale, optionally resize, resample to sim_hz."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

    raw_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = _resize_gray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            max_size,
        )
        raw_frames.append(gray)
        elapsed = len(raw_frames) / source_fps
        if max_seconds is not None and elapsed >= max_seconds:
            break
    cap.release()

    if not raw_frames:
        raise ValueError(f"No frames read from {path} at t={start_seconds}s")

    stack = np.stack(raw_frames, axis=0)
    n_source = stack.shape[0]
    duration = n_source / source_fps
    n_target = max(2, int(round(duration * sim_hz)))
    target_times = start_seconds + np.linspace(0.0, duration, n_target, endpoint=False)

    indices = np.clip(
        np.round(np.linspace(0, n_source - 1, n_target)).astype(int), 0, n_source - 1
    )
    resampled = stack[indices]

    return VideoData(
        frames=resampled,
        timestamps=target_times.astype(np.float32),
        source_fps=source_fps,
        sim_hz=sim_hz,
    )
