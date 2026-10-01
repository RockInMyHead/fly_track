"""Browser-friendly video preview helpers (frame JPEG extraction)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe_codec(path: str | Path) -> str:
    """Return video codec name via ffprobe, or empty string."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or "").strip().lower()
    except FileNotFoundError:
        return ""


def is_browser_playable(path: str | Path) -> bool:
    """True if the file is likely playable in HTML5 <video> without transcoding."""
    ext = Path(path).suffix.lower()
    if ext in {".webm", ".m4v"}:
        return True
    if ext == ".mp4":
        codec = probe_codec(path)
        return codec in {"", "h264", "avc1", "hev1", "h265"}
    return False


def encode_preview_jpeg(bgr: np.ndarray, max_width: int = 960, quality: int = 78) -> bytes:
    """Resize BGR frame and return JPEG bytes."""
    h, w = bgr.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        bgr = cv2.resize(bgr, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def extract_frame_jpeg(path: str | Path, t_seconds: float, max_width: int = 960) -> bytes:
    """Seek to timestamp and return a preview JPEG."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_seconds) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise ValueError(f"No frame at t={t_seconds}s in {path}")
        return encode_preview_jpeg(frame, max_width=max_width)
    finally:
        cap.release()
