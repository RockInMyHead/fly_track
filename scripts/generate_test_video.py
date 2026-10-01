#!/usr/bin/env python3
"""Generate a synthetic walking video for P0 smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate(path: Path, duration: float = 10.0, fps: float = 30.0, size: int = 480) -> None:
    """Simulate forward walk + 90° right turn + forward walk."""
    n = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))

    for i in range(n):
        t = i / fps
        frame = np.ones((size, size, 3), dtype=np.uint8) * 180

        # Grid floor pattern
        grid = 40
        for g in range(0, size, grid):
            cv2.line(frame, (g, 0), (g, size), (160, 160, 160), 1)
            cv2.line(frame, (0, g), (size, g), (160, 160, 160), 1)

        # Phase 1: forward (0-4s), Phase 2: turn right (4-6s), Phase 3: forward (6-10s)
        if t < 4.0:
            offset = int(t * 30) % grid
            M = np.float32([[1, 0, 0], [0, 1, offset]])
        elif t < 6.0:
            angle = (t - 4.0) / 2.0 * 90
            center = (size // 2, size // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        else:
            offset = int((t - 6.0) * 30) % grid
            center = (size // 2, size // 2)
            R = cv2.getRotationMatrix2D(center, -90, 1.0)
            T = np.float32([[1, 0, 0], [0, 1, offset]])
            M = T @ np.vstack([R, [0, 0, 1]])[:2]

        frame = cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REFLECT)

        # Landmarks
        cv2.rectangle(frame, (50, 50), (100, 100), (80, 60, 40), -1)
        cv2.rectangle(frame, (size - 120, size - 120), (size - 60, size - 60), (60, 80, 100), -1)

        writer.write(frame)

    writer.release()
    print(f"Wrote {path} ({n} frames, {duration}s @ {fps} fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-o", type=Path, default=Path("data/test_walk.mp4"))
    p.add_argument("--duration", type=float, default=10.0)
    args = p.parse_args()
    args.o.parent.mkdir(parents=True, exist_ok=True)
    generate(args.o, duration=args.duration)
