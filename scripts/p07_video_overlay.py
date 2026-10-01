#!/usr/bin/env python3
"""
P07 step 8 — the trajectory film: the clip on the left, the route on the right.

The route is drawn as it accumulates, one point per brain step, so the moving dot on the
map corresponds to the frame being shown. Nothing is interpolated or smoothed for
display; the map is exactly `trajectory_<video>.csv`.

Usage:
    PYTHONPATH=. python scripts/p07_video_overlay.py --video VID00001
    PYTHONPATH=. python scripts/p07_video_overlay.py --video VID00002 --stride 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/p07"
FPS_OUT = 10      # every third source frame, so playback is real time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="VID00001")
    ap.add_argument("--stride", type=int, default=6,
                    help="use every Nth brain step; the brain runs at 50 Hz and the "
                         "clip at 30, so any footage faster than real time needs one")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    from fly_vo.video_reader import VideoReader

    src = ROOT / f"data/p01r/{args.video}.AVI"
    # prefer the P07.1 filtered trajectory, which advances the heading only inside
    # detected turns; fall back to the unfiltered one
    traj = OUT / f"trajectory_filtered_{args.video}.csv"
    if not traj.exists():
        traj = OUT / f"trajectory_{args.video}.csv"
    if not src.exists() or not traj.exists():
        print(f"нет {src} или {traj}")
        return

    rows = list(csv.DictReader(traj.open()))
    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    n_active = (np.array([float(r["n_active"]) for r in rows])
                if "n_active" in rows[0] else np.zeros(len(rows)))
    in_turn = (np.array([float(r["in_turn"]) for r in rows])
               if "in_turn" in rows[0] else None)

    # the map is drawn in a fixed frame with a margin, so the view never jumps
    pad = 0.08 * max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    x0, x1 = x.min() - pad, x.max() + pad
    y0, y1 = y.min() - pad, y.max() + pad
    span = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    w = args.width
    h = int(round(w * 9 / 16))
    mapw = h
    out_path = OUT / f"raw_{args.video}.mp4"
    # the writer must be told the true frame size: the clip panel plus the map panel.
    # A mismatch makes every write fail silently apart from a warning, leaving a file
    # with only a header in it.
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS_OUT, (w + mapw, h))
    if not vw.isOpened():
        print("не удалось открыть VideoWriter")
        return

    def to_px(px, py):
        u = (px - cx) / span * (mapw * 0.9) + mapw / 2
        v = (py - cy) / span * (mapw * 0.9) + h / 2
        return int(u), int(h - v)

    print(f"  {args.video}: {len(rows)} шагов траектории, шаг {args.stride}")
    # stream the clip and consume trajectory rows in step with the wall clock
    idx = 0
    written = 0
    with VideoReader(src) as vr:
        for fi, frame in vr.read_all():
            while idx + 1 < len(t) and t[idx + 1] <= fi / 30.0:
                idx += 1
            if fi % 3 != 0:
                continue
            left = cv2.resize(frame, (w, h))
            canvas = np.zeros((h, mapw, 3), np.uint8)
            canvas[:] = (18, 18, 18)
            end = min(idx + 1, len(x))
            for k in range(1, end):
                if k % args.stride:
                    continue
                cv2.line(canvas, to_px(x[k - 1], y[k - 1]), to_px(x[k], y[k]),
                         (200, 170, 60), 1, cv2.LINE_AA)
            col = ((80, 220, 255) if in_turn is not None and in_turn[end - 1] > 0
                   else (80, 80, 235))
            cv2.circle(canvas, to_px(x[end - 1], y[end - 1]), 5, col, -1)
            cv2.circle(canvas, to_px(x[0], y[0]), 5, (90, 200, 90), -1)
            cv2.putText(canvas, f"t={t[end - 1]:.0f}s  n_active={int(n_active[end - 1])}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            tag = ("TURN" if in_turn is not None and in_turn[end - 1] > 0 else "")
            cv2.putText(canvas, tag, (mapw - 70, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
            cv2.putText(canvas, "trajectory from MaleCNS outputs",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
            vw.write(np.hstack([left, canvas]))
            written += 1
    vw.release()
    print(f"  записано {written} кадров -> {out_path}")
    print(f"  (масштаб условный; ось не в метрах)")


if __name__ == "__main__":
    main()
