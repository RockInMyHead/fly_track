#!/usr/bin/env python3
"""
P0.1S — side-by-side audit of every labelled event: frames vs measurement.

Three different local-motion estimators (1-D Reichardt, 2-D Reichardt,
Lucas-Kanade) agree with each other on the sign of every one of the seven labelled
events, and the fly's 20 frozen visual cells give the same ~60% accuracy whichever
estimator feeds them. So the remaining disagreement is not an estimator quality
problem, and it has to be settled by looking at what is actually in the video.

This script produces one panel per event containing:

  frames    a strip of the actual footage every 0.25 s across the window
  a(t)      per-frame camera rotation (Lucas-Kanade field, uniform component)
  b(t)      per-frame forward-motion expansion (the azimuth slope)
  totals    accumulated rotation over the window, its z-score against the
            label-free random-walk noise, and the accumulated expansion

Reading it: a real turn is a sustained one-sided excursion of a(t). Forward
motion is a sustained positive b(t) and *no* sustained a(t). Where the frames show
a turn but a(t) sits on the other side of zero, the label or the window is wrong;
where a(t) is one-sided and the frames agree, the estimator is right.

Frontend only: no MaleCNS, no injection, nothing trained.

Usage:
    PYTHONPATH=. python scripts/p01s_disputed_review.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.brain_clock import iter_video_at_brain_hz
from fly_vo.lk_motion_field import LKMotionField

VIDEO = ROOT / "data/p01r/VID00001.AVI"
EVENTS = {
    "left_65":   (63.5, "LEFT"),
    "left_81":   (78.5, "LEFT"),
    "left_108":  (106.0, "LEFT"),
    "right_67":  (66.0, "RIGHT"),
    "right_210": (209.0, "RIGHT"),
    "right_216": (214.0, "RIGHT"),
    "fwd_140":   (137.5, "FWD"),
}
WANT = {"LEFT": -1, "RIGHT": +1, "FWD": 0}
PRE_ROLL_S = 1.5
EVENT_DUR_S = 3.0
DT = 0.02
GRID_W, GRID_H = 16, 8
XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
PINV = np.linalg.pinv(np.stack([np.ones_like(XN), XN], axis=1))
CROP = (0.30, 0.70)          # the frozen pipeline's sensor band
FRAME_STEP_S = 0.25
PLOT_W = 1400


def frame_strip(video: Path, t0: float, t1: float, step_s: float, width: int) -> np.ndarray:
    imgs, times = [], []
    t = t0
    while t < t1:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        if ok:
            imgs.append(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)))
            times.append(t)
        t += step_s
    if not imgs:
        return np.zeros((10, width, 3), np.uint8)
    h = max(i.shape[0] for i in imgs)
    total_w = sum(i.shape[1] for i in imgs) + 3 * (len(imgs) - 1)
    canvas = np.full((h + 20, total_w, 3), 250, np.uint8)
    x = 0
    for tt, im in zip(times, imgs):
        canvas[20:20 + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, f"{tt:.2f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        x += im.shape[1] + 3
    scale = width / canvas.shape[1]
    return cv2.resize(canvas, (width, int(canvas.shape[0] * scale)))


def trace_panel(t: np.ndarray, a: np.ndarray, b: np.ndarray, want: int,
                t0: float, height: int = 300, width: int = PLOT_W) -> np.ndarray:
    img = np.full((height, width, 3), 255, np.uint8)
    pad_l, pad_r, pad_t, pad_b = 78, 20, 40, 34
    xw, yh = width - pad_l - pad_r, height - pad_t - pad_b
    mid = pad_t + yh // 2
    lim = max(float(np.percentile(np.abs(a), 98)), 1e-6)

    def px(i: int) -> int:
        return pad_l + int(i / max(len(t) - 1, 1) * xw)

    def py(v: float) -> int:
        return int(mid - np.clip(v / lim, -1, 1) * (yh / 2 - 4))

    x_ev = px(int(round(PRE_ROLL_S / DT)))
    cv2.rectangle(img, (pad_l, pad_t), (x_ev, pad_t + yh), (244, 244, 232), -1)
    cv2.line(img, (pad_l, mid), (pad_l + xw, mid), (160, 160, 160), 1)
    cv2.line(img, (x_ev, pad_t), (x_ev, pad_t + yh), (110, 110, 110), 1)
    cv2.putText(img, "pre-roll", (pad_l + 3, pad_t + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (150, 150, 150), 1)
    cv2.putText(img, "event window", (x_ev + 3, pad_t + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (80, 80, 80), 1)
    cv2.putText(img, "LEFT (-)", (pad_l - 66, pad_t + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (60, 60, 200), 1)
    cv2.putText(img, "RIGHT (+)", (pad_l - 66, pad_t + yh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 90, 60), 1)

    for i in range(len(t) - 1):
        if want:
            c = (70, 160, 70) if np.sign(a[i]) == want else (60, 60, 220)
        else:
            c = (150, 150, 150)
        cv2.line(img, (px(i), py(a[i])), (px(i + 1), py(a[i + 1])), c, 1)
    # expansion drawn to the same scale but thinner, dashed look via lighter colour
    lim_b = max(float(np.percentile(np.abs(b), 98)), 1e-6)
    for i in range(len(t) - 1):
        y1 = int(mid - np.clip(b[i] / lim_b, -1, 1) * (yh / 2 - 4))
        y2 = int(mid - np.clip(b[i + 1] / lim_b, -1, 1) * (yh / 2 - 4))
        if i % 2 == 0:
            cv2.line(img, (px(i), y1), (px(i + 1), y2), (120, 190, 120), 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("-o", "--output", default="output/p01s_audit")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("P0.1S — labelled events, frames next to the measurement")
    print(f"  rotation = uniform component a of dx(x,y) = a + b*azimuth, "
          f"crop {CROP[0]:.0%}-{CROP[1]:.0%} (frozen band)")

    summary = []
    for eid, (start, lab) in EVENTS.items():
        want = WANT[lab]
        t0 = max(0.0, start - PRE_ROLL_S)
        t1 = start + EVENT_DUR_S
        lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
        ts, a_vals, b_vals = [], [], []
        for t, frame in iter_video_at_brain_hz(args.video, t0, t1, DT):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            h = img.shape[0]
            band = np.ascontiguousarray(img[int(CROP[0] * h): int(CROP[1] * h)])
            dxf, _dy, _d = lk.process(band)
            ab = dxf.reshape(-1) @ PINV.T
            ts.append(t)
            a_vals.append(float(ab[0]))
            b_vals.append(float(ab[1]))
        t = np.asarray(ts)
        a = np.asarray(a_vals)
        b = np.asarray(b_vals)

        n_pre = int(round(PRE_ROLL_S / DT))
        a_ev = a[n_pre:]
        k = 9
        pad = k // 2
        sm = np.array([np.median(a_ev[max(0, i - pad): i + pad + 1]) for i in range(len(a_ev))])
        sigma = float(1.4826 * np.median(np.abs(a_ev - sm))) if len(a_ev) else 0.0
        total = float(a_ev.sum())
        z = total / max(sigma * np.sqrt(len(a_ev)), 1e-9)
        total_b = float(b[n_pre:].sum())

        strip = frame_strip(Path(args.video), t0, t1, FRAME_STEP_S, PLOT_W)
        panel = trace_panel(t, a, b, want, t0)
        head = np.full((46, PLOT_W, 3), 255, np.uint8)
        ok_txt = "sign matches label" if (want and np.sign(total) == want) else (
            "SIGN OPPOSITE TO LABEL" if want else "n/a (forward window)")
        col = (60, 140, 60) if (want and np.sign(total) == want) else (
            (40, 40, 210) if want else (120, 120, 120))
        cv2.putText(head, f"{eid}   blind label {lab}   window t={t0:.1f}-{t1:.1f}s   "
                         f"{ok_txt}",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(head, f"accumulated rotation {total:+.1f} px  (z = {z:+.1f} vs label-free "
                         f"noise, sigma {sigma:.2f} px/frame, N = {len(a_ev)})    "
                         f"accumulated expansion {total_b:+.1f}    "
                         f"green line = a(t), light green = b(t)",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
        combo = np.vstack([head, panel, np.full((4, PLOT_W, 3), 255, np.uint8), strip])
        cv2.imwrite(str(out / f"{eid}_{lab}.png"), combo)

        summary.append({
            "event": eid, "label": lab, "window": [t0, t1],
            "accumulated_rotation_px": total, "z": z, "sigma_px_per_frame": sigma,
            "n_frames": len(a_ev), "accumulated_expansion": total_b,
            "sign_matches_label": bool(want and np.sign(total) == want),
        })
        print(f"  {eid:10s} {lab:5s} accumulated rotation {total:+8.1f} px  z {z:+7.1f}  "
              f"expansion {total_b:+8.1f}  "
              f"{'match' if summary[-1]['sign_matches_label'] else 'opposite'}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/  (one PNG per event: frames on top of the a(t)/b(t) trace)")


if __name__ == "__main__":
    main()
