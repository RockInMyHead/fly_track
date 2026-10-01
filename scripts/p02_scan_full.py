#!/usr/bin/env python3
"""
P0.2 — label the whole clip as turn / forward / stationary / mixed.

The first scan could only ever return yaw candidates: it ranked frames by sideways
signal strength and kept the top 3%, so "forward" and "standing still" were excluded
by construction. This scan classifies every moment instead, using the whole
decomposition of the motion field rather than one component of it.

Per frame, over the pipeline's crop:

    dx(x) = a + b * azimuth        a = uniform, the yaw-like part
    dy(y) = d + c * elevation      c = vertical spread
    divergence = b + c             the radial part, forward motion

    a large, divergence small          -> the camera rotated      LEFT / RIGHT
    divergence large, a small          -> the camera moved ahead  FWD
    nothing large anywhere             -> nothing is happening    STATIC
    neither dominates, residual large  -> MIXED

`residual` is what neither term explains: independently moving objects, parallax,
a person crossing the view. A window with a large residual is not a clean ego-motion
window even when one term dominates, so it is reported rather than discarded.

The output is a labelled timeline. Every window of every class goes to the review
page, so forward and stationary moments can be confirmed by eye exactly like turns.

Reading is sequential through `VideoReader`: seeking on this file lands up to 222 s
early (see fly_vo/video_reader.py).

Usage:
    PYTHONPATH=. python scripts/p02_scan_full.py            # scan + label
    PYTHONPATH=. python scripts/p02_scan_full.py --only label
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from fly_vo.content_motion import CROP, ENCODE, GRID_H, GRID_W
from fly_vo.lk_motion_field import LKMotionField
from fly_vo.video_reader import VideoReader

VIDEO = ROOT / "data/p01r/VID00001.AVI"
CACHE = ROOT / "output/p02_full/full_signal.npz"
OUT_JSON = ROOT / "data/p01r/windows_labelled.json"
FPS = 30.0

XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
YN = np.repeat(np.linspace(-1.0, 1.0, GRID_H), GRID_W)
P_A = np.linalg.pinv(np.stack([np.ones_like(XN), XN], axis=1))
P_C = np.linalg.pinv(np.stack([np.ones_like(YN), YN], axis=1))

SMOOTH_S = 0.8
TURN_MIN = 1.0        # |a| needed to call a rotation
FWD_MIN = 0.35        # |divergence| needed to call forward motion
STATIC_MAX = 0.35     # local motion below this = nothing happening
DOMINANCE = 1.3       # how much larger the winning term must be
MIN_SEG_S = 0.8


def scan(video: Path) -> dict:
    vr = VideoReader(video, declared_fps=FPS)
    n = vr.frame_count
    print(f"  frames: {n} ({vr.duration_s:.0f} s at {FPS:.0f} fps)")
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    A, B, C, RES, AX, AY = [], [], [], [], [], []
    t0 = time.time()
    for i, frame in vr.read_all():
        g = frame  # BGR
        import cv2
        gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        sg = cv2.resize(gray, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        h = sg.shape[0]
        band = np.ascontiguousarray(sg[int(CROP[0] * h): int(CROP[1] * h)])
        dxf, dyf, _d = lk.process(band)
        dx = dxf.reshape(-1); dy = dyf.reshape(-1)
        ab = dx @ P_A.T
        cd = dy @ P_C.T
        pred = ab[0] + ab[1] * XN
        A.append(float(ab[0])); B.append(float(ab[1])); C.append(float(cd[1]))
        RES.append(float(np.abs(dx - pred).mean()))
        AX.append(float(np.abs(dx).mean())); AY.append(float(np.abs(dy).mean()))
        if i % 3000 == 0 and i:
            el = time.time() - t0
            print(f"    {i}/{n}  {el:.0f}s  {i / max(el, 1e-9):.0f} fps")
    vr.close()
    return {"a": np.array(A), "b": np.array(B), "c": np.array(C),
            "residual": np.array(RES), "act_x": np.array(AX), "act_y": np.array(AY),
            "t": np.arange(len(A), dtype=np.float32) / FPS}


def box(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def classify(a: np.ndarray, div: np.ndarray, act: np.ndarray,
             act_y: np.ndarray) -> np.ndarray:
    kind = np.full(len(a), "MIXED", dtype=object)
    rot = np.abs(a)
    fwd = np.abs(div)
    still = (act < STATIC_MAX) & (act_y < STATIC_MAX)
    ahead = (fwd >= FWD_MIN) & (rot < DOMINANCE * fwd)
    turn_l = (rot >= TURN_MIN) & (a > 0) & (fwd < DOMINANCE * rot)
    turn_r = (rot >= TURN_MIN) & (a < 0) & (fwd < DOMINANCE * rot)
    kind[still] = "STATIC"
    kind[ahead] = "FWD"
    kind[turn_l] = "LEFT"
    kind[turn_r] = "RIGHT"
    return kind


def label(sig: dict) -> list[dict]:
    n_sm = max(int(round(SMOOTH_S * FPS)), 1)
    a = box(sig["a"], n_sm)
    div = box(sig["b"] + sig["c"], n_sm)
    act = box(sig["act_x"], n_sm)
    act_y = box(sig["act_y"], n_sm)
    res = box(sig["residual"], n_sm)
    kind = classify(a, div, act, act_y)
    t = sig["t"]

    segs = []
    i = 0
    while i < len(kind):
        j = i
        while j + 1 < len(kind) and kind[j + 1] == kind[i]:
            j += 1
        dur = (j - i + 1) / FPS
        if dur >= MIN_SEG_S:
            k = kind[i]
            seg = {
                "kind": k, "t0": round(float(t[i]), 2), "t1": round(float(t[j]), 2),
                "duration_s": round(dur, 2),
                "a": round(float(np.median(a[i:j + 1])), 3),
                "divergence": round(float(np.median(div[i:j + 1])), 3),
                "residual": round(float(np.median(res[i:j + 1])), 3),
                "activity": round(float(np.median(act[i:j + 1])), 3),
                "activity_vertical": round(float(np.median(act_y[i:j + 1])), 3),
            }
            if k in ("LEFT", "RIGHT"):
                seg["camera_direction"] = k
                seg["confidence"] = ("high" if abs(seg["a"]) >= 3.0 else
                                     "medium" if abs(seg["a"]) >= 1.5 else "low")
            elif k == "FWD":
                seg["camera_direction"] = "FWD"
                seg["confidence"] = ("high" if abs(seg["divergence"]) >= 0.8 else "medium")
            elif k == "STATIC":
                seg["camera_direction"] = "STATIC"
                seg["confidence"] = "high"
            else:
                seg["camera_direction"] = "UNCLEAR"
                seg["confidence"] = "low"
            seg["id"] = f"w{len(segs) + 1:04d}"
            segs.append(seg)
        i = j + 1
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["scan", "label", "all"], default="all")
    args = ap.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    if args.only in ("scan", "all") or not CACHE.exists():
        print("полный проход по видео (последовательное чтение, без seek)")
        sig = scan(Path(VIDEO))
        np.savez_compressed(CACHE, **sig)
        print(f"  кэш -> {CACHE}")
    else:
        d = np.load(CACHE)
        sig = {k: d[k] for k in d.files}
        print(f"кэш: {len(sig['t'])} кадров")

    segs = label(sig)
    from collections import Counter
    cnt = Counter(s["kind"] for s in segs)
    total = sum(s["duration_s"] for s in segs)
    print(f"\nразмечено окон: {len(segs)}, покрытие {total:.0f} с")
    print(f"  {dict(cnt)}")
    print(f"\n  {'#':>5s} {'класс':>7s} {'окно, с':>14s} {'длит':>6s} {'a':>7s} "
          f"{'div':>6s} {'resid':>6s} {'actX':>5s} {'actY':>5s} {'увер':>7s}")
    for s in segs:
        print(f"  {s['id']:>5s} {s['kind']:>7s} {s['t0']:8.1f}-{s['t1']:<8.1f} "
              f"{s['duration_s']:6.2f} {s['a']:+7.2f} {s['divergence']:+6.2f} "
              f"{s['residual']:6.2f} {s['activity']:5.2f} {s['activity_vertical']:5.2f} "
              f"{s['confidence']:>7s}")

    OUT_JSON.write_text(json.dumps({
        "source": str(VIDEO), "frames": int(len(sig["t"])), "fps": FPS,
        "note": "Классы размечены системой. Меткой становится только то, что подтвердил "
                "человек. Знак: a>0 значит содержимое едет вправо, то есть камера влево.",
        "thresholds": {"turn_min": TURN_MIN, "fwd_min": FWD_MIN,
                       "static_max": STATIC_MAX, "dominance": DOMINANCE,
                       "smooth_s": SMOOTH_S, "min_seg_s": MIN_SEG_S},
        "n_windows": len(segs), "counts": dict(cnt),
        "windows": segs,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    with (ROOT / "output/p02_full/windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(segs[0].keys()))
        w.writeheader(); w.writerows(segs)
    print(f"\nWrote {OUT_JSON} ({len(segs)} окон)")


if __name__ == "__main__":
    main()
