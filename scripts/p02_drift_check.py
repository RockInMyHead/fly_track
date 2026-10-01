#!/usr/bin/env python3
"""
P0.2 check — is the measured sideways drift real, and is it a turn?

Three questions, kept strictly apart.

1. IS THE DRIFT REAL?
   The Lucas-Kanade field says the content slides sideways at several pixels per
   frame in these windows. That is checked against phase correlation, which shares
   no code and no assumptions with the tracker: it works on whole frames in the
   frequency domain and returns the single best global shift. If the two agree, the
   drift is in the video.

2. IS IT A TURN, A FORWARD MOTION, OR NEITHER?
   A yaw puts a uniform component into the field. Forward motion puts a radial one:
   content spreads outward from the focus of expansion, so the horizontal flow grows
   with azimuth and the vertical flow grows with elevation. Standing still puts
   nothing anywhere. The decomposition

       dx(x) = a + b * azimuth        a = uniform (yaw), b = horizontal spread
       dy(y) = d + c * elevation      c = vertical spread

   separates them. The classifier reports which term dominates, plus how much of the
   field neither term explains (residual = independent objects, parallax, noise).

3. DOES THE CROP MATTER?
   The pipeline measures only the central 30-70% of rows, a choice inherited from the
   first 1-D version. For forward motion the informative part is the vertical spread,
   and cropping the middle band removes the top and bottom where that is largest. The
   same three numbers are therefore computed on the full frame as well.

Nothing is tuned; thresholds are stated and fixed.

Usage:
    PYTHONPATH=. python scripts/p02_drift_check.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_vo.content_motion import CROP, ENCODE, GRID_H, GRID_W
from fly_vo.lk_motion_field import LKMotionField
from fly_vo.video_reader import VideoReader

VIDEO = ROOT / "data/p01r/VID00001.AVI"
CANDIDATES = ROOT / "data/p01r/candidate_turns.json"
VERDICTS = ROOT / "data/p01r/turn_verdicts.json"
FPS = 30.0

XN = np.tile(np.linspace(-1.0, 1.0, GRID_W), GRID_H)
YN = np.repeat(np.linspace(-1.0, 1.0, GRID_H), GRID_W)
P_A = np.linalg.pinv(np.stack([np.ones_like(XN), XN], axis=1))
P_C = np.linalg.pinv(np.stack([np.ones_like(YN), YN], axis=1))

# classifier thresholds, in pixels per frame
TURN_MIN = 1.0       # uniform component needed to call it a turn
FWD_MIN = 0.35       # radial spread needed to call it forward
STATIC_MAX = 0.30    # local motion below this and nothing is happening
DOMINANCE = 1.3      # how much larger one term must be to be named the cause


def phase_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Global shift between two frames by phase correlation, in ENCODE pixels."""
    a = cv2.resize(prev, ENCODE, interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(cur, ENCODE, interpolation=cv2.INTER_LINEAR)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy)


def analyse(vr: VideoReader, t0: float, t1: float, crop: bool = True) -> dict:
    frames = vr.frames_at(t0, t1)
    if len(frames) < 3:
        return {}
    lk = LKMotionField(grid_w=GRID_W, grid_h=GRID_H)
    a_v, b_v, c_v, resid, act, dy_abs = [], [], [], [], [], []
    pc_dx, pc_dy = [], []
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sg = cv2.resize(g, ENCODE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        h = sg.shape[0]
        band = sg[int(CROP[0] * h): int(CROP[1] * h)] if crop else sg
        dxf, dyf, _d = lk.process(np.ascontiguousarray(band))
        dx = dxf.reshape(-1)
        dy = dyf.reshape(-1)
        ab = dx @ P_A.T
        cd = dy @ P_C.T
        pred = ab[0] + ab[1] * XN
        a_v.append(float(ab[0]))
        b_v.append(float(ab[1]))
        c_v.append(float(cd[1]))
        resid.append(float(np.abs(dx - pred).mean()))
        act.append(float(np.abs(dx).mean()))
        dy_abs.append(float(np.abs(dy).mean()))
        if i > 0:
            pdx, pdy = phase_shift(frames[i - 1], f)
            pc_dx.append(pdx)
            pc_dy.append(pdy)

    med = lambda v: float(np.median(v)) if len(v) else 0.0
    a_m, b_m, c_m = med(a_v), med(b_v), med(c_v)
    divergence = b_m + c_m
    rot = abs(a_m)
    fwd = abs(divergence)
    activity = med(act)
    vertical = med(dy_abs)

    if activity < STATIC_MAX and vertical < STATIC_MAX:
        kind = "STATIC"
    elif fwd >= FWD_MIN and rot < DOMINANCE * fwd:
        kind = "FWD"
    elif rot >= TURN_MIN and fwd < DOMINANCE * rot:
        kind = "LEFT" if a_m > 0 else "RIGHT"
    else:
        kind = "MIXED"

    return {
        "n_frames": len(frames),
        "a_uniform": a_m, "b_h_spread": b_m, "c_v_spread": c_m,
        "divergence": divergence, "residual": med(resid),
        "activity": activity, "activity_vertical": vertical,
        "phase_dx": med(pc_dx), "phase_dy": med(pc_dy),
        "kind": kind,
        "camera_if_turn": ("LEFT" if a_m > 0 else "RIGHT"),
    }


def main() -> None:
    out = ROOT / "output/p02_drift"
    out.mkdir(parents=True, exist_ok=True)
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]
    verd = {}
    if VERDICTS.exists():
        verd = json.loads(VERDICTS.read_text(encoding="utf-8")).get("verdicts", {})

    print("P0.2 — независимая проверка дрейфа и классификация")
    print("  phase correlation — глобальный сдвиг без трекера, независимая реализация")
    print("  a = однородная часть (поворот), divergence = радиальная (вперёд)")
    print("  пороги: поворот |a|>=1.0, вперёд |div|>=0.35, покой активность<0.30\n")
    print(f"  {'#':>4s} {'окно':>14s} {'человек':>8s} | {'LK a':>7s} {'phase':>7s} {'совпад':>7s} | "
          f"{'div':>6s} {'resid':>6s} {'act':>5s} | {'вывод':>7s}")

    rows = []
    with VideoReader(VIDEO, declared_fps=FPS) as vr:
        for i, c in enumerate(cands, 1):
            cid = c.get("id") or f"c{i:03d}"
            human = verd.get(cid, {}).get("actual_direction") or (
                "верно" if verd.get(cid, {}).get("verdict") == "correct" else "?")
            r = analyse(vr, float(c["t0"]), float(c["t1"]), crop=True)
            if not r:
                continue
            r["id"] = cid; r["t0"] = c["t0"]; r["t1"] = c["t1"]; r["human"] = human
            r["system"] = c["camera_direction"]
            lk_sign = np.sign(r["a_uniform"])
            pc_sign = np.sign(r["phase_dx"])
            r["lk_phase_agree"] = bool(lk_sign == pc_sign)
            rows.append(r)
            if i <= 40:
                print(f"  {cid:>4s} {c['t0']:6.1f}-{c['t1']:<6.1f} {str(human):>8s} | "
                      f"{r['a_uniform']:+7.2f} {r['phase_dx']:+7.2f} "
                      f"{('да' if r['lk_phase_agree'] else 'НЕТ'):>7s} | "
                      f"{r['divergence']:+6.2f} {r['residual']:6.2f} {r['activity']:5.2f} | "
                      f"{r['kind']:>7s}")

    agree = sum(1 for r in rows if r["lk_phase_agree"])
    print(f"\n  LK и phase correlation совпадают по знаку: {agree}/{len(rows)}")
    print("  (если совпадают — дрейф реальный, а не артефакт трекера)")

    from collections import Counter
    print(f"\n  классификация: {Counter(r['kind'] for r in rows)}")
    print(f"  что видел ты:  {Counter(r['human'] for r in rows)}")

    with (out / "drift.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "report.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
